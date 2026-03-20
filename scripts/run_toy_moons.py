#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.datasets import make_moons
except ModuleNotFoundError as exc:  # pragma: no cover - dependency error path
    raise SystemExit(
        f"Missing dependency: {exc}. Install requirements with `pip install -r requirements.txt`."
    ) from exc

from mmd_gan_experiments.kernels import build_kernel
from mmd_gan_experiments.mmd import mmd2_unbiased, witness_values
from mmd_gan_experiments.models_toy import ToyFeatureCritic, ToyGenerator
from mmd_gan_experiments.utils import (
    device_summary,
    ensure_dir,
    grad_penalty_features,
    pick_device,
    save_json,
    seed_everything,
    timestamp,
)


try:
    from tqdm.auto import trange
except Exception:  # pragma: no cover
    trange = range


@dataclass
class ToyRunConfig:
    kernel: str
    use_gp: bool
    gp_lambda: float
    steps: int
    batch_size: int
    critic_steps: int
    lr_g: float
    lr_c: float
    z_dim: int
    hidden_dim: int
    feat_dim: int
    num_real_samples: int
    noise: float
    seed: int
    device: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Toy make_moons MMD-GAN experiment")
    p.add_argument("--mode", choices=["grid", "single"], default="grid")
    p.add_argument("--kernel", choices=["linear", "rbf", "rq"], default="rq")
    p.add_argument("--gp", action="store_true", help="Use gradient penalty (single mode)")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--critic-steps", type=int, default=5)
    p.add_argument("--lr-g", type=float, default=1e-4)
    p.add_argument("--lr-c", type=float, default=1e-4)
    p.add_argument("--z-dim", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--feat-dim", type=int, default=16)
    p.add_argument("--gp-lambda", type=float, default=1.0)
    p.add_argument("--num-real-samples", type=int, default=10000)
    p.add_argument("--noise", type=float, default=0.06)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--witness-grid", type=int, default=140)
    p.add_argument("--outdir", type=str, default="results/toy_moons")
    return p.parse_args()


def sample_real_batch(real_data: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    idx = torch.randint(0, real_data.shape[0], (batch_size,), device=real_data.device)
    return real_data[idx].to(device)


def train_one(
    cfg: ToyRunConfig,
    real_data: torch.Tensor,
    outdir: Path,
    log_every: int,
    witness_grid: int,
) -> Dict:
    device = torch.device(cfg.device)
    kernel = build_kernel(cfg.kernel)

    G = ToyGenerator(z_dim=cfg.z_dim, hidden_dim=cfg.hidden_dim).to(device)
    C = ToyFeatureCritic(hidden_dim=cfg.hidden_dim, feat_dim=cfg.feat_dim).to(device)

    opt_g = torch.optim.Adam(G.parameters(), lr=cfg.lr_g, betas=(0.5, 0.9))
    opt_c = torch.optim.Adam(C.parameters(), lr=cfg.lr_c, betas=(0.5, 0.9))

    hist = {"step": [], "mmd2": [], "loss_g": [], "loss_c": [], "gp": []}

    iterator = trange(1, cfg.steps + 1, desc=f"kernel={cfg.kernel},gp={cfg.use_gp}")
    for step in iterator:
        for _ in range(cfg.critic_steps):
            real = sample_real_batch(real_data, cfg.batch_size, device)
            z = torch.randn(cfg.batch_size, cfg.z_dim, device=device)
            fake = G(z).detach()

            f_real = C(real)
            f_fake = C(fake)
            mmd2 = mmd2_unbiased(f_real, f_fake, kernel)
            gp = (
                grad_penalty_features(C, real, fake, cfg.gp_lambda)
                if cfg.use_gp
                else torch.zeros((), device=device)
            )
            loss_c = -mmd2 + gp

            opt_c.zero_grad(set_to_none=True)
            loss_c.backward()
            opt_c.step()

        real = sample_real_batch(real_data, cfg.batch_size, device)
        z = torch.randn(cfg.batch_size, cfg.z_dim, device=device)
        fake = G(z)
        loss_g = mmd2_unbiased(C(real), C(fake), kernel)

        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        opt_g.step()

        if step % log_every == 0 or step == cfg.steps:
            with torch.no_grad():
                eval_real = sample_real_batch(real_data, cfg.batch_size, device)
                eval_fake = G(torch.randn(cfg.batch_size, cfg.z_dim, device=device))
                eval_mmd2 = mmd2_unbiased(C(eval_real), C(eval_fake), kernel)
            hist["step"].append(step)
            hist["mmd2"].append(float(eval_mmd2.detach().cpu()))
            hist["loss_g"].append(float(loss_g.detach().cpu()))
            hist["loss_c"].append(float(loss_c.detach().cpu()))
            hist["gp"].append(float(gp.detach().cpu()))
            if hasattr(iterator, "set_postfix"):
                iterator.set_postfix(mmd2=f"{hist['mmd2'][-1]:.4f}")

    with torch.no_grad():
        fake_all = G(torch.randn(real_data.shape[0], cfg.z_dim, device=device)).cpu().numpy()
        real_np = real_data.cpu().numpy()

    x_min, x_max = real_np[:, 0].min() - 0.5, real_np[:, 0].max() + 0.5
    y_min, y_max = real_np[:, 1].min() - 0.5, real_np[:, 1].max() + 0.5

    gx = np.linspace(x_min, x_max, witness_grid)
    gy = np.linspace(y_min, y_max, witness_grid)
    xx, yy = np.meshgrid(gx, gy)
    grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)

    with torch.no_grad():
        grid_t = torch.from_numpy(grid_points).float().to(device)
        real_t = torch.from_numpy(real_np[: min(2000, len(real_np))]).float().to(device)
        fake_t = torch.from_numpy(fake_all[: min(2000, len(fake_all))]).float().to(device)
        w = witness_values(C(grid_t), C(real_t), C(fake_t), kernel).cpu().numpy().reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(real_np[:, 0], real_np[:, 1], s=4, alpha=0.3, label="real")
    axes[0].scatter(fake_all[:, 0], fake_all[:, 1], s=4, alpha=0.3, label="generated")
    axes[0].set_title(f"Samples | kernel={cfg.kernel} | GP={cfg.use_gp}")
    axes[0].legend(loc="best")

    contour = axes[1].contourf(xx, yy, w, levels=80, cmap="coolwarm")
    axes[1].scatter(real_np[::20, 0], real_np[::20, 1], s=3, c="k", alpha=0.15)
    axes[1].set_title("Witness function")
    fig.colorbar(contour, ax=axes[1], shrink=0.8)

    fig.tight_layout()
    fig_path = outdir / f"toy_kernel-{cfg.kernel}_gp-{int(cfg.use_gp)}.png"
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    ckpt_path = outdir / f"toy_kernel-{cfg.kernel}_gp-{int(cfg.use_gp)}.pt"
    torch.save(
        {
            "config": asdict(cfg),
            "generator": G.state_dict(),
            "critic": C.state_dict(),
            "history": hist,
        },
        ckpt_path,
    )

    return {
        "config": asdict(cfg),
        "history": hist,
        "figure": str(fig_path),
        "checkpoint": str(ckpt_path),
        "final_mmd2": hist["mmd2"][-1],
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = pick_device(prefer_cuda=True)

    run_root = ensure_dir(Path(args.outdir) / timestamp())

    moons_x, _ = make_moons(n_samples=args.num_real_samples, noise=args.noise, random_state=args.seed)
    real_data = torch.from_numpy(moons_x).float()

    print(f"[info] device: {device_summary(device)}")
    print(f"[info] outputs: {run_root}")

    if args.mode == "grid":
        run_specs: List[Tuple[str, bool]] = [(k, gp) for gp in (False, True) for k in ("linear", "rbf", "rq")]
    else:
        run_specs = [(args.kernel, bool(args.gp))]

    results: List[Dict] = []
    for kernel_name, use_gp in run_specs:
        cfg = ToyRunConfig(
            kernel=kernel_name,
            use_gp=use_gp,
            gp_lambda=args.gp_lambda,
            steps=args.steps,
            batch_size=args.batch_size,
            critic_steps=args.critic_steps,
            lr_g=args.lr_g,
            lr_c=args.lr_c,
            z_dim=args.z_dim,
            hidden_dim=args.hidden_dim,
            feat_dim=args.feat_dim,
            num_real_samples=args.num_real_samples,
            noise=args.noise,
            seed=args.seed,
            device=str(device),
        )
        out = train_one(
            cfg=cfg,
            real_data=real_data,
            outdir=run_root,
            log_every=args.log_every,
            witness_grid=args.witness_grid,
        )
        results.append(out)

    summary = {
        "device": device_summary(device),
        "args": vars(args),
        "results": sorted(results, key=lambda d: (d["config"]["use_gp"], d["config"]["kernel"])),
    }
    save_json(summary, run_root / "summary.json")

    print("\n[summary] final unbiased MMD^2")
    for r in summary["results"]:
        c = r["config"]
        print(f"  kernel={c['kernel']:>6} | gp={int(c['use_gp'])} | final_mmd2={r['final_mmd2']:.6f}")


if __name__ == "__main__":
    main()
