#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Sequence, Tuple

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
from mmd_gan_experiments.protocol_v2_helpers import (
    SmallToyFeatureCritic,
    SmallToyGenerator,
    mean_feature_norm,
    witness_gradient_penalty,
)
from mmd_gan_experiments.utils import device_summary, ensure_dir, pick_device, save_json, seed_everything, timestamp

try:
    from tqdm.auto import trange
except Exception:  # pragma: no cover
    trange = range


TOY_RBF_SIGMAS = (0.1, 0.2, 0.5, 1.0, 2.0)
RQ_ALPHAS = (0.2, 0.5, 1.0, 2.0, 5.0)


@dataclass
class ToyProtocolConfig:
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
    train_size: int
    val_size: int
    noise: float
    seed: int
    device: str
    log_every: int
    sample_every: int
    sample_points: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Protocol-oriented make_moons MMD-GAN experiment (v2)")
    p.add_argument("--mode", choices=["suite", "single"], default="suite")
    p.add_argument("--kernel", choices=["linear", "rbf", "rq", "rq_linear"], default="rq")
    p.add_argument("--gp", action="store_true", help="Use witness GP in single mode")
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--critic-steps", type=int, default=5)
    p.add_argument("--lr-g", type=float, default=1e-3)
    p.add_argument("--lr-c", type=float, default=1e-3)
    p.add_argument("--z-dim", type=int, default=4)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--feat-dim", type=int, default=8)
    p.add_argument("--gp-lambda", type=float, default=1.0)
    p.add_argument("--train-size", type=int, default=10_000)
    p.add_argument("--val-size", type=int, default=2_000)
    p.add_argument("--noise", type=float, default=0.08)
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--sample-every", type=int, default=1_000)
    p.add_argument("--sample-points", type=int, default=2_000)
    p.add_argument("--witness-grid", type=int, default=160)
    p.add_argument("--outdir", type=str, default="results/toy_moons_v2")
    return p.parse_args()


def build_toy_kernel(name: str):
    lname = name.lower()
    if lname == "linear":
        return build_kernel("linear")
    if lname == "rbf":
        return build_kernel("rbf", rbf_sigmas=TOY_RBF_SIGMAS)
    if lname == "rq":
        return build_kernel("rq", rq_alphas=RQ_ALPHAS, rq_add_linear=False)
    if lname == "rq_linear":
        return build_kernel("rq", rq_alphas=RQ_ALPHAS, rq_add_linear=True)
    raise ValueError(f"Unsupported toy kernel: {name}")


def parse_seeds(raw: str) -> List[int]:
    return [int(chunk.strip()) for chunk in raw.split(",") if chunk.strip()]


def prepare_moons(
    train_size: int,
    val_size: int,
    noise: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, List[float]]]:
    train_x, _ = make_moons(n_samples=train_size, noise=noise, random_state=seed)
    val_x, _ = make_moons(n_samples=val_size, noise=noise, random_state=seed + 10_000)

    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True) + 1e-6

    train_x = (train_x - mean) / std
    val_x = (val_x - mean) / std

    stats = {
        "mean": mean.reshape(-1).tolist(),
        "std": std.reshape(-1).tolist(),
    }
    return torch.from_numpy(train_x).float(), torch.from_numpy(val_x).float(), stats


def sample_real_batch(real_data: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    idx = torch.randint(0, real_data.shape[0], (batch_size,), device=real_data.device)
    return real_data[idx].to(device)


def checkpoint_steps(total_steps: int, sample_every: int) -> List[int]:
    picks = [0]
    step = sample_every
    while step < total_steps:
        picks.append(step)
        step += sample_every
    if total_steps not in picks:
        picks.append(total_steps)
    if len(picks) <= 4:
        return picks

    lin = np.linspace(0, len(picks) - 1, 4).round().astype(int)
    return [picks[idx] for idx in np.unique(lin)]


@torch.no_grad()
def evaluate_validation_mmd(
    generator: torch.nn.Module,
    critic: torch.nn.Module,
    val_data: torch.Tensor,
    z_eval: torch.Tensor,
    kernel,
    device: torch.device,
) -> float:
    real = val_data.to(device)
    fake = generator(z_eval)
    return float(mmd2_unbiased(critic(real), critic(fake), kernel).detach().cpu())


def save_scatter(
    path: Path,
    real_np: np.ndarray,
    fake_np: np.ndarray,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.scatter(real_np[:, 0], real_np[:, 1], s=5, alpha=0.25, label="real")
    ax.scatter(fake_np[:, 0], fake_np[:, 1], s=5, alpha=0.25, label="generated")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_witness_heatmap(
    path: Path,
    critic: torch.nn.Module,
    kernel,
    real_data: torch.Tensor,
    fake_data: torch.Tensor,
    grid_size: int,
    device: torch.device,
) -> None:
    real_np = real_data.cpu().numpy()
    fake_np = fake_data.cpu().numpy()
    x_min = min(real_np[:, 0].min(), fake_np[:, 0].min()) - 0.6
    x_max = max(real_np[:, 0].max(), fake_np[:, 0].max()) + 0.6
    y_min = min(real_np[:, 1].min(), fake_np[:, 1].min()) - 0.6
    y_max = max(real_np[:, 1].max(), fake_np[:, 1].max()) + 0.6

    gx = np.linspace(x_min, x_max, grid_size)
    gy = np.linspace(y_min, y_max, grid_size)
    xx, yy = np.meshgrid(gx, gy)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)

    with torch.no_grad():
        grid_t = torch.from_numpy(grid).float().to(device)
        real_t = real_data[: min(2000, real_data.shape[0])].to(device)
        fake_t = fake_data[: min(2000, fake_data.shape[0])].to(device)
        witness = witness_values(critic(grid_t), critic(real_t), critic(fake_t), kernel)
        witness = witness.cpu().numpy().reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    contour = ax.contourf(xx, yy, witness, levels=80, cmap="coolwarm")
    ax.scatter(real_np[::20, 0], real_np[::20, 1], s=5, c="k", alpha=0.15)
    ax.scatter(fake_np[::20, 0], fake_np[::20, 1], s=5, c="white", alpha=0.12)
    ax.set_title("Witness heatmap")
    fig.colorbar(contour, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_trajectory_panel(path: Path, snapshots: Sequence[Tuple[int, np.ndarray, np.ndarray]]) -> None:
    cols = len(snapshots)
    fig, axes = plt.subplots(1, cols, figsize=(4.2 * cols, 4.2), squeeze=False)
    for ax, (step, real_np, fake_np) in zip(axes[0], snapshots):
        ax.scatter(real_np[:, 0], real_np[:, 1], s=5, alpha=0.22, label="real")
        ax.scatter(fake_np[:, 0], fake_np[:, 1], s=5, alpha=0.22, label="generated")
        ax.set_title(f"step {step}")
        ax.grid(alpha=0.2)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def train_one(
    cfg: ToyProtocolConfig,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    run_dir: Path,
    witness_grid: int,
) -> Dict:
    device = torch.device(cfg.device)
    kernel = build_toy_kernel(cfg.kernel)

    seed_everything(cfg.seed)
    G = SmallToyGenerator(z_dim=cfg.z_dim, hidden_dim=cfg.hidden_dim).to(device)
    C = SmallToyFeatureCritic(in_dim=2, hidden_dim=cfg.hidden_dim, feat_dim=cfg.feat_dim).to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=cfg.lr_g, betas=(0.5, 0.9))
    opt_c = torch.optim.Adam(C.parameters(), lr=cfg.lr_c, betas=(0.5, 0.9))

    hist: Dict[str, List[float]] = {
        "step": [],
        "mmd2_train": [],
        "mmd2_val": [],
        "loss_g": [],
        "loss_c": [],
        "gp": [],
        "feat_norm_real": [],
        "feat_norm_fake": [],
        "elapsed_sec": [],
    }

    sample_dir = ensure_dir(run_dir / "samples")
    z_val = torch.randn(val_data.shape[0], cfg.z_dim, device=device)
    z_samples = torch.randn(cfg.sample_points, cfg.z_dim, device=device)
    sample_schedule = set(checkpoint_steps(cfg.steps, cfg.sample_every))
    real_preview = val_data[: cfg.sample_points].cpu().numpy()
    trajectory_snapshots: List[Tuple[int, np.ndarray, np.ndarray]] = []

    with torch.no_grad():
        initial_fake = G(z_samples).cpu().numpy()
    trajectory_snapshots.append((0, real_preview, initial_fake))

    start = perf_counter()
    iterator = trange(1, cfg.steps + 1, desc=f"toy-v2 {cfg.kernel} gp={int(cfg.use_gp)} seed={cfg.seed}")
    for step in iterator:
        gp_value = torch.zeros((), device=device)
        critic_loss = torch.zeros((), device=device)
        mmd2 = torch.zeros((), device=device)

        for _ in range(cfg.critic_steps):
            real = sample_real_batch(train_data, cfg.batch_size, device)
            z = torch.randn(cfg.batch_size, cfg.z_dim, device=device)
            fake = G(z).detach()

            feat_real = C(real)
            feat_fake = C(fake)
            mmd2 = mmd2_unbiased(feat_real, feat_fake, kernel)
            gp_value = (
                witness_gradient_penalty(C, real, fake, kernel, cfg.gp_lambda)
                if cfg.use_gp
                else torch.zeros((), device=device)
            )
            critic_loss = -mmd2 + gp_value

            opt_c.zero_grad(set_to_none=True)
            critic_loss.backward()
            opt_c.step()

        real = sample_real_batch(train_data, cfg.batch_size, device)
        z = torch.randn(cfg.batch_size, cfg.z_dim, device=device)
        fake = G(z)
        feat_real = C(real)
        feat_fake = C(fake)
        gen_loss = mmd2_unbiased(feat_real, feat_fake, kernel)

        opt_g.zero_grad(set_to_none=True)
        gen_loss.backward()
        opt_g.step()

        if step % cfg.log_every == 0 or step == cfg.steps:
            with torch.no_grad():
                val_mmd2 = evaluate_validation_mmd(G, C, val_data, z_val, kernel, device)
                hist["step"].append(step)
                hist["mmd2_train"].append(float(mmd2.detach().cpu()))
                hist["mmd2_val"].append(val_mmd2)
                hist["loss_g"].append(float(gen_loss.detach().cpu()))
                hist["loss_c"].append(float(critic_loss.detach().cpu()))
                hist["gp"].append(float(gp_value.detach().cpu()))
                hist["feat_norm_real"].append(float(mean_feature_norm(feat_real).detach().cpu()))
                hist["feat_norm_fake"].append(float(mean_feature_norm(feat_fake).detach().cpu()))
                hist["elapsed_sec"].append(perf_counter() - start)
            if hasattr(iterator, "set_postfix"):
                iterator.set_postfix(val_mmd2=f"{val_mmd2:.4f}")

        if step % cfg.sample_every == 0 or step == cfg.steps:
            with torch.no_grad():
                fake_samples = G(z_samples).cpu().numpy()
            save_scatter(
                sample_dir / f"samples_step-{step:07d}.png",
                real_preview,
                fake_samples,
                title=f"{cfg.kernel} | GP={int(cfg.use_gp)} | seed={cfg.seed} | step={step}",
            )
            if step in sample_schedule:
                trajectory_snapshots.append((step, real_preview, fake_samples))

    with torch.no_grad():
        final_fake = G(z_samples).cpu()
        save_witness_heatmap(
            run_dir / "witness_heatmap.png",
            C,
            kernel,
            val_data,
            final_fake,
            grid_size=witness_grid,
            device=device,
        )

    save_trajectory_panel(run_dir / "sample_trajectory.png", trajectory_snapshots)
    torch.save(
        {
            "config": asdict(cfg),
            "generator": G.state_dict(),
            "critic": C.state_dict(),
            "history": hist,
        },
        run_dir / "checkpoint_final.pt",
    )

    result = {
        "config": asdict(cfg),
        "history": hist,
        "run_dir": str(run_dir),
        "final_val_mmd2": hist["mmd2_val"][-1],
        "best_val_mmd2": float(min(hist["mmd2_val"])),
    }
    save_json(result, run_dir / "summary.json")
    return result


def aggregate_histories(results: Sequence[Dict], metric: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps = np.asarray(results[0]["history"]["step"], dtype=np.int64)
    curves = np.asarray([r["history"][metric] for r in results], dtype=np.float64)
    return steps, curves.mean(axis=0), curves.std(axis=0)


def plot_group_curves(path: Path, grouped: Dict[str, Sequence[Dict]], metric: str, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    palette = ["C0", "C1", "C2", "C3", "C4", "C5"]
    for color, (label, runs) in zip(palette, grouped.items()):
        steps, mean, std = aggregate_histories(runs, metric)
        ax.plot(steps, mean, color=color, linewidth=2.0, label=label)
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.16, linewidth=0)
    ax.set_title(title)
    ax.set_xlabel("generator step")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_kernel_comparison(path: Path, grouped: Dict[str, Sequence[Dict]], use_gp: bool) -> None:
    selected = {
        kernel: runs
        for kernel, runs in grouped.items()
        if kernel.endswith(f"|gp={int(use_gp)}")
    }
    plot_group_curves(
        path=path,
        grouped=selected,
        metric="mmd2_val",
        title=f"Kernel comparison on validation MMD (GP={int(use_gp)})",
        ylabel="validation MMD^2",
    )


def plot_gp_comparison(path: Path, grouped: Dict[str, Sequence[Dict]], kernel_name: str) -> None:
    selected = {
        label: runs
        for label, runs in grouped.items()
        if label.startswith(f"{kernel_name}|")
    }
    plot_group_curves(
        path=path,
        grouped=selected,
        metric="mmd2_val",
        title=f"GP comparison for {kernel_name}",
        ylabel="validation MMD^2",
    )


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    device = pick_device(prefer_cuda=True)
    run_root = ensure_dir(Path(args.outdir) / timestamp())

    print(f"[info] device: {device_summary(device)}")
    print(f"[info] output root: {run_root}")

    if args.mode == "suite":
        kernels = ["linear", "rbf", "rq"]
        run_specs = [(kernel, use_gp) for use_gp in (False, True) for kernel in kernels]
    else:
        run_specs = [(args.kernel, bool(args.gp))]

    all_results: List[Dict] = []
    grouped: Dict[str, List[Dict]] = {}

    for kernel_name, use_gp in run_specs:
        for seed in seeds:
            train_data, val_data, norm_stats = prepare_moons(
                train_size=args.train_size,
                val_size=args.val_size,
                noise=args.noise,
                seed=seed,
            )
            run_dir = ensure_dir(run_root / f"kernel-{kernel_name}_gp-{int(use_gp)}_seed-{seed}")
            cfg = ToyProtocolConfig(
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
                train_size=args.train_size,
                val_size=args.val_size,
                noise=args.noise,
                seed=seed,
                device=str(device),
                log_every=args.log_every,
                sample_every=args.sample_every,
                sample_points=args.sample_points,
            )
            result = train_one(cfg, train_data, val_data, run_dir, witness_grid=args.witness_grid)
            result["normalization"] = norm_stats
            all_results.append(result)
            grouped.setdefault(f"{kernel_name}|gp={int(use_gp)}", []).append(result)

    if all_results:
        plot_group_curves(
            run_root / "training_curves_all.png",
            grouped,
            metric="mmd2_val",
            title="Validation MMD across kernels and GP settings",
            ylabel="validation MMD^2",
        )
        plot_kernel_comparison(run_root / "kernel_comparison_gp-0.png", grouped, use_gp=False)
        plot_kernel_comparison(run_root / "kernel_comparison_gp-1.png", grouped, use_gp=True)
        if "rq|gp=0" in grouped and "rq|gp=1" in grouped:
            plot_gp_comparison(run_root / "gp_comparison_rq.png", grouped, kernel_name="rq")

    summary = {
        "args": vars(args),
        "device": device_summary(device),
        "results": all_results,
        "run_root": str(run_root),
    }
    save_json(summary, run_root / "summary_all.json")

    print("\n[summary] final validation MMD^2")
    for result in all_results:
        cfg = result["config"]
        print(
            f"  kernel={cfg['kernel']:>6} | gp={int(cfg['use_gp'])} | seed={cfg['seed']:>2} | "
            f"final_val_mmd2={result['final_val_mmd2']:.6f}"
        )


if __name__ == "__main__":
    main()
