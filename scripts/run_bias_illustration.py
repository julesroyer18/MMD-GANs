#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - dependency error path
    raise SystemExit(
        f"Missing dependency: {exc}. Install requirements with `pip install -r requirements.txt`."
    ) from exc

from mmd_gan_experiments.kernels import build_kernel
from mmd_gan_experiments.mmd import mmd2_unbiased
from mmd_gan_experiments.models_toy import FixedRandomFeatureMap, ToyFeatureCritic
from mmd_gan_experiments.utils import ensure_dir, pick_device, save_json, seed_everything, timestamp


try:
    from tqdm.auto import trange
except Exception:  # pragma: no cover
    trange = range


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bias illustration: fixed critic vs learned critic")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--kernel", choices=["linear", "rbf", "rq"], default="rq")
    p.add_argument("--delta", type=float, default=0.0, help="Mean shift of fake distribution")
    p.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated seeds for repeated runs",
    )
    p.add_argument("--outdir", type=str, default="results/bias_illustration")
    return p.parse_args()


def sample_pair(batch_size: int, device: torch.device, delta: float) -> tuple[torch.Tensor, torch.Tensor]:
    real = torch.randn(batch_size, 1, device=device)
    fake = torch.randn(batch_size, 1, device=device) + delta
    return real, fake


def run_once(seed: int, args: argparse.Namespace, device: torch.device) -> Dict[str, List[float]]:
    seed_everything(seed)
    kernel = build_kernel(args.kernel)

    learned = ToyFeatureCritic(in_dim=1, hidden_dim=64, feat_dim=8).to(device)
    fixed = FixedRandomFeatureMap(in_dim=1, hidden_dim=64, feat_dim=8).to(device)
    opt = torch.optim.Adam(learned.parameters(), lr=args.lr, betas=(0.5, 0.9))

    out: Dict[str, List[float]] = {
        "step": [],
        "learned_train": [],
        "learned_holdout": [],
        "fixed_train": [],
        "fixed_holdout": [],
    }

    for step in trange(1, args.steps + 1, desc=f"seed={seed}"):
        real_train, fake_train = sample_pair(args.batch_size, device, args.delta)
        loss = -mmd2_unbiased(learned(real_train), learned(fake_train), kernel)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % args.eval_every == 0 or step == args.steps:
            real_test, fake_test = sample_pair(args.batch_size, device, args.delta)
            with torch.no_grad():
                out["step"].append(step)
                out["learned_train"].append(
                    float(mmd2_unbiased(learned(real_train), learned(fake_train), kernel).cpu())
                )
                out["learned_holdout"].append(
                    float(mmd2_unbiased(learned(real_test), learned(fake_test), kernel).cpu())
                )
                out["fixed_train"].append(
                    float(mmd2_unbiased(fixed(real_train), fixed(fake_train), kernel).cpu())
                )
                out["fixed_holdout"].append(
                    float(mmd2_unbiased(fixed(real_test), fixed(fake_test), kernel).cpu())
                )

    return out


def mean_std(curves: List[List[float]]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(curves, dtype=np.float64)
    return arr.mean(axis=0), arr.std(axis=0)


def main() -> None:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    device = pick_device(prefer_cuda=True)

    run_root = ensure_dir(Path(args.outdir) / timestamp())
    print(f"[info] device: {device}")
    print(f"[info] output: {run_root}")

    runs = [run_once(seed, args, device=device) for seed in seeds]

    steps = runs[0]["step"]
    keys = ["learned_train", "learned_holdout", "fixed_train", "fixed_holdout"]
    stats = {}
    for key in keys:
        mu, sd = mean_std([r[key] for r in runs])
        stats[key] = {"mean": mu.tolist(), "std": sd.tolist()}

    fig, ax = plt.subplots(figsize=(9, 5))
    styles = {
        "learned_train": ("C3", "Learned critic (fit batch)"),
        "learned_holdout": ("C0", "Learned critic (holdout batch)"),
        "fixed_train": ("C2", "Fixed critic (fit batch)"),
        "fixed_holdout": ("C1", "Fixed critic (holdout batch)"),
    }
    for key in keys:
        color, label = styles[key]
        mu = np.asarray(stats[key]["mean"])
        sd = np.asarray(stats[key]["std"])
        ax.plot(steps, mu, color=color, label=label)
        ax.fill_between(steps, mu - sd, mu + sd, color=color, alpha=0.18, linewidth=0)

    ax.set_title(f"Bias illustration (delta={args.delta}, kernel={args.kernel})")
    ax.set_xlabel("critic optimization step")
    ax.set_ylabel("unbiased MMD^2 estimate")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_root / "bias_curves.png", dpi=180)
    plt.close(fig)

    final_summary = {k: float(stats[k]["mean"][-1]) for k in keys}
    final_summary["learned_gap"] = final_summary["learned_train"] - final_summary["learned_holdout"]
    final_summary["fixed_gap"] = final_summary["fixed_train"] - final_summary["fixed_holdout"]

    payload = {
        "args": vars(args),
        "device": str(device),
        "seeds": seeds,
        "steps": steps,
        "stats": stats,
        "final_summary": final_summary,
    }
    save_json(payload, run_root / "summary.json")

    print("\n[summary]")
    for key, value in final_summary.items():
        print(f"  {key:16s}: {value:.6f}")


if __name__ == "__main__":
    main()
