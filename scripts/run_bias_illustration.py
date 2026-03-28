#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
from mmd_gan_experiments.protocol_v2_helpers import BiasFeatureCritic
from mmd_gan_experiments.utils import (
    ensure_dir,
    pick_device,
    save_json,
    seed_everything,
    timestamp,
)

try:
    from tqdm.auto import trange
except Exception:  # pragma: no cover
    trange = range


RQ_ALPHAS = (0.2, 0.5, 1.0, 2.0, 5.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Finite-sample critic-bias illustration (v3)"
    )
    p.add_argument("--delta-grid", type=str, default="-2,-1.5,-1,-0.5,0,0.5,1,1.5,2")
    p.add_argument("--num-seeds", type=int, default=20)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--train-size", type=int, default=512)
    p.add_argument("--holdout-size", type=int, default=512)
    p.add_argument("--kernel", choices=["rq", "rbf", "linear"], default="rq")
    p.add_argument("--critic-hidden-dim", type=int, default=8)
    p.add_argument("--critic-feat-dim", type=int, default=4)
    p.add_argument("--critic-steps", type=int, default=1500)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--activation-penalty", type=float, default=1e-3)
    p.add_argument(
        "--gradient-mode", choices=["autograd", "finite-diff"], default="autograd"
    )
    p.add_argument("--finite-diff-eps", type=float, default=1e-2)
    p.add_argument(
        "--fixed-critic-mode", choices=["trained", "random"], default="trained"
    )
    p.add_argument("--fixed-critic-delta0", type=float, default=1.0)
    p.add_argument("--fixed-critic-seed", type=int, default=1234)
    p.add_argument("--fixed-critic-train-size", type=int, default=2048)
    p.add_argument("--fixed-critic-steps", type=int, default=2000)
    p.add_argument("--outdir", type=str, default="results/bias_illustration_v3")
    return p.parse_args()


def parse_delta_grid(raw: str) -> List[float]:
    return [float(chunk.strip()) for chunk in raw.split(",") if chunk.strip()]


def build_bias_kernel(name: str):
    if name == "rq":
        return build_kernel("rq", rq_alphas=RQ_ALPHAS, rq_add_linear=False)
    if name == "rbf":
        return build_kernel("rbf")
    if name == "linear":
        return build_kernel("linear")
    raise ValueError(f"Unsupported bias kernel: {name}")


def sample_location_family(
    batch_size: int,
    delta: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    real = torch.randn(batch_size, 1, device=device)
    noise = torch.randn(batch_size, 1, device=device)
    fake = noise + float(delta)
    return real, fake


def sample_location_family_with_noise(
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    real = torch.randn(batch_size, 1, device=device)
    noise = torch.randn(batch_size, 1, device=device)
    return real, noise


def freeze_module(module: torch.nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def mmd_value(
    critic: torch.nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    kernel,
) -> torch.Tensor:
    return mmd2_unbiased(critic(real), critic(fake), kernel)


def train_critic(
    *,
    real_train: torch.Tensor,
    noise_train: torch.Tensor,
    delta: float,
    hidden_dim: int,
    feat_dim: int,
    steps: int,
    lr: float,
    activation_penalty: float,
    kernel,
) -> torch.nn.Module:
    critic = BiasFeatureCritic(hidden_dim=hidden_dim, feat_dim=feat_dim).to(
        real_train.device
    )
    optimizer = torch.optim.Adam(critic.parameters(), lr=lr, betas=(0.5, 0.9))
    delta_value = torch.tensor(float(delta), device=real_train.device)

    for _ in range(steps):
        fake_train = noise_train + delta_value
        feat_real = critic(real_train)
        feat_fake = critic(fake_train)
        mmd2 = mmd2_unbiased(feat_real, feat_fake, kernel)
        penalty = activation_penalty * (
            feat_real.pow(2).mean() + feat_fake.pow(2).mean()
        )
        loss = -mmd2 + penalty

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    freeze_module(critic)
    return critic


def objective_with_delta(
    critic: torch.nn.Module,
    real: torch.Tensor,
    noise: torch.Tensor,
    delta: torch.Tensor,
    kernel,
) -> torch.Tensor:
    fake = noise + delta.view(1, 1)
    return mmd_value(critic, real, fake, kernel)


def gradient_estimate(
    critic: torch.nn.Module,
    real: torch.Tensor,
    noise: torch.Tensor,
    delta: float,
    kernel,
    mode: str,
    eps: float,
) -> float:
    if mode == "autograd":
        delta_value = torch.tensor(float(delta), device=real.device, requires_grad=True)
        loss = objective_with_delta(critic, real, noise, delta_value, kernel)
        (grad,) = torch.autograd.grad(loss, delta_value)
        return float(grad.detach().cpu())

    with torch.no_grad():
        delta_plus = torch.tensor(float(delta + eps), device=real.device)
        delta_minus = torch.tensor(float(delta - eps), device=real.device)
        loss_plus = objective_with_delta(critic, real, noise, delta_plus, kernel)
        loss_minus = objective_with_delta(critic, real, noise, delta_minus, kernel)
    return float(((loss_plus - loss_minus) / (2.0 * eps)).cpu())


def train_or_build_fixed_critic(
    args: argparse.Namespace, kernel, device: torch.device
) -> torch.nn.Module:
    seed_everything(args.fixed_critic_seed)
    critic = BiasFeatureCritic(
        hidden_dim=args.critic_hidden_dim, feat_dim=args.critic_feat_dim
    ).to(device)
    if args.fixed_critic_mode == "random":
        freeze_module(critic)
        return critic

    real_ref, noise_ref = sample_location_family_with_noise(
        args.fixed_critic_train_size, device
    )
    trained = train_critic(
        real_train=real_ref,
        noise_train=noise_ref,
        delta=args.fixed_critic_delta0,
        hidden_dim=args.critic_hidden_dim,
        feat_dim=args.critic_feat_dim,
        steps=args.fixed_critic_steps,
        lr=args.critic_lr,
        activation_penalty=args.activation_penalty,
        kernel=kernel,
    )
    return trained


def curve_mean_std(curves: Sequence[Sequence[float]]) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(curves, dtype=np.float64)
    return arr.mean(axis=0), arr.std(axis=0)


def plot_with_bands(
    ax,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    label: str,
    color: str,
    linestyle: str = "-",
) -> None:
    ax.plot(x, mean, color=color, linewidth=2.0, linestyle=linestyle, label=label)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)


def build_markdown_report(
    *,
    delta_grid: np.ndarray,
    objective_fixed_mean: np.ndarray,
    objective_holdout_mean: np.ndarray,
    gradient_fixed_mean: np.ndarray,
    gradient_holdout_mean: np.ndarray,
    learned_gap_mean: np.ndarray,
    learned_grad_gap_mean: np.ndarray,
) -> str:
    zero_idx = int(np.argmin(np.abs(delta_grid)))
    return "\n".join(
        [
            "# Bias Illustration v3",
            "",
            "This experiment keeps the finite-sample split explicit. For each seed and each delta, one critic is fit on a fixed train set and then evaluated on both the train and holdout sets. The fixed-critic baseline removes the critic-selection step by freezing a single feature map for the whole sweep.",
            "",
            "## Short Interpretation",
            "",
            "The fixed-critic curves are the clean baseline: the only randomness comes from finite datasets, not from re-selecting the critic. The learned-critic curves add the critic-selection step on top of finite-sample noise. The train-holdout gap is therefore the direct visual signature of overfitting in the critic selection stage, which is the theory point this subsection is meant to illustrate.",
            "",
            "## One-Point Diagnostic",
            "",
            (
                f"At delta={delta_grid[zero_idx]:+.2f}, the fixed-critic objective mean is "
                f"{objective_fixed_mean[zero_idx]:+.4f}, the learned holdout objective mean is "
                f"{objective_holdout_mean[zero_idx]:+.4f}, the fixed-critic gradient mean is "
                f"{gradient_fixed_mean[zero_idx]:+.4f}, and the learned holdout gradient mean is "
                f"{gradient_holdout_mean[zero_idx]:+.4f}. The learned objective gap is "
                f"{learned_gap_mean[zero_idx]:+.4f} and the learned gradient gap is "
                f"{learned_grad_gap_mean[zero_idx]:+.4f}."
            ),
        ]
    )


def main() -> None:
    args = parse_args()
    delta_grid = np.asarray(parse_delta_grid(args.delta_grid), dtype=np.float64)
    device = pick_device(prefer_cuda=True)
    kernel = build_bias_kernel(args.kernel)
    run_root = ensure_dir(Path(args.outdir) / timestamp())

    print(f"[info] device: {device}")
    print(f"[info] output: {run_root}")

    fixed_critic = train_or_build_fixed_critic(args, kernel, device)

    per_seed: Dict[str, List[List[float]]] = {
        "objective_fixed": [],
        "objective_train": [],
        "objective_holdout": [],
        "gradient_fixed": [],
        "gradient_train": [],
        "gradient_holdout": [],
    }

    for seed in trange(
        args.seed_offset, args.seed_offset + args.num_seeds, desc="bias-v3"
    ):
        objective_fixed_curve: List[float] = []
        objective_train_curve: List[float] = []
        objective_holdout_curve: List[float] = []
        gradient_fixed_curve: List[float] = []
        gradient_train_curve: List[float] = []
        gradient_holdout_curve: List[float] = []

        for delta_idx, delta in enumerate(delta_grid):
            data_seed = seed * 1000 + delta_idx
            seed_everything(data_seed)
            real_train, noise_train = sample_location_family_with_noise(
                args.train_size, device
            )
            real_holdout, noise_holdout = sample_location_family_with_noise(
                args.holdout_size, device
            )

            critic_seed = 1_000_000 + seed * 1000 + delta_idx
            seed_everything(critic_seed)
            learned_critic = train_critic(
                real_train=real_train,
                noise_train=noise_train,
                delta=float(delta),
                hidden_dim=args.critic_hidden_dim,
                feat_dim=args.critic_feat_dim,
                steps=args.critic_steps,
                lr=args.critic_lr,
                activation_penalty=args.activation_penalty,
                kernel=kernel,
            )

            with torch.no_grad():
                delta_value = torch.tensor(float(delta), device=device)
                objective_fixed_curve.append(
                    float(
                        objective_with_delta(
                            fixed_critic,
                            real_holdout,
                            noise_holdout,
                            delta_value,
                            kernel,
                        ).cpu()
                    )
                )
                objective_train_curve.append(
                    float(
                        objective_with_delta(
                            learned_critic, real_train, noise_train, delta_value, kernel
                        ).cpu()
                    )
                )
                objective_holdout_curve.append(
                    float(
                        objective_with_delta(
                            learned_critic,
                            real_holdout,
                            noise_holdout,
                            delta_value,
                            kernel,
                        ).cpu()
                    )
                )

            gradient_fixed_curve.append(
                gradient_estimate(
                    fixed_critic,
                    real_holdout,
                    noise_holdout,
                    float(delta),
                    kernel,
                    args.gradient_mode,
                    args.finite_diff_eps,
                )
            )
            gradient_train_curve.append(
                gradient_estimate(
                    learned_critic,
                    real_train,
                    noise_train,
                    float(delta),
                    kernel,
                    args.gradient_mode,
                    args.finite_diff_eps,
                )
            )
            gradient_holdout_curve.append(
                gradient_estimate(
                    learned_critic,
                    real_holdout,
                    noise_holdout,
                    float(delta),
                    kernel,
                    args.gradient_mode,
                    args.finite_diff_eps,
                )
            )

        per_seed["objective_fixed"].append(objective_fixed_curve)
        per_seed["objective_train"].append(objective_train_curve)
        per_seed["objective_holdout"].append(objective_holdout_curve)
        per_seed["gradient_fixed"].append(gradient_fixed_curve)
        per_seed["gradient_train"].append(gradient_train_curve)
        per_seed["gradient_holdout"].append(gradient_holdout_curve)

    objective_fixed_mean, objective_fixed_std = curve_mean_std(
        per_seed["objective_fixed"]
    )
    objective_train_mean, objective_train_std = curve_mean_std(
        per_seed["objective_train"]
    )
    objective_holdout_mean, objective_holdout_std = curve_mean_std(
        per_seed["objective_holdout"]
    )
    gradient_fixed_mean, gradient_fixed_std = curve_mean_std(per_seed["gradient_fixed"])
    gradient_train_mean, gradient_train_std = curve_mean_std(per_seed["gradient_train"])
    gradient_holdout_mean, gradient_holdout_std = curve_mean_std(
        per_seed["gradient_holdout"]
    )

    learned_gap = np.asarray(per_seed["objective_train"]) - np.asarray(
        per_seed["objective_holdout"]
    )
    learned_grad_gap = np.asarray(per_seed["gradient_train"]) - np.asarray(
        per_seed["gradient_holdout"]
    )
    learned_gap_mean, learned_gap_std = curve_mean_std(learned_gap.tolist())
    learned_grad_gap_mean, learned_grad_gap_std = curve_mean_std(
        learned_grad_gap.tolist()
    )

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    plot_with_bands(
        ax,
        delta_grid,
        objective_fixed_mean,
        objective_fixed_std,
        "Fixed critic",
        "black",
    )
    plot_with_bands(
        ax,
        delta_grid,
        objective_train_mean,
        objective_train_std,
        "Learned critic, train",
        "C3",
    )
    plot_with_bands(
        ax,
        delta_grid,
        objective_holdout_mean,
        objective_holdout_std,
        "Learned critic, holdout",
        "C0",
    )
    ax.set_title("Objective vs delta")
    ax.set_xlabel("delta")
    ax.set_ylabel("empirical MMD^2")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_root / "objective_vs_delta.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    plot_with_bands(
        ax, delta_grid, gradient_fixed_mean, gradient_fixed_std, "Fixed critic", "black"
    )
    plot_with_bands(
        ax,
        delta_grid,
        gradient_train_mean,
        gradient_train_std,
        "Learned critic, train",
        "C3",
    )
    plot_with_bands(
        ax,
        delta_grid,
        gradient_holdout_mean,
        gradient_holdout_std,
        "Learned critic, holdout",
        "C0",
    )
    ax.axhline(0.0, color="0.65", linewidth=1.0)
    ax.set_title("Gradient vs delta")
    ax.set_xlabel("delta")
    ax.set_ylabel("d/d delta empirical MMD^2")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_root / "gradient_vs_delta.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    plot_with_bands(
        axes[0], delta_grid, learned_gap_mean, learned_gap_std, "Train - holdout", "C4"
    )
    axes[0].axhline(0.0, color="0.65", linewidth=1.0)
    axes[0].set_title("Objective gap")
    axes[0].set_xlabel("delta")
    axes[0].set_ylabel("MMD^2 gap")
    axes[0].grid(alpha=0.25)

    plot_with_bands(
        axes[1],
        delta_grid,
        learned_grad_gap_mean,
        learned_grad_gap_std,
        "Train - holdout",
        "C5",
    )
    axes[1].axhline(0.0, color="0.65", linewidth=1.0)
    axes[1].set_title("Gradient gap")
    axes[1].set_xlabel("delta")
    axes[1].set_ylabel("gradient gap")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(run_root / "gap_vs_delta.png", dpi=180)
    plt.close(fig)

    payload = {
        "args": vars(args),
        "device": str(device),
        "delta_grid": delta_grid.tolist(),
        "objective_fixed": {
            "mean": objective_fixed_mean.tolist(),
            "std": objective_fixed_std.tolist(),
        },
        "objective_train": {
            "mean": objective_train_mean.tolist(),
            "std": objective_train_std.tolist(),
        },
        "objective_holdout": {
            "mean": objective_holdout_mean.tolist(),
            "std": objective_holdout_std.tolist(),
        },
        "gradient_fixed": {
            "mean": gradient_fixed_mean.tolist(),
            "std": gradient_fixed_std.tolist(),
        },
        "gradient_train": {
            "mean": gradient_train_mean.tolist(),
            "std": gradient_train_std.tolist(),
        },
        "gradient_holdout": {
            "mean": gradient_holdout_mean.tolist(),
            "std": gradient_holdout_std.tolist(),
        },
        "learned_objective_gap": {
            "mean": learned_gap_mean.tolist(),
            "std": learned_gap_std.tolist(),
        },
        "learned_gradient_gap": {
            "mean": learned_grad_gap_mean.tolist(),
            "std": learned_grad_gap_std.tolist(),
        },
    }
    save_json(payload, run_root / "summary.json")

    report = build_markdown_report(
        delta_grid=delta_grid,
        objective_fixed_mean=objective_fixed_mean,
        objective_holdout_mean=objective_holdout_mean,
        gradient_fixed_mean=gradient_fixed_mean,
        gradient_holdout_mean=gradient_holdout_mean,
        learned_gap_mean=learned_gap_mean,
        learned_grad_gap_mean=learned_grad_gap_mean,
    )
    (run_root / "deliverables.md").write_text(report)

    print("\n[summary]")
    print(f"  fixed critic mode     : {args.fixed_critic_mode}")
    print(f"  num seeds             : {args.num_seeds}")
    print(f"  delta grid            : {delta_grid.tolist()}")
    print(f"  output                : {run_root}")


if __name__ == "__main__":
    main()
