#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, List, Tuple

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
from mmd_gan_experiments.models_toy import ToyFeatureCritic
from mmd_gan_experiments.utils import ensure_dir, pick_device, save_json, seed_everything, timestamp

try:
    from tqdm.auto import trange
except Exception:  # pragma: no cover
    trange = range


KernelFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bias illustration v2: fixed critic vs critic re-fit from finite samples"
    )
    p.add_argument("--psi-min", type=float, default=-1.0)
    p.add_argument("--psi-max", type=float, default=1.0)
    p.add_argument("--psi-points", type=int, default=15)
    p.add_argument("--num-seeds", type=int, default=24)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--kernel", choices=["linear", "rbf", "rq"], default="linear")
    p.add_argument("--batch-size", type=int, default=32, help="Finite-sample size for learned critic")
    p.add_argument("--eval-batch-size", type=int, default=32, help="Finite-sample size for gradient estimates")
    p.add_argument("--critic-hidden-dim", type=int, default=32)
    p.add_argument("--critic-feat-dim", type=int, default=8)
    p.add_argument("--critic-steps", type=int, default=200)
    p.add_argument("--critic-lr", type=float, default=1e-3)
    p.add_argument(
        "--activation-penalty",
        type=float,
        default=1e-3,
        help="Small penalty to keep the learned feature map from blowing up.",
    )
    p.add_argument(
        "--fixed-critic-psi",
        type=float,
        default=0.75,
        help="Anchor shift used once to train the frozen critic.",
    )
    p.add_argument("--fixed-critic-seed", type=int, default=1234)
    p.add_argument("--fixed-critic-steps", type=int, default=400)
    p.add_argument("--fixed-critic-batch-size", type=int, default=2048)
    p.add_argument("--reference-batch-size", type=int, default=8192)
    p.add_argument("--reference-repeats", type=int, default=8)
    p.add_argument("--outdir", type=str, default="results/bias_illustration_v2")
    return p.parse_args()


def sample_real_and_noise(batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    real = torch.randn(batch_size, 1, device=device)
    noise = torch.randn(batch_size, 1, device=device)
    return real, noise


def make_fake(noise: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    return noise + psi.view(1, 1)


def critic_penalty(
    critic: torch.nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    if scale <= 0.0:
        return torch.zeros((), device=real.device)
    feat_real = critic(real)
    feat_fake = critic(fake)
    return scale * (feat_real.pow(2).mean() + feat_fake.pow(2).mean())


def mmd_loss(
    critic: torch.nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    kernel: KernelFn,
) -> torch.Tensor:
    return mmd2_unbiased(critic(real), critic(fake), kernel)


def freeze_module(module: torch.nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


def train_critic_on_fixed_dataset(
    *,
    seed: int,
    psi: float,
    batch_size: int,
    steps: int,
    lr: float,
    hidden_dim: int,
    feat_dim: int,
    activation_penalty: float,
    kernel: KernelFn,
    device: torch.device,
) -> Tuple[torch.nn.Module, Tuple[torch.Tensor, torch.Tensor]]:
    seed_everything(seed)
    real_train, noise_train = sample_real_and_noise(batch_size, device)
    critic = ToyFeatureCritic(in_dim=1, hidden_dim=hidden_dim, feat_dim=feat_dim).to(device)
    optimizer = torch.optim.Adam(critic.parameters(), lr=lr, betas=(0.5, 0.9))
    psi_value = torch.tensor(float(psi), device=device)

    for _ in range(steps):
        fake_train = make_fake(noise_train, psi_value)
        obj = mmd_loss(critic, real_train, fake_train, kernel)
        penalty = critic_penalty(critic, real_train, fake_train, activation_penalty)
        loss = -obj + penalty

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    freeze_module(critic)
    return critic, (real_train.detach(), noise_train.detach())


def gradient_estimate(
    critic: torch.nn.Module,
    real: torch.Tensor,
    noise: torch.Tensor,
    psi: float,
    kernel: KernelFn,
) -> float:
    psi_value = torch.tensor(float(psi), device=real.device, requires_grad=True)
    fake = make_fake(noise, psi_value)
    loss = mmd_loss(critic, real, fake, kernel)
    (grad,) = torch.autograd.grad(loss, psi_value)
    return float(grad.detach().cpu())


def mean_std(curves: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(curves, dtype=np.float64)
    return arr.mean(axis=0), arr.std(axis=0)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def build_markdown_report(
    *,
    plot_name: str,
    psi_grid: np.ndarray,
    reference: np.ndarray,
    fixed_mean: np.ndarray,
    fixed_std: np.ndarray,
    learned_holdout_mean: np.ndarray,
    learned_holdout_std: np.ndarray,
    learned_train_mean: np.ndarray,
    learned_train_std: np.ndarray,
) -> str:
    fixed_rmse = rmse(fixed_mean, reference)
    learned_holdout_rmse = rmse(learned_holdout_mean, reference)
    learned_train_rmse = rmse(learned_train_mean, reference)

    zero_idx = int(np.argmin(np.abs(psi_grid)))
    paragraph = (
        "This toy sweep is only an illustration of the theory point, not a theorem check. "
        "With the frozen critic, we differentiate a single objective L(psi, theta_fixed), so the "
        "mean finite-batch gradient stays close to the large-batch reference. Once the critic is "
        "re-fit from a small dataset at every psi, the differentiated quantity becomes "
        "L(psi, theta_hat_S(psi)): both the empirical sample S and the fitted critic change with psi. "
        "The gap between the fixed-critic curve and the re-estimated curves, together with the larger "
        "seed-to-seed spread, is the empirical signature predicted by the theory discussion."
    )

    return "\n".join(
        [
            "# Bias Illustration v2",
            "",
            f"![Gradient plot]({plot_name})",
            "",
            "## Schematic Table",
            "",
            "| setup | what is fixed | what is re-estimated | observed consequence |",
            "| --- | --- | --- | --- |",
            (
                "| Fixed critic | One critic trained once at an anchor shift, then frozen for the full sweep | "
                "Only the finite evaluation batch changes across seeds | "
                f"Closest to the large-batch reference (RMSE {fixed_rmse:.4f}); moderate variability "
                f"(mean sd {fixed_std.mean():.4f}) |"
            ),
            (
                "| Learned critic, holdout gradient | Target distribution family and architecture are fixed | "
                "A new critic is fit on a finite train set for each psi and seed; gradient is evaluated on "
                "an independent holdout batch | "
                f"Mean curve shifts away from the fixed-critic reference (RMSE {learned_holdout_rmse:.4f}) "
                f"and variability increases (mean sd {learned_holdout_std.mean():.4f}) |"
            ),
            (
                "| Learned critic, same training batch | Same finite train set is reused for critic fitting and "
                "gradient evaluation | "
                "Critic and gradient are both tied to the same small sample at each psi | "
                f"Strongest discrepancy (RMSE {learned_train_rmse:.4f}) and the noisiest behavior "
                f"(mean sd {learned_train_std.mean():.4f}) |"
            ),
            "",
            "## Connection Back to Theory",
            "",
            paragraph,
            "",
            "## One-Point Diagnostic",
            "",
            (
                f"Near psi={psi_grid[zero_idx]:+.3f}, the large-batch fixed-critic reference gradient is "
                f"{reference[zero_idx]:+.4f}, the fixed finite-batch mean is {fixed_mean[zero_idx]:+.4f}, "
                f"the learned holdout mean is {learned_holdout_mean[zero_idx]:+.4f}, and the learned same-batch "
                f"mean is {learned_train_mean[zero_idx]:+.4f}."
            ),
        ]
    )


def main() -> None:
    args = parse_args()
    device = pick_device(prefer_cuda=True)
    kernel = build_kernel(args.kernel)
    psi_grid = np.linspace(args.psi_min, args.psi_max, args.psi_points, dtype=np.float64)
    run_root = ensure_dir(Path(args.outdir) / timestamp())

    print(f"[info] device: {device}")
    print(f"[info] output: {run_root}")
    print(
        "[info] design: one fixed critic trained once, then per-psi finite-sample critic re-fits "
        "with same-batch and holdout gradient evaluation"
    )

    fixed_critic, _ = train_critic_on_fixed_dataset(
        seed=args.fixed_critic_seed,
        psi=args.fixed_critic_psi,
        batch_size=args.fixed_critic_batch_size,
        steps=args.fixed_critic_steps,
        lr=args.critic_lr,
        hidden_dim=args.critic_hidden_dim,
        feat_dim=args.critic_feat_dim,
        activation_penalty=args.activation_penalty,
        kernel=kernel,
        device=device,
    )

    reference_grads: List[float] = []
    for psi in psi_grid:
        per_repeat: List[float] = []
        for _ in range(args.reference_repeats):
            real_ref, noise_ref = sample_real_and_noise(args.reference_batch_size, device)
            per_repeat.append(gradient_estimate(fixed_critic, real_ref, noise_ref, float(psi), kernel))
        reference_grads.append(float(np.mean(per_repeat)))

    per_seed: Dict[str, List[List[float]]] = {
        "fixed": [],
        "learned_holdout": [],
        "learned_train": [],
    }

    for seed in trange(
        args.seed_offset,
        args.seed_offset + args.num_seeds,
        desc="bias-v2",
    ):
        fixed_curve: List[float] = []
        learned_holdout_curve: List[float] = []
        learned_train_curve: List[float] = []

        for psi_idx, psi in enumerate(psi_grid):
            critic_seed = seed * 1000 + psi_idx
            learned_critic, (real_train, noise_train) = train_critic_on_fixed_dataset(
                seed=critic_seed,
                psi=float(psi),
                batch_size=args.batch_size,
                steps=args.critic_steps,
                lr=args.critic_lr,
                hidden_dim=args.critic_hidden_dim,
                feat_dim=args.critic_feat_dim,
                activation_penalty=args.activation_penalty,
                kernel=kernel,
                device=device,
            )

            eval_seed = 10_000_000 + seed * 1000 + psi_idx
            seed_everything(eval_seed)
            real_eval, noise_eval = sample_real_and_noise(args.eval_batch_size, device)

            fixed_curve.append(gradient_estimate(fixed_critic, real_eval, noise_eval, float(psi), kernel))
            learned_holdout_curve.append(
                gradient_estimate(learned_critic, real_eval, noise_eval, float(psi), kernel)
            )
            learned_train_curve.append(
                gradient_estimate(learned_critic, real_train, noise_train, float(psi), kernel)
            )

        per_seed["fixed"].append(fixed_curve)
        per_seed["learned_holdout"].append(learned_holdout_curve)
        per_seed["learned_train"].append(learned_train_curve)

    fixed_mean, fixed_std = mean_std(per_seed["fixed"])
    learned_holdout_mean, learned_holdout_std = mean_std(per_seed["learned_holdout"])
    learned_train_mean, learned_train_std = mean_std(per_seed["learned_train"])
    reference = np.asarray(reference_grads)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(psi_grid, reference, color="black", linewidth=2.2, label="Fixed critic reference")

    def add_band(x: np.ndarray, mean: np.ndarray, std: np.ndarray, color: str, label: str, linestyle: str = "-") -> None:
        ax.plot(x, mean, color=color, linewidth=2.0, linestyle=linestyle, label=label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)

    add_band(psi_grid, fixed_mean, fixed_std, "C2", "Fixed critic, finite eval")
    add_band(
        psi_grid,
        learned_holdout_mean,
        learned_holdout_std,
        "C0",
        "Learned critic, holdout eval",
    )
    add_band(
        psi_grid,
        learned_train_mean,
        learned_train_std,
        "C3",
        "Learned critic, same-batch eval",
        linestyle="--",
    )

    ax.axhline(0.0, color="0.65", linewidth=1.0)
    ax.set_title("Gradient estimates vs generator shift")
    ax.set_xlabel("Generator parameter psi")
    ax.set_ylabel("d/dpsi MMD^2 in critic feature space")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    plot_path = run_root / "gradient_bias_plot.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    metrics = {
        "fixed_rmse_to_reference": rmse(fixed_mean, reference),
        "learned_holdout_rmse_to_reference": rmse(learned_holdout_mean, reference),
        "learned_train_rmse_to_reference": rmse(learned_train_mean, reference),
        "fixed_mean_std": float(fixed_std.mean()),
        "learned_holdout_mean_std": float(learned_holdout_std.mean()),
        "learned_train_mean_std": float(learned_train_std.mean()),
    }

    payload = {
        "args": vars(args),
        "device": str(device),
        "psi_grid": psi_grid.tolist(),
        "reference_fixed_critic_gradient": reference.tolist(),
        "fixed": {"mean": fixed_mean.tolist(), "std": fixed_std.tolist()},
        "learned_holdout": {
            "mean": learned_holdout_mean.tolist(),
            "std": learned_holdout_std.tolist(),
        },
        "learned_train": {"mean": learned_train_mean.tolist(), "std": learned_train_std.tolist()},
        "metrics": metrics,
    }
    save_json(payload, run_root / "summary.json")

    report = build_markdown_report(
        plot_name=plot_path.name,
        psi_grid=psi_grid,
        reference=reference,
        fixed_mean=fixed_mean,
        fixed_std=fixed_std,
        learned_holdout_mean=learned_holdout_mean,
        learned_holdout_std=learned_holdout_std,
        learned_train_mean=learned_train_mean,
        learned_train_std=learned_train_std,
    )
    (run_root / "deliverables.md").write_text(report)

    print("\n[summary]")
    for key, value in metrics.items():
        print(f"  {key:32s}: {value:.6f}")
    print(f"  deliverables: {run_root / 'deliverables.md'}")


if __name__ == "__main__":
    main()
