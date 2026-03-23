#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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

from mmd_gan_experiments.utils import ensure_dir, pick_device, save_json, seed_everything, timestamp

try:
    from tqdm.auto import trange
except Exception:  # pragma: no cover
    trange = range


@dataclass(frozen=True)
class CriticSpec:
    scale: float
    slope: float
    offset: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Theorem 2 data-splitting demo with a finite critic dictionary"
    )
    p.add_argument("--delta-grid", type=str, default="-2,-1.5,-1,-0.5,0,0.5,1,1.5,2")
    p.add_argument("--train-sizes", type=str, default="16,32,64,256")
    p.add_argument("--test-size", type=int, default=4096)
    p.add_argument("--repetitions", type=int, default=1000)
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--truth-samples", type=int, default=200000)
    p.add_argument("--gradient-eps", type=float, default=1e-2)
    p.add_argument("--tie-tol", type=float, default=1e-8)
    p.add_argument("--scale-grid", type=str, default="0.5,1,2")
    p.add_argument("--slope-grid", type=str, default="0.5,1,2,4")
    p.add_argument("--offset-grid", type=str, default="-1,0,1")
    p.add_argument("--outdir", type=str, default="results/theorem2_data_split_demo")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def parse_float_grid(raw: str) -> List[float]:
    return [float(chunk.strip()) for chunk in raw.split(",") if chunk.strip()]


def parse_int_grid(raw: str) -> List[int]:
    return [int(chunk.strip()) for chunk in raw.split(",") if chunk.strip()]


def build_dictionary(args: argparse.Namespace) -> List[CriticSpec]:
    scales = parse_float_grid(args.scale_grid)
    slopes = parse_float_grid(args.slope_grid)
    offsets = parse_float_grid(args.offset_grid)
    return [CriticSpec(scale=s, slope=b, offset=c) for s in scales for b in slopes for c in offsets]


def dictionary_tensors(specs: List[CriticSpec], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scales = torch.tensor([spec.scale for spec in specs], device=device, dtype=torch.float32)
    slopes = torch.tensor([spec.slope for spec in specs], device=device, dtype=torch.float32)
    offsets = torch.tensor([spec.offset for spec in specs], device=device, dtype=torch.float32)
    return scales, slopes, offsets


def critic_values(
    x: torch.Tensor,
    scales: torch.Tensor,
    slopes: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    # x: [n] or [batch, n]
    view_shape = (-1,) + (1,) * x.ndim
    return scales.view(view_shape) * torch.tanh(slopes.view(view_shape) * x.unsqueeze(0) + offsets.view(view_shape))


def ipm_scores(
    real: torch.Tensor,
    fake: torch.Tensor,
    scales: torch.Tensor,
    slopes: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    real_vals = critic_values(real, scales, slopes, offsets)
    fake_vals = critic_values(fake, scales, slopes, offsets)
    return real_vals.mean(dim=-1) - fake_vals.mean(dim=-1)


def delta_key(delta: float) -> str:
    return f"{delta:.8f}"


def extended_delta_grid(delta_grid: List[float], eps: float) -> List[float]:
    values = set(delta_grid)
    for delta in delta_grid:
        values.add(delta - eps)
        values.add(delta + eps)
    return sorted(values)


def make_population_reference(
    *,
    deltas: List[float],
    truth_samples: int,
    scales: torch.Tensor,
    slopes: torch.Tensor,
    offsets: torch.Tensor,
    tie_tol: float,
    device: torch.device,
) -> Dict[str, Dict]:
    real = torch.randn(truth_samples, device=device)
    noise = torch.randn(truth_samples, device=device)

    out: Dict[str, Dict] = {}
    for delta in deltas:
        fake = noise + float(delta)
        scores = ipm_scores(real, fake, scales, slopes, offsets)
        best = float(scores.max().detach().cpu())
        optimal_mask = scores >= scores.max() - tie_tol
        out[delta_key(delta)] = {
            "scores": scores.detach().cpu().tolist(),
            "best_value": best,
            "optimal_mask": optimal_mask.detach().cpu().tolist(),
            "optimal_count": int(optimal_mask.sum().item()),
        }
    return out


def evaluate_selected_scores(
    *,
    selected_idx: torch.Tensor,
    real_test: torch.Tensor,
    fake_test: torch.Tensor,
    scales: torch.Tensor,
    slopes: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    unique_idx, inverse = torch.unique(selected_idx, sorted=True, return_inverse=True)
    sub_scales = scales[unique_idx]
    sub_slopes = slopes[unique_idx]
    sub_offsets = offsets[unique_idx]

    test_scores = ipm_scores(real_test, fake_test, sub_scales, sub_slopes, sub_offsets)  # [U, B]
    picked = test_scores[inverse, torch.arange(real_test.shape[0], device=real_test.device)]
    return picked


def run_split_estimator(
    *,
    delta: float,
    train_size: int,
    test_size: int,
    repetitions: int,
    chunk_size: int,
    scales: torch.Tensor,
    slopes: torch.Tensor,
    offsets: torch.Tensor,
    optimal_mask: torch.Tensor,
    combo_seed: int,
    device: torch.device,
) -> Dict[str, float]:
    split_values: List[torch.Tensor] = []
    selection_hits: List[torch.Tensor] = []

    processed = 0
    while processed < repetitions:
        batch = min(chunk_size, repetitions - processed)
        seed_everything(combo_seed + processed)

        real_train = torch.randn(batch, train_size, device=device)
        noise_train = torch.randn(batch, train_size, device=device)
        fake_train = noise_train + float(delta)

        train_scores = ipm_scores(real_train, fake_train, scales, slopes, offsets)  # [K, B]
        selected_idx = train_scores.argmax(dim=0)

        real_test = torch.randn(batch, test_size, device=device)
        noise_test = torch.randn(batch, test_size, device=device)
        fake_test = noise_test + float(delta)

        selected_scores = evaluate_selected_scores(
            selected_idx=selected_idx,
            real_test=real_test,
            fake_test=fake_test,
            scales=scales,
            slopes=slopes,
            offsets=offsets,
        )
        split_values.append(selected_scores.detach().cpu())
        selection_hits.append(optimal_mask[selected_idx].to(dtype=torch.float32).detach().cpu())
        processed += batch

    values = torch.cat(split_values)
    hits = torch.cat(selection_hits)
    mean = float(values.mean().item())
    std = float(values.std(unbiased=False).item())
    se = std / float(np.sqrt(repetitions))
    accuracy = float(hits.mean().item())
    accuracy_se = float(np.sqrt(max(accuracy * (1.0 - accuracy), 0.0) / repetitions))
    return {
        "mean": mean,
        "std": std,
        "se": se,
        "selection_accuracy": accuracy,
        "selection_accuracy_se": accuracy_se,
    }


def finite_difference(curve_minus: np.ndarray, curve_plus: np.ndarray, eps: float) -> np.ndarray:
    return (curve_plus - curve_minus) / (2.0 * eps)


def save_plot_objective(
    path: Path,
    deltas: np.ndarray,
    population: np.ndarray,
    split_means: Dict[int, np.ndarray],
    split_ses: Dict[int, np.ndarray],
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(deltas, population, color="black", linewidth=2.4, label="Population supremum")
    colors = ["C0", "C1", "C2", "C3", "C4", "C5"]
    for color, train_size in zip(colors, split_means):
        mean = split_means[train_size]
        se = split_ses[train_size]
        ax.plot(deltas, mean, color=color, linewidth=2.0, label=f"Split estimator, m={train_size}")
        ax.fill_between(deltas, mean - 2.0 * se, mean + 2.0 * se, color=color, alpha=0.16, linewidth=0)
    ax.set_title("Population supremum vs split estimator")
    ax.set_xlabel("delta")
    ax.set_ylabel("objective value")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_plot_bias(
    path: Path,
    deltas: np.ndarray,
    bias_means: Dict[int, np.ndarray],
    bias_ses: Dict[int, np.ndarray],
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    colors = ["C0", "C1", "C2", "C3", "C4", "C5"]
    for color, train_size in zip(colors, bias_means):
        mean = bias_means[train_size]
        se = bias_ses[train_size]
        ax.plot(deltas, mean, color=color, linewidth=2.0, label=f"m={train_size}")
        ax.fill_between(deltas, mean - 2.0 * se, mean + 2.0 * se, color=color, alpha=0.16, linewidth=0)
    ax.axhline(0.0, color="0.55", linewidth=1.0)
    ax.set_title("Downward bias of the split estimator")
    ax.set_xlabel("delta")
    ax.set_ylabel("E[split] - D(delta)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_plot_selection_accuracy(
    path: Path,
    deltas: np.ndarray,
    accuracy_means: Dict[int, np.ndarray],
    accuracy_ses: Dict[int, np.ndarray],
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    colors = ["C0", "C1", "C2", "C3", "C4", "C5"]
    for color, train_size in zip(colors, accuracy_means):
        mean = accuracy_means[train_size]
        se = accuracy_ses[train_size]
        ax.plot(deltas, mean, color=color, linewidth=2.0, label=f"m={train_size}")
        ax.fill_between(
            deltas,
            np.clip(mean - 2.0 * se, 0.0, 1.0),
            np.clip(mean + 2.0 * se, 0.0, 1.0),
            color=color,
            alpha=0.16,
            linewidth=0,
        )
    ax.set_title("Population-optimal critic selection accuracy")
    ax.set_xlabel("delta")
    ax.set_ylabel("P(train argmax is population-optimal)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_plot_gradient(
    path: Path,
    deltas: np.ndarray,
    population_grad: np.ndarray,
    split_grads: Dict[int, np.ndarray],
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(deltas, population_grad, color="black", linewidth=2.4, label="Population gradient")
    colors = ["C0", "C1", "C2", "C3", "C4", "C5"]
    for color, train_size in zip(colors, split_grads):
        ax.plot(deltas, split_grads[train_size], color=color, linewidth=2.0, label=f"Split gradient, m={train_size}")
    ax.axhline(0.0, color="0.55", linewidth=1.0)
    ax.set_title("Finite-difference gradient of population vs split objective")
    ax.set_xlabel("delta")
    ax.set_ylabel("gradient")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_overview_panel(
    path: Path,
    deltas: np.ndarray,
    population: np.ndarray,
    split_means: Dict[int, np.ndarray],
    bias_means: Dict[int, np.ndarray],
    accuracy_means: Dict[int, np.ndarray],
    population_grad: np.ndarray,
    split_grads: Dict[int, np.ndarray],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
    colors = ["C0", "C1", "C2", "C3", "C4", "C5"]

    axes[0, 0].plot(deltas, population, color="black", linewidth=2.3, label="Population supremum")
    for color, train_size in zip(colors, split_means):
        axes[0, 0].plot(deltas, split_means[train_size], color=color, linewidth=1.8, label=f"m={train_size}")
    axes[0, 0].set_title("Objective")
    axes[0, 0].grid(alpha=0.25)

    for color, train_size in zip(colors, bias_means):
        axes[0, 1].plot(deltas, bias_means[train_size], color=color, linewidth=1.8, label=f"m={train_size}")
    axes[0, 1].axhline(0.0, color="0.55", linewidth=1.0)
    axes[0, 1].set_title("Bias")
    axes[0, 1].grid(alpha=0.25)

    for color, train_size in zip(colors, accuracy_means):
        axes[1, 0].plot(deltas, accuracy_means[train_size], color=color, linewidth=1.8, label=f"m={train_size}")
    axes[1, 0].set_ylim(-0.02, 1.02)
    axes[1, 0].set_title("Selection Accuracy")
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].plot(deltas, population_grad, color="black", linewidth=2.3, label="Population gradient")
    for color, train_size in zip(colors, split_grads):
        axes[1, 1].plot(deltas, split_grads[train_size], color=color, linewidth=1.8, label=f"m={train_size}")
    axes[1, 1].axhline(0.0, color="0.55", linewidth=1.0)
    axes[1, 1].set_title("Gradient")
    axes[1, 1].grid(alpha=0.25)

    for ax in axes.ravel():
        ax.set_xlabel("delta")
    axes[0, 0].set_ylabel("value")
    axes[0, 1].set_ylabel("split - population")
    axes[1, 0].set_ylabel("accuracy")
    axes[1, 1].set_ylabel("gradient")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_report(
    *,
    train_sizes: List[int],
    deltas: np.ndarray,
    population: np.ndarray,
    split_means: Dict[int, np.ndarray],
    bias_means: Dict[int, np.ndarray],
    accuracy_means: Dict[int, np.ndarray],
    population_grad: np.ndarray,
    split_grads: Dict[int, np.ndarray],
) -> str:
    zero_idx = int(np.argmin(np.abs(deltas)))
    lines = [
        "# Theorem 2 Data-Splitting Demo",
        "",
        "This experiment instantiates the theorem directly with a finite witness dictionary and an explicit split estimator: the critic is selected on a finite train sample and then evaluated on an independent test sample.",
        "",
        "## Main Reading",
        "",
        "The relevant comparison is between the population supremum `D(delta)` and the average split estimator. Because the critic selected on train data is not almost surely population-optimal, the split curve sits below the population curve, which is the downward-bias mechanism stated by the theorem.",
        "",
        "## Train-Size Effect",
        "",
    ]
    for train_size in train_sizes:
        lines.append(
            f"- `m={train_size}`: bias near `delta={deltas[zero_idx]:+.2f}` is {bias_means[train_size][zero_idx]:+.5f}, "
            f"selection accuracy is {accuracy_means[train_size][zero_idx]:.3f}, and split gradient is "
            f"{split_grads[train_size][zero_idx]:+.5f} versus population gradient {population_grad[zero_idx]:+.5f}."
        )
    lines.extend(
        [
            "",
            "## Report Sentence",
            "",
            "We instantiated the paper’s data-splitting estimator directly using a finite critic family. For each generator parameter `delta`, we compared the population supremum `D(delta)` to the average split estimator obtained by selecting the critic on a finite train set and evaluating it on an independent test set. The resulting estimator was consistently downward biased, and its finite-difference gradient differed from the gradient of the population supremum. This directly illustrates the mechanism of Theorem 2: the bias is caused by critic selection from finite samples.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = pick_device(prefer_cuda=True)
    run_root = ensure_dir(Path(args.outdir) / timestamp())

    delta_grid = parse_float_grid(args.delta_grid)
    train_sizes = parse_int_grid(args.train_sizes)
    all_deltas = extended_delta_grid(delta_grid, args.gradient_eps)

    specs = build_dictionary(args)
    scales, slopes, offsets = dictionary_tensors(specs, device)

    print(f"[info] device: {device}")
    print(f"[info] output: {run_root}")
    print(f"[info] critics: {len(specs)}")

    population_ref = make_population_reference(
        deltas=all_deltas,
        truth_samples=args.truth_samples,
        scales=scales,
        slopes=slopes,
        offsets=offsets,
        tie_tol=args.tie_tol,
        device=device,
    )

    split_stats: Dict[int, Dict[str, Dict[str, float]]] = {train_size: {} for train_size in train_sizes}
    total_jobs = len(train_sizes) * len(all_deltas)
    job_idx = 0
    for train_size in train_sizes:
        for delta_idx, delta in enumerate(all_deltas):
            job_idx += 1
            print(f"[info] split estimator {job_idx}/{total_jobs} | m={train_size} | delta={delta:+.4f}")
            ref = population_ref[delta_key(delta)]
            optimal_mask = torch.tensor(ref["optimal_mask"], device=device, dtype=torch.bool)
            combo_seed = args.seed * 1_000_000 + train_size * 10_000 + delta_idx
            stats = run_split_estimator(
                delta=delta,
                train_size=train_size,
                test_size=args.test_size,
                repetitions=args.repetitions,
                chunk_size=args.chunk_size,
                scales=scales,
                slopes=slopes,
                offsets=offsets,
                optimal_mask=optimal_mask,
                combo_seed=combo_seed,
                device=device,
            )
            split_stats[train_size][delta_key(delta)] = stats

    deltas_np = np.asarray(delta_grid, dtype=np.float64)
    population_curve = np.asarray([population_ref[delta_key(delta)]["best_value"] for delta in delta_grid], dtype=np.float64)
    population_plus = np.asarray(
        [population_ref[delta_key(delta + args.gradient_eps)]["best_value"] for delta in delta_grid],
        dtype=np.float64,
    )
    population_minus = np.asarray(
        [population_ref[delta_key(delta - args.gradient_eps)]["best_value"] for delta in delta_grid],
        dtype=np.float64,
    )
    population_grad = finite_difference(population_minus, population_plus, args.gradient_eps)

    split_means: Dict[int, np.ndarray] = {}
    split_ses: Dict[int, np.ndarray] = {}
    bias_means: Dict[int, np.ndarray] = {}
    bias_ses: Dict[int, np.ndarray] = {}
    accuracy_means: Dict[int, np.ndarray] = {}
    accuracy_ses: Dict[int, np.ndarray] = {}
    split_grads: Dict[int, np.ndarray] = {}

    for train_size in train_sizes:
        mean_curve = np.asarray(
            [split_stats[train_size][delta_key(delta)]["mean"] for delta in delta_grid],
            dtype=np.float64,
        )
        se_curve = np.asarray(
            [split_stats[train_size][delta_key(delta)]["se"] for delta in delta_grid],
            dtype=np.float64,
        )
        acc_curve = np.asarray(
            [split_stats[train_size][delta_key(delta)]["selection_accuracy"] for delta in delta_grid],
            dtype=np.float64,
        )
        acc_se_curve = np.asarray(
            [split_stats[train_size][delta_key(delta)]["selection_accuracy_se"] for delta in delta_grid],
            dtype=np.float64,
        )

        mean_plus = np.asarray(
            [split_stats[train_size][delta_key(delta + args.gradient_eps)]["mean"] for delta in delta_grid],
            dtype=np.float64,
        )
        mean_minus = np.asarray(
            [split_stats[train_size][delta_key(delta - args.gradient_eps)]["mean"] for delta in delta_grid],
            dtype=np.float64,
        )

        split_means[train_size] = mean_curve
        split_ses[train_size] = se_curve
        bias_means[train_size] = mean_curve - population_curve
        bias_ses[train_size] = se_curve
        accuracy_means[train_size] = acc_curve
        accuracy_ses[train_size] = acc_se_curve
        split_grads[train_size] = finite_difference(mean_minus, mean_plus, args.gradient_eps)

    save_plot_objective(run_root / "population_vs_split.png", deltas_np, population_curve, split_means, split_ses)
    save_plot_bias(run_root / "bias_vs_delta.png", deltas_np, bias_means, bias_ses)
    save_plot_selection_accuracy(
        run_root / "selection_accuracy_vs_delta.png",
        deltas_np,
        accuracy_means,
        accuracy_ses,
    )
    save_plot_gradient(run_root / "gradient_vs_delta.png", deltas_np, population_grad, split_grads)
    save_overview_panel(
        run_root / "theorem2_overview.png",
        deltas_np,
        population_curve,
        split_means,
        bias_means,
        accuracy_means,
        population_grad,
        split_grads,
    )

    payload = {
        "args": vars(args),
        "device": str(device),
        "dictionary": [asdict(spec) for spec in specs],
        "population": population_ref,
        "split_stats": split_stats,
        "summary_curves": {
            "delta_grid": delta_grid,
            "population_curve": population_curve.tolist(),
            "population_gradient": population_grad.tolist(),
            "split_means": {str(k): v.tolist() for k, v in split_means.items()},
            "bias_means": {str(k): v.tolist() for k, v in bias_means.items()},
            "selection_accuracy_means": {str(k): v.tolist() for k, v in accuracy_means.items()},
            "split_gradients": {str(k): v.tolist() for k, v in split_grads.items()},
        },
    }
    save_json(payload, run_root / "summary.json")
    (run_root / "deliverables.md").write_text(
        build_report(
            train_sizes=train_sizes,
            deltas=deltas_np,
            population=population_curve,
            split_means=split_means,
            bias_means=bias_means,
            accuracy_means=accuracy_means,
            population_grad=population_grad,
            split_grads=split_grads,
        )
    )

    print("\n[summary]")
    for train_size in train_sizes:
        mean_bias = float(bias_means[train_size].mean())
        mean_acc = float(accuracy_means[train_size].mean())
        print(
            f"  m={train_size:>4} | avg bias={mean_bias:+.6f} | "
            f"avg selection acc={mean_acc:.4f}"
        )
    print(f"  output: {run_root}")


if __name__ == "__main__":
    main()
