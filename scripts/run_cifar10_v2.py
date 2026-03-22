#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from torchvision import datasets, transforms
    from torchvision.utils import save_image
except ModuleNotFoundError as exc:  # pragma: no cover - dependency error path
    raise SystemExit(
        f"Missing dependency: {exc}. Install requirements with `pip install -r requirements.txt`."
    ) from exc

from mmd_gan_experiments.kernels import build_kernel
from mmd_gan_experiments.metrics import compute_fid_kid
from mmd_gan_experiments.mmd import mmd2_unbiased
from mmd_gan_experiments.models_cifar import DCGANFeatureCritic, DCGANGenerator, DCGANScalarCritic
from mmd_gan_experiments.utils import (
    device_summary,
    ensure_dir,
    grad_penalty_features,
    grad_penalty_scalar,
    infinite_loader,
    maybe_deterministic,
    pick_device,
    save_json,
    seed_everything,
    timestamp,
)

try:
    from tqdm.auto import trange
except Exception:  # pragma: no cover
    trange = range


RBF_SIGMAS = (2.0, 5.0, 10.0, 20.0, 40.0, 80.0)
RQ_ALPHAS = (0.2, 0.5, 1.0, 2.0, 5.0)


@dataclass
class CifarRunSpec:
    name: str
    method: str
    critic_size: str
    critic_base_channels: int
    kernel: str | None
    feature_dim: int | None
    use_gp: bool
    activation_penalty: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Protocol-oriented CIFAR-10 experiment suite (v2)")
    p.add_argument("--steps", type=int, default=30_000, help="Generator updates")
    p.add_argument("--critic-steps", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr-g", type=float, default=1e-4)
    p.add_argument("--lr-c", type=float, default=1e-4)
    p.add_argument("--z-dim", type=int, default=128)
    p.add_argument("--generator-base-channels", type=int, default=64)
    p.add_argument("--feature-dim", type=int, default=16)
    p.add_argument("--gp-lambda-mmd", type=float, default=1.0)
    p.add_argument("--gp-lambda-wgan", type=float, default=10.0)
    p.add_argument("--activation-penalty", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--sample-every", type=int, default=5_000)
    p.add_argument("--metrics-every", type=int, default=5_000)
    p.add_argument("--save-every", type=int, default=5_000)
    p.add_argument("--checkpoint-metric-samples", type=int, default=10_000)
    p.add_argument("--final-metric-samples", type=int, default=25_000)
    p.add_argument("--data-root", type=str, default="data/cifar10")
    p.add_argument("--outdir", type=str, default="results/cifar10_v2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--include-wgan", action="store_true")
    p.add_argument(
        "--suite",
        choices=["course", "paper"],
        default="course",
        help="course: three MMD runs; paper: add WGAN-GP baseline automatically",
    )
    return p.parse_args()


def get_cifar_loaders(
    data_root: Path,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_ds = datasets.CIFAR10(root=str(data_root), train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR10(root=str(data_root), train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    eval_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, eval_loader


def build_mmd_kernel(name: str):
    if name == "rq_star":
        return build_kernel("rq", rq_alphas=RQ_ALPHAS, rq_add_linear=True)
    if name == "rbf":
        return build_kernel("rbf", rbf_sigmas=RBF_SIGMAS)
    raise ValueError(f"Unsupported MMD kernel: {name}")


def mean_feature_norm(feats: torch.Tensor) -> float:
    return float(feats.norm(dim=1).mean().detach().cpu())


def save_grid(generator: nn.Module, fixed_noise: torch.Tensor, path: Path) -> None:
    generator.eval()
    with torch.no_grad():
        fake = generator(fixed_noise)
    generator.train()
    save_image((fake + 1.0) * 0.5, str(path), nrow=8)


def metric_sample_budget(args: argparse.Namespace, step: int) -> int:
    return args.final_metric_samples if step == args.steps else args.checkpoint_metric_samples


def build_suite(args: argparse.Namespace) -> List[CifarRunSpec]:
    suite = [
        CifarRunSpec(
            name="mmd_small_rq_star",
            method="mmd",
            critic_size="small",
            critic_base_channels=16,
            kernel="rq_star",
            feature_dim=args.feature_dim,
            use_gp=True,
            activation_penalty=args.activation_penalty,
        ),
        CifarRunSpec(
            name="mmd_large_rq_star",
            method="mmd",
            critic_size="large",
            critic_base_channels=64,
            kernel="rq_star",
            feature_dim=args.feature_dim,
            use_gp=True,
            activation_penalty=args.activation_penalty,
        ),
        CifarRunSpec(
            name="mmd_small_rbf",
            method="mmd",
            critic_size="small",
            critic_base_channels=16,
            kernel="rbf",
            feature_dim=args.feature_dim,
            use_gp=True,
            activation_penalty=args.activation_penalty,
        ),
    ]

    if args.suite == "paper" or args.include_wgan:
        suite.append(
            CifarRunSpec(
                name="wgan_large",
                method="wgan",
                critic_size="large",
                critic_base_channels=64,
                kernel=None,
                feature_dim=None,
                use_gp=True,
                activation_penalty=0.0,
            )
        )
    return suite


def train_mmd_run(
    spec: CifarRunSpec,
    args: argparse.Namespace,
    device: torch.device,
    run_root: Path,
    train_loader: DataLoader,
    eval_loader: DataLoader,
) -> Dict:
    assert spec.kernel is not None
    assert spec.feature_dim is not None

    kernel = build_mmd_kernel(spec.kernel)
    run_dir = ensure_dir(run_root / spec.name)

    G = DCGANGenerator(z_dim=args.z_dim, base_channels=args.generator_base_channels).to(device)
    C = DCGANFeatureCritic(base_channels=spec.critic_base_channels, feature_dim=spec.feature_dim).to(device)

    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.9))
    opt_c = torch.optim.Adam(C.parameters(), lr=args.lr_c, betas=(0.5, 0.9))

    logs: Dict[str, List[float]] = {
        "step": [],
        "mmd2": [],
        "loss_g": [],
        "loss_c": [],
        "gp": [],
        "feature_norm_real": [],
        "feature_norm_fake": [],
        "elapsed_sec": [],
    }
    metric_logs: List[Dict] = []

    data_iter = infinite_loader(train_loader)
    fixed_noise = torch.randn(64, args.z_dim, device=device)
    start = perf_counter()

    for step in trange(1, args.steps + 1, desc=spec.name):
        gp_value = torch.zeros((), device=device)
        loss_c = torch.zeros((), device=device)
        mmd2 = torch.zeros((), device=device)
        feat_real = torch.zeros((), device=device)
        feat_fake = torch.zeros((), device=device)

        for _ in range(args.critic_steps):
            real = next(data_iter)[0].to(device, non_blocking=True)
            fake = G(torch.randn(args.batch_size, args.z_dim, device=device)).detach()

            feat_real = C(real)
            feat_fake = C(fake)
            mmd2 = mmd2_unbiased(feat_real, feat_fake, kernel)
            gp_value = (
                grad_penalty_features(C, real, fake, args.gp_lambda_mmd)
                if spec.use_gp
                else torch.zeros((), device=device)
            )
            activation_pen = spec.activation_penalty * (feat_real.pow(2).mean() + feat_fake.pow(2).mean())
            loss_c = -mmd2 + gp_value + activation_pen

            opt_c.zero_grad(set_to_none=True)
            loss_c.backward()
            opt_c.step()

        real = next(data_iter)[0].to(device, non_blocking=True)
        fake = G(torch.randn(args.batch_size, args.z_dim, device=device))
        feat_real = C(real)
        feat_fake = C(fake)
        loss_g = mmd2_unbiased(feat_real, feat_fake, kernel)

        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        opt_g.step()

        if step % args.log_every == 0 or step == args.steps:
            logs["step"].append(step)
            logs["mmd2"].append(float(mmd2.detach().cpu()))
            logs["loss_g"].append(float(loss_g.detach().cpu()))
            logs["loss_c"].append(float(loss_c.detach().cpu()))
            logs["gp"].append(float(gp_value.detach().cpu()))
            logs["feature_norm_real"].append(mean_feature_norm(feat_real))
            logs["feature_norm_fake"].append(mean_feature_norm(feat_fake))
            logs["elapsed_sec"].append(perf_counter() - start)

        if step % args.sample_every == 0 or step == args.steps:
            save_grid(G, fixed_noise, run_dir / f"samples_step-{step:07d}.png")

        if args.metrics_every > 0 and (step % args.metrics_every == 0 or step == args.steps):
            G.eval()
            metrics, warning = compute_fid_kid(
                generator=G,
                real_loader=eval_loader,
                z_dim=args.z_dim,
                device=device,
                num_samples=metric_sample_budget(args, step),
            )
            G.train()

            entry = {
                "step": step,
                "num_samples": metric_sample_budget(args, step),
                "elapsed_sec": perf_counter() - start,
                "feature_norm_real": mean_feature_norm(feat_real),
                "feature_norm_fake": mean_feature_norm(feat_fake),
                **metrics,
            }
            if warning is not None:
                entry["warning"] = warning
                print(f"[warn] {warning}")
            metric_logs.append(entry)

        if step % args.save_every == 0 or step == args.steps:
            torch.save(
                {
                    "step": step,
                    "spec": asdict(spec),
                    "args": vars(args),
                    "generator": G.state_dict(),
                    "critic": C.state_dict(),
                    "logs": logs,
                    "metric_logs": metric_logs,
                },
                run_dir / f"checkpoint_step-{step:07d}.pt",
            )

    result = {
        "spec": asdict(spec),
        "logs": logs,
        "metrics": metric_logs,
        "run_dir": str(run_dir),
    }
    save_json(result, run_dir / "summary.json")
    return result


def train_wgan_run(
    spec: CifarRunSpec,
    args: argparse.Namespace,
    device: torch.device,
    run_root: Path,
    train_loader: DataLoader,
    eval_loader: DataLoader,
) -> Dict:
    run_dir = ensure_dir(run_root / spec.name)

    G = DCGANGenerator(z_dim=args.z_dim, base_channels=args.generator_base_channels).to(device)
    D = DCGANScalarCritic(base_channels=spec.critic_base_channels).to(device)

    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr_c, betas=(0.5, 0.9))

    logs: Dict[str, List[float]] = {
        "step": [],
        "wass": [],
        "loss_g": [],
        "loss_d": [],
        "gp": [],
        "elapsed_sec": [],
    }
    metric_logs: List[Dict] = []

    data_iter = infinite_loader(train_loader)
    fixed_noise = torch.randn(64, args.z_dim, device=device)
    start = perf_counter()

    for step in trange(1, args.steps + 1, desc=spec.name):
        gp_value = torch.zeros((), device=device)
        wass = torch.zeros((), device=device)

        for _ in range(args.critic_steps):
            real = next(data_iter)[0].to(device, non_blocking=True)
            fake = G(torch.randn(args.batch_size, args.z_dim, device=device)).detach()

            d_real = D(real)
            d_fake = D(fake)
            wass = d_real.mean() - d_fake.mean()
            gp_value = grad_penalty_scalar(D, real, fake, args.gp_lambda_wgan)
            loss_d = -wass + gp_value

            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

        fake = G(torch.randn(args.batch_size, args.z_dim, device=device))
        loss_g = -D(fake).mean()

        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        opt_g.step()

        if step % args.log_every == 0 or step == args.steps:
            logs["step"].append(step)
            logs["wass"].append(float(wass.detach().cpu()))
            logs["loss_g"].append(float(loss_g.detach().cpu()))
            logs["loss_d"].append(float(loss_d.detach().cpu()))
            logs["gp"].append(float(gp_value.detach().cpu()))
            logs["elapsed_sec"].append(perf_counter() - start)

        if step % args.sample_every == 0 or step == args.steps:
            save_grid(G, fixed_noise, run_dir / f"samples_step-{step:07d}.png")

        if args.metrics_every > 0 and (step % args.metrics_every == 0 or step == args.steps):
            G.eval()
            metrics, warning = compute_fid_kid(
                generator=G,
                real_loader=eval_loader,
                z_dim=args.z_dim,
                device=device,
                num_samples=metric_sample_budget(args, step),
            )
            G.train()

            entry = {
                "step": step,
                "num_samples": metric_sample_budget(args, step),
                "elapsed_sec": perf_counter() - start,
                **metrics,
            }
            if warning is not None:
                entry["warning"] = warning
                print(f"[warn] {warning}")
            metric_logs.append(entry)

        if step % args.save_every == 0 or step == args.steps:
            torch.save(
                {
                    "step": step,
                    "spec": asdict(spec),
                    "args": vars(args),
                    "generator": G.state_dict(),
                    "critic": D.state_dict(),
                    "logs": logs,
                    "metric_logs": metric_logs,
                },
                run_dir / f"checkpoint_step-{step:07d}.pt",
            )

    result = {
        "spec": asdict(spec),
        "logs": logs,
        "metrics": metric_logs,
        "run_dir": str(run_dir),
    }
    save_json(result, run_dir / "summary.json")
    return result


def build_matrix_markdown(specs: List[CifarRunSpec]) -> str:
    lines = [
        "# CIFAR-10 v2 Matrix",
        "",
        "| Model | Critic size | Kernel | GP | Activation penalty |",
        "| --- | --- | --- | --- | --- |",
    ]
    for spec in specs:
        kernel = spec.kernel if spec.kernel is not None else "-"
        activation = "yes" if spec.activation_penalty > 0 else "-"
        lines.append(
            f"| {spec.method.upper()} | {spec.critic_size} | {kernel} | "
            f"{'yes' if spec.use_gp else 'no'} | {activation} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    maybe_deterministic(args.deterministic)

    device = pick_device(prefer_cuda=True)
    run_root = ensure_dir(Path(args.outdir) / timestamp())
    suite = build_suite(args)

    print(f"[info] device: {device_summary(device)}")
    print(f"[info] output root: {run_root}")
    print(f"[info] runs: {[spec.name for spec in suite]}")

    train_loader, eval_loader = get_cifar_loaders(
        data_root=Path(args.data_root),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    all_results: List[Dict] = []
    for spec in suite:
        if spec.method == "mmd":
            result = train_mmd_run(spec, args, device, run_root, train_loader, eval_loader)
        else:
            result = train_wgan_run(spec, args, device, run_root, train_loader, eval_loader)
        all_results.append(result)

    summary = {
        "args": vars(args),
        "device": device_summary(device),
        "suite": [asdict(spec) for spec in suite],
        "results": all_results,
        "run_root": str(run_root),
    }
    save_json(summary, run_root / "summary_all.json")
    (run_root / "comparison_matrix.md").write_text(build_matrix_markdown(suite))

    print("\n[summary] completed runs")
    for result in all_results:
        spec = result["spec"]
        print(f"  {spec['name']}: {result['run_dir']}")


if __name__ == "__main__":
    main()
