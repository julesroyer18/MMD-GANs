#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CIFAR-10 MMD-GAN / WGAN-GP experiments")
    p.add_argument("--method", choices=["mmd", "wgan"], default="mmd")
    p.add_argument("--kernel-list", type=str, default="rq", help="Comma-separated kernels for MMD method")
    p.add_argument("--run-baseline-wgan", action="store_true", help="Also run WGAN-GP after MMD runs")

    p.add_argument("--steps", type=int, default=30000, help="Generator updates")
    p.add_argument("--critic-steps", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)

    p.add_argument("--lr-g", type=float, default=1e-4)
    p.add_argument("--lr-c", type=float, default=1e-4)
    p.add_argument("--z-dim", type=int, default=128)
    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--feature-dim", type=int, default=128)

    p.add_argument("--no-gp", action="store_false", dest="use_gp", help="Disable gradient penalty")
    p.set_defaults(use_gp=True)
    p.add_argument("--gp-lambda-mmd", type=float, default=1.0)
    p.add_argument("--gp-lambda-wgan", type=float, default=10.0)
    p.add_argument("--activation-penalty", type=float, default=1.0)

    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--sample-every", type=int, default=2000)
    p.add_argument("--metrics-every", type=int, default=5000)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--metric-samples", type=int, default=10000)

    p.add_argument("--data-root", type=str, default="data/cifar10")
    p.add_argument("--outdir", type=str, default="results/cifar10")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")
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


def _sample_and_save(generator: nn.Module, fixed_noise: torch.Tensor, path: Path) -> None:
    generator.eval()
    with torch.no_grad():
        fake = generator(fixed_noise)
    generator.train()
    save_image((fake + 1.0) * 0.5, str(path), nrow=8)


def train_mmd(
    kernel_name: str,
    args: argparse.Namespace,
    device: torch.device,
    run_root: Path,
    train_loader: DataLoader,
    eval_loader: DataLoader,
) -> Dict:
    run_dir = ensure_dir(run_root / f"mmd_{kernel_name}")
    kernel = build_kernel(kernel_name)

    G = DCGANGenerator(z_dim=args.z_dim, base_channels=args.base_channels).to(device)
    C = DCGANFeatureCritic(base_channels=args.base_channels, feature_dim=args.feature_dim).to(device)

    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.9))
    opt_c = torch.optim.Adam(C.parameters(), lr=args.lr_c, betas=(0.5, 0.9))

    logs: Dict[str, List[float]] = {"step": [], "mmd2": [], "loss_g": [], "loss_c": [], "gp": []}
    metric_logs: List[Dict] = []

    data_iter = infinite_loader(train_loader)
    fixed_noise = torch.randn(64, args.z_dim, device=device)

    for step in trange(1, args.steps + 1, desc=f"MMD-{kernel_name}"):
        gp_value = torch.zeros((), device=device)

        for _ in range(args.critic_steps):
            real = next(data_iter)[0].to(device, non_blocking=True)
            fake = G(torch.randn(args.batch_size, args.z_dim, device=device)).detach()

            feat_real = C(real)
            feat_fake = C(fake)

            mmd2 = mmd2_unbiased(feat_real, feat_fake, kernel)
            gp_value = (
                grad_penalty_features(C, real, fake, args.gp_lambda_mmd)
                if args.use_gp
                else torch.zeros((), device=device)
            )
            activation_pen = args.activation_penalty * (feat_real.pow(2).mean() + feat_fake.pow(2).mean())
            loss_c = -mmd2 + gp_value + activation_pen

            opt_c.zero_grad(set_to_none=True)
            loss_c.backward()
            opt_c.step()

        real = next(data_iter)[0].to(device, non_blocking=True)
        fake = G(torch.randn(args.batch_size, args.z_dim, device=device))
        loss_g = mmd2_unbiased(C(real), C(fake), kernel)

        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        opt_g.step()

        if step % args.log_every == 0 or step == args.steps:
            logs["step"].append(step)
            logs["mmd2"].append(float(mmd2.detach().cpu()))
            logs["loss_g"].append(float(loss_g.detach().cpu()))
            logs["loss_c"].append(float(loss_c.detach().cpu()))
            logs["gp"].append(float(gp_value.detach().cpu()))

        if step % args.sample_every == 0 or step == args.steps:
            _sample_and_save(G, fixed_noise, run_dir / f"samples_step-{step:07d}.png")

        if args.metrics_every > 0 and (step % args.metrics_every == 0 or step == args.steps):
            G.eval()
            metrics, warning = compute_fid_kid(
                generator=G,
                real_loader=eval_loader,
                z_dim=args.z_dim,
                device=device,
                num_samples=args.metric_samples,
            )
            G.train()

            entry = {"step": step, **metrics}
            if warning is not None:
                entry["warning"] = warning
            metric_logs.append(entry)
            if warning is not None:
                print(f"[warn] {warning}")

        if step % args.save_every == 0 or step == args.steps:
            torch.save(
                {
                    "step": step,
                    "generator": G.state_dict(),
                    "critic": C.state_dict(),
                    "args": vars(args),
                    "kernel": kernel_name,
                    "logs": logs,
                    "metric_logs": metric_logs,
                },
                run_dir / f"checkpoint_step-{step:07d}.pt",
            )

    result = {
        "method": "mmd",
        "kernel": kernel_name,
        "logs": logs,
        "metrics": metric_logs,
        "run_dir": str(run_dir),
    }
    save_json(result, run_dir / "summary.json")
    return result


def train_wgan(
    args: argparse.Namespace,
    device: torch.device,
    run_root: Path,
    train_loader: DataLoader,
    eval_loader: DataLoader,
) -> Dict:
    run_dir = ensure_dir(run_root / "wgan_gp")

    G = DCGANGenerator(z_dim=args.z_dim, base_channels=args.base_channels).to(device)
    D = DCGANScalarCritic(base_channels=args.base_channels).to(device)

    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr_c, betas=(0.5, 0.9))

    logs: Dict[str, List[float]] = {"step": [], "wass": [], "loss_g": [], "loss_d": [], "gp": []}
    metric_logs: List[Dict] = []

    data_iter = infinite_loader(train_loader)
    fixed_noise = torch.randn(64, args.z_dim, device=device)

    for step in trange(1, args.steps + 1, desc="WGAN-GP"):
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

        if step % args.sample_every == 0 or step == args.steps:
            _sample_and_save(G, fixed_noise, run_dir / f"samples_step-{step:07d}.png")

        if args.metrics_every > 0 and (step % args.metrics_every == 0 or step == args.steps):
            G.eval()
            metrics, warning = compute_fid_kid(
                generator=G,
                real_loader=eval_loader,
                z_dim=args.z_dim,
                device=device,
                num_samples=args.metric_samples,
            )
            G.train()

            entry = {"step": step, **metrics}
            if warning is not None:
                entry["warning"] = warning
            metric_logs.append(entry)
            if warning is not None:
                print(f"[warn] {warning}")

        if step % args.save_every == 0 or step == args.steps:
            torch.save(
                {
                    "step": step,
                    "generator": G.state_dict(),
                    "critic": D.state_dict(),
                    "args": vars(args),
                    "logs": logs,
                    "metric_logs": metric_logs,
                },
                run_dir / f"checkpoint_step-{step:07d}.pt",
            )

    result = {
        "method": "wgan-gp",
        "logs": logs,
        "metrics": metric_logs,
        "run_dir": str(run_dir),
    }
    save_json(result, run_dir / "summary.json")
    return result


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    maybe_deterministic(args.deterministic)

    device = pick_device(prefer_cuda=True)
    run_root = ensure_dir(Path(args.outdir) / timestamp())

    print(f"[info] device: {device_summary(device)}")
    print(f"[info] output root: {run_root}")

    train_loader, eval_loader = get_cifar_loaders(
        data_root=Path(args.data_root),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    all_results: List[Dict] = []

    if args.method == "mmd":
        kernels = [k.strip().lower() for k in args.kernel_list.split(",") if k.strip()]
        for kernel in kernels:
            all_results.append(train_mmd(kernel, args, device, run_root, train_loader, eval_loader))

        if args.run_baseline_wgan:
            all_results.append(train_wgan(args, device, run_root, train_loader, eval_loader))
    else:
        all_results.append(train_wgan(args, device, run_root, train_loader, eval_loader))

    summary = {
        "args": vars(args),
        "device": device_summary(device),
        "results": all_results,
        "run_root": str(run_root),
    }
    save_json(summary, run_root / "summary_all.json")

    print("\n[summary] runs completed:")
    for r in all_results:
        tag = f"{r['method']} ({r.get('kernel', '-')})"
        print(f"  - {tag}: {r['run_dir']}")


if __name__ == "__main__":
    main()
