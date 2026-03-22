#!/usr/bin/env python3
"""
Minimal PyTorch MMD-GAN demo.

Features
--------
- MMD-GAN objective with linear / RBF / rational-quadratic kernels.
- Witness-function gradient penalty.
- Low-dimensional `make_moons` experiment.
- Image experiment path for MNIST / CIFAR-10 / ImageFolder datasets.

Notes
-----
This script is designed as a compact research starter rather than a polished training
framework. The image pipeline is intentionally simple and easy to adapt.

Examples
--------
# 2D make_moons
python mmd_gan_demo.py --dataset moons --epochs 300 --kernel rq --use-gp

# CIFAR-10 (downloads locally when run on your machine)
python mmd_gan_demo.py --dataset cifar10 --data-root ./data --epochs 100 --kernel rq --use-gp

# MNIST
python mmd_gan_demo.py --dataset mnist --data-root ./data --epochs 50 --kernel rq --use-gp

# Custom folder dataset (expects class subfolders, ImageFolder style)
python mmd_gan_demo.py --dataset folder --folder-root /path/to/images --image-size 64
"""
from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Optional imports used only by some paths.
try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:
    from sklearn.datasets import make_moons
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover
    make_moons = None
    StandardScaler = None

try:
    import torchvision
    import torchvision.transforms as T
    import torchvision.utils as vutils
except Exception:  # pragma: no cover
    torchvision = None
    T = None
    vutils = None


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainStats:
    critic_losses: List[float]
    gen_losses: List[float]
    mmd_values: List[float]
    gp_values: List[float]


# -----------------------------------------------------------------------------
# Kernels and MMD
# -----------------------------------------------------------------------------


def pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # x: [n, d], y: [m, d]
    x_norm = (x ** 2).sum(dim=1, keepdim=True)
    y_norm = (y ** 2).sum(dim=1, keepdim=True).T
    dist = x_norm + y_norm - 2.0 * x @ y.T
    return torch.clamp(dist, min=0.0)


class KernelMMD(nn.Module):
    def __init__(
        self,
        kernel: str = "rq",
        scales: Sequence[float] = (0.2, 0.5, 1.0, 2.0, 5.0),
        alpha: float = 1.0,
        linear_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.kernel = kernel.lower()
        self.scales = tuple(float(s) for s in scales)
        self.alpha = float(alpha)
        self.linear_weight = float(linear_weight)
        if self.kernel not in {"linear", "rbf", "rq"}:
            raise ValueError(f"Unsupported kernel: {self.kernel}")

    def kernel_matrix(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.kernel == "linear":
            return x @ y.T

        d2 = pairwise_sq_dists(x, y)
        mats = []
        for s in self.scales:
            if self.kernel == "rbf":
                mats.append(torch.exp(-d2 / (2.0 * s * s)))
            elif self.kernel == "rq":
                mats.append((1.0 + d2 / (2.0 * self.alpha * s * s)).pow(-self.alpha))
        out = torch.stack(mats, dim=0).mean(dim=0)
        if self.linear_weight != 0.0:
            out = out + self.linear_weight * (x @ y.T)
        return out

    def mmd2_biased(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        k_xx = self.kernel_matrix(x, x)
        k_yy = self.kernel_matrix(y, y)
        k_xy = self.kernel_matrix(x, y)
        return k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()

    def mmd2_unbiased(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n, m = x.size(0), y.size(0)
        if n < 2 or m < 2:
            return self.mmd2_biased(x, y)
        k_xx = self.kernel_matrix(x, x)
        k_yy = self.kernel_matrix(y, y)
        k_xy = self.kernel_matrix(x, y)
        sum_xx = (k_xx.sum() - torch.diag(k_xx).sum()) / (n * (n - 1))
        sum_yy = (k_yy.sum() - torch.diag(k_yy).sum()) / (m * (m - 1))
        sum_xy = k_xy.mean()
        return sum_xx + sum_yy - 2.0 * sum_xy


# -----------------------------------------------------------------------------
# Networks
# -----------------------------------------------------------------------------


class MLPGenerator2D(nn.Module):
    def __init__(self, z_dim: int = 8, hidden: int = 128, out_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class MLPFeatureMap2D(nn.Module):
    def __init__(self, in_dim: int = 2, hidden: int = 128, feat_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, feat_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvGenerator(nn.Module):
    def __init__(self, z_dim: int = 128, channels: int = 3, base: int = 64, image_size: int = 32) -> None:
        super().__init__()
        if image_size not in {28, 32, 64}:
            raise ValueError("ConvGenerator currently supports image sizes 28, 32, 64")
        self.z_dim = z_dim
        self.image_size = image_size
        if image_size == 28:
            self.net = nn.Sequential(
                nn.ConvTranspose2d(z_dim, base * 4, 7, 1, 0, bias=False),
                nn.BatchNorm2d(base * 4),
                nn.ReLU(True),
                nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(base * 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(base * 2, channels, 4, 2, 1, bias=False),
                nn.Tanh(),
            )
        elif image_size == 32:
            self.net = nn.Sequential(
                nn.ConvTranspose2d(z_dim, base * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(base * 8),
                nn.ReLU(True),
                nn.ConvTranspose2d(base * 8, base * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(base * 4),
                nn.ReLU(True),
                nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(base * 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(base * 2, channels, 4, 2, 1, bias=False),
                nn.Tanh(),
            )
        else:  # 64
            self.net = nn.Sequential(
                nn.ConvTranspose2d(z_dim, base * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(base * 8),
                nn.ReLU(True),
                nn.ConvTranspose2d(base * 8, base * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(base * 4),
                nn.ReLU(True),
                nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(base * 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(base * 2, base, 4, 2, 1, bias=False),
                nn.BatchNorm2d(base),
                nn.ReLU(True),
                nn.ConvTranspose2d(base, channels, 4, 2, 1, bias=False),
                nn.Tanh(),
            )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.view(z.size(0), self.z_dim, 1, 1))


class ConvFeatureMap(nn.Module):
    def __init__(self, channels: int = 3, base: int = 64, feat_dim: int = 16, image_size: int = 32) -> None:
        super().__init__()
        if image_size not in {28, 32, 64}:
            raise ValueError("ConvFeatureMap currently supports image sizes 28, 32, 64")
        blocks = [
            nn.Conv2d(channels, base, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if image_size in {32, 64}:
            blocks += [
                nn.Conv2d(base * 2, base * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(base * 4),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        if image_size == 64:
            blocks += [
                nn.Conv2d(base * 4, base * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(base * 8),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        self.features = nn.Sequential(*blocks)
        with torch.no_grad():
            dummy = torch.zeros(1, channels, image_size, image_size)
            flat_dim = self.features(dummy).view(1, -1).size(1)
        self.proj = nn.Linear(flat_dim, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = h.view(x.size(0), -1)
        return self.proj(h)


# -----------------------------------------------------------------------------
# Witness and gradient penalty
# -----------------------------------------------------------------------------


def empirical_witness_fn(query_feats: torch.Tensor, real_feats: torch.Tensor, fake_feats: torch.Tensor, mmd: KernelMMD) -> torch.Tensor:
    """
    Empirical witness function evaluated at query features:
        mean_i k(real_i, q) - mean_j k(fake_j, q)
    Returns shape [num_query].
    """
    k_rq = mmd.kernel_matrix(real_feats, query_feats).mean(dim=0)
    k_fq = mmd.kernel_matrix(fake_feats, query_feats).mean(dim=0)
    return k_rq - k_fq


def witness_gradient_penalty(
    feature_map: nn.Module,
    real_x: torch.Tensor,
    fake_x: torch.Tensor,
    mmd: KernelMMD,
    target_norm: float = 1.0,
) -> torch.Tensor:
    batch_size = min(real_x.size(0), fake_x.size(0))
    if real_x.size(0) != batch_size:
        real_x = real_x[:batch_size]
    if fake_x.size(0) != batch_size:
        fake_x = fake_x[:batch_size]

    shape = [batch_size] + [1] * (real_x.ndim - 1)
    eps = torch.rand(shape, device=real_x.device)
    x_hat = eps * real_x + (1.0 - eps) * fake_x
    x_hat.requires_grad_(True)

    real_feats = feature_map(real_x).detach()
    fake_feats = feature_map(fake_x).detach()
    query_feats = feature_map(x_hat)
    witness_vals = empirical_witness_fn(query_feats, real_feats, fake_feats, mmd)

    grads = autograd.grad(
        outputs=witness_vals.sum(),
        inputs=x_hat,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grads = grads.view(batch_size, -1)
    grad_norm = grads.norm(2, dim=1)
    gp = ((grad_norm - target_norm) ** 2).mean()
    return gp


# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------


def build_moons_loader(n_samples: int, noise: float, batch_size: int, seed: int) -> DataLoader:
    if make_moons is None or StandardScaler is None:
        raise ImportError("scikit-learn is required for the moons dataset path")
    x, _ = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    x = StandardScaler().fit_transform(x).astype(np.float32)
    dataset = TensorDataset(torch.from_numpy(x))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


def build_image_loader(args: argparse.Namespace) -> Tuple[DataLoader, int, int]:
    if torchvision is None or T is None:
        raise ImportError("torchvision is required for image datasets")

    if args.dataset == "mnist":
        channels = 1
        image_size = 28
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,)),
        ])
        dataset = torchvision.datasets.MNIST(root=args.data_root, train=True, download=True, transform=transform)
    elif args.dataset == "cifar10":
        channels = 3
        image_size = 32
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = torchvision.datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=transform)
    elif args.dataset == "folder":
        channels = 3
        image_size = args.image_size
        transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = torchvision.datasets.ImageFolder(root=args.folder_root, transform=transform)
    else:
        raise ValueError(f"Unsupported image dataset: {args.dataset}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader, channels, image_size


# -----------------------------------------------------------------------------
# Training loops
# -----------------------------------------------------------------------------


def train_moons(args: argparse.Namespace, device: torch.device) -> TrainStats:
    loader = build_moons_loader(args.n_samples, args.moons_noise, args.batch_size, args.seed)

    G = MLPGenerator2D(z_dim=args.z_dim, hidden=args.hidden_dim).to(device)
    H = MLPFeatureMap2D(in_dim=2, hidden=args.hidden_dim, feat_dim=args.feat_dim).to(device)
    mmd = KernelMMD(args.kernel, scales=args.kernel_scales, alpha=args.rq_alpha, linear_weight=args.linear_weight)

    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    opt_h = torch.optim.Adam(H.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    stats = TrainStats([], [], [], [])
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    fixed_z = torch.randn(args.eval_batch_size, args.z_dim, device=device)

    step = 0
    for epoch in range(args.epochs):
        for (real_x,) in loader:
            real_x = real_x.to(device)
            batch_size = real_x.size(0)

            # Critic / feature map updates
            for _ in range(args.n_critic):
                z = torch.randn(batch_size, args.z_dim, device=device)
                fake_x = G(z).detach()

                real_feats = H(real_x)
                fake_feats = H(fake_x)
                mmd2 = mmd.mmd2_unbiased(real_feats, fake_feats)
                loss_h = -mmd2
                gp_val = torch.tensor(0.0, device=device)
                if args.use_gp:
                    gp_val = witness_gradient_penalty(H, real_x, fake_x, mmd, target_norm=args.gp_target)
                    loss_h = loss_h + args.lambda_gp * gp_val
                if args.lambda_act > 0.0:
                    loss_h = loss_h + args.lambda_act * (real_feats.pow(2).mean() + fake_feats.pow(2).mean())

                opt_h.zero_grad(set_to_none=True)
                loss_h.backward()
                opt_h.step()

            # Generator update
            z = torch.randn(batch_size, args.z_dim, device=device)
            fake_x = G(z)
            real_feats = H(real_x).detach() if args.detach_real_features else H(real_x)
            fake_feats = H(fake_x)
            gen_mmd2 = mmd.mmd2_unbiased(real_feats, fake_feats)
            loss_g = gen_mmd2

            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

            stats.critic_losses.append(float(loss_h.item()))
            stats.gen_losses.append(float(loss_g.item()))
            stats.mmd_values.append(float(gen_mmd2.item()))
            stats.gp_values.append(float(gp_val.item()))

            if step % args.log_every == 0:
                print(
                    f"[moons] epoch {epoch:03d} step {step:05d} | "
                    f"critic {loss_h.item():+.4f} | gen {loss_g.item():+.4f} | "
                    f"mmd {gen_mmd2.item():.4f} | gp {gp_val.item():.4f}"
                )
            step += 1

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            save_moons_visuals(G, H, mmd, fixed_z, loader, device, out_dir, epoch)

    save_training_curves(stats, out_dir, prefix="moons")
    return stats


@torch.no_grad()
def save_moons_visuals(
    G: nn.Module,
    H: nn.Module,
    mmd: KernelMMD,
    fixed_z: torch.Tensor,
    loader: DataLoader,
    device: torch.device,
    out_dir: str,
    epoch: int,
) -> None:
    if plt is None:
        return

    # Get a fresh batch of real data for plotting
    real_x = next(iter(loader))[0].to(device)
    fake_x = G(fixed_z).detach()

    plt.figure(figsize=(6, 6))
    plt.scatter(real_x[:, 0].cpu().numpy(), real_x[:, 1].cpu().numpy(), s=12, alpha=0.5, label="real")
    plt.scatter(fake_x[:, 0].cpu().numpy(), fake_x[:, 1].cpu().numpy(), s=12, alpha=0.5, label="fake")
    plt.legend()
    plt.title(f"make_moons: real vs fake (epoch {epoch + 1})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"moons_scatter_epoch_{epoch + 1:04d}.png"), dpi=160)
    plt.close()

    # Optional witness contour in input space.
    xmin = min(real_x[:, 0].min(), fake_x[:, 0].min()).item() - 1.0
    xmax = max(real_x[:, 0].max(), fake_x[:, 0].max()).item() + 1.0
    ymin = min(real_x[:, 1].min(), fake_x[:, 1].min()).item() - 1.0
    ymax = max(real_x[:, 1].max(), fake_x[:, 1].max()).item() + 1.0
    gx, gy = torch.meshgrid(
        torch.linspace(xmin, xmax, 120, device=device),
        torch.linspace(ymin, ymax, 120, device=device),
        indexing="ij",
    )
    grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    grid_feats = H(grid)
    real_feats = H(real_x)
    fake_feats = H(fake_x)
    witness = empirical_witness_fn(grid_feats, real_feats, fake_feats, mmd).view(120, 120).cpu().numpy()

    plt.figure(figsize=(7, 6))
    plt.contourf(gx.cpu().numpy(), gy.cpu().numpy(), witness, levels=40)
    plt.colorbar(label="empirical witness")
    plt.scatter(real_x[:, 0].cpu().numpy(), real_x[:, 1].cpu().numpy(), s=8, c="white", alpha=0.7)
    plt.scatter(fake_x[:, 0].cpu().numpy(), fake_x[:, 1].cpu().numpy(), s=8, c="red", alpha=0.7)
    plt.title(f"Empirical witness contour (epoch {epoch + 1})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"moons_witness_epoch_{epoch + 1:04d}.png"), dpi=160)
    plt.close()



def train_images(args: argparse.Namespace, device: torch.device) -> TrainStats:
    loader, channels, image_size = build_image_loader(args)

    G = ConvGenerator(z_dim=args.z_dim, channels=channels, base=args.g_base, image_size=image_size).to(device)
    H = ConvFeatureMap(channels=channels, base=args.d_base, feat_dim=args.feat_dim, image_size=image_size).to(device)
    mmd = KernelMMD(args.kernel, scales=args.kernel_scales, alpha=args.rq_alpha, linear_weight=args.linear_weight)

    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    opt_h = torch.optim.Adam(H.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    stats = TrainStats([], [], [], [])
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    fixed_z = torch.randn(args.eval_batch_size, args.z_dim, device=device)

    step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            real_x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            batch_size = real_x.size(0)

            for _ in range(args.n_critic):
                z = torch.randn(batch_size, args.z_dim, device=device)
                fake_x = G(z).detach()

                real_feats = H(real_x)
                fake_feats = H(fake_x)
                mmd2 = mmd.mmd2_unbiased(real_feats, fake_feats)
                loss_h = -mmd2
                gp_val = torch.tensor(0.0, device=device)
                if args.use_gp:
                    gp_val = witness_gradient_penalty(H, real_x, fake_x, mmd, target_norm=args.gp_target)
                    loss_h = loss_h + args.lambda_gp * gp_val
                if args.lambda_act > 0.0:
                    loss_h = loss_h + args.lambda_act * (real_feats.pow(2).mean() + fake_feats.pow(2).mean())

                opt_h.zero_grad(set_to_none=True)
                loss_h.backward()
                opt_h.step()

            z = torch.randn(batch_size, args.z_dim, device=device)
            fake_x = G(z)
            real_feats = H(real_x).detach() if args.detach_real_features else H(real_x)
            fake_feats = H(fake_x)
            gen_mmd2 = mmd.mmd2_unbiased(real_feats, fake_feats)
            loss_g = gen_mmd2

            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

            stats.critic_losses.append(float(loss_h.item()))
            stats.gen_losses.append(float(loss_g.item()))
            stats.mmd_values.append(float(gen_mmd2.item()))
            stats.gp_values.append(float(gp_val.item()))

            if step % args.log_every == 0:
                print(
                    f"[{args.dataset}] epoch {epoch:03d} step {step:05d} | "
                    f"critic {loss_h.item():+.4f} | gen {loss_g.item():+.4f} | "
                    f"mmd {gen_mmd2.item():.4f} | gp {gp_val.item():.4f}"
                )
            step += 1

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            save_image_grid(G, fixed_z, out_dir, epoch, args.dataset)

        if args.max_steps > 0 and step >= args.max_steps:
            break

    save_training_curves(stats, out_dir, prefix=args.dataset)
    save_checkpoint({
        "G": G.state_dict(),
        "H": H.state_dict(),
        "args": vars(args),
    }, os.path.join(out_dir, f"{args.dataset}_final.pt"))
    return stats


@torch.no_grad()
def save_image_grid(G: nn.Module, fixed_z: torch.Tensor, out_dir: str, epoch: int, prefix: str) -> None:
    if vutils is None:
        return
    fake = G(fixed_z).detach().cpu()
    grid = vutils.make_grid(fake, nrow=int(math.sqrt(fake.size(0))), normalize=True, value_range=(-1, 1))
    vutils.save_image(grid, os.path.join(out_dir, f"{prefix}_grid_epoch_{epoch + 1:04d}.png"))



def save_training_curves(stats: TrainStats, out_dir: str, prefix: str) -> None:
    if plt is None:
        return
    curves = {
        f"{prefix}_critic_loss.png": stats.critic_losses,
        f"{prefix}_gen_loss.png": stats.gen_losses,
        f"{prefix}_mmd.png": stats.mmd_values,
        f"{prefix}_gp.png": stats.gp_values,
    }
    for name, values in curves.items():
        if not values:
            continue
        plt.figure(figsize=(7, 4))
        plt.plot(values)
        plt.title(name.replace("_", " ").replace(".png", ""))
        plt.xlabel("training step")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, name), dpi=160)
        plt.close()



def save_checkpoint(obj: dict, path: str) -> None:
    torch.save(obj, path)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Minimal MMD-GAN demo")
    p.add_argument("--dataset", type=str, default="moons", choices=["moons", "mnist", "cifar10", "folder"])
    p.add_argument("--folder-root", type=str, default="", help="Path for ImageFolder datasets")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--out-dir", type=str, default="./mmd_gan_outputs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Data / optimization
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--n-critic", type=int, default=5)
    p.add_argument("--lr-g", type=float, default=1e-4)
    p.add_argument("--lr-d", type=float, default=1e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--beta2", type=float, default=0.9)
    p.add_argument("--max-steps", type=int, default=-1, help="For debugging; negative means no limit")

    # Kernel / MMD
    p.add_argument("--kernel", type=str, default="rq", choices=["linear", "rbf", "rq"])
    p.add_argument("--kernel-scales", type=float, nargs="+", default=[0.2, 0.5, 1.0, 2.0, 5.0])
    p.add_argument("--rq-alpha", type=float, default=1.0)
    p.add_argument("--linear-weight", type=float, default=0.0)
    p.add_argument("--feat-dim", type=int, default=16)

    # Regularization
    p.add_argument("--use-gp", action="store_true")
    p.add_argument("--lambda-gp", type=float, default=1.0)
    p.add_argument("--gp-target", type=float, default=1.0)
    p.add_argument("--lambda-act", type=float, default=0.0, help="Optional feature activation penalty")
    p.add_argument("--detach-real-features", action="store_true", help="Detach H(real) during generator step")

    # Moons path
    p.add_argument("--n-samples", type=int, default=10_000)
    p.add_argument("--moons-noise", type=float, default=0.08)
    p.add_argument("--z-dim", type=int, default=8)
    p.add_argument("--hidden-dim", type=int, default=128)

    # Image path
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--g-base", type=int, default=64)
    p.add_argument("--d-base", type=int, default=64)

    # Logging
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=10)
    return p


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Output directory: {args.out_dir}")

    if args.dataset == "moons":
        train_moons(args, device)
    else:
        train_images(args, device)

    print("Done.")


if __name__ == "__main__":
    main()
