from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from .mmd import witness_values


KernelFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class SmallToyGenerator(nn.Module):
    """Small MLP generator used by the protocol-oriented make_moons experiment."""

    def __init__(self, z_dim: int = 4, hidden_dim: int = 64, out_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class SmallToyFeatureCritic(nn.Module):
    """Two-hidden-layer feature map for low-dimensional MMD experiments."""

    def __init__(self, in_dim: int = 2, hidden_dim: int = 64, feat_dim: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, feat_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BiasFeatureCritic(nn.Module):
    """Tiny 1D feature map for the finite-sample critic-bias illustration."""

    def __init__(self, hidden_dim: int = 64, feat_dim: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, feat_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def mean_feature_norm(features: torch.Tensor) -> torch.Tensor:
    return features.norm(dim=1).mean()


def witness_gradient_penalty(
    critic: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    kernel: KernelFn,
    lambda_gp: float,
) -> torch.Tensor:
    if lambda_gp <= 0.0:
        return torch.zeros((), device=real.device)

    bsz = real.shape[0]
    alpha_shape = [bsz] + [1] * (real.ndim - 1)
    alpha = torch.rand(alpha_shape, device=real.device)
    interp = alpha * real + (1.0 - alpha) * fake
    interp.requires_grad_(True)

    feat_interp = critic(interp)
    feat_real = critic(real).detach()
    feat_fake = critic(fake).detach()
    witness = witness_values(feat_interp, feat_real, feat_fake, kernel)

    (grads,) = torch.autograd.grad(
        outputs=witness,
        inputs=interp,
        grad_outputs=torch.ones_like(witness),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    grads = grads.view(bsz, -1)
    return lambda_gp * (grads.norm(2, dim=1) - 1.0).pow(2).mean()
