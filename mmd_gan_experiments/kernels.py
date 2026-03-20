from __future__ import annotations

from typing import Callable, Iterable

import torch


KernelFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1, keepdim=True).t()
    d2 = x2 + y2 - 2.0 * x @ y.t()
    return d2.clamp_min(0.0)


def linear_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x @ y.t()


def rbf_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    sigmas: Iterable[float] = (2.0, 5.0, 10.0, 20.0, 40.0, 80.0),
) -> torch.Tensor:
    d2 = _pairwise_sq_dists(x, y)
    out = torch.zeros_like(d2)
    for sigma in sigmas:
        gamma = 1.0 / (2.0 * (sigma**2))
        out = out + torch.exp(-gamma * d2)
    return out


def rq_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    alphas: Iterable[float] = (0.2, 0.5, 1.0, 2.0, 5.0),
    add_linear: bool = True,
) -> torch.Tensor:
    d2 = _pairwise_sq_dists(x, y)
    out = torch.zeros_like(d2)
    for alpha in alphas:
        out = out + (1.0 + d2 / (2.0 * alpha)).pow(-alpha)
    if add_linear:
        out = out + linear_kernel(x, y)
    return out


def build_kernel(
    name: str,
    rbf_sigmas: Iterable[float] = (2.0, 5.0, 10.0, 20.0, 40.0, 80.0),
    rq_alphas: Iterable[float] = (0.2, 0.5, 1.0, 2.0, 5.0),
    rq_add_linear: bool = True,
) -> KernelFn:
    lname = name.lower()
    if lname == "linear":
        return linear_kernel
    if lname == "rbf":
        return lambda a, b: rbf_kernel(a, b, sigmas=rbf_sigmas)
    if lname == "rq":
        return lambda a, b: rq_kernel(a, b, alphas=rq_alphas, add_linear=rq_add_linear)
    raise ValueError(f"Unknown kernel '{name}'. Choose from: linear, rbf, rq")
