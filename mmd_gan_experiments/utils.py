from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator

import numpy as np
import torch


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_summary(device: torch.device) -> str:
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        return f"cuda ({name})"
    return "cpu"


def infinite_loader(loader: Iterable) -> Iterator:
    while True:
        for batch in loader:
            yield batch


def grad_penalty_features(
    critic: torch.nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    lambda_gp: float,
) -> torch.Tensor:
    if lambda_gp <= 0.0:
        return torch.zeros((), device=real.device)

    bsz = real.shape[0]
    alpha_shape = [bsz] + [1] * (real.ndim - 1)
    alpha = torch.rand(alpha_shape, device=real.device)
    interp = alpha * real + (1.0 - alpha) * fake
    interp.requires_grad_(True)

    feat = critic(interp)
    grad_outputs = torch.ones_like(feat)
    (grads,) = torch.autograd.grad(
        outputs=feat,
        inputs=interp,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    grads = grads.view(bsz, -1)
    penalty = (grads.norm(2, dim=1) - 1.0).pow(2).mean()
    return lambda_gp * penalty


def grad_penalty_scalar(
    critic: torch.nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    lambda_gp: float,
) -> torch.Tensor:
    if lambda_gp <= 0.0:
        return torch.zeros((), device=real.device)

    bsz = real.shape[0]
    alpha_shape = [bsz] + [1] * (real.ndim - 1)
    alpha = torch.rand(alpha_shape, device=real.device)
    interp = alpha * real + (1.0 - alpha) * fake
    interp.requires_grad_(True)

    out = critic(interp)
    grad_outputs = torch.ones_like(out)
    (grads,) = torch.autograd.grad(
        outputs=out,
        inputs=interp,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    grads = grads.view(bsz, -1)
    penalty = (grads.norm(2, dim=1) - 1.0).pow(2).mean()
    return lambda_gp * penalty


def to_uint8_image(x: torch.Tensor) -> torch.Tensor:
    x = (x.clamp(-1.0, 1.0) + 1.0) * 127.5
    return x.to(torch.uint8)


def maybe_deterministic(deterministic: bool) -> None:
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
