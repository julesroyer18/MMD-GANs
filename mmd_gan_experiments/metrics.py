from __future__ import annotations

from typing import Dict, Tuple

import torch

from .utils import to_uint8_image


def _load_torchmetrics():
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance
    except Exception as exc:  # pragma: no cover - dependency-dependent path
        return None, None, str(exc)
    return FrechetInceptionDistance, KernelInceptionDistance, None


@torch.no_grad()
def compute_fid_kid(
    generator: torch.nn.Module,
    real_loader,
    z_dim: int,
    device: torch.device,
    num_samples: int = 10_000,
) -> Tuple[Dict[str, float], str | None]:
    FID, KID, err = _load_torchmetrics()
    if err is not None:
        return {}, f"Skipping FID/KID: torchmetrics unavailable ({err})"

    fid = FID(normalize=False).to(device)
    kid = KID(subset_size=1000, normalize=False).to(device)

    seen_real = 0
    for batch in real_loader:
        imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
        imgs = imgs.to(device)
        n = min(imgs.shape[0], num_samples - seen_real)
        imgs = to_uint8_image(imgs[:n])
        fid.update(imgs, real=True)
        kid.update(imgs, real=True)
        seen_real += n
        if seen_real >= num_samples:
            break

    seen_fake = 0
    while seen_fake < num_samples:
        bsz = min(256, num_samples - seen_fake)
        z = torch.randn(bsz, z_dim, device=device)
        fake = generator(z)
        fake = to_uint8_image(fake)
        fid.update(fake, real=False)
        kid.update(fake, real=False)
        seen_fake += bsz

    kid_mean, kid_std = kid.compute()
    out = {
        "fid": float(fid.compute().detach().cpu()),
        "kid_mean": float(kid_mean.detach().cpu()),
        "kid_std": float(kid_std.detach().cpu()),
    }
    return out, None
