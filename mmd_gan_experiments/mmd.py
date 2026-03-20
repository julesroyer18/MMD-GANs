from __future__ import annotations

from typing import Callable

import torch


KernelFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def mmd2_biased(x: torch.Tensor, y: torch.Tensor, kernel: KernelFn) -> torch.Tensor:
    k_xx = kernel(x, x)
    k_yy = kernel(y, y)
    k_xy = kernel(x, y)
    return k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()


def mmd2_unbiased(x: torch.Tensor, y: torch.Tensor, kernel: KernelFn) -> torch.Tensor:
    n = x.shape[0]
    m = y.shape[0]
    if n < 2 or m < 2:
        raise ValueError("Need at least two samples per set for unbiased MMD estimator.")

    k_xx = kernel(x, x)
    k_yy = kernel(y, y)
    k_xy = kernel(x, y)

    sum_xx = (k_xx.sum() - torch.diagonal(k_xx).sum()) / (n * (n - 1))
    sum_yy = (k_yy.sum() - torch.diagonal(k_yy).sum()) / (m * (m - 1))
    sum_xy = k_xy.mean()
    return sum_xx + sum_yy - 2.0 * sum_xy


def witness_values(
    eval_feats: torch.Tensor,
    real_feats: torch.Tensor,
    fake_feats: torch.Tensor,
    kernel: KernelFn,
) -> torch.Tensor:
    # f*(t) = E[k(t, x)] - E[k(t, y)] in RKHS witness form.
    k_tr = kernel(eval_feats, real_feats)
    k_tf = kernel(eval_feats, fake_feats)
    return k_tr.mean(dim=1) - k_tf.mean(dim=1)
