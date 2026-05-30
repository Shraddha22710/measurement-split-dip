from __future__ import annotations

import math

import numpy as np
import torch
from skimage.metrics import structural_similarity


def psnr(x_hat: torch.Tensor, x: torch.Tensor, data_range: float = 1.0) -> float:
    mse = torch.mean((x_hat.detach().clamp(0, 1) - x.detach().clamp(0, 1)) ** 2).item()
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def ssim(x_hat: torch.Tensor, x: torch.Tensor) -> float:
    a = x_hat.detach().clamp(0, 1).cpu().numpy()[0]
    b = x.detach().clamp(0, 1).cpu().numpy()[0]
    if a.shape[0] == 1:
        return float(structural_similarity(a[0], b[0], data_range=1.0))
    return float(
        structural_similarity(
            np.moveaxis(a, 0, -1),
            np.moveaxis(b, 0, -1),
            channel_axis=-1,
            data_range=1.0,
        )
    )


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) < window:
        return values[:]
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out.append(float(np.mean(values[lo : i + 1])))
    return out

