from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def _rand_mask(shape: tuple[int, ...], keep: float, device: torch.device | str, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=device).manual_seed(seed)
    return torch.rand(shape, generator=gen, device=device) < keep


def _split_mask(base: torch.Tensor, train_ratio: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    device = base.device
    gen = torch.Generator(device=device).manual_seed(seed)
    score = torch.rand(base.shape, generator=gen, device=device)
    train = base & (score < train_ratio)
    val = base & ~train
    return train, val


def _complex_mse(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (a - b) * mask
    denom = mask.sum().clamp_min(1)
    return diff.abs().square().sum() / denom


def _masked_mse(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (a - b) * mask
    denom = mask.sum().clamp_min(1)
    return diff.square().sum() / denom


def _gaussian_kernel(size: int, sigma: float, channels: int, device: torch.device | str) -> torch.Tensor:
    coords = torch.arange(size, device=device) - (size - 1) / 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel[None, None].repeat(channels, 1, 1, 1)


@dataclass
class SplitOperator:
    name: str
    y: torch.Tensor
    x_shape: tuple[int, int, int, int]
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    split_domain: str = "pixel"
    matrix: torch.Tensor | None = None
    kernel: torch.Tensor | None = None
    sr_factor: int = 2

    def A(self, x: torch.Tensor) -> torch.Tensor:
        if self.name == "inpainting":
            return x
        if self.name == "fourier":
            return torch.fft.fft2(x, norm="ortho")
        if self.name == "compressed_sensing":
            assert self.matrix is not None
            flat = x.flatten(1)
            return flat @ self.matrix.T
        if self.name == "deblur":
            assert self.kernel is not None
            pad = self.kernel.shape[-1] // 2
            return F.conv2d(x, self.kernel, padding=pad, groups=x.shape[1])
        if self.name == "superres":
            return F.avg_pool2d(x, self.sr_factor)
        raise ValueError(f"Unknown operator {self.name}")

    def loss(self, x: torch.Tensor, split: str) -> torch.Tensor:
        mask = self.train_mask if split == "train" else self.val_mask
        pred = self.A(x)
        target = self.y
        if self.split_domain == "fourier" and self.name in {"deblur", "superres"}:
            pred = torch.fft.fft2(pred, norm="ortho")
            target = torch.fft.fft2(target, norm="ortho")
            return _complex_mse(pred, target, mask)
        if torch.is_complex(pred):
            return _complex_mse(pred, target, mask)
        return _masked_mse(pred, target, mask)


def build_operator(
    x: torch.Tensor,
    operator: str,
    split_ratio: float = 0.8,
    noise_sigma: float = 0.03,
    seed: int = 0,
    split_domain: str = "pixel",
    blur_sigma: float = 1.6,
    sr_factor: int = 2,
    sampling_ratio: float = 0.5,
) -> SplitOperator:
    device = x.device
    _, c, h, w = x.shape
    gen = torch.Generator(device=device).manual_seed(seed + 13)

    if operator == "inpainting":
        obs = _rand_mask((1, 1, h, w), sampling_ratio, device, seed + 1).float()
        y = x * obs + noise_sigma * torch.randn(x.shape, generator=gen, device=device) * obs
        train, val = _split_mask(obs.bool(), split_ratio, seed + 2)
        return SplitOperator(operator, y, tuple(x.shape), train.float(), val.float(), "pixel")

    if operator == "fourier":
        full = torch.ones((1, 1, h, w), dtype=torch.bool, device=device)
        obs = _rand_mask((1, 1, h, w), sampling_ratio, device, seed + 3) | _center_mask(h, w, 0.08, device)
        y_clean = torch.fft.fft2(x, norm="ortho")
        noise = noise_sigma * (torch.randn(x.shape, generator=gen, device=device) + 1j * torch.randn(x.shape, generator=gen, device=device))
        y = (y_clean + noise) * obs
        train, val = _split_mask(full & obs, split_ratio, seed + 4)
        return SplitOperator(operator, y, tuple(x.shape), train.float(), val.float(), "fourier")

    if operator == "compressed_sensing":
        n = c * h * w
        m = max(8, int(sampling_ratio * n))
        matrix = torch.randn(m, n, generator=gen, device=device) / (m**0.5)
        y = x.flatten(1) @ matrix.T
        y = y + noise_sigma * torch.randn(y.shape, generator=gen, device=device)
        all_rows = torch.ones_like(y, dtype=torch.bool)
        train, val = _split_mask(all_rows, split_ratio, seed + 5)
        return SplitOperator(operator, y, tuple(x.shape), train.float(), val.float(), "vector", matrix=matrix)

    if operator == "deblur":
        kernel = _gaussian_kernel(size=9, sigma=blur_sigma, channels=c, device=device)
        y_clean = F.conv2d(x, kernel, padding=4, groups=c)
        y = y_clean + noise_sigma * torch.randn(x.shape, generator=gen, device=device)
        mask_shape = (1, 1, h, w)
        base = torch.ones(mask_shape, dtype=torch.bool, device=device)
        train, val = _split_mask(base, split_ratio, seed + 6)
        if split_domain == "fourier":
            train, val = _split_mask(base, split_ratio, seed + 7)
        return SplitOperator(operator, y, tuple(x.shape), train.float(), val.float(), split_domain, kernel=kernel)

    if operator == "superres":
        y_clean = F.avg_pool2d(x, sr_factor)
        y = y_clean + noise_sigma * torch.randn(y_clean.shape, generator=gen, device=device)
        _, _, lh, lw = y.shape
        base = torch.ones((1, 1, lh, lw), dtype=torch.bool, device=device)
        train, val = _split_mask(base, split_ratio, seed + 8)
        return SplitOperator(operator, y, tuple(x.shape), train.float(), val.float(), split_domain, sr_factor=sr_factor)

    raise ValueError(f"Unsupported operator: {operator}")


def _center_mask(h: int, w: int, fraction: float, device: torch.device | str) -> torch.Tensor:
    ch = max(1, int(h * fraction))
    cw = max(1, int(w * fraction))
    mask = torch.zeros((1, 1, h, w), dtype=torch.bool, device=device)
    hs = h // 2 - ch // 2
    ws = w // 2 - cw // 2
    mask[:, :, hs : hs + ch, ws : ws + cw] = True
    return mask

