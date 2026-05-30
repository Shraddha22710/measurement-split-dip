from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvDIP(nn.Module):
    """Small convolutional decoder suitable for DIP pilot experiments."""

    def __init__(
        self,
        out_channels: int,
        out_size: int,
        latent_channels: int = 64,
        hidden_channels: int = 64,
        start_size: int = 8,
    ) -> None:
        super().__init__()
        self.out_size = out_size
        layers: list[nn.Module] = []
        channels = latent_channels
        size = start_size
        while size < out_size:
            layers.extend(
                [
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(channels, hidden_channels, 3, padding=1),
                    nn.BatchNorm2d(hidden_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                    nn.BatchNorm2d(hidden_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            channels = hidden_channels
            size *= 2
        layers.append(nn.Conv2d(channels, out_channels, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.net(z)
        if x.shape[-1] != self.out_size:
            x = F.interpolate(x, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        return torch.sigmoid(x)


def make_latent(
    batch: int,
    channels: int,
    start_size: int,
    device: torch.device | str,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.rand(batch, channels, start_size, start_size, generator=generator, device=device)

