from __future__ import annotations

import numpy as np
import torch
from skimage import color, data, transform


def load_demo_image(img_size: int = 64, channels: int = 1, name: str = "camera") -> torch.Tensor:
    """Load a deterministic demo image as a BCHW tensor in [0, 1]."""
    if name == "camera":
        img = data.camera()
    elif name == "astronaut":
        img = data.astronaut()
    elif name == "coins":
        img = data.coins()
    else:
        img = data.camera()

    img = img.astype(np.float32) / 255.0
    if img.ndim == 3 and channels == 1:
        img = color.rgb2gray(img).astype(np.float32)
    if img.ndim == 2 and channels == 3:
        img = np.repeat(img[..., None], 3, axis=2)

    img = transform.resize(
        img,
        (img_size, img_size) if channels == 1 else (img_size, img_size, channels),
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)

    if channels == 1:
        tensor = torch.from_numpy(img)[None, None]
    else:
        tensor = torch.from_numpy(img).permute(2, 0, 1)[None]
    return tensor.clamp(0, 1)

