from __future__ import annotations

import numpy as np
import torch
from PIL import Image
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


def load_image_path(path: str, img_size: int = 96, channels: int = 3) -> torch.Tensor:
    """Load an image file as a BCHW tensor in [0, 1]."""
    mode = "RGB" if channels == 3 else "L"
    img = Image.open(path).convert(mode)
    arr = np.asarray(img).astype(np.float32) / 255.0
    if channels == 1 and arr.ndim == 3:
        arr = color.rgb2gray(arr).astype(np.float32)
    arr = center_crop_square(arr)
    arr = transform.resize(
        arr,
        (img_size, img_size) if channels == 1 else (img_size, img_size, channels),
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)
    if channels == 1:
        return torch.from_numpy(arr)[None, None].clamp(0, 1)
    return torch.from_numpy(arr).permute(2, 0, 1)[None].clamp(0, 1)


def center_crop_square(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return arr[y0 : y0 + side, x0 : x0 + side, ...]
