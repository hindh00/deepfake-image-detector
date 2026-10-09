"""Image transforms for the dual-branch model.

Spatial branch gets a standard ImageNet-normalized RGB tensor (ViT input).
Frequency branch gets a log-scaled 2D FFT magnitude spectrum of the grayscale image.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

IMAGE_SIZE = 224

# Matches timm's ImageNet-21k pretraining stats for vit_small_patch16_224.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def build_spatial_transform(train: bool) -> T.Compose:
    """RGB transform pipeline for the spatial (ViT) branch."""
    if train:
        return T.Compose(
            [
                T.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            ]
        )
    return T.Compose(
        [
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )


def fft_magnitude_log1p(image: Image.Image, size: int = IMAGE_SIZE) -> torch.Tensor:
    """Grayscale 2D FFT magnitude, log1p-scaled and per-image standardized.

    Per-image standardization: It removes overall exposure/brightness scale as a signal, so the frequency branch
    learns from spectral *shape* (where GAN vs diffusion artifacts actually differ),
    not from incidental brightness differences between real/fake sources.

    Returns a (1, size, size) float32 tensor.
    """
    gray = image.convert("L").resize((size, size), Image.BILINEAR)
    arr = torch.from_numpy(np.array(gray, dtype=np.float32)) / 255.0

    spectrum = torch.fft.fftshift(torch.fft.fft2(arr))
    magnitude = torch.log1p(spectrum.abs())
    magnitude = (magnitude - magnitude.mean()) / (magnitude.std() + 1e-6)

    return magnitude.unsqueeze(0)
