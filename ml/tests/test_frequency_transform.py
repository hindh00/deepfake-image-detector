import numpy as np
import torch
from PIL import Image

from src.datasets.transforms import IMAGE_SIZE, build_spatial_transform, fft_magnitude_log1p


def _random_image(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def test_fft_magnitude_shape_and_dtype():
    out = fft_magnitude_log1p(_random_image(0))
    assert out.shape == (1, IMAGE_SIZE, IMAGE_SIZE)
    assert out.dtype == torch.float32


def test_fft_magnitude_no_nan_or_inf():
    out = fft_magnitude_log1p(_random_image(1))
    assert torch.isfinite(out).all()


def test_fft_magnitude_is_deterministic():
    img = _random_image(2)
    out1 = fft_magnitude_log1p(img)
    out2 = fft_magnitude_log1p(img)
    assert torch.equal(out1, out2)


def test_fft_magnitude_differs_across_distinct_images():
    out_a = fft_magnitude_log1p(_random_image(3))
    out_b = fft_magnitude_log1p(_random_image(4))
    assert not torch.equal(out_a, out_b)


def test_fft_magnitude_is_per_image_standardized():
    out = fft_magnitude_log1p(_random_image(5))
    assert out.mean().abs() < 1e-4
    assert abs(out.std().item() - 1.0) < 1e-3


def test_spatial_transform_eval_shape_and_range():
    transform = build_spatial_transform(train=False)
    out = transform(_random_image(6))
    assert out.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert out.dtype == torch.float32
    assert out.min() > -3.5
    assert out.max() < 3.5


def test_spatial_transform_train_shape():
    transform = build_spatial_transform(train=True)
    out = transform(_random_image(7))
    assert out.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
