"""Tests for gaussian blob training utilities."""

import torch

from .data import generate_multi_blob_batch, hsv_to_rgb, perlin_noise_2d


def test_perlin_noise_shape_and_range() -> None:
    """Perlin noise has correct shape and range."""
    B, H, W = 4, 64, 64
    noise = perlin_noise_2d((B, H, W), scale=16, octaves=3, persistence=0.5, device=torch.device("cpu"))
    assert noise.shape == (B, H, W)
    assert noise.min() >= 0
    assert noise.max() <= 1


def test_hsv_to_rgb() -> None:
    """HSV to RGB conversion produces valid RGB values."""
    h = torch.tensor([0.0, 0.33, 0.66, 1.0])
    s = torch.ones(4)
    v = torch.ones(4)
    rgb = hsv_to_rgb(h, s, v)
    assert rgb.shape == (4, 3)
    assert (rgb >= 0).all() and (rgb <= 1).all()


def test_generate_multi_blob_batch_shapes() -> None:
    """Batch generation produces correct shapes."""
    B, size, n_blobs = 4, 64, 3
    images, colors, centers, all_centers = generate_multi_blob_batch(
        B, size, n_blobs, torch.device("cpu")
    )
    assert images.shape == (B, 3, size, size)
    assert colors.shape == (B, 3)
    assert centers.shape == (B, 2)
    assert all_centers.shape == (B, n_blobs, 2)


def test_generate_multi_blob_batch_centers_in_bounds() -> None:
    """Blob centers are within valid range."""
    B, size, n_blobs = 8, 64, 2
    margin = 0.3
    _, _, centers, all_centers = generate_multi_blob_batch(
        B, size, n_blobs, torch.device("cpu"), margin=margin
    )
    valid_range = 1 - margin
    assert (centers.abs() <= valid_range + 1e-6).all()
    assert (all_centers.abs() <= valid_range + 1e-6).all()


def test_generate_multi_blob_independent_positions() -> None:
    """Each sample has independent blob positions."""
    B, size, n_blobs = 4, 64, 2
    _, _, _, all_centers = generate_multi_blob_batch(
        B, size, n_blobs, torch.device("cpu")
    )
    # Samples should have different positions
    assert (all_centers[0] != all_centers[1]).any()
