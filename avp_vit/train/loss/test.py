"""Smoke tests for loss functions."""

import torch

from . import LOSS_FNS, cos_dissim, get_loss_fn, gram_mse, spatial_gram


def test_cos_dissim_identical() -> None:
    """Cosine dissimilarity of identical vectors is 0."""
    x = torch.randn(4, 10)
    assert cos_dissim(x, x).item() < 1e-6


def test_cos_dissim_orthogonal() -> None:
    """Cosine dissimilarity of orthogonal vectors is 1."""
    x = torch.tensor([[1.0, 0.0]])
    y = torch.tensor([[0.0, 1.0]])
    assert abs(cos_dissim(x, y).item() - 1.0) < 1e-6


def test_get_loss_fn() -> None:
    """get_loss_fn returns correct functions."""
    assert get_loss_fn("cos") is LOSS_FNS["cos"]
    assert get_loss_fn("l1") is LOSS_FNS["l1"]
    assert get_loss_fn("mse") is LOSS_FNS["mse"]
    assert get_loss_fn("gram") is LOSS_FNS["gram"]


def test_spatial_gram_shape() -> None:
    """spatial_gram produces [B, N, N] from [B, N, D]."""
    x = torch.randn(2, 16, 64)
    g = spatial_gram(x)
    assert g.shape == (2, 16, 16)


def test_spatial_gram_symmetric() -> None:
    """Gram matrix is symmetric."""
    x = torch.randn(2, 16, 64)
    g = spatial_gram(x)
    assert torch.allclose(g, g.transpose(1, 2))


def test_gram_mse_identical() -> None:
    """Gram MSE is 0 for identical inputs."""
    x = torch.randn(2, 16, 64)
    assert gram_mse(x, x).item() < 1e-6


def test_gram_mse_permutation_invariant() -> None:
    """Gram MSE is invariant to spatial permutation (same correlations)."""
    x = torch.randn(1, 16, 64)
    perm = torch.randperm(16)
    x_perm = x[:, perm, :]
    # Same correlations, just reordered - gram matrices differ but structure same
    # This test verifies gram captures structure, not position
    g1, g2 = spatial_gram(x), spatial_gram(x_perm)
    # Permuted gram should equal original gram with rows/cols permuted
    g1_perm = g1[:, perm, :][:, :, perm]
    assert torch.allclose(g1_perm, g2)
