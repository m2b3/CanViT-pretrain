"""Smoke tests for loss functions."""

import torch

from . import LOSS_FNS, cos_dissim, get_loss_fn


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
