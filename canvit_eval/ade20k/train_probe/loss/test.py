"""Tests for loss functions."""

import torch

from . import focal_loss, upsample_preds


def test_focal_loss_shape():
    """Focal loss returns scalar."""
    B, C, H, W = 2, 150, 32, 32
    logits = torch.randn(B, C, H, W)
    masks = torch.randint(0, C, (B, H, W))
    loss = focal_loss(logits, masks, gamma=2.0)
    assert loss.shape == ()
    assert loss.item() > 0


def test_focal_loss_downsamples_masks():
    """Focal loss handles mask resolution mismatch."""
    B, C = 2, 150
    logits = torch.randn(B, C, 16, 16)
    masks = torch.randint(0, C, (B, 32, 32))
    loss = focal_loss(logits, masks, gamma=2.0)
    assert loss.shape == ()


def test_upsample_preds_noop():
    """Upsample is noop when already correct size."""
    preds = torch.randint(0, 150, (2, 32, 32))
    out = upsample_preds(preds, 32, 32)
    assert torch.equal(out, preds)


def test_upsample_preds_upscale():
    """Upsample correctly upscales."""
    preds = torch.randint(0, 150, (2, 16, 16))
    out = upsample_preds(preds, 32, 32)
    assert out.shape == (2, 32, 32)
    assert out.dtype == torch.int64
