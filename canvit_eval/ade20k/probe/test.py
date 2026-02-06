"""Tests for probe head."""

import torch

from canvit_eval.ade20k.dataset import NUM_CLASSES
from canvit_eval.ade20k.probe import ProbeHead


def test_forward_shape() -> None:
    """forward() outputs at patch resolution."""
    probe = ProbeHead(768, dropout=0.1, use_ln=True)
    x = torch.randn(2, 32, 32, 768)  # [B, Hp, Wp, D]
    out = probe(x)
    assert out.shape == (2, NUM_CLASSES, 32, 32)  # [B, C, Hp, Wp]


def test_predict_rescales() -> None:
    """predict() rescales to target resolution."""
    probe = ProbeHead(768, dropout=0.1, use_ln=True).eval()
    x = torch.randn(1, 32, 32, 768)  # 32x32 patches
    out = probe.predict(x, rescale_to=(512, 512))
    assert out.shape == (1, NUM_CLASSES, 512, 512)


def test_predict_no_dropout() -> None:
    """predict() has no dropout - output is deterministic in eval mode."""
    probe = ProbeHead(768, dropout=0.5, use_ln=False).eval()
    x = torch.randn(1, 8, 8, 768)
    with torch.no_grad():
        out1 = probe.predict(x, rescale_to=(64, 64))
        out2 = probe.predict(x, rescale_to=(64, 64))
    assert torch.allclose(out1, out2)


