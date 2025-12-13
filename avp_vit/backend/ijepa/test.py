"""Tests for IJEPABackend - must match native forward exactly."""
import pytest
import torch
from src.models.vision_transformer import vit_small

from . import IJEPABackend


def test_backend_forward_matches_native():
    """Backend block-by-block forward must exactly match native forward."""
    torch.manual_seed(42)
    backbone = vit_small(img_size=[112], patch_size=16)
    backbone.eval()

    backend = IJEPABackend(backbone)

    B = 2
    img = torch.randn(B, 3, 112, 112)

    with torch.no_grad():
        native_out = backbone(img, masks=None)

    x, H, W = backend.prepare_tokens(img)

    with torch.no_grad():
        for i in range(backend.n_blocks):
            x = backend.forward_block(i, x, rope=None)
        x = backbone.norm(x)

    assert torch.allclose(x, native_out, atol=1e-5)


def test_backend_properties():
    """Backend exposes correct properties from backbone."""
    backbone = vit_small(img_size=[112], patch_size=16)
    backend = IJEPABackend(backbone)

    assert backend.embed_dim == 384
    assert backend.num_heads == 6
    assert backend.n_blocks == 12
    assert backend.n_prefix_tokens == 0
    assert backend.rope_periods.shape[0] == backend.embed_dim // backend.num_heads // 4


def test_ijepa_variable_resolution_crashes():
    """I-JEPA's interpolate_pos_encoding is buggy for variable resolution.

    Their code assumes CLS token exists (does `npatch = x.shape[1] - 1`) but
    I-JEPA has no CLS token. When image size matches init size, the early
    return `if npatch == N: return pos_embed` saves it. Different sizes crash.

    This test documents this known upstream limitation. We only use fixed sizes.
    """
    backbone = vit_small(img_size=[112], patch_size=16)

    # Same size works: interpolate_pos_encoding does `if npatch == N: return pos_embed`
    # The -1 bug (npatch = x.shape[1] - 1) affects both sides equally, so they match
    # and the early return avoids the broken reshape logic.
    backbone(torch.randn(1, 3, 112, 112))

    # Different size crashes in their interpolate_pos_encoding
    with pytest.raises(RuntimeError, match="invalid for input of size"):
        backbone(torch.randn(1, 3, 224, 224))
