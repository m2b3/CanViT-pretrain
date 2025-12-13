"""Tests for IJEPABackbone - must match native forward exactly."""

import pytest
import torch
from ijepa.models.vision_transformer import vit_small

from avp_vit.backbone.ijepa import IJEPABackbone


def test_forward_matches_native():
    """Backbone block-by-block forward must exactly match native forward."""
    torch.manual_seed(42)
    native = vit_small(img_size=[112], patch_size=16)
    native.eval()

    backbone = IJEPABackbone(native, rope_dtype=torch.float32)

    B = 2
    img = torch.randn(B, 3, 112, 112)

    with torch.no_grad():
        native_out = native(img, masks=None)

    x, H, W = backbone.prepare_tokens(img)

    with torch.no_grad():
        for i in range(backbone.n_blocks):
            x = backbone.forward_block(i, x, rope=None)
        x = native.norm(x)

    assert torch.allclose(x, native_out, atol=1e-5)


def test_properties():
    """Backbone exposes correct properties."""
    native = vit_small(img_size=[112], patch_size=16)
    backbone = IJEPABackbone(native, rope_dtype=torch.float32)

    assert backbone.embed_dim == 384
    assert backbone.num_heads == 6
    assert backbone.n_blocks == 12
    assert backbone.n_prefix_tokens == 0
    assert (
        backbone.rope_periods.shape[0] == backbone.embed_dim // backbone.num_heads // 4
    )


def test_ijepa_variable_resolution_crashes():
    """I-JEPA's interpolate_pos_encoding is buggy for variable resolution.

    Their code assumes CLS token exists (does `npatch = x.shape[1] - 1`) but
    I-JEPA has no CLS token. When image size matches init size, the early
    return `if npatch == N: return pos_embed` saves it. Different sizes crash.

    This test documents this known upstream limitation. We only use fixed sizes.
    """
    native = vit_small(img_size=[112], patch_size=16)

    # Same size works: the -1 bug affects both sides of the comparison equally,
    # so they match and the early return avoids the broken reshape logic.
    native(torch.randn(1, 3, 112, 112))

    # Different size crashes in their interpolate_pos_encoding
    with pytest.raises(RuntimeError, match="invalid for input of size"):
        native(torch.randn(1, 3, 224, 224))
