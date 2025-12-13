"""Tests for DINOv3Backbone - must match native forward exactly."""

from typing import Any, cast

import torch
from dinov3.models.vision_transformer import vit_small
from torch import Tensor

from avp_vit.rope import compute_rope, glimpse_positions

from . import DINOv3Backbone


def test_forward_matches_native():
    """Backbone block-by-block forward must exactly match native forward_features."""
    torch.manual_seed(42)
    native = vit_small(img_size=112, patch_size=16)
    native.init_weights()
    native.eval()

    backbone = DINOv3Backbone(native)

    B = 2
    img = torch.randn(B, 3, 112, 112)

    with torch.no_grad():
        native_out = cast(dict[str, Any], native.forward_features(img))
    native_tokens: Tensor = native_out["x_prenorm"]

    x, H, W = backbone.prepare_tokens(img)
    centers = torch.zeros(B, 2)
    scales = torch.ones(B)
    positions = glimpse_positions(centers, scales, H, W, dtype=backbone.rope_dtype)
    rope = compute_rope(positions, backbone.rope_periods)

    with torch.no_grad():
        for i in range(backbone.n_blocks):
            x = backbone.forward_block(i, x, rope)

    assert torch.allclose(x, native_tokens, atol=1e-5)


def test_properties():
    """Backbone exposes correct properties."""
    native = vit_small(img_size=112, patch_size=16)
    backbone = DINOv3Backbone(native)

    assert backbone.embed_dim == 384
    assert backbone.num_heads == 6
    assert backbone.n_blocks == 12
    assert backbone.n_prefix_tokens == 1 + native.n_storage_tokens
    assert (
        backbone.rope_periods.shape[0] == backbone.embed_dim // backbone.num_heads // 4
    )
