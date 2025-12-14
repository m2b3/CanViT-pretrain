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


def test_block_flops():
    """Block FLOPs follow standard ViT formula: 24ND² + 4N²D."""
    native = vit_small(img_size=112, patch_size=16)
    backbone = DINOv3Backbone(native)
    N, D = 54, 384  # typical: 49 patches + 5 prefix tokens
    f = backbone.block_flops(N)
    assert f == 24 * N * D * D + 4 * N * N * D


def test_patch_embed_flops():
    """Patch embed FLOPs: 2 * D * 3 * P² * n_patches."""
    native = vit_small(img_size=112, patch_size=16)
    backbone = DINOv3Backbone(native)
    n_patches = 49  # 7x7
    D, P = 384, 16
    f = backbone.patch_embed_flops(n_patches)
    assert f == 2 * D * 3 * P * P * n_patches
