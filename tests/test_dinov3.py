"""Integration tests with DINOv3 backbone."""

import math
from typing import Any, cast

import pytest
import torch
from dinov3.models.vision_transformer import vit_small
from torch import Tensor

from avp_vit.backbone.dinov3 import DINOv3Backbone
from avp_vit.rope import compute_rope, glimpse_positions


@pytest.mark.parametrize("rope_dtype", ["fp32", "bf16"])
def test_rope_matches_dinov3(rope_dtype: str) -> None:
    """Our RoPE sin/cos must exactly match DINOv3's computation."""
    H, W = 7, 7
    native = vit_small(img_size=112, patch_size=16, pos_embed_rope_dtype=rope_dtype)
    native.init_weights()

    rope_embed = native.rope_embed
    dtype = rope_embed.dtype
    assert dtype is not None
    periods = cast(Tensor, rope_embed.periods)
    device = periods.device

    coords_h = torch.arange(0.5, H, device=device, dtype=dtype) / H
    coords_w = torch.arange(0.5, W, device=device, dtype=dtype) / W
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
    coords = coords.flatten(0, 1)
    coords = 2.0 * coords - 1.0

    angles_dino = 2 * math.pi * coords[:, :, None] / periods[None, None, :]
    angles_dino = angles_dino.flatten(1, 2).tile((2,))
    sin_dino, cos_dino = torch.sin(angles_dino), torch.cos(angles_dino)

    centers = torch.zeros(1, 2, device=device)
    scales = torch.ones(1, device=device)
    our_pos = glimpse_positions(centers, scales, H, W, dtype=dtype)
    our_sin, our_cos = compute_rope(our_pos, periods)

    assert torch.allclose(our_pos[0], coords)
    assert torch.allclose(our_sin[0, 0], sin_dino)
    assert torch.allclose(our_cos[0, 0], cos_dino)


@pytest.mark.parametrize("rope_dtype", ["fp32", "bf16"])
def test_rope_matches_backbone_forward(rope_dtype: str) -> None:
    """Full forward with our RoPE must match backbone's native forward."""
    torch.manual_seed(42)
    native = vit_small(img_size=112, patch_size=16, pos_embed_rope_dtype=rope_dtype)
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


def test_per_batch_rope_differs() -> None:
    """Different glimpse positions must produce different outputs."""
    torch.manual_seed(42)
    native = vit_small(img_size=112, patch_size=16, pos_embed_rope_dtype="fp32")
    native.init_weights()

    backbone = DINOv3Backbone(native)
    B, H, W = 2, 7, 7

    local = (
        torch.randn(1, backbone.n_prefix_tokens + H * W, backbone.embed_dim)
        .expand(B, -1, -1)
        .clone()
    )
    centers = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
    scales = torch.tensor([0.3, 0.7])

    positions = glimpse_positions(centers, scales, H, W, dtype=backbone.rope_dtype)
    rope = compute_rope(positions, backbone.rope_periods)

    out = local.clone()
    for i in range(backbone.n_blocks):
        out = backbone.forward_block(i, out, rope)

    assert not torch.allclose(out[0], out[1], atol=1e-3)
