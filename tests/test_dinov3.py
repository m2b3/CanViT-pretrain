"""Integration tests with DINOv3 backbone."""
import math

import pytest
import torch
from dinov3.models.vision_transformer import vit_small

from avp_vit import AVPConfig, AVPViT
from avp_vit.backend.dinov3 import DINOv3Backend
from avp_vit.rope import compute_rope, glimpse_positions


@pytest.mark.parametrize("rope_dtype", ["fp32", "bf16"])
def test_rope_matches_dinov3(rope_dtype: str):
    """Our RoPE sin/cos must exactly match DINOv3's computation."""
    H, W = 7, 7
    backbone = vit_small(img_size=112, patch_size=16, pos_embed_rope_dtype=rope_dtype)
    backbone.init_weights()

    rope_embed = backbone.rope_embed
    dtype = rope_embed.dtype
    device = rope_embed.periods.device

    # DINOv3's computation (from rope_position_encoding.py)
    dd = {"device": device, "dtype": dtype}
    coords_h = torch.arange(0.5, H, **dd) / H
    coords_w = torch.arange(0.5, W, **dd) / W
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
    coords = coords.flatten(0, 1)
    coords = 2.0 * coords - 1.0

    angles_dino = 2 * math.pi * coords[:, :, None] / rope_embed.periods[None, None, :]
    angles_dino = angles_dino.flatten(1, 2).tile((2,))
    sin_dino, cos_dino = torch.sin(angles_dino), torch.cos(angles_dino)

    # Ours with center=0, scale=1
    centers = torch.zeros(1, 2, device=device)
    scales = torch.ones(1, device=device)
    our_pos = glimpse_positions(centers, scales, H, W, dtype=dtype)
    our_sin, our_cos = compute_rope(our_pos, rope_embed.periods)

    assert torch.allclose(our_pos[0], coords)
    assert torch.allclose(our_sin[0, 0], sin_dino)
    assert torch.allclose(our_cos[0, 0], cos_dino)


@pytest.mark.parametrize("rope_dtype", ["fp32", "bf16"])
def test_rope_matches_backbone_forward(rope_dtype: str):
    """Full forward with our RoPE must match backbone's native forward."""
    torch.manual_seed(42)
    backbone = vit_small(img_size=112, patch_size=16, pos_embed_rope_dtype=rope_dtype)
    backbone.init_weights()
    backbone.eval()

    backend = DINOv3Backend(backbone)
    B = 2
    img = torch.randn(B, 3, 112, 112)

    with torch.no_grad():
        native_out = backbone.forward_features(img)
    native_tokens = native_out["x_prenorm"]

    x, H, W = backend.prepare_tokens(img)
    centers = torch.zeros(B, 2)
    scales = torch.ones(B)
    positions = glimpse_positions(centers, scales, H, W, dtype=backend.rope_dtype)
    rope = compute_rope(positions, backend.rope_periods)

    with torch.no_grad():
        for i in range(backend.n_blocks):
            x = backend.forward_block(i, x, rope)

    assert torch.allclose(x, native_tokens, atol=1e-5)


def test_per_batch_rope_differs():
    """Different glimpse positions must produce different outputs."""
    torch.manual_seed(42)
    backbone = vit_small(img_size=112, patch_size=16, pos_embed_rope_dtype="fp32")
    backbone.init_weights()

    backend = DINOv3Backend(backbone)
    B, H, W = 2, 7, 7

    local = torch.randn(1, backend.n_prefix_tokens + H * W, backend.embed_dim).expand(B, -1, -1).clone()
    centers = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
    scales = torch.tensor([0.3, 0.7])

    positions = glimpse_positions(centers, scales, H, W, dtype=backend.rope_dtype)
    rope = compute_rope(positions, backend.rope_periods)

    out = local.clone()
    for i in range(backend.n_blocks):
        out = backend.forward_block(i, out, rope)

    assert not torch.allclose(out[0], out[1], atol=1e-3)


def test_avp_identity_init():
    """With γ=0, AVP should be identity: local = backbone(local), scene = scene."""
    torch.manual_seed(42)
    backbone = vit_small(img_size=112, patch_size=16, pos_embed_rope_dtype="fp32")
    backbone.init_weights()

    backend = DINOv3Backend(backbone)
    cfg = AVPConfig(scene_grid_size=8, glimpse_grid_size=7, gate_init=0.0)
    avp = AVPViT(backend, cfg)

    B, H, W = 2, 7, 7

    local = torch.randn(B, backend.n_prefix_tokens + H * W, backend.embed_dim)
    centers = torch.zeros(B, 2)
    scales = torch.ones(B)

    positions = glimpse_positions(centers, scales, H, W, dtype=backend.rope_dtype)
    rope = compute_rope(positions, backend.rope_periods)

    expected = local.clone()
    for i in range(backend.n_blocks):
        expected = backend.forward_block(i, expected, rope)

    actual_local, actual_scene = avp(local.clone(), centers, scales)

    assert torch.allclose(actual_local, expected, atol=1e-5)
    assert torch.allclose(actual_scene, avp.scene_tokens.expand(B, -1, -1), atol=1e-5)
