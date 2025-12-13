"""Tests for DINOv3Backend - must match native forward exactly."""
import torch
from dinov3.models.vision_transformer import vit_small

from avp_vit.rope import compute_rope, glimpse_positions

from . import DINOv3Backend


def test_backend_forward_matches_native():
    """Backend block-by-block forward must exactly match native forward_features."""
    torch.manual_seed(42)
    backbone = vit_small(img_size=112, patch_size=16)
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


def test_backend_properties():
    """Backend exposes correct properties from backbone."""
    backbone = vit_small(img_size=112, patch_size=16)
    backend = DINOv3Backend(backbone)

    assert backend.embed_dim == 384
    assert backend.num_heads == 6
    assert backend.n_blocks == 12
    assert backend.n_prefix_tokens == 1 + backbone.n_storage_tokens
    assert backend.rope_periods.shape[0] == backend.embed_dim // backend.num_heads // 4
