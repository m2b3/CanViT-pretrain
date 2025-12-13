import torch
from dinov3.models.vision_transformer import vit_small

from .model import AVPConfig, AVPViT
from .rope import compute_rope, glimpse_positions, make_grid_positions


def test_rope_shapes():
    pos = make_grid_positions(7, 7, torch.device("cpu"))
    assert pos.shape == (49, 2)

    centers = torch.rand(4, 2)
    scales = torch.rand(4)
    local_pos = glimpse_positions(centers, scales, 7, 7)
    assert local_pos.shape == (4, 49, 2)

    sin, cos = compute_rope(local_pos, head_dim=64)
    assert sin.shape == (4, 1, 49, 64)
    assert cos.shape == (4, 1, 49, 64)


def test_avp_identity_init():
    """With γ=0, AVP should be identity: local = backbone(local), scene = scene."""
    torch.manual_seed(42)
    backbone = vit_small(img_size=112, patch_size=16)
    backbone.init_weights()

    cfg = AVPConfig(scene_grid_size=8, glimpse_grid_size=7, gate_init=0.0)
    avp = AVPViT.from_dinov3(backbone, cfg)

    # Verify gates are zero (as configured)
    for g in avp.read_gate:
        assert (g == 0).all()
    for g in avp.write_gate:
        assert (g == 0).all()

    B, H, W, D = 2, 7, 7, backbone.embed_dim
    n_prefix = 1 + backbone.n_storage_tokens

    # Random input: prefix + patch tokens
    local = torch.randn(B, n_prefix + H * W, D)
    centers = torch.rand(B, 2)
    scales = torch.full((B,), 0.5)

    # Run backbone directly
    local_pos = glimpse_positions(centers, scales, H, W)
    local_rope = compute_rope(local_pos, backbone.embed_dim // backbone.num_heads)

    expected = local.clone()
    for block in backbone.blocks:
        expected = block(expected, local_rope)

    # Run AVP
    actual_local, actual_scene = avp(local.clone(), centers, scales)

    # Local should match backbone output
    assert torch.allclose(actual_local, expected, atol=1e-5), "Local stream should match backbone"

    # Scene should be unchanged (identity passthrough)
    expected_scene = avp.scene_tokens.expand(B, -1, -1)
    assert torch.allclose(actual_scene, expected_scene, atol=1e-5), "Scene should be identity"


def test_per_batch_rope_differs():
    """Verify different glimpse positions produce different outputs."""
    torch.manual_seed(42)
    backbone = vit_small(img_size=112, patch_size=16)
    backbone.init_weights()

    B, H, W, D = 2, 7, 7, backbone.embed_dim
    n_prefix = 1 + backbone.n_storage_tokens

    # Same input, different positions
    local = torch.randn(1, n_prefix + H * W, D).expand(B, -1, -1).clone()
    centers = torch.tensor([[0.2, 0.2], [0.8, 0.8]])  # very different
    scales = torch.tensor([0.3, 0.7])  # very different

    local_pos = glimpse_positions(centers, scales, H, W)
    local_rope = compute_rope(local_pos, backbone.embed_dim // backbone.num_heads)

    # Run backbone
    out = local.clone()
    for block in backbone.blocks:
        out = block(out, local_rope)

    # Outputs should differ because RoPE differs
    assert not torch.allclose(out[0], out[1], atol=1e-3), "Per-batch RoPE not applied correctly"


def test_avp_forward_shapes():
    backbone = vit_small(img_size=112, patch_size=16)
    backbone.init_weights()
    cfg = AVPConfig(scene_grid_size=8, glimpse_grid_size=7)
    avp = AVPViT.from_dinov3(backbone, cfg)

    B = 2
    n_prefix = 1 + backbone.n_storage_tokens
    local = torch.randn(B, n_prefix + 49, backbone.embed_dim)
    centers = torch.rand(B, 2)
    scales = torch.rand(B)

    out_local, out_scene = avp(local, centers, scales)

    assert out_local.shape == local.shape
    assert out_scene.shape == (B, 64, backbone.embed_dim)
