"""Tests for attention module."""

import torch


def test_to_from_multihead_roundtrip():
    from ytch.attention.mh import to_multihead, from_multihead
    x = torch.randn(2, 16, 64)
    mh = to_multihead(x, num_heads=8)
    assert mh.shape == (2, 8, 16, 8)
    back = from_multihead(mh)
    assert torch.allclose(back, x)


def test_elementwise_affine():
    from ytch.nn.elementwise_affine import ElementwiseAffine
    ewa = ElementwiseAffine(64, scale=2.0)
    x = torch.ones(2, 16, 64)
    out = ewa(x)
    assert out.shape == x.shape
    assert torch.allclose(out, torch.full_like(out, 2.0))


def test_layer_scale():
    from ytch.nn.layer_scale import LayerScale
    ls = LayerScale(64, init_values=0.5)
    x = torch.ones(2, 16, 64)
    out = ls(x)
    assert torch.allclose(out, torch.full_like(out, 0.5))


def test_read_cross_attention():
    from canvit.attention import ReadCrossAttention, CrossAttentionConfig
    from canvit.rope import compute_rope, make_rope_periods
    local_dim, canvas_dim, num_heads = 64, 64, 8
    canvas_head_dim = canvas_dim // num_heads
    attn = ReadCrossAttention(local_dim, canvas_dim, num_heads, CrossAttentionConfig())
    q_in = torch.randn(2, 16, local_dim)
    kv_in = torch.randn(2, 32, canvas_dim)
    periods = make_rope_periods(canvas_head_dim, torch.float32)
    q_rope = compute_rope(torch.randn(2, 16, 2), periods)
    kv_rope = compute_rope(torch.randn(2, 32, 2), periods)
    out = attn(q_in, kv_in, q_rope, kv_rope)
    assert out.shape == q_in.shape  # output is local_dim


def test_write_cross_attention():
    from canvit.attention import WriteCrossAttention, CrossAttentionConfig
    from canvit.rope import compute_rope, make_rope_periods
    local_dim, canvas_dim, num_heads = 64, 64, 8
    canvas_head_dim = canvas_dim // num_heads
    attn = WriteCrossAttention(local_dim, canvas_dim, num_heads, CrossAttentionConfig())
    q_in = torch.randn(2, 32, canvas_dim)
    kv_in = torch.randn(2, 16, local_dim)
    periods = make_rope_periods(canvas_head_dim, torch.float32)
    q_rope = compute_rope(torch.randn(2, 32, 2), periods)
    kv_rope = compute_rope(torch.randn(2, 16, 2), periods)
    out = attn(q_in, kv_in, q_rope, kv_rope)
    assert out.shape == q_in.shape  # output is canvas_dim


def test_scaled_residual_attention():
    from canvit.attention import ScaledResidualAttention, ReadCrossAttention, CrossAttentionConfig
    from canvit.rope import compute_rope, make_rope_periods
    torch.manual_seed(42)
    local_dim, canvas_dim, num_heads = 64, 64, 8
    canvas_head_dim = canvas_dim // num_heads
    attn = ReadCrossAttention(local_dim, canvas_dim, num_heads, CrossAttentionConfig())
    scaled = ScaledResidualAttention(attn, scale_init=1e-6)
    x = torch.randn(2, 16, local_dim)
    kv = torch.randn(2, 32, canvas_dim)
    periods = make_rope_periods(canvas_head_dim, torch.float32)
    x_rope = compute_rope(torch.randn(2, 16, 2), periods)
    kv_rope = compute_rope(torch.randn(2, 32, 2), periods)
    out = scaled(x, kv, x_rope, kv_rope)
    assert out.shape == x.shape
    assert torch.allclose(out, x, atol=1e-4)


def test_asymmetric_dimensions():
    """Test cross-attention with canvas_dim > local_dim (canvas_dim_mult > 1)."""
    from canvit.attention import ReadCrossAttention, WriteCrossAttention, CrossAttentionConfig
    from canvit.rope import compute_rope, make_rope_periods
    local_dim, canvas_dim, num_heads = 64, 128, 8  # canvas is 2x local
    canvas_head_dim = canvas_dim // num_heads
    periods = make_rope_periods(canvas_head_dim, torch.float32)

    # Read: local queries canvas → output is local_dim
    read = ReadCrossAttention(local_dim, canvas_dim, num_heads, CrossAttentionConfig())
    local_in = torch.randn(2, 16, local_dim)
    canvas_in = torch.randn(2, 32, canvas_dim)
    local_rope = compute_rope(torch.randn(2, 16, 2), periods)
    canvas_rope = compute_rope(torch.randn(2, 32, 2), periods)
    out = read(local_in, canvas_in, local_rope, canvas_rope)
    assert out.shape == (2, 16, local_dim)

    # Write: canvas queries local → output is canvas_dim
    write = WriteCrossAttention(local_dim, canvas_dim, num_heads, CrossAttentionConfig())
    out = write(canvas_in, local_in, canvas_rope, local_rope)
    assert out.shape == (2, 32, canvas_dim)
