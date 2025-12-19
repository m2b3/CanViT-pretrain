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


def test_rope_read_cross_attention():
    from canvit.attention import RoPEReadCrossAttention, CrossAttentionConfig
    from canvit.rope import compute_rope, make_rope_periods
    attn = RoPEReadCrossAttention(64, num_heads=8, cfg=CrossAttentionConfig())
    q_in = torch.randn(2, 16, 64)
    kv_in = torch.randn(2, 32, 64)
    periods = make_rope_periods(8, torch.float32)
    q_rope = compute_rope(torch.randn(2, 16, 2), periods)
    kv_rope = compute_rope(torch.randn(2, 32, 2), periods)
    out = attn(q_in, kv_in, q_rope, kv_rope)
    assert out.shape == q_in.shape


def test_rope_write_cross_attention():
    from canvit.attention import RoPEWriteCrossAttention, CrossAttentionConfig
    from canvit.rope import compute_rope, make_rope_periods
    attn = RoPEWriteCrossAttention(64, num_heads=8, cfg=CrossAttentionConfig())
    q_in = torch.randn(2, 32, 64)
    kv_in = torch.randn(2, 16, 64)
    periods = make_rope_periods(8, torch.float32)
    q_rope = compute_rope(torch.randn(2, 32, 2), periods)
    kv_rope = compute_rope(torch.randn(2, 16, 2), periods)
    out = attn(q_in, kv_in, q_rope, kv_rope)
    assert out.shape == q_in.shape


def test_scaled_residual_attention():
    from canvit.attention import ScaledResidualAttention, RoPEReadCrossAttention, CrossAttentionConfig
    from canvit.rope import compute_rope, make_rope_periods
    torch.manual_seed(42)
    attn = RoPEReadCrossAttention(64, num_heads=8, cfg=CrossAttentionConfig())
    scaled = ScaledResidualAttention(attn, scale_init=1e-6)
    x = torch.randn(2, 16, 64)
    kv = torch.randn(2, 32, 64)
    periods = make_rope_periods(8, torch.float32)
    x_rope = compute_rope(torch.randn(2, 16, 2), periods)
    kv_rope = compute_rope(torch.randn(2, 32, 2), periods)
    out = scaled(x, kv, x_rope, kv_rope)
    assert out.shape == x.shape
    assert torch.allclose(out, x, atol=1e-4)
