import torch
from torch import nn

from avp_vit.attention import CrossAttentionConfig, RoPEReadCrossAttention, RoPEWriteCrossAttention


def _default_cfg() -> CrossAttentionConfig:
    return CrossAttentionConfig()


def test_read_shapes():
    B, N_q, N_kv, D, heads = 2, 10, 20, 64, 4
    head_dim = D // heads
    attn = RoPEReadCrossAttention(D, heads, _default_cfg())
    q = torch.randn(B, N_q, D)
    kv = torch.randn(B, N_kv, D)
    q_rope = (torch.randn(B, 1, N_q, head_dim), torch.randn(B, 1, N_q, head_dim))
    kv_rope = (torch.randn(B, 1, N_kv, head_dim), torch.randn(B, 1, N_kv, head_dim))
    out = attn(q, kv, q_rope, kv_rope)
    assert out.shape == (B, N_q, D)


def test_write_shapes():
    B, N_q, N_kv, D, heads = 2, 20, 10, 64, 4
    head_dim = D // heads
    attn = RoPEWriteCrossAttention(D, heads, _default_cfg())
    q = torch.randn(B, N_q, D)
    kv = torch.randn(B, N_kv, D)
    q_rope = (torch.randn(B, 1, N_q, head_dim), torch.randn(B, 1, N_q, head_dim))
    kv_rope = (torch.randn(B, 1, N_kv, head_dim), torch.randn(B, 1, N_kv, head_dim))
    out = attn(q, kv, q_rope, kv_rope)
    assert out.shape == (B, N_q, D)


def test_flops_read():
    """Read attention: Q and O projections on queries."""
    D = 64
    attn = RoPEReadCrossAttention(D, num_heads=4, cfg=_default_cfg())
    f = attn.flops(n_q=10, n_kv=20)
    # attention + Q proj + O proj
    assert f == 4 * 10 * 20 * D + 2 * 10 * D * D + 2 * 10 * D * D


def test_flops_write():
    """Write attention: K and V projections on keys/values."""
    D = 64
    attn = RoPEWriteCrossAttention(D, num_heads=4, cfg=_default_cfg())
    f = attn.flops(n_q=20, n_kv=10)
    # attention + K proj + V proj
    assert f == 4 * 20 * 10 * D + 2 * 10 * D * D + 2 * 10 * D * D


def test_flops_projection_placement_matters():
    """Which tokens get projected affects FLOPs significantly."""
    D = 64
    cfg = _default_cfg()
    read = RoPEReadCrossAttention(D, 4, cfg)
    write = RoPEWriteCrossAttention(D, 4, cfg)
    # Asymmetric: 10 queries, 100 keys/values
    read_f = read.flops(n_q=10, n_kv=100)
    write_f = write.flops(n_q=10, n_kv=100)
    # Read projects small tensor (queries), write projects large tensor (keys/values)
    assert write_f > read_f * 4


def test_ewa_transforms_enabled():
    """Unprojected transforms are EWA when use_ewa_transforms=True."""
    from ytch.nn.elementwise_affine import ElementwiseAffine

    D = 64
    cfg = CrossAttentionConfig(use_ewa_transforms=True)
    read = RoPEReadCrossAttention(D, 4, cfg)
    write = RoPEWriteCrossAttention(D, 4, cfg)
    # Read: K and V unprojected
    assert isinstance(read.k_transform, ElementwiseAffine)
    assert isinstance(read.v_transform, ElementwiseAffine)
    # Write: Q and O unprojected
    assert isinstance(write.q_transform, ElementwiseAffine)
    assert isinstance(write.out_transform, ElementwiseAffine)


def test_ewa_transforms_disabled():
    """Unprojected transforms are Identity when use_ewa_transforms=False."""
    D = 64
    cfg = CrossAttentionConfig(use_ewa_transforms=False)
    read = RoPEReadCrossAttention(D, 4, cfg)
    write = RoPEWriteCrossAttention(D, 4, cfg)
    # Read: K and V unprojected
    assert isinstance(read.k_transform, nn.Identity)
    assert isinstance(read.v_transform, nn.Identity)
    # Write: Q and O unprojected
    assert isinstance(write.q_transform, nn.Identity)
    assert isinstance(write.out_transform, nn.Identity)


def test_normalize_q_enabled():
    """Q normalization uses LayerNorm when enabled."""
    D = 64
    cfg = CrossAttentionConfig(normalize_q=True)
    attn = RoPEReadCrossAttention(D, 4, cfg)
    assert isinstance(attn.q_norm, nn.LayerNorm)


def test_normalize_q_disabled():
    """Q normalization is Identity when disabled."""
    D = 64
    cfg = CrossAttentionConfig(normalize_q=False)
    attn = RoPEReadCrossAttention(D, 4, cfg)
    assert isinstance(attn.q_norm, nn.Identity)


def test_normalize_k_enabled():
    """K normalization uses LayerNorm when enabled."""
    D = 64
    cfg = CrossAttentionConfig(normalize_k=True)
    attn = RoPEReadCrossAttention(D, 4, cfg)
    assert isinstance(attn.k_norm, nn.LayerNorm)


def test_normalize_k_disabled():
    """K normalization is Identity when disabled."""
    D = 64
    cfg = CrossAttentionConfig(normalize_k=False)
    attn = RoPEReadCrossAttention(D, 4, cfg)
    assert isinstance(attn.k_norm, nn.Identity)


def test_normalize_v_enabled():
    """V normalization uses LayerNorm when enabled."""
    D = 64
    cfg = CrossAttentionConfig(normalize_v=True)
    attn = RoPEReadCrossAttention(D, 4, cfg)
    assert isinstance(attn.v_norm, nn.LayerNorm)


def test_normalize_v_disabled():
    """V normalization is Identity when disabled."""
    D = 64
    cfg = CrossAttentionConfig(normalize_v=False)
    attn = RoPEReadCrossAttention(D, 4, cfg)
    assert isinstance(attn.v_norm, nn.Identity)


# ==================== ScaledResidualAttention ====================


def test_scaled_residual_attention():
    from avp_vit.attention import ScaledResidualAttention

    B, N_q, N_kv, D, heads = 2, 10, 20, 64, 4
    head_dim = D // heads
    scale_init = 0.1

    attn = RoPEReadCrossAttention(D, heads, _default_cfg())
    wrapped = ScaledResidualAttention(attn, scale_init)

    x = torch.randn(B, N_q, D)
    kv = torch.randn(B, N_kv, D)
    x_rope = (torch.randn(B, 1, N_q, head_dim), torch.randn(B, 1, N_q, head_dim))
    kv_rope = (torch.randn(B, 1, N_kv, head_dim), torch.randn(B, 1, N_kv, head_dim))

    out = wrapped(x, kv, x_rope, kv_rope)
    assert out.shape == (B, N_q, D)

    # Verify it's x + scale * attn(...)
    with torch.no_grad():
        raw = attn(x, kv, x_rope, kv_rope)
        expected = x + scale_init * raw
        assert torch.allclose(out, expected, rtol=1e-4)
