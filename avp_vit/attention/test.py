import torch
from torch import nn

from avp_vit.attention import AttentionConfig, RoPEReadCrossAttention, RoPEWriteCrossAttention


def _default_cfg() -> AttentionConfig:
    return AttentionConfig()


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


def test_write_v_identity_init():
    """V projection starts as identity when vo_identity_init=True."""
    D = 64
    cfg = AttentionConfig(vo_identity_init=True)
    attn = RoPEWriteCrossAttention(D, num_heads=4, cfg=cfg)
    v = attn.v_transform
    assert isinstance(v, nn.Linear)
    assert torch.allclose(v.weight, torch.eye(D))
    assert torch.allclose(v.bias, torch.zeros(D))


def test_write_v_default_init():
    """V projection uses default init when vo_identity_init=False."""
    D = 64
    cfg = AttentionConfig(vo_identity_init=False)
    attn = RoPEWriteCrossAttention(D, num_heads=4, cfg=cfg)
    v = attn.v_transform
    assert isinstance(v, nn.Linear)
    assert not torch.allclose(v.weight, torch.eye(D))


def test_read_o_identity_init():
    """O projection starts as identity when vo_identity_init=True."""
    D = 64
    cfg = AttentionConfig(vo_identity_init=True)
    attn = RoPEReadCrossAttention(D, num_heads=4, cfg=cfg)
    assert torch.allclose(attn.out_transform.weight, torch.eye(D))
    assert torch.allclose(attn.out_transform.bias, torch.zeros(D))


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
    cfg = AttentionConfig(use_ewa_transforms=True)
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
    cfg = AttentionConfig(use_ewa_transforms=False)
    read = RoPEReadCrossAttention(D, 4, cfg)
    write = RoPEWriteCrossAttention(D, 4, cfg)
    # Read: K and V unprojected
    assert isinstance(read.k_transform, nn.Identity)
    assert isinstance(read.v_transform, nn.Identity)
    # Write: Q and O unprojected
    assert isinstance(write.q_transform, nn.Identity)
    assert isinstance(write.out_transform, nn.Identity)


def test_post_rope_ewa_enabled():
    """Post-RoPE EWAs are ElementwiseAffine when use_post_rope_ewa=True."""
    from ytch.nn.elementwise_affine import ElementwiseAffine

    D, heads = 64, 4
    cfg = AttentionConfig(use_post_rope_ewa=True)
    attn = RoPEReadCrossAttention(D, heads, cfg)
    assert isinstance(attn.post_rope_q, ElementwiseAffine)
    assert isinstance(attn.post_rope_k, ElementwiseAffine)


def test_post_rope_ewa_disabled():
    """Post-RoPE EWAs are Identity when use_post_rope_ewa=False."""
    D = 64
    cfg = AttentionConfig(use_post_rope_ewa=False)
    attn = RoPEReadCrossAttention(D, 4, cfg)
    assert isinstance(attn.post_rope_q, nn.Identity)
    assert isinstance(attn.post_rope_k, nn.Identity)
