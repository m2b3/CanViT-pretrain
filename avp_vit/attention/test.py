import torch

from avp_vit.attention import RoPEReadCrossAttention, RoPEWriteCrossAttention


def test_read_shapes():
    B, N_q, N_kv, D, heads = 2, 10, 20, 64, 4
    head_dim = D // heads
    attn = RoPEReadCrossAttention(D, heads)
    q = torch.randn(B, N_q, D)
    kv = torch.randn(B, N_kv, D)
    q_rope = (torch.randn(B, 1, N_q, head_dim), torch.randn(B, 1, N_q, head_dim))
    kv_rope = (torch.randn(B, 1, N_kv, head_dim), torch.randn(B, 1, N_kv, head_dim))
    out = attn(q, kv, q_rope, kv_rope)
    assert out.shape == (B, N_q, D)


def test_write_shapes():
    B, N_q, N_kv, D, heads = 2, 20, 10, 64, 4
    head_dim = D // heads
    attn = RoPEWriteCrossAttention(D, heads)
    q = torch.randn(B, N_q, D)
    kv = torch.randn(B, N_kv, D)
    q_rope = (torch.randn(B, 1, N_q, head_dim), torch.randn(B, 1, N_q, head_dim))
    kv_rope = (torch.randn(B, 1, N_kv, head_dim), torch.randn(B, 1, N_kv, head_dim))
    out = attn(q, kv, q_rope, kv_rope)
    assert out.shape == (B, N_q, D)


def test_write_v_identity_init():
    """V projection starts as identity so writes copy glimpse content directly."""
    D = 64
    attn = RoPEWriteCrossAttention(D, num_heads=4)
    v = attn.v_transform
    assert isinstance(v, torch.nn.Linear)
    assert torch.allclose(v.weight, torch.eye(D))
    assert torch.allclose(v.bias, torch.zeros(D))


def test_flops_read():
    """Read attention: Q and O projections on queries."""
    D = 64
    attn = RoPEReadCrossAttention(D, num_heads=4)
    f = attn.flops(n_q=10, n_kv=20)
    # attention + Q proj + O proj
    assert f == 4 * 10 * 20 * D + 2 * 10 * D * D + 2 * 10 * D * D


def test_flops_write():
    """Write attention: K and V projections on keys/values."""
    D = 64
    attn = RoPEWriteCrossAttention(D, num_heads=4)
    f = attn.flops(n_q=20, n_kv=10)
    # attention + K proj + V proj
    assert f == 4 * 20 * 10 * D + 2 * 10 * D * D + 2 * 10 * D * D


def test_flops_projection_placement_matters():
    """Which tokens get projected affects FLOPs significantly."""
    D = 64
    read = RoPEReadCrossAttention(D, 4)
    write = RoPEWriteCrossAttention(D, 4)
    # Asymmetric: 10 queries, 100 keys/values
    read_f = read.flops(n_q=10, n_kv=100)
    write_f = write.flops(n_q=10, n_kv=100)
    # Read projects small tensor (queries), write projects large tensor (keys/values)
    assert write_f > read_f * 4
