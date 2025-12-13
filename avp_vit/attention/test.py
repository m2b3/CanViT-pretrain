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
