import math

import torch

from avp_vit.attention import AttentionConfig, RoPEReadCrossAttention, RoPEWriteCrossAttention
from avp_vit.attention.convex import ConvexGatedAttention


def _make_rope(B: int, N: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (torch.randn(B, 1, N, head_dim), torch.randn(B, 1, N, head_dim))


def test_output_shape():
    B, N_x, N_kv, D, heads = 2, 10, 20, 64, 4
    head_dim = D // heads
    cfg = AttentionConfig()

    proposal = RoPEReadCrossAttention(D, heads, cfg)
    gate = RoPEReadCrossAttention(D, heads, cfg)
    cvx = ConvexGatedAttention(proposal, gate, gate_init=1e-5)

    x = torch.randn(B, N_x, D)
    kv = torch.randn(B, N_kv, D)
    x_rope = _make_rope(B, N_x, head_dim)
    kv_rope = _make_rope(B, N_kv, head_dim)

    out = cvx(x, kv, x_rope, kv_rope)
    assert out.shape == (B, N_x, D)


def test_gate_init_value():
    """At init, gate = sigmoid(0 * gate_attn + bias) = sigmoid(bias) = gate_init."""
    D, heads = 64, 4
    gate_init = 1e-5
    cfg = AttentionConfig()

    proposal = RoPEReadCrossAttention(D, heads, cfg)
    gate_attn = RoPEReadCrossAttention(D, heads, cfg)
    cvx = ConvexGatedAttention(proposal, gate_attn, gate_init=gate_init)

    # Check bias is logit(gate_init)
    expected_bias = math.log(gate_init / (1 - gate_init))
    assert torch.allclose(cvx.gate_bias, torch.full((D,), expected_bias))

    # Check scale is 0
    assert torch.allclose(cvx.gate_scale, torch.zeros(D))

    # Therefore at init: gate = sigmoid(bias) = gate_init
    gate_at_init = torch.sigmoid(cvx.gate_bias)
    assert torch.allclose(gate_at_init, torch.full((D,), gate_init), rtol=1e-5)


def test_init_behavior_matches_layerscale():
    """At init with scale=0, convex behaves like x + gate_init * proposal."""
    B, N_x, N_kv, D, heads = 2, 10, 20, 64, 4
    head_dim = D // heads
    gate_init = 1e-3
    cfg = AttentionConfig()

    proposal_attn = RoPEReadCrossAttention(D, heads, cfg)
    gate_attn = RoPEReadCrossAttention(D, heads, cfg)
    cvx = ConvexGatedAttention(proposal_attn, gate_attn, gate_init=gate_init)

    x = torch.randn(B, N_x, D)
    kv = torch.randn(B, N_kv, D)
    x_rope = _make_rope(B, N_x, head_dim)
    kv_rope = _make_rope(B, N_kv, head_dim)

    # At init: gate = gate_init (constant)
    # x_new = (1 - gate_init) * x + gate_init * proposal
    #       = x - gate_init * x + gate_init * proposal
    #       = x + gate_init * (proposal - x)
    with torch.no_grad():
        proposal = proposal_attn(x, kv, x_rope, kv_rope)
        out = cvx(x, kv, x_rope, kv_rope)

        expected = (1 - gate_init) * x + gate_init * proposal
        assert torch.allclose(out, expected, rtol=1e-5)


def test_gradient_flow():
    """Gradients flow through both proposal and gate paths."""
    B, N_x, N_kv, D, heads = 2, 5, 10, 32, 4
    head_dim = D // heads
    cfg = AttentionConfig()

    proposal = RoPEReadCrossAttention(D, heads, cfg)
    gate_attn = RoPEReadCrossAttention(D, heads, cfg)
    cvx = ConvexGatedAttention(proposal, gate_attn, gate_init=0.5)

    x = torch.randn(B, N_x, D, requires_grad=True)
    kv = torch.randn(B, N_kv, D)
    x_rope = _make_rope(B, N_x, head_dim)
    kv_rope = _make_rope(B, N_kv, head_dim)

    out = cvx(x, kv, x_rope, kv_rope)
    out.sum().backward()

    assert x.grad is not None
    assert cvx.gate_scale.grad is not None
    assert cvx.gate_bias.grad is not None


def test_works_with_write_attention():
    """ConvexGatedAttention works with RoPEWriteCrossAttention too."""
    B, N_x, N_kv, D, heads = 2, 20, 10, 64, 4
    head_dim = D // heads
    cfg = AttentionConfig()

    proposal = RoPEWriteCrossAttention(D, heads, cfg)
    gate_attn = RoPEWriteCrossAttention(D, heads, cfg)
    cvx = ConvexGatedAttention(proposal, gate_attn, gate_init=1e-5)

    x = torch.randn(B, N_x, D)
    kv = torch.randn(B, N_kv, D)
    x_rope = _make_rope(B, N_x, head_dim)
    kv_rope = _make_rope(B, N_kv, head_dim)

    out = cvx(x, kv, x_rope, kv_rope)
    assert out.shape == (B, N_x, D)


def test_flops_is_sum_of_inner():
    """ConvexGatedAttention.flops() = proposal.flops() + gate.flops()."""
    D, heads = 64, 4
    cfg = AttentionConfig()

    proposal = RoPEReadCrossAttention(D, heads, cfg)
    gate_attn = RoPEReadCrossAttention(D, heads, cfg)
    cvx = ConvexGatedAttention(proposal, gate_attn, gate_init=0.5)

    n_q, n_kv = 50, 256
    expected = proposal.flops(n_q, n_kv) + gate_attn.flops(n_q, n_kv)
    assert cvx.flops(n_q, n_kv) == expected
    assert cvx.flops(n_q, n_kv) == 2 * proposal.flops(n_q, n_kv)  # Same arch → 2x
