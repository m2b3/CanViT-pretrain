import math

import torch

from avp_vit.attention import CrossAttentionConfig, RoPEReadCrossAttention, RoPEWriteCrossAttention
from avp_vit.attention.convex import CheapConvexGatedAttention, ConvexGatedAttention


def _make_rope(B: int, N: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (torch.randn(B, 1, N, head_dim), torch.randn(B, 1, N, head_dim))


# ==================== ConvexGatedAttention ====================


def test_output_shape():
    B, N_x, N_kv, D, heads = 2, 10, 20, 64, 4
    head_dim = D // heads
    cfg = CrossAttentionConfig()

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
    """At init, gate = sigmoid(bias) = gate_init (gate_attn output doesn't matter initially)."""
    D, heads = 64, 4
    gate_init = 1e-5
    cfg = CrossAttentionConfig()

    proposal = RoPEReadCrossAttention(D, heads, cfg)
    gate_attn = RoPEReadCrossAttention(D, heads, cfg)
    cvx = ConvexGatedAttention(proposal, gate_attn, gate_init=gate_init)

    expected_bias = math.log(gate_init / (1 - gate_init))
    assert torch.allclose(cvx.gate_bias, torch.full((D,), expected_bias))


def test_gradient_flow():
    """Gradients flow through both proposal and gate paths."""
    B, N_x, N_kv, D, heads = 2, 5, 10, 32, 4
    head_dim = D // heads
    cfg = CrossAttentionConfig()

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
    assert cvx.gate_bias.grad is not None


def test_works_with_write_attention():
    """ConvexGatedAttention works with RoPEWriteCrossAttention too."""
    B, N_x, N_kv, D, heads = 2, 20, 10, 64, 4
    head_dim = D // heads
    cfg = CrossAttentionConfig()

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
    cfg = CrossAttentionConfig()

    proposal = RoPEReadCrossAttention(D, heads, cfg)
    gate_attn = RoPEReadCrossAttention(D, heads, cfg)
    cvx = ConvexGatedAttention(proposal, gate_attn, gate_init=0.5)

    n_q, n_kv = 50, 256
    expected = proposal.flops(n_q, n_kv) + gate_attn.flops(n_q, n_kv)
    assert cvx.flops(n_q, n_kv) == expected


# ==================== CheapConvexGatedAttention ====================


def test_cheap_output_shape():
    B, N_x, N_kv, D, heads = 2, 10, 20, 64, 4
    head_dim = D // heads
    cfg = CrossAttentionConfig()

    attn = RoPEReadCrossAttention(D, heads, cfg)
    cvx = CheapConvexGatedAttention(attn, gate_init=1e-5)

    x = torch.randn(B, N_x, D)
    kv = torch.randn(B, N_kv, D)
    x_rope = _make_rope(B, N_x, head_dim)
    kv_rope = _make_rope(B, N_kv, head_dim)

    out = cvx(x, kv, x_rope, kv_rope)
    assert out.shape == (B, N_x, D)


def test_cheap_gate_init():
    """At init: gate_proj.weight=0, bias=logit(gate_init)."""
    D, heads = 64, 4
    gate_init = 1e-3
    cfg = CrossAttentionConfig()

    attn = RoPEReadCrossAttention(D, heads, cfg)
    cvx = CheapConvexGatedAttention(attn, gate_init=gate_init)

    assert torch.allclose(cvx.gate_proj.weight, torch.zeros(1, D))
    expected_bias = math.log(gate_init / (1 - gate_init))
    assert torch.allclose(cvx.gate_bias, torch.tensor(expected_bias))


def test_cheap_init_behavior():
    """At init with zero gate_proj.weight, gate ≈ gate_init for all tokens."""
    B, N_x, N_kv, D, heads = 2, 10, 20, 64, 4
    head_dim = D // heads
    gate_init = 1e-3
    cfg = CrossAttentionConfig()

    attn = RoPEReadCrossAttention(D, heads, cfg)
    cvx = CheapConvexGatedAttention(attn, gate_init=gate_init)

    x = torch.randn(B, N_x, D)
    kv = torch.randn(B, N_kv, D)
    x_rope = _make_rope(B, N_x, head_dim)
    kv_rope = _make_rope(B, N_kv, head_dim)

    with torch.no_grad():
        proposal = attn(x, kv, x_rope, kv_rope)
        out = cvx(x, kv, x_rope, kv_rope)
        expected = (1 - gate_init) * x + gate_init * proposal
        assert torch.allclose(out, expected, rtol=1e-5)


def test_cheap_gradient_flow():
    """Gradients flow to gate_proj and gate_bias."""
    B, N_x, N_kv, D, heads = 2, 5, 10, 32, 4
    head_dim = D // heads
    cfg = CrossAttentionConfig()

    attn = RoPEReadCrossAttention(D, heads, cfg)
    cvx = CheapConvexGatedAttention(attn, gate_init=0.5)

    x = torch.randn(B, N_x, D, requires_grad=True)
    kv = torch.randn(B, N_kv, D)
    x_rope = _make_rope(B, N_x, head_dim)
    kv_rope = _make_rope(B, N_kv, head_dim)

    out = cvx(x, kv, x_rope, kv_rope)
    out.sum().backward()

    assert x.grad is not None
    assert cvx.gate_proj.weight.grad is not None
    assert cvx.gate_bias.grad is not None


def test_cheap_flops_less_than_full():
    """CheapConvexGatedAttention uses ~half the FLOPs of ConvexGatedAttention."""
    D, heads = 64, 4
    cfg = CrossAttentionConfig()
    n_q, n_kv = 50, 256

    attn = RoPEReadCrossAttention(D, heads, cfg)
    cheap = CheapConvexGatedAttention(attn, gate_init=0.5)

    full_proposal = RoPEReadCrossAttention(D, heads, cfg)
    full_gate = RoPEReadCrossAttention(D, heads, cfg)
    full = ConvexGatedAttention(full_proposal, full_gate, gate_init=0.5)

    # Cheap should be ~half the FLOPs (1 attn vs 2)
    assert cheap.flops(n_q, n_kv) < full.flops(n_q, n_kv)
    # Linear(D,1) adds only 2*n_q*D FLOPs, negligible vs attention
    assert cheap.flops(n_q, n_kv) == attn.flops(n_q, n_kv) + 2 * n_q * D
