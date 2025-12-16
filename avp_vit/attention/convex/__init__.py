"""Convex gated attention: dynamic per-token gating."""

import math
from typing import final

import torch
from torch import Tensor, nn

from avp_vit.attention import RoPECrossAttention


@final
class ConvexGatedAttention(nn.Module):
    """Convex update with TWO attention ops: proposal and gate.

    x_new = (1-g)*x + g*proposal where g = sigmoid(gate_attn + bias).
    Per-token, per-dimension gating. Expensive but most expressive.

    At init: gate_bias = logit(gate_init), so sigmoid(bias) = gate_init.
    """

    proposal_attn: RoPECrossAttention
    gate_attn: RoPECrossAttention
    gate_bias: nn.Parameter  # [D], starts at logit(gate_init)

    def __init__(
        self,
        proposal_attn: RoPECrossAttention,
        gate_attn: RoPECrossAttention,
        gate_init: float,
    ) -> None:
        super().__init__()
        assert 0 < gate_init < 1, f"gate_init must be in (0, 1), got {gate_init}"
        assert proposal_attn.dim == gate_attn.dim

        self.proposal_attn = proposal_attn
        self.gate_attn = gate_attn

        dim = proposal_attn.dim
        bias_init = math.log(gate_init / (1 - gate_init))  # logit(gate_init)
        self.gate_bias = nn.Parameter(torch.full((dim,), bias_init))

    def forward(
        self,
        x: Tensor,
        kv: Tensor,
        x_rope: tuple[Tensor, Tensor],
        kv_rope: tuple[Tensor, Tensor],
    ) -> Tensor:
        proposal = self.proposal_attn(x, kv, x_rope, kv_rope)
        gate_raw = self.gate_attn(x, kv, x_rope, kv_rope)
        gate = torch.sigmoid(gate_raw + self.gate_bias)
        return (1 - gate) * x + gate * proposal

    def flops(self, n_q: int, n_kv: int) -> int:
        """FLOPs for forward pass: proposal + gate attention."""
        return self.proposal_attn.flops(n_q, n_kv) + self.gate_attn.flops(n_q, n_kv)


@final
class CheapConvexGatedAttention(nn.Module):
    """Convex update with ONE attention op + Linear(D→1) gate.

    x_new = (1-g)*x + g*proposal where g = sigmoid(gate_proj(proposal) + bias).
    Per-token (not per-dimension) gating. Much cheaper than ConvexGatedAttention.

    Gate is computed from proposal: "how confident/relevant is this update?"

    At init: gate_proj.weight=0, bias=logit(gate_init), so gate ≈ gate_init.
    """

    attn: RoPECrossAttention
    gate_proj: nn.Linear  # [D] -> [1], no bias (bias is separate param)
    gate_bias: nn.Parameter  # [] scalar

    def __init__(self, attn: RoPECrossAttention, gate_init: float) -> None:
        super().__init__()
        assert 0 < gate_init < 1, f"gate_init must be in (0, 1), got {gate_init}"
        self.attn = attn
        self.gate_proj = nn.Linear(attn.dim, 1, bias=False)
        nn.init.zeros_(self.gate_proj.weight)
        bias_init = math.log(gate_init / (1 - gate_init))
        self.gate_bias = nn.Parameter(torch.tensor(bias_init))

    def forward(
        self,
        x: Tensor,
        kv: Tensor,
        x_rope: tuple[Tensor, Tensor],
        kv_rope: tuple[Tensor, Tensor],
    ) -> Tensor:
        proposal = self.attn(x, kv, x_rope, kv_rope)
        gate = torch.sigmoid(self.gate_proj(proposal) + self.gate_bias)
        return (1 - gate) * x + gate * proposal

    def flops(self, n_q: int, n_kv: int) -> int:
        """FLOPs: attention + gate projection."""
        return self.attn.flops(n_q, n_kv) + 2 * n_q * self.attn.dim
