"""Convex gated attention: dynamic per-token, per-dimension gating."""

import math
from typing import final

import torch
from torch import Tensor, nn

from avp_vit.attention import RoPECrossAttention


@final
class ConvexGatedAttention(nn.Module):
    """Convex update: x_new = (1-g)*x + g*proposal where g = sigmoid(scale*gate_attn + bias).

    At initialization, gate H gate_init (matching LayerScale behavior):
    - gate_scale starts at 0, so gate_attn contribution is zero
    - gate_bias = logit(gate_init), so sigmoid(gate_bias) = gate_init
    """

    proposal_attn: RoPECrossAttention
    gate_attn: RoPECrossAttention
    gate_scale: nn.Parameter  # [D], starts at 0
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
        self.gate_scale = nn.Parameter(torch.zeros(dim))
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
        gate = torch.sigmoid(self.gate_scale * gate_raw + self.gate_bias)
        return (1 - gate) * x + gate * proposal

    def flops(self, n_q: int, n_kv: int) -> int:
        """FLOPs for forward pass: proposal + gate attention."""
        return self.proposal_attn.flops(n_q, n_kv) + self.gate_attn.flops(n_q, n_kv)
