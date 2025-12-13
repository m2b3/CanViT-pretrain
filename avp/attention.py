import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .rope import rope_apply


class CrossAttention(nn.Module):
    """Cross-attention with RoPE. Q attends to KV."""

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)

    def forward(
        self,
        q_tokens: Tensor,  # [B, N_q, D]
        kv_tokens: Tensor,  # [B, N_kv, D]
        q_rope: tuple[Tensor, Tensor] | None = None,  # (sin, cos) for Q positions
        kv_rope: tuple[Tensor, Tensor] | None = None,  # (sin, cos) for KV positions
        q_prefix: int = 0,  # number of Q tokens without RoPE (CLS, registers)
        kv_prefix: int = 0,  # number of KV tokens without RoPE
    ) -> Tensor:
        B, N_q, D = q_tokens.shape
        N_kv = kv_tokens.shape[1]

        q = self.q_proj(q_tokens).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(kv_tokens).view(B, N_kv, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE to non-prefix tokens
        if q_rope is not None:
            q = self._apply_rope(q, q_rope, q_prefix)
        if kv_rope is not None:
            k = self._apply_rope(k, kv_rope, kv_prefix)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N_q, D)
        return self.out_proj(out)

    def _apply_rope(
        self, x: Tensor, rope: tuple[Tensor, Tensor], prefix: int
    ) -> Tensor:
        """Apply RoPE, skipping prefix tokens. x: [B, heads, N, head_dim], rope: [B, 1, N, D]"""
        sin, cos = rope
        x_prefix, x_rest = x[:, :, :prefix], x[:, :, prefix:]
        x_rest = rope_apply(x_rest, sin, cos)
        return torch.cat([x_prefix, x_rest], dim=2)
