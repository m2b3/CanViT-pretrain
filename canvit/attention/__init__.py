"""Cross-attention with RoPE and asymmetric projections."""

from dataclasses import dataclass
from typing import final

import torch.nn.functional as F
from torch import Tensor, nn
from ytch.attention.mh import from_multihead, to_multihead
from ytch.nn.elementwise_affine import ElementwiseAffine
from ytch.nn.layer_scale import LayerScale

from canvit.rope import rope_apply_with_prefix


# === Cross-attention ===


@final
@dataclass
class CrossAttentionConfig:
    """Config for a single cross-attention module (read OR write)."""

    pre_proj_q_ln: bool = True
    pre_proj_k_ln: bool = True
    pre_proj_v_ln: bool = True
    post_proj_qk_ln: bool = False
    use_ewa_transforms: bool = False


def _ln_or_identity(dim: int, normalize: bool) -> nn.Module:
    return nn.LayerNorm(dim, elementwise_affine=False) if normalize else nn.Identity()


def _ewa_or_identity(dim: int, use_ewa: bool) -> nn.Module:
    return ElementwiseAffine(dim) if use_ewa else nn.Identity()


class RoPECrossAttention(nn.Module):
    """Cross-attention with RoPE. Subclasses configure Q/K/V/O transforms."""

    dim: int
    num_heads: int
    q_transform: nn.Module
    k_transform: nn.Module
    v_transform: nn.Module
    out_transform: nn.Module

    def __init__(self, dim: int, num_heads: int, cfg: CrossAttentionConfig) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.pre_proj_q_ln = _ln_or_identity(dim=dim, normalize=cfg.pre_proj_q_ln)
        self.pre_proj_k_ln = _ln_or_identity(dim=dim, normalize=cfg.pre_proj_k_ln)
        self.pre_proj_v_ln = _ln_or_identity(dim=dim, normalize=cfg.pre_proj_v_ln)
        self.post_proj_q_ln = _ln_or_identity(dim=head_dim, normalize=cfg.post_proj_qk_ln)
        self.post_proj_k_ln = _ln_or_identity(dim=head_dim, normalize=cfg.post_proj_qk_ln)

    def forward(
        self,
        q_in: Tensor,
        kv_in: Tensor,
        q_rope: tuple[Tensor, Tensor],
        kv_rope: tuple[Tensor, Tensor],
    ) -> Tensor:
        q = to_multihead(self.q_transform(self.pre_proj_q_ln(q_in)), self.num_heads)
        k = to_multihead(self.k_transform(self.pre_proj_k_ln(kv_in)), self.num_heads)
        v = to_multihead(self.v_transform(self.pre_proj_v_ln(kv_in)), self.num_heads)

        q = self.post_proj_q_ln(q)
        k = self.post_proj_k_ln(k)

        q = rope_apply_with_prefix(q, q_rope)
        k = rope_apply_with_prefix(k, kv_rope)

        out = F.scaled_dot_product_attention(q, k, v)
        return self.out_transform(from_multihead(out))

    def _proj_flops(self, module: nn.Module, n_tokens: int) -> int:
        if isinstance(module, nn.Linear):
            return 2 * n_tokens * module.in_features * module.out_features
        if isinstance(module, nn.Sequential):
            return sum(
                2 * n_tokens * m.in_features * m.out_features
                for m in module
                if isinstance(m, nn.Linear)
            )
        return 0

    def flops(self, n_q: int, n_kv: int) -> int:
        """FLOPs for one forward pass.

        Counts: SDPA matmuls (Q@K^T, softmax@V), Linear projections.
        Ignores (<<1% of total): LayerNorm (~5D/token), EWA (~2D/token), softmax (~3 n_q*n_kv).
        """
        f = 4 * n_q * n_kv * self.dim  # Q@K^T + softmax@V
        f += self._proj_flops(self.q_transform, n_q)
        f += self._proj_flops(self.k_transform, n_kv)
        f += self._proj_flops(self.v_transform, n_kv)
        f += self._proj_flops(self.out_transform, n_q)
        return f


@final
class RoPEReadCrossAttention(RoPECrossAttention):
    """Read: Q, O are Linear on local; K, V are EWA on canvas."""

    def __init__(self, dim: int, num_heads: int, cfg: CrossAttentionConfig) -> None:
        super().__init__(dim, num_heads, cfg)
        self.q_transform = nn.Linear(dim, dim)
        self.k_transform = _ewa_or_identity(dim=dim, use_ewa=cfg.use_ewa_transforms)
        self.v_transform = _ewa_or_identity(dim=dim, use_ewa=cfg.use_ewa_transforms)
        self.out_transform = nn.Linear(dim, dim)


@final
class RoPEWriteCrossAttention(RoPECrossAttention):
    """Write: K, V are Linear on local; Q, O are EWA on canvas."""

    def __init__(self, dim: int, num_heads: int, cfg: CrossAttentionConfig) -> None:
        super().__init__(dim, num_heads, cfg)
        self.q_transform = _ewa_or_identity(dim=dim, use_ewa=cfg.use_ewa_transforms)
        self.k_transform = nn.Linear(dim, dim)
        self.v_transform = nn.Linear(dim, dim)
        self.out_transform = _ewa_or_identity(dim=dim, use_ewa=cfg.use_ewa_transforms)


@final
class ScaledResidualAttention(nn.Module):
    """Attention with residual + LayerScale: x_new = x + scale * attn(x, kv)."""

    def __init__(self, attn: RoPECrossAttention, scale_init: float) -> None:
        super().__init__()
        self.attn = attn
        self.scale = LayerScale(attn.dim, init_values=scale_init)

    def forward(
        self,
        x: Tensor,
        kv: Tensor,
        x_rope: tuple[Tensor, Tensor],
        kv_rope: tuple[Tensor, Tensor],
    ) -> Tensor:
        return x + self.scale(self.attn(x, kv, x_rope, kv_rope))

    def flops(self, n_q: int, n_kv: int) -> int:
        return self.attn.flops(n_q, n_kv)
