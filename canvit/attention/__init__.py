"""Cross-attention with RoPE and asymmetric projections."""

from dataclasses import dataclass
from typing import final

import torch.nn.functional as F
from torch import Tensor, nn
from ytch.attention.mh import from_multihead, to_multihead
from ytch.nn.elementwise_affine import ElementwiseAffine
from ytch.nn.layer_scale import LayerScale

from canvit.rope import rope_apply_with_prefix


@final
@dataclass
class CrossAttentionConfig:
    pre_proj_q_ln: bool = True
    pre_proj_k_ln: bool = True
    pre_proj_v_ln: bool = True
    post_proj_qk_ln: bool = False
    use_ewa_transforms: bool = True


class CanvasCrossAttention(nn.Module):
    """Canvas cross-attention with RoPE. Subclasses configure Q/K/V/O transforms."""

    canvas_dim: int
    out_dim: int  # for LayerScale (local_dim for Read, canvas_dim for Write)
    num_heads: int
    q_transform: nn.Module
    k_transform: nn.Module
    v_transform: nn.Module
    out_transform: nn.Module

    def __init__(
        self,
        q_in_dim: int,
        kv_in_dim: int,
        canvas_dim: int,
        out_dim: int,
        num_heads: int,
        cfg: CrossAttentionConfig,
    ) -> None:
        super().__init__()
        assert canvas_dim % num_heads == 0
        self.canvas_dim = canvas_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        head_dim = canvas_dim // num_heads

        def ln_or_id(d: int, use: bool) -> nn.Module:
            return nn.LayerNorm(d, elementwise_affine=False) if use else nn.Identity()

        self.pre_proj_q_ln = ln_or_id(q_in_dim, cfg.pre_proj_q_ln)
        self.pre_proj_k_ln = ln_or_id(kv_in_dim, cfg.pre_proj_k_ln)
        self.pre_proj_v_ln = ln_or_id(kv_in_dim, cfg.pre_proj_v_ln)
        self.post_proj_q_ln = ln_or_id(head_dim, cfg.post_proj_qk_ln)
        self.post_proj_k_ln = ln_or_id(head_dim, cfg.post_proj_qk_ln)

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
            return sum(self._proj_flops(m, n_tokens) for m in module)
        if isinstance(module, (nn.Identity, nn.LayerNorm, ElementwiseAffine)):
            return 0
        raise TypeError(f"Unexpected module type for FLOPs: {type(module)}")

    def flops(self, n_q: int, n_kv: int) -> int:
        """FLOPs for one forward pass.

        Counts: SDPA matmuls (Q@K^T, softmax@V), Linear projections.
        Ignores (<<1% of total): LayerNorm (~5D/token), EWA (~2D/token), softmax (~3 n_q*n_kv).
        """
        f = 4 * n_q * n_kv * self.canvas_dim  # Q@K^T + softmax@V
        f += self._proj_flops(self.q_transform, n_q)
        f += self._proj_flops(self.k_transform, n_kv)
        f += self._proj_flops(self.v_transform, n_kv)
        f += self._proj_flops(self.out_transform, n_q)
        return f


@final
class ReadCrossAttention(CanvasCrossAttention):
    """Read: local queries canvas. Q/O Linear on local, K/V EWA on canvas.

    Attention in canvas_dim space. Local projected up, output projected back down.
    """

    def __init__(
        self,
        local_dim: int,
        canvas_dim: int,
        num_heads: int,
        cfg: CrossAttentionConfig,
    ) -> None:
        super().__init__(
            q_in_dim=local_dim,
            kv_in_dim=canvas_dim,
            canvas_dim=canvas_dim,
            out_dim=local_dim,
            num_heads=num_heads,
            cfg=cfg,
        )
        self.q_transform = nn.Linear(local_dim, canvas_dim)
        self.k_transform = ElementwiseAffine(canvas_dim) if cfg.use_ewa_transforms else nn.Identity()
        self.v_transform = ElementwiseAffine(canvas_dim) if cfg.use_ewa_transforms else nn.Identity()
        self.out_transform = nn.Linear(canvas_dim, local_dim)


@final
class WriteCrossAttention(CanvasCrossAttention):
    """Write: canvas queries local. K/V Linear on local, Q/O EWA on canvas.

    Attention in canvas_dim space. Local K/V projected up, canvas stays in canvas_dim.
    """

    def __init__(
        self,
        local_dim: int,
        canvas_dim: int,
        num_heads: int,
        cfg: CrossAttentionConfig,
    ) -> None:
        super().__init__(
            q_in_dim=canvas_dim,
            kv_in_dim=local_dim,
            canvas_dim=canvas_dim,
            out_dim=canvas_dim,
            num_heads=num_heads,
            cfg=cfg,
        )
        self.q_transform = ElementwiseAffine(canvas_dim) if cfg.use_ewa_transforms else nn.Identity()
        self.k_transform = nn.Linear(local_dim, canvas_dim)
        self.v_transform = nn.Linear(local_dim, canvas_dim)
        self.out_transform = ElementwiseAffine(canvas_dim) if cfg.use_ewa_transforms else nn.Identity()


@final
class ScaledResidualAttention(nn.Module):
    """Attention with residual + LayerScale: x_new = x + scale * attn(x, kv)."""

    def __init__(self, attn: CanvasCrossAttention, scale_init: float) -> None:
        super().__init__()
        self.attn = attn
        self.scale = LayerScale(attn.out_dim, init_values=scale_init)

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
