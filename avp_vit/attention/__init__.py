from typing import final

import torch.nn.functional as F
from torch import Tensor, nn
from ytch.attention.mh import from_multihead, to_multihead
from ytch.nn.elementwise_affine import ElementwiseAffine

from avp_vit.rope import rope_apply_with_prefix


class RoPECrossAttention(nn.Module):
    """Cross-attention with RoPE. Subclasses configure Q/K/V/O transforms."""

    dim: int
    num_heads: int
    affine_q: ElementwiseAffine
    affine_k: ElementwiseAffine
    affine_v: ElementwiseAffine
    q_transform: nn.Module
    k_transform: nn.Module
    v_transform: nn.Module
    out_transform: nn.Module

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.affine_q = ElementwiseAffine(dim)
        self.affine_k = ElementwiseAffine(dim)
        self.affine_v = ElementwiseAffine(dim)

    def forward(
        self,
        q_in: Tensor,
        kv_in: Tensor,
        q_rope: tuple[Tensor, Tensor],
        kv_rope: tuple[Tensor, Tensor],
    ) -> Tensor:
        q_normed = F.layer_norm(q_in, (self.dim,))
        kv_normed = F.layer_norm(kv_in, (self.dim,))

        q = to_multihead(self.q_transform(self.affine_q(q_normed)), self.num_heads)
        k = to_multihead(self.k_transform(self.affine_k(kv_normed)), self.num_heads)
        v = to_multihead(self.v_transform(self.affine_v(kv_normed)), self.num_heads)

        q = rope_apply_with_prefix(q, q_rope)
        k = rope_apply_with_prefix(k, kv_rope)

        out = F.scaled_dot_product_attention(q, k, v)
        return self.out_transform(from_multihead(out))

    def flops(self, n_q: int, n_kv: int) -> int:
        """FLOPs for one forward pass. Introspects actual transform structure."""
        D = self.dim
        f = 4 * n_q * n_kv * D  # Q @ Kᵀ + attn @ V
        if not isinstance(self.q_transform, nn.Identity):
            f += 2 * n_q * D * D
        if not isinstance(self.k_transform, nn.Identity):
            f += 2 * n_kv * D * D
        if not isinstance(self.v_transform, nn.Identity):
            f += 2 * n_kv * D * D
        if not isinstance(self.out_transform, nn.Identity):
            f += 2 * n_q * D * D
        return f


@final
class RoPEReadCrossAttention(RoPECrossAttention):
    """For reading: Q and O projected, K and V unprojected."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__(dim, num_heads)
        self.q_transform = nn.Linear(dim, dim)
        self.k_transform = nn.Identity()
        self.v_transform = nn.Identity()
        self.out_transform = nn.Linear(dim, dim)


@final
class RoPEWriteCrossAttention(RoPECrossAttention):
    """For writing: K and V projected, Q and O unprojected.

    V is identity-initialized so initial writes copy glimpse features directly.
    As training progresses, V learns what content to actually write.
    """

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__(dim, num_heads)
        self.q_transform = nn.Identity()
        self.k_transform = nn.Linear(dim, dim)
        v_proj = nn.Linear(dim, dim)
        nn.init.eye_(v_proj.weight)
        nn.init.zeros_(v_proj.bias)
        self.v_transform = v_proj
        self.out_transform = nn.Identity()
