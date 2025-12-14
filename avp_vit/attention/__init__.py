from dataclasses import dataclass
from typing import final

import torch.nn.functional as F
from torch import Tensor, nn
from ytch.attention.mh import from_multihead, to_multihead
from ytch.nn.elementwise_affine import ElementwiseAffine

from avp_vit.rope import rope_apply_with_prefix


@final
@dataclass
class AttentionConfig:
    """Ablation flags for cross-attention modules."""

    use_pre_affine: bool = True  # EWA before Q/K/V transforms
    use_post_rope_affine: bool = False  # EWA after RoPE on Q/K
    identity_init_v: bool = False  # Identity-init V projection (write attn only)


class RoPECrossAttention(nn.Module):
    """Cross-attention with RoPE. Subclasses configure Q/K/V/O transforms."""

    dim: int
    num_heads: int
    cfg: AttentionConfig
    affine_q: nn.Module
    affine_k: nn.Module
    affine_v: nn.Module
    post_rope_q: nn.Module
    post_rope_k: nn.Module
    q_transform: nn.Module
    k_transform: nn.Module
    v_transform: nn.Module
    out_transform: nn.Module

    def __init__(self, dim: int, num_heads: int, cfg: AttentionConfig) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.cfg = cfg

        # Pre-transform affines
        pre = ElementwiseAffine if cfg.use_pre_affine else nn.Identity
        self.affine_q = pre(dim)
        self.affine_k = pre(dim)
        self.affine_v = pre(dim)

        # Post-RoPE affines (operate on head_dim)
        head_dim = dim // num_heads
        post = ElementwiseAffine if cfg.use_post_rope_affine else nn.Identity
        self.post_rope_q = post(head_dim)
        self.post_rope_k = post(head_dim)

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

        q = self.post_rope_q(rope_apply_with_prefix(q, q_rope))
        k = self.post_rope_k(rope_apply_with_prefix(k, kv_rope))

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

    def __init__(self, dim: int, num_heads: int, cfg: AttentionConfig) -> None:
        super().__init__(dim, num_heads, cfg)
        self.q_transform = nn.Linear(dim, dim)
        self.k_transform = nn.Identity()
        self.v_transform = nn.Identity()
        self.out_transform = nn.Linear(dim, dim)


@final
class RoPEWriteCrossAttention(RoPECrossAttention):
    """For writing: K and V projected, Q and O unprojected."""

    def __init__(self, dim: int, num_heads: int, cfg: AttentionConfig) -> None:
        super().__init__(dim, num_heads, cfg)
        self.q_transform = nn.Identity()
        self.k_transform = nn.Linear(dim, dim)
        self.v_transform = self._make_v_proj(dim, cfg.identity_init_v)
        self.out_transform = nn.Identity()

    @staticmethod
    def _make_v_proj(dim: int, identity_init: bool) -> nn.Linear:
        proj = nn.Linear(dim, dim)
        if identity_init:
            nn.init.eye_(proj.weight)
            nn.init.zeros_(proj.bias)
        return proj


