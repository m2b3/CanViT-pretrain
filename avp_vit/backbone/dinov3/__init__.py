"""DINOv3 backbone wrapper for AVP."""

from typing import Any, cast, override

import torch
from dinov3.models.vision_transformer import DinoVisionTransformer
from torch import Tensor, nn

from avp_vit.backbone import ViTBackbone


class DINOv3Backbone(ViTBackbone, nn.Module):
    """Wraps a DINOv3 ViT for use with AVP.

    The output_norm property exposes the LayerNorm used to produce x_norm_patchtokens
    in forward_features. Apply it to raw block outputs before comparison with teacher.
    """

    _backbone: DinoVisionTransformer
    _rope_periods: Tensor
    _rope_dtype: torch.dtype

    def __init__(self, backbone: DinoVisionTransformer) -> None:
        nn.Module.__init__(self)
        self._backbone = backbone

        rope_embed = backbone.rope_embed
        periods = rope_embed.periods
        dtype = rope_embed.dtype
        assert isinstance(periods, Tensor)
        assert dtype is not None
        self.register_buffer("_rope_periods", periods)
        self._rope_periods = periods
        self._rope_dtype = dtype

    @property
    @override
    def embed_dim(self) -> int:
        return self._backbone.embed_dim

    @property
    @override
    def num_heads(self) -> int:
        return self._backbone.num_heads

    @property
    @override
    def n_prefix_tokens(self) -> int:
        return 1 + self._backbone.n_storage_tokens

    @property
    @override
    def n_register_tokens(self) -> int:
        return self._backbone.n_storage_tokens

    @property
    @override
    def n_blocks(self) -> int:
        return len(self._backbone.blocks)

    @property
    @override
    def patch_size(self) -> int:
        return self._backbone.patch_size

    @property
    @override
    def rope_periods(self) -> Tensor:
        return self._rope_periods

    @property
    @override
    def rope_dtype(self) -> torch.dtype:
        return self._rope_dtype

    @override
    def forward_block(
        self, idx: int, x: Tensor, rope: tuple[Tensor, Tensor] | None
    ) -> Tensor:
        out: Tensor = self._backbone.blocks[idx](x, rope)
        return out

    def forward_norm_patches(self, images: Tensor) -> Tensor:
        """Forward pass returning normalized patch tokens [B, H*W, D]."""
        out = cast(dict[str, Any], self._backbone.forward_features(images))
        return out["x_norm_patchtokens"]

    @override
    def prepare_tokens(self, images: Tensor) -> tuple[Tensor, int, int]:
        x, hw = self._backbone.prepare_tokens_with_masks(images, masks=None)
        H, W = cast(tuple[int, int], hw)
        return x, H, W

    def block_flops(self, n_tokens: int) -> int:
        """FLOPs for one DINOv3 transformer block (standard ViT: 4× MLP)."""
        N, D = n_tokens, self.embed_dim
        # Self-attention: 8ND² + 4N²D (QKV+O projections + attention)
        # MLP (4× expansion): 16ND²
        return 24 * N * D * D + 4 * N * N * D

    def patch_embed_flops(self, n_patches: int) -> int:
        """FLOPs for patch embedding Conv2d(3, D, kernel=P, stride=P)."""
        D, P = self.embed_dim, self.patch_size
        return 2 * D * 3 * P * P * n_patches
