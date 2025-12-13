"""DINOv3 backbone wrapper for AVP."""

from typing import cast, override

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

    @property
    def output_norm(self) -> nn.Module:
        """The LayerNorm applied to produce x_norm_patchtokens."""
        return self._backbone.norm

    @override
    def prepare_tokens(self, images: Tensor) -> tuple[Tensor, int, int]:
        x, hw = self._backbone.prepare_tokens_with_masks(images, masks=None)
        H, W = cast(tuple[int, int], hw)
        return x, H, W
