"""DINOv3 backbone wrapper for CanViT."""

import logging
from typing import Any, NamedTuple, cast, override

import torch
from dinov3.models.vision_transformer import DinoVisionTransformer
from torch import Tensor, nn

from canvit.backbone import ViTBackbone

log = logging.getLogger(__name__)


class NormFeatures(NamedTuple):
    """Normalized features from DINOv3 forward pass."""

    patches: Tensor  # [B, H*W, D]
    cls: Tensor  # [B, D]


class DINOv3Backbone(ViTBackbone, nn.Module):
    """Wraps a DINOv3 ViT for use with CanViT.

    The output_norm property exposes the LayerNorm used to produce x_norm_patchtokens
    in forward_features. Apply it to raw block outputs before comparison with teacher.
    """

    vit: DinoVisionTransformer

    def __init__(self, vit: DinoVisionTransformer) -> None:
        nn.Module.__init__(self)
        self.vit = vit

    @property
    @override
    def embed_dim(self) -> int:
        return self.vit.embed_dim

    @property
    @override
    def num_heads(self) -> int:
        return self.vit.num_heads

    @property
    @override
    def n_prefix_tokens(self) -> int:
        return 1 + self.vit.n_storage_tokens

    @property
    def n_register_tokens(self) -> int:
        return self.vit.n_storage_tokens

    @property
    @override
    def n_blocks(self) -> int:
        return len(self.vit.blocks)

    @property
    @override
    def patch_size_px(self) -> int:
        return self.vit.patch_size

    @property
    @override
    def rope_periods(self) -> Tensor:
        periods = self.vit.rope_embed.periods
        assert isinstance(periods, Tensor)
        return periods

    @property
    @override
    def rope_dtype(self) -> torch.dtype:
        dtype = self.vit.rope_embed.dtype
        assert dtype is not None
        return dtype

    @override
    def forward_block(
        self, idx: int, x: Tensor, rope: tuple[Tensor, Tensor] | None
    ) -> Tensor:
        out: Tensor = self.vit.blocks[idx](x, rope)
        return out

    @property
    def output_norm(self) -> nn.Module:
        """The LayerNorm applied to produce x_norm_patchtokens."""
        return self.vit.norm

    def forward_norm_features(self, images: Tensor) -> NormFeatures:
        """Forward pass returning normalized patches and cls token."""
        out = cast(dict[str, Any], self.vit.forward_features(images))
        return NormFeatures(patches=out["x_norm_patchtokens"], cls=out["x_norm_clstoken"])

    @override
    def prepare_tokens(self, images: Tensor) -> tuple[Tensor, int, int]:
        x, hw = self.vit.prepare_tokens_with_masks(images, masks=None)
        H, W = cast(tuple[int, int], hw)
        return x, H, W

    @property
    def ffn_ratio(self) -> float:
        """FFN hidden dimension ratio, queried from first block's MLP."""
        fc1: nn.Linear = self.vit.blocks[0].mlp.fc1  # type: ignore[assignment]
        return fc1.out_features / fc1.in_features

    def block_flops(self, n_tokens: int) -> int:
        """FLOPs for one DINOv3 transformer block."""
        N, D = n_tokens, self.embed_dim
        # Self-attention: 4*N*D² (Q,K,V,O projections) + 2*N²*D (Q@K^T, softmax@V)
        # MLP: 2*ffn_ratio*N*D² (up + down projections)
        attn = 4 * N * D * D + 2 * N * N * D
        mlp = int(2 * self.ffn_ratio * N * D * D)
        return attn + mlp

    def patch_embed_flops(self, n_patches: int) -> int:
        """FLOPs for patch embedding Conv2d(3, D, kernel=P, stride=P)."""
        D, P = self.embed_dim, self.patch_size_px
        return 2 * D * 3 * P * P * n_patches

    def compile(self, **kwargs) -> None:
        """Compile all transformer blocks."""
        log.info(f"Compiling {self.n_blocks} backbone blocks")
        for block in self.vit.blocks:
            block.compile(**kwargs)
