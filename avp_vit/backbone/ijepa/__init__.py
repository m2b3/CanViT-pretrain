"""I-JEPA backbone wrapper for AVP."""

from typing import override

import torch
from ijepa.models.vision_transformer import VisionTransformer as IJEPAVisionTransformer
from torch import Tensor, nn

from avp_vit.backbone import ViTBackbone
from avp_vit.rope import make_rope_periods


class IJEPABackbone(ViTBackbone, nn.Module):
    """Wraps an I-JEPA ViT for use with AVP.

    I-JEPA uses sincos positional embeddings (added once at tokenization).
    Blocks don't use RoPE - the rope argument is ignored.
    We create our own rope_periods for AVP cross-attention.
    """

    _backbone: IJEPAVisionTransformer
    _rope_periods: Tensor
    _rope_dtype: torch.dtype

    def __init__(self, backbone: IJEPAVisionTransformer, rope_dtype: torch.dtype) -> None:
        nn.Module.__init__(self)
        self._backbone = backbone
        self._rope_dtype = rope_dtype
        head_dim = backbone.embed_dim // backbone.num_heads
        periods = make_rope_periods(head_dim, dtype=rope_dtype)
        self.register_buffer("_rope_periods", periods)
        self._rope_periods = periods

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
        return 0

    @property
    @override
    def n_register_tokens(self) -> int:
        return 0

    @property
    @override
    def n_blocks(self) -> int:
        return len(self._backbone.blocks)

    @property
    @override
    def patch_size(self) -> int:
        return self._backbone.patch_embed.patch_size

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
        out: Tensor = self._backbone.blocks[idx](x)
        return out

    @override
    def prepare_tokens(self, images: Tensor) -> tuple[Tensor, int, int]:
        """Patchify and add sincos positional embeddings.

        Note: I-JEPA's interpolate_pos_encoding has a bug - it assumes CLS token exists
        (does `npatch = x.shape[1] - 1`) but I-JEPA has no CLS. Works for fixed sizes
        because the early return `if npatch == N: return pos_embed` saves it.
        See test_ijepa_variable_resolution_crashes for details.
        """
        x: Tensor = self._backbone.patch_embed(images)
        N = x.shape[1]
        H = W = int(N**0.5)
        assert H * W == N, f"expected square grid, got {N} patches"
        pos_embed: Tensor = self._backbone.interpolate_pos_encoding(
            x, self._backbone.pos_embed
        )
        x = x + pos_embed
        return x, H, W
