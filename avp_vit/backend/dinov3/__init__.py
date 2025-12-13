"""DINOv3 backend for AVP."""
import torch
from torch import Tensor, nn

from .. import ViTBackend


class DINOv3Backend(ViTBackend, nn.Module):
    """Wraps a DINOv3 backbone for use with AVP."""

    def __init__(self, backbone: nn.Module):
        nn.Module.__init__(self)
        self.backbone = backbone

    @property
    def embed_dim(self) -> int:
        return self.backbone.embed_dim

    @property
    def num_heads(self) -> int:
        return self.backbone.num_heads

    @property
    def n_prefix_tokens(self) -> int:
        return 1 + self.backbone.n_storage_tokens

    @property
    def n_blocks(self) -> int:
        return len(self.backbone.blocks)

    @property
    def rope_periods(self) -> Tensor:
        return self.backbone.rope_embed.periods

    @property
    def rope_dtype(self) -> torch.dtype:
        return self.backbone.rope_embed.dtype

    def forward_block(self, idx: int, x: Tensor, rope: tuple[Tensor, Tensor] | None) -> Tensor:
        return self.backbone.blocks[idx](x, rope)

    def prepare_tokens(self, images: Tensor) -> tuple[Tensor, int, int]:
        x, (H, W) = self.backbone.prepare_tokens_with_masks(images, masks=None)
        return x, H, W
