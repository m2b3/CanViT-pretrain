"""I-JEPA backend for AVP."""
import torch
from torch import Tensor, nn

from .. import ViTBackend, make_rope_periods


class IJEPABackend(ViTBackend, nn.Module):
    """Wraps an I-JEPA backbone for use with AVP.

    I-JEPA uses sincos positional embeddings (added once at tokenization).
    Blocks don't use RoPE - the rope argument is ignored.
    We create our own rope_periods for AVP cross-attention.
    """

    def __init__(self, backbone: nn.Module, rope_dtype: torch.dtype = torch.float32):
        nn.Module.__init__(self)
        self.backbone = backbone
        self._rope_dtype = rope_dtype
        head_dim = backbone.embed_dim // backbone.num_heads
        self.register_buffer("_rope_periods", make_rope_periods(head_dim, dtype=rope_dtype))

    @property
    def embed_dim(self) -> int:
        return self.backbone.embed_dim

    @property
    def num_heads(self) -> int:
        return self.backbone.num_heads

    @property
    def n_prefix_tokens(self) -> int:
        return 0

    @property
    def n_blocks(self) -> int:
        return len(self.backbone.blocks)

    @property
    def rope_periods(self) -> Tensor:
        return self._rope_periods

    @property
    def rope_dtype(self) -> torch.dtype:
        return self._rope_dtype

    def forward_block(self, idx: int, x: Tensor, rope: tuple[Tensor, Tensor] | None) -> Tensor:
        return self.backbone.blocks[idx](x)

    def prepare_tokens(self, images: Tensor) -> tuple[Tensor, int, int]:
        x = self.backbone.patch_embed(images)
        B, N, D = x.shape
        H = W = int(N**0.5)
        pos_embed = self.backbone.interpolate_pos_encoding(x, self.backbone.pos_embed)
        x = x + pos_embed
        return x, H, W
