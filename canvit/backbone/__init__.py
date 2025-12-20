"""ViT backbone protocol for CanViT."""

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class ViTBackbone(ABC):
    """Abstract interface for ViT backbones used with CanViT.

    Implementations must provide block-by-block forward and token preparation.
    RoPE parameters exposed for cross-attention position encoding.
    """

    @property
    @abstractmethod
    def embed_dim(self) -> int: ...

    @property
    @abstractmethod
    def num_heads(self) -> int: ...

    @property
    @abstractmethod
    def n_prefix_tokens(self) -> int:
        """Number of prefix tokens (CLS, registers) before patch tokens."""
        ...

    @property
    @abstractmethod
    def n_blocks(self) -> int: ...

    @property
    @abstractmethod
    def patch_size_px(self) -> int:
        """Patch size in pixels."""
        ...

    @property
    @abstractmethod
    def rope_periods(self) -> Tensor:
        """RoPE frequency periods for backbone self-attention."""
        ...

    @property
    @abstractmethod
    def rope_dtype(self) -> torch.dtype:
        """dtype for RoPE computation."""
        ...

    @abstractmethod
    def forward_block(
        self, idx: int, x: Tensor, rope: tuple[Tensor, Tensor] | None
    ) -> Tensor:
        """Run single transformer block. Backbone decides whether to use rope."""
        ...

    @abstractmethod
    def prepare_tokens(self, images: Tensor) -> tuple[Tensor, int, int]:
        """Patchify images. Returns (tokens, grid_h, grid_w)."""
        ...

    def compile(self, **kwargs) -> None:
        """Compile backbone blocks for faster execution. Override in subclasses."""
        pass
