"""ViT backbone protocol for AVP."""

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class ViTBackbone(ABC):
    """Abstract interface for ViT backbones."""

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
    def n_register_tokens(self) -> int:
        """Number of register/storage tokens (excluding CLS)."""
        ...

    @property
    @abstractmethod
    def n_blocks(self) -> int: ...

    @property
    @abstractmethod
    def rope_periods(self) -> Tensor:
        """RoPE frequency periods for cross-attention."""
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
