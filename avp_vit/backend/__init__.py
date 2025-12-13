"""ViT backend abstraction for AVP.

Unifies DINOv3 and I-JEPA (and future backends) behind a common interface.
"""
from abc import ABC, abstractmethod

import torch
from torch import Tensor


def make_rope_periods(
    head_dim: int, base: float = 100.0, device: torch.device | None = None, dtype: torch.dtype = torch.float32
) -> Tensor:
    """Create RoPE frequency periods (DINOv3-style)."""
    n_freqs = head_dim // 4
    exponents = torch.arange(n_freqs, device=device, dtype=dtype) / n_freqs
    return base**exponents


class ViTBackend(ABC):
    """Abstract base for ViT backends."""

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        ...

    @property
    @abstractmethod
    def num_heads(self) -> int:
        ...

    @property
    @abstractmethod
    def n_prefix_tokens(self) -> int:
        """Number of prefix tokens (CLS, registers) before patch tokens."""
        ...

    @property
    @abstractmethod
    def n_blocks(self) -> int:
        ...

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
    def forward_block(self, idx: int, x: Tensor, rope: tuple[Tensor, Tensor] | None) -> Tensor:
        """Run single block. Backend decides whether to use rope."""
        ...

    @abstractmethod
    def prepare_tokens(self, images: Tensor) -> tuple[Tensor, int, int]:
        """Patchify images. Returns (tokens, grid_h, grid_w)."""
        ...
