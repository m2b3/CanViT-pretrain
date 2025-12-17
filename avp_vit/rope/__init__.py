"""RoPE utilities for AVP.

We reimplement RoPE computation here because DINOv3's rope_embed(H=H, W=W) computes
positions internally from the grid size. We need per-batch-item positions for glimpses
with varying (center, scale), which their API doesn't support.

We use their `periods` buffer directly to stay numerically consistent.
"""

import math

import torch
from torch import Tensor
from ytch.correctness import assert_shape


def make_rope_periods(
    head_dim: int,
    dtype: torch.dtype,
    base: float = 100.0,
    device: torch.device | None = None,
) -> Tensor:
    """Create RoPE frequency periods (DINOv3-style)."""
    n_freqs = head_dim // 4
    exponents = torch.arange(n_freqs, device=device, dtype=dtype) / n_freqs
    return base**exponents


def grid_offsets(grid_h: int, grid_w: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Normalized grid offsets in [-1, 1]^2 (DINOv3 convention).

    This is the SINGLE SOURCE OF TRUTH for patch center coordinates.
    Used by both glimpse_positions (for RoPE) and sample_at_viewpoint (for cropping).

    Convention: (idx + 0.5) / grid_size * 2 - 1
    - Maps patch centers to [-1, 1] (not edges)
    - Grid indexed as (row, col) = (y, x)
    - Output shape: [H*W, 2] with [..., 0] = y, [..., 1] = x

    Returns:
        Tensor of shape [grid_h * grid_w, 2] with (y, x) coordinates
    """
    h = torch.arange(grid_h, device=device, dtype=dtype)
    w = torch.arange(grid_w, device=device, dtype=dtype)
    h = (h + 0.5) / grid_h * 2 - 1
    w = (w + 0.5) / grid_w * 2 - 1
    return torch.stack(torch.meshgrid(h, w, indexing="ij"), dim=-1).flatten(0, 1)


def make_grid_positions(
    grid_h: int, grid_w: int, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """Fixed grid positions in [-1, 1]^2 (DINOv3 convention)."""
    out = grid_offsets(grid_h, grid_w, device, dtype)
    assert_shape(out, (grid_h * grid_w, 2))
    return out


def glimpse_positions(
    centers: Tensor, scales: Tensor, grid_h: int, grid_w: int, dtype: torch.dtype
) -> Tensor:
    """Compute scene-space positions for glimpse patch tokens."""
    B = centers.shape[0]
    device = centers.device
    scales = scales.view(B, 1, 1).to(dtype)
    centers = centers.to(dtype)
    offsets = grid_offsets(grid_h, grid_w, device, dtype)
    positions = centers.unsqueeze(1) + scales * offsets.unsqueeze(0)
    assert_shape(positions, (B, grid_h * grid_w, 2))
    return positions


def compute_rope(positions: Tensor, periods: Tensor) -> tuple[Tensor, Tensor]:
    """Compute RoPE sin/cos from positions and frequency periods."""
    B, N, _ = positions.shape
    assert_shape(positions, (B, N, 2))
    assert positions.device == periods.device
    n_freqs = periods.shape[0]
    head_dim = 4 * n_freqs

    angles = 2 * math.pi * positions.unsqueeze(-1) / periods
    angles = angles.flatten(-2, -1).tile((2,))
    sin, cos = torch.sin(angles).unsqueeze(1), torch.cos(angles).unsqueeze(1)
    assert_shape(sin, (B, 1, N, head_dim))
    return sin, cos


def rope_rotate_half(x: Tensor) -> Tensor:
    """RoPE helper: [-x2, x1] from [x1, x2]."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    """Apply RoPE rotation to tensor."""
    assert x.shape[-2:] == sin.shape[-2:] == cos.shape[-2:]
    return (x * cos) + (rope_rotate_half(x) * sin)


def rope_apply_with_prefix(x: Tensor, rope: tuple[Tensor, Tensor]) -> Tensor:
    """Apply RoPE, skipping prefix tokens (CLS, registers).

    Prefix count inferred from shape: x has N tokens, rope has N_rope, prefix = N - N_rope.
    """
    sin, cos = rope
    prefix = x.shape[2] - sin.shape[2]
    x_prefix, x_rest = x[:, :, :prefix], x[:, :, prefix:]
    x_rest = rope_apply(x_rest, sin, cos)
    return torch.cat([x_prefix, x_rest], dim=2)
