"""
RoPE utilities for AVP.

We reimplement RoPE computation here because DINOv3's rope_embed(H=H, W=W) computes
positions internally from the grid size. We need per-batch-item positions for glimpses
with varying (center, scale), which their API doesn't support.

We use their `periods` buffer directly to stay numerically consistent.
"""
import math

import torch
from torch import Tensor
from ytch.correctness import assert_shape


def make_grid_positions(
    grid_h: int, grid_w: int, device: torch.device, dtype: torch.dtype = torch.float32
) -> Tensor:
    """Fixed grid positions in [-1, 1]^2 (DINOv3 convention)."""
    h = torch.arange(grid_h, device=device, dtype=dtype)
    w = torch.arange(grid_w, device=device, dtype=dtype)
    h = (h + 0.5) / grid_h * 2 - 1
    w = (w + 0.5) / grid_w * 2 - 1
    grid = torch.stack(torch.meshgrid(h, w, indexing="ij"), dim=-1)
    out = grid.flatten(0, 1)
    assert_shape(out, (grid_h * grid_w, 2))
    return out


def glimpse_positions(
    centers: Tensor,
    scales: Tensor,
    grid_h: int,
    grid_w: int,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Compute scene-space positions for glimpse patch tokens."""
    B = centers.shape[0]
    device = centers.device
    scales = scales.view(B, 1, 1).to(dtype)
    centers = centers.to(dtype)

    h = torch.arange(grid_h, device=device, dtype=dtype)
    w = torch.arange(grid_w, device=device, dtype=dtype)
    h = (h + 0.5) / grid_h * 2 - 1
    w = (w + 0.5) / grid_w * 2 - 1
    offsets = torch.stack(torch.meshgrid(h, w, indexing="ij"), dim=-1).flatten(0, 1)

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
