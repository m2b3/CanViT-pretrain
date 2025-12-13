import math
import torch
from torch import Tensor


def make_grid_positions(grid_h: int, grid_w: int, device: torch.device) -> Tensor:
    """Fixed grid positions in [0, 1]^2. Returns [H*W, 2]."""
    h = torch.arange(grid_h, device=device, dtype=torch.float32)
    w = torch.arange(grid_w, device=device, dtype=torch.float32)
    h = (h + 0.5) / grid_h
    w = (w + 0.5) / grid_w
    grid = torch.stack(torch.meshgrid(h, w, indexing="ij"), dim=-1)  # [H, W, 2]
    return grid.flatten(0, 1)  # [H*W, 2]


def glimpse_positions(
    centers: Tensor,  # [B, 2] in [0, 1]^2
    scales: Tensor,  # [B] or [B, 1]
    grid_h: int,
    grid_w: int,
) -> Tensor:
    """Compute scene-space positions for glimpse patch tokens. Returns [B, H*W, 2]."""
    B = centers.shape[0]
    device = centers.device
    scales = scales.view(B, 1, 1)

    # Local offsets in [-0.5, 0.5]^2
    h = torch.arange(grid_h, device=device, dtype=torch.float32)
    w = torch.arange(grid_w, device=device, dtype=torch.float32)
    h = (h + 0.5) / grid_h - 0.5
    w = (w + 0.5) / grid_w - 0.5
    offsets = torch.stack(torch.meshgrid(h, w, indexing="ij"), dim=-1)  # [H, W, 2]
    offsets = offsets.flatten(0, 1)  # [HW, 2]

    # Scene coords = center + scale * offset
    positions = centers.unsqueeze(1) + scales * offsets.unsqueeze(0)  # [B, HW, 2]
    return positions


def compute_rope(
    positions: Tensor,  # [B, N, 2] coordinates
    head_dim: int,
    base: float = 100.0,
) -> tuple[Tensor, Tensor]:
    """Compute RoPE sin/cos. Returns [B, 1, N, head_dim]."""
    assert positions.ndim == 3 and head_dim % 4 == 0
    n_freqs = head_dim // 4
    periods = base ** (2 * torch.arange(n_freqs, device=positions.device, dtype=positions.dtype) / (head_dim // 2))
    angles = 2 * math.pi * positions.unsqueeze(-1) / periods  # [B, N, 2, n_freqs]
    angles = angles.flatten(-2, -1).tile((2,))  # [B, N, head_dim]
    return torch.sin(angles).unsqueeze(1), torch.cos(angles).unsqueeze(1)


def rope_rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    return (x * cos) + (rope_rotate_half(x) * sin)
