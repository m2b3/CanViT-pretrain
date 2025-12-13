"""Glimpse extraction and tokenization for AVP.

Coordinate convention (shared with avp_vit.rope):
- Coordinates are in [-1, 1]^2 (DINOv3/PyTorch convention)
- (0, 0) is image center, (-1, -1) is top-left, (1, 1) is bottom-right
- centers specify where the glimpse is centered in scene space
- scales specify the glimpse size relative to scene (1 = full scene)
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from avp_vit.rope import grid_offsets


@dataclass
class Viewpoint:
    """A viewpoint specifying where to look in an image.

    Coordinates use the same convention as grid_offsets/glimpse_positions.
    """

    name: str
    centers: Tensor  # [B, 2] in [-1, 1], (y, x) order
    scales: Tensor  # [B] in (0, 1]

    @staticmethod
    def full_scene(B: int, device: torch.device) -> "Viewpoint":
        """Full scene viewpoint: center=(0,0), scale=1."""
        return Viewpoint(
            name="full",
            centers=torch.zeros(B, 2, device=device),
            scales=torch.ones(B, device=device),
        )

    @staticmethod
    def quadrant(B: int, device: torch.device, qx: int, qy: int) -> "Viewpoint":
        """Quadrant viewpoint: qx,qy in {0,1} -> center, scale=0.5."""
        cx = -0.5 + qx  # 0 -> -0.5, 1 -> 0.5
        cy = -0.5 + qy
        name = ["TL", "TR", "BL", "BR"][qy * 2 + qx]
        # Note: centers are (y, x) to match grid_offsets convention
        centers = torch.tensor([[cy, cx]], device=device).expand(B, -1)
        return Viewpoint(
            name=name,
            centers=centers,
            scales=torch.full((B,), 0.5, device=device),
        )


def extract_glimpse(img: Tensor, viewpoint: Viewpoint, size: int) -> Tensor:
    """Extract glimpse crop from image using grid_sample.

    Uses grid_offsets from avp_vit.rope as the SINGLE SOURCE OF TRUTH for
    coordinate computation. This ensures pixel sampling matches RoPE positions.

    Args:
        img: [B, C, H, W] scene image
        viewpoint: Viewpoint with centers [B, 2] and scales [B]
        size: output size (size x size)

    Returns:
        [B, C, size, size] bilinearly interpolated crop
    """
    B = img.shape[0]
    device = img.device
    centers, scales = viewpoint.centers, viewpoint.scales

    # Get grid offsets from the single source of truth
    # Shape: [size*size, 2] with (y, x) coordinates
    offsets = grid_offsets(size, size, device, dtype=torch.float32)

    # Reshape to [1, size, size, 2]
    offsets = offsets.view(size, size, 2).unsqueeze(0)

    # Transform: positions = centers + scales * offsets
    # centers: [B, 2], scales: [B], offsets: [1, size, size, 2]
    grid = centers.view(B, 1, 1, 2) + scales.view(B, 1, 1, 1) * offsets

    # grid_sample expects (x, y) order, but our offsets are (y, x)
    # Flip the last dimension: [..., 0] = y -> [..., 1], [..., 1] = x -> [..., 0]
    grid = grid.flip(-1)

    return nn.functional.grid_sample(img, grid, mode="bilinear", align_corners=False)
