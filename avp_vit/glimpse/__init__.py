"""Glimpse extraction and tokenization for AVP.

Coordinate convention (shared with avp_vit.rope):
- Coordinates are in [-1, 1]^2 (DINOv3/PyTorch convention)
- (0, 0) is image center, (-1, -1) is top-left, (1, 1) is bottom-right
- centers specify where the glimpse is centered in scene space
- scales specify the glimpse size relative to scene (1 = full scene)
"""

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor, nn

from avp_vit.rope import grid_offsets


class PixelBox(NamedTuple):
    """Axis-aligned bounding box in pixel coordinates.

    Coordinate system:
    - Origin (0, 0) at top-left corner
    - x increases rightward (horizontal)
    - y increases downward (vertical)
    """

    left: float  # x of left edge
    top: float  # y of top edge
    width: float  # horizontal extent
    height: float  # vertical extent
    center_x: float  # x of center
    center_y: float  # y of center


def normalized_to_pixel(coord: float, size: int) -> float:
    """Convert normalized [-1, 1] coordinate to pixel [0, size]."""
    return (coord + 1) / 2 * size


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

    def to_pixel_box(self, batch_idx: int, H: int, W: int) -> PixelBox:
        """Convert viewpoint for a single batch item to pixel coordinates.

        Args:
            batch_idx: Index into the batch dimension.
            H: Image height in pixels.
            W: Image width in pixels.

        Returns:
            PixelBox with coordinates in pixel space.
        """
        cy, cx = self.centers[batch_idx].tolist()
        scale = self.scales[batch_idx].item()

        center_x = normalized_to_pixel(cx, W)
        center_y = normalized_to_pixel(cy, H)
        width = scale * W
        height = scale * H

        return PixelBox(
            left=center_x - width / 2,
            top=center_y - height / 2,
            width=width,
            height=height,
            center_x=center_x,
            center_y=center_y,
        )


def sample_at_viewpoint(spatial: Tensor, viewpoint: Viewpoint, out_size: int) -> Tensor:
    """Sample from spatial tensor at viewpoint positions.

    THE primitive for viewpoint-based sampling. Works for images [B,C,H,W]
    or latent feature maps [B,D,G,G] — any spatial tensor.

    Args:
        spatial: [B, C, H, W] spatial tensor (images or feature maps)
        viewpoint: Viewpoint with centers [B, 2] and scales [B]
        out_size: output spatial size (out_size x out_size)

    Returns:
        [B, C, out_size, out_size] bilinearly sampled crop
    """
    B = viewpoint.centers.shape[0]
    device = spatial.device

    offsets = grid_offsets(out_size, out_size, device, dtype=torch.float32)
    offsets = offsets.view(out_size, out_size, 2).unsqueeze(0)

    centers, scales = viewpoint.centers, viewpoint.scales
    grid = centers.view(B, 1, 1, 2) + scales.view(B, 1, 1, 1) * offsets

    # grid_sample expects (x, y), our offsets are (y, x)
    grid = grid.flip(-1)

    return nn.functional.grid_sample(spatial, grid, mode="bilinear", align_corners=False)


