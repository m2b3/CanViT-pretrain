"""Viewpoint sampling for training and evaluation.

Extends canvit.viewpoint with training-specific utilities:
- Named viewpoints for debugging/logging
- Random viewpoint sampling with safe-box-area distribution
- Evaluation viewpoint sequences
"""

import random
from dataclasses import dataclass

import torch
from canvit.viewpoint import Viewpoint as CoreViewpoint
from canvit.viewpoint import sample_at_viewpoint

from avp_vit.train.viz import PixelBox, viewpoint_to_pixel_box

__all__ = [
    "Viewpoint",
    "sample_at_viewpoint",
    "random_viewpoint",
    "make_eval_viewpoints",
]


@dataclass
class Viewpoint(CoreViewpoint):
    """Viewpoint with name for debugging/logging."""

    name: str = ""

    def to_pixel_box(self, batch_idx: int, H: int, W: int) -> PixelBox:
        """Convert to pixel coordinates for visualization."""
        return viewpoint_to_pixel_box(self.centers, self.scales, batch_idx, H, W)

    @staticmethod
    def full_scene(*, batch_size: int, device: torch.device) -> "Viewpoint":
        return Viewpoint(
            name="full",
            centers=torch.zeros(batch_size, 2, device=device),
            scales=torch.ones(batch_size, device=device),
        )

    @staticmethod
    def quadrant(B: int, device: torch.device, qx: int, qy: int) -> "Viewpoint":
        """Quadrant viewpoint: qx,qy in {0,1} -> center, scale=0.5."""
        cx = -0.5 + qx
        cy = -0.5 + qy
        name = ["TL", "TR", "BL", "BR"][qy * 2 + qx]
        centers = torch.tensor([[cy, cx]], device=device).expand(B, -1)
        return Viewpoint(
            name=name,
            centers=centers,
            scales=torch.full((B,), 0.5, device=device),
        )


def random_viewpoint(
    B: int,
    device: torch.device,
    min_scale: float = 0.0,
    max_scale: float = 1.0,
) -> Viewpoint:
    """Sample random viewpoints with uniform safe-box-area distribution.

    Geometry: viewpoint has center (x, y) ∈ [-1, 1]² and scale s ∈ [min_scale, max_scale].
    Constraint: |x| + s ≤ 1 and |y| + s ≤ 1 (viewpoint must fit in scene).
    Given scale s, valid centers form a "safe box": [-(1-s), (1-s)]² with area A = 4·(1-s)².

    We sample UNIFORMLY OVER SAFE-BOX AREA because large scales have fewer valid centers.
    """
    assert 0.0 <= min_scale <= max_scale <= 1.0

    L_min = 1 - max_scale
    L_max = 1 - min_scale

    u = torch.rand(B, device=device)
    L_sq = L_min**2 + u * (L_max**2 - L_min**2)
    L = torch.sqrt(L_sq)

    scales = 1 - L
    centers = (torch.rand(B, 2, device=device) * 2 - 1) * L.unsqueeze(1)

    return Viewpoint(name="random", centers=centers, scales=scales)


def make_eval_viewpoints(B: int, device: torch.device) -> list[Viewpoint]:
    """Full scene followed by 4 quadrants in shuffled order."""
    vps = [Viewpoint.full_scene(batch_size=B, device=device)]
    quadrants = [(0, 0), (0, 1), (1, 0), (1, 1)]
    random.shuffle(quadrants)
    for qx, qy in quadrants:
        vps.append(Viewpoint.quadrant(B, device, qx, qy))
    return vps
