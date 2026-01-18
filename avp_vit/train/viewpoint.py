"""Viewpoint sampling for training and evaluation.

Extends canvit.viewpoint with training-specific utilities:
- Named viewpoints for debugging/logging
- Random viewpoint sampling with safe-box-area distribution
- Evaluation viewpoint sequences
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import NamedTuple

import torch
from canvit.viewpoint import Viewpoint as CoreViewpoint
from torch import Tensor

__all__ = [
    "PixelBox",
    "Viewpoint",
    "ViewpointType",
    "random_viewpoint",
    "make_eval_viewpoints",
    "viewpoint_to_pixel_box",
]


class ViewpointType(Enum):
    """Type of viewpoint for training branches."""

    RANDOM = auto()
    FULL = auto()
    POLICY = auto()


class PixelBox(NamedTuple):
    """Axis-aligned bounding box in pixel coordinates."""

    left: float
    top: float
    width: float
    height: float
    center_x: float
    center_y: float


def viewpoint_to_pixel_box(
    centers: Tensor, scales: Tensor, batch_idx: int, H: int, W: int
) -> PixelBox:
    """Convert viewpoint geometry to pixel coordinates for visualization.

    Maps normalized [-1, 1] to pixel centers [0, W-1] and [0, H-1].
    """
    cy, cx = centers[batch_idx].tolist()
    scale = scales[batch_idx].item()
    # Map normalized [-1, 1] to pixel [0, W-1] (pixel center convention)
    center_x = (cx + 1) / 2 * (W - 1)
    center_y = (cy + 1) / 2 * (H - 1)
    width = scale * (W - 1)
    height = scale * (H - 1)
    return PixelBox(
        left=center_x - width / 2,
        top=center_y - height / 2,
        width=width,
        height=height,
        center_x=center_x,
        center_y=center_y,
    )


@dataclass
class Viewpoint(CoreViewpoint):
    """Viewpoint with name for debugging/logging."""

    name: str = ""

    def to_pixel_box(self, batch_idx: int, H: int, W: int) -> "PixelBox":
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

    @staticmethod
    def random(
        *, batch_size: int, device: torch.device, min_scale: float, max_scale: float = 1.0
    ) -> "Viewpoint":
        """Sample random viewpoints with uniform safe-box-area distribution.

        Geometry: viewpoint has center (x, y) ∈ [-1, 1]² and scale s ∈ [min_scale, max_scale].
        Constraint: |x| + s ≤ 1 and |y| + s ≤ 1 (viewpoint must fit in scene).
        Given scale s, valid centers form a "safe box": [-(1-s), (1-s)]² with area A = 4·(1-s)².

        We sample UNIFORMLY OVER SAFE-BOX AREA because large scales have fewer valid centers.
        """
        assert 0.0 <= min_scale <= max_scale <= 1.0

        L_min = 1 - max_scale
        L_max = 1 - min_scale

        u = torch.rand(batch_size, device=device)
        L_sq = L_min**2 + u * (L_max**2 - L_min**2)
        L = torch.sqrt(L_sq)

        scales = 1 - L
        centers = (torch.rand(batch_size, 2, device=device) * 2 - 1) * L.unsqueeze(1)

        return Viewpoint(name="random", centers=centers, scales=scales)


def random_viewpoint(
    B: int,
    device: torch.device,
    min_scale: float = 0.0,
    max_scale: float = 1.0,
) -> Viewpoint:
    """Sample random viewpoints. Thin wrapper for backward compat."""
    return Viewpoint.random(
        batch_size=B, device=device, min_scale=min_scale, max_scale=max_scale
    )


def make_eval_viewpoints(
    B: int, device: torch.device, n_viewpoints: int = 10
) -> list[Viewpoint]:
    """Generate quadtree viewpoints with random ordering WITHIN each level, per batch item.

    Quadtree structure:
    - Level 0: Full scene (1 viewpoint, scale=1) - always first
    - Level 1: 4 quadrants (scale=0.5) - shuffled per batch item
    - Level 2: 16 sub-quadrants (scale=0.25) - shuffled per batch item
    - Level L: 4^L viewpoints (scale=0.5^L)

    Each batch item gets the same level ordering but different shuffle within levels.
    """
    assert n_viewpoints >= 1

    # Build levels of quadtree nodes
    levels: list[list[tuple[float, float, float]]] = [[(0.0, 0.0, 1.0)]]
    while sum(len(lvl) for lvl in levels) < n_viewpoints:
        parent_level = levels[-1]
        child_level: list[tuple[float, float, float]] = []
        for parent_cy, parent_cx, parent_scale in parent_level:
            child_scale = parent_scale / 2
            for qy, qx in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                child_cy = parent_cy + (qy - 0.5) * parent_scale
                child_cx = parent_cx + (qx - 0.5) * parent_scale
                child_level.append((child_cy, child_cx, child_scale))
        levels.append(child_level)

    # Build viewpoints with per-batch-item shuffling within each level
    result: list[Viewpoint] = []
    for level_idx, level in enumerate(levels):
        level_tensor = torch.tensor(level, device=device, dtype=torch.float32)  # (L, 3)
        L = len(level)

        # Random permutation per batch item for this level
        if L == 1:
            # Only one node in level (e.g., full scene) - no shuffle needed
            perms = torch.zeros(B, 1, dtype=torch.long, device=device)
        else:
            perms = torch.stack([torch.randperm(L, device=device) for _ in range(B)])  # (B, L)

        for i in range(L):
            if len(result) >= n_viewpoints:
                break
            indices = perms[:, i]  # (B,) - which node in this level each batch item sees
            centers = level_tensor[indices, :2]  # (B, 2)
            scales = level_tensor[indices, 2]  # (B,)
            if level_idx == 0:
                name = "full"
            else:
                name = f"L{level_idx}_{i}"
            result.append(Viewpoint(name=name, centers=centers, scales=scales))

        if len(result) >= n_viewpoints:
            break

    return result[:n_viewpoints]
