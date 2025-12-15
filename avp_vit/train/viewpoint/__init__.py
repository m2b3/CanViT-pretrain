"""Viewpoint sampling for training and evaluation."""

import math
import random

import torch

from avp_vit.glimpse import Viewpoint


def random_viewpoint(
    B: int, device: torch.device, min_scale: float, max_scale: float
) -> Viewpoint:
    """Random viewpoint with log-uniform scale, center constrained to stay in bounds.

    Scale is sampled log-uniformly in [min_scale, max_scale].
    Center is sampled uniformly within valid bounds (so glimpse stays in image).
    """
    log_min, log_max = math.log(min_scale), math.log(max_scale)
    scales = torch.exp(torch.rand(B, device=device) * (log_max - log_min) + log_min)
    max_offset = (1 - scales).unsqueeze(1)
    centers = (torch.rand(B, 2, device=device) * 2 - 1) * max_offset
    return Viewpoint(name="random", centers=centers, scales=scales)


def make_eval_viewpoints(B: int, device: torch.device) -> list[Viewpoint]:
    """Full scene followed by 4 quadrants in shuffled order.

    Starts with global view, then visits all quadrants (shuffled to avoid
    order bias). Final MSE reflects quality after seeing all viewpoints.
    """
    vps = [Viewpoint.full_scene(B, device)]
    quadrants = [(0, 0), (0, 1), (1, 0), (1, 1)]
    random.shuffle(quadrants)
    for qx, qy in quadrants:
        vps.append(Viewpoint.quadrant(B, device, qx, qy))
    return vps


def make_curriculum_eval_viewpoints(
    B: int, G: int, g: int, device: torch.device
) -> list[Viewpoint]:
    """Generate eval viewpoints for curriculum stage via quadrant recursion.

    Args:
        B: Batch size
        G: Scene grid size
        g: Glimpse grid size (unused but kept for API consistency)
        device: Torch device

    Returns:
        List of n_eval viewpoints: full scene + recursive quadrants (shuffled per depth)
    """
    from avp_vit.train.curriculum import n_eval_viewpoints

    _ = g  # API consistency with curriculum module
    n_eval = n_eval_viewpoints(G, g)
    all_vps: list[Viewpoint] = [Viewpoint.full_scene(B, device)]

    # Depth needed: cumulative viewpoints at depth d = 1 + 4 + 16 + ... = (4^(d+1) - 1) / 3
    # Solve: (4^(d+1) - 1) / 3 >= n_eval => d >= log_4(3*n_eval + 1) - 1
    max_depth = max(1, math.ceil(math.log(3 * n_eval + 1, 4)) - 1)

    for depth in range(1, max_depth + 1):
        level_vps = _quadrants_at_depth(B, depth, device)
        random.shuffle(level_vps)
        all_vps.extend(level_vps)

    return all_vps[:n_eval]


def _quadrants_at_depth(B: int, depth: int, device: torch.device) -> list[Viewpoint]:
    """Generate all 4^depth quadrants at given depth.

    depth=1: 4 quadrants (scale=0.5)
    depth=2: 16 sub-quadrants (scale=0.25)
    """
    scale = 0.5**depth
    n = 2**depth  # grid divisions per side
    vps: list[Viewpoint] = []

    for i in range(n):
        for j in range(n):
            # Center of cell (i,j) in [-1,1] coords
            cx = -1 + scale + 2 * scale * j
            cy = -1 + scale + 2 * scale * i
            centers = torch.full((B, 2), 0.0, device=device)
            centers[:, 0] = cx
            centers[:, 1] = cy
            scales = torch.full((B,), scale, device=device)
            vps.append(Viewpoint(f"depth{depth}_{i}_{j}", centers, scales))

    return vps
