"""Viewpoint sampling for training and evaluation."""

import random
from dataclasses import dataclass
from typing import final

import torch
from torch import Tensor

from avp_vit.glimpse import Viewpoint


@final
@dataclass
class ViewpointScaleConfig:
    """Viewpoint scale distribution: Beta over area fraction.

    For each sample:
    1. Center sampled uniformly in [-1, 1]²
    2. max_scale = min(1 - |y|, 1 - |x|) (viewpoint must fit)
    3. area ~ Beta(alpha, beta) on [min_area, max_scale²]
    4. scale = sqrt(area)
    """

    alpha: float = 1.0  # Uniform by default
    beta: float = 1.0
    min_area: float = (1 / 128) ** 2  # scale=1/128 → area=1/16384


def _sample_unit_beta(B: int, device: torch.device, alpha: float, beta: float) -> Tensor:
    """Sample from Beta(alpha, beta) on [0, 1], directly on device (no sync)."""
    if device.type == "mps":
        # MPS doesn't support Dirichlet (used by Beta internally)
        return torch.distributions.Beta(alpha, beta).sample((B,)).to(device)
    else:
        a = torch.tensor(alpha, device=device)
        b = torch.tensor(beta, device=device)
        return torch.distributions.Beta(a, b).sample((B,))


def random_viewpoint(B: int, device: torch.device, cfg: ViewpointScaleConfig) -> Viewpoint:
    """Random viewpoint: uniform center (constrained), then scale from Beta over area.

    1. min_scale = sqrt(min_area)
    2. Sample centers uniformly in [-(1 - min_scale), (1 - min_scale)]²
    3. max_scale = min(1 - |y|, 1 - |x|) ≥ min_scale (guaranteed by center constraint)
    4. Sample area from Beta on [min_area, max_scale²], scale = sqrt(area)
    """
    min_scale = cfg.min_area ** 0.5
    max_center = 1 - min_scale

    # 1. Uniform centers in valid range
    centers = (torch.rand(B, 2, device=device) * 2 - 1) * max_center  # (y, x)

    # 2. Max scale per sample (≥ min_scale by construction)
    max_scale = (1 - centers.abs()).min(dim=1).values  # [B]

    # 3. Sample area from Beta on [min_area, max_scale²]
    unit = _sample_unit_beta(B, device, cfg.alpha, cfg.beta)
    min_area = cfg.min_area
    max_area = max_scale.square()
    area = min_area + (max_area - min_area) * unit
    scales = torch.sqrt(area)

    return Viewpoint(name="random", centers=centers, scales=scales)


def make_eval_viewpoints(B: int, device: torch.device) -> list[Viewpoint]:
    """Full scene followed by 4 quadrants in shuffled order.

    Intentionally non-deterministic: quadrant order varies each call to test
    that the model generalizes across orderings, not just a fixed sequence.
    Uses Python random (not torch) since reproducibility is NOT desired here.
    """
    vps = [Viewpoint.full_scene(B, device)]
    quadrants = [(0, 0), (0, 1), (1, 0), (1, 1)]
    random.shuffle(quadrants)
    for qx, qy in quadrants:
        vps.append(Viewpoint.quadrant(B, device, qx, qy))
    return vps
