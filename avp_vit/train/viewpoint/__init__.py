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

    Samples area_fraction ~ Beta(alpha, beta) on [min_area, max_area],
    then computes scale = sqrt(area_fraction).

    Relationship: area_fraction = scale², so scale = sqrt(area_fraction).
    """

    alpha: float = 1.0  # Uniform by default
    beta: float = 1.0
    min_area: float = (1 / 128) ** 2  # scale=1/128 → area=1/16384
    max_area: float = 1.0


def sample_scales(B: int, device: torch.device, cfg: ViewpointScaleConfig) -> Tensor:
    """Sample scales from Beta distribution over area fraction."""
    # MPS doesn't support Dirichlet (used by Beta), sample on CPU and transfer
    # CUDA/CPU sample directly on device (no sync)
    if device.type == "mps":
        unit_area = torch.distributions.Beta(cfg.alpha, cfg.beta).sample((B,)).to(device)
    else:
        alpha = torch.tensor(cfg.alpha, device=device)
        beta = torch.tensor(cfg.beta, device=device)
        unit_area = torch.distributions.Beta(alpha, beta).sample((B,))
    # Scale to [min_area, max_area]
    area = cfg.min_area + (cfg.max_area - cfg.min_area) * unit_area
    # area = scale² → scale = sqrt(area)
    return torch.sqrt(area)


def random_viewpoint(B: int, device: torch.device, cfg: ViewpointScaleConfig) -> Viewpoint:
    """Random viewpoint with Beta-distributed area, center constrained to stay in bounds."""
    scales = sample_scales(B, device, cfg)
    max_offset = (1 - scales).unsqueeze(1)
    centers = (torch.rand(B, 2, device=device) * 2 - 1) * max_offset
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
