"""Shared utilities for CanViT evaluation."""

import os
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

import torch
from canvit import Viewpoint
from canvit_utils.policies import coarse_to_fine_viewpoints, random_viewpoints

from typing import Literal

# Narrower type for make_viewpoints — only supports static pre-generated policies.
# For evaluation policies (including fine_to_coarse, entropy_coarse_to_fine),
# use canvit_eval.policies.PolicyName + make_eval_policy instead.
ViewpointPolicyName = Literal["coarse_to_fine", "random", "full_then_random"]


def get_git_commit() -> str | None:
    """Get current git commit hash, or None if not in a git repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def collect_metadata(cfg: Any) -> dict:
    """Collect run metadata for reproducibility.

    Args:
        cfg: Any dataclass config (will be converted via asdict)
    """
    return {
        "config": asdict(cfg),
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": get_git_commit(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": os.environ.get("HOSTNAME") or os.environ.get("SLURMD_NODENAME"),
        "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }


def make_viewpoints(
    policy: ViewpointPolicyName,
    batch_size: int,
    device: torch.device,
    n_viewpoints: int,
    *,
    min_scale: float = 0.05,
    max_scale: float = 1.0,
    start_with_full_scene: bool = True,
) -> list[Viewpoint]:
    """Create viewpoints for given policy.

    Args:
        policy: One of "coarse_to_fine", "random", "full_then_random"
        batch_size: Batch size
        device: Target device
        n_viewpoints: Number of viewpoints to generate
        min_scale: Min scale for random policies
        max_scale: Max scale for random policies
        start_with_full_scene: Whether t=0 is full scene (for random/full_then_random)
    """
    if policy == "coarse_to_fine":
        return coarse_to_fine_viewpoints(batch_size, device, n_viewpoints)
    if policy == "random":
        return random_viewpoints(
            batch_size, device, n_viewpoints,
            min_scale=min_scale, max_scale=max_scale,
            start_with_full_scene=start_with_full_scene,
        )
    if policy == "full_then_random":
        return random_viewpoints(
            batch_size, device, n_viewpoints,
            min_scale=min_scale, max_scale=max_scale,
            start_with_full_scene=True,  # Always start full
        )
    raise ValueError(f"Unknown policy: {policy}")
