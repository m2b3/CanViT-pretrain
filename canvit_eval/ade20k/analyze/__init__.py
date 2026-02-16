"""Post-hoc analysis of ADE20K evaluation results.

Aggregates global mIoU from multiple runs for variance estimation.

IMPORTANT: mIoU computation methods differ significantly:
- GLOBAL (standard): sum intersection/union across ALL images, then divide.
  This weights images by pixel count. Used by DINOv3, benchmarks, papers.
- PER-IMAGE (non-standard): compute mIoU per image, then average.
  Treats each image equally. ~5% lower than global on ADE20K.

This module uses GLOBAL mIoU from data['mious'], NOT per-image averages.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from canvit_eval.ade20k.train_probe.config import FeatureType


@dataclass
class PolicyStats:
    """Aggregated statistics for one policy across multiple runs."""

    policy: str
    n_runs: int
    n_timesteps: int
    # Per-timestep global mIoU: mean and std across runs
    mious: np.ndarray  # [T] mean
    std: np.ndarray  # [T] std across runs
    # Raw per-run data
    per_run: np.ndarray  # [n_runs, T]


def load_run(path: Path, feature: FeatureType) -> dict:
    """Load mIoU curve from a single run."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    mious_dict = data["mious"][feature]
    T = len([k for k in mious_dict if k.startswith("t")])
    mious = np.array([mious_dict[f"t{t}"] for t in range(T)])
    return {"mious": mious, "metadata": data["metadata"]}


def compute_stats(paths: list[Path], feature: FeatureType = "predicted_norm") -> PolicyStats:
    """Compute aggregated statistics across multiple runs.

    Args:
        paths: list of .pt files from SLURM jobs (same policy, different seeds)
        feature: which feature type to analyze

    Returns:
        PolicyStats with mean/std mIoU curves across runs
    """
    runs = [load_run(p, feature) for p in paths]
    per_run = np.stack([r["mious"] for r in runs])  # [n_runs, T]

    # Extract policy name from first run's metadata
    cfg = runs[0]["metadata"].get("config", {})
    policy = cfg.get("policy", "unknown")
    if policy == "random" and cfg.get("start_full"):
        policy = "fullrand"
    elif policy == "random":
        policy = "iid"

    return PolicyStats(
        policy=policy,
        n_runs=len(runs),
        n_timesteps=per_run.shape[1],
        mious=per_run.mean(axis=0),
        std=per_run.std(axis=0),
        per_run=per_run,
    )


def find_result_files(output_dir: Path, policy_prefix: str) -> list[Path]:
    """Find all .pt result files for a given policy prefix."""
    return sorted(output_dir.glob(f"ade20k_{policy_prefix}_*.pt"))
