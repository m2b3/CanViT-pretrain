"""Numerical metrics and statistics for visualization."""

import numpy as np
from numpy.typing import NDArray
from torch import Tensor


def cosine_dissimilarity(a: NDArray, b: NDArray) -> NDArray:
    """Compute 1 - cosine_similarity along last axis.

    Args:
        a: [N, D] array
        b: [N, D] array

    Returns:
        [N] array of dissimilarities in [0, 2]
    """
    dot = (a * b).sum(axis=-1)
    norm_a = np.linalg.norm(a, axis=-1)
    norm_b = np.linalg.norm(b, axis=-1)
    cos_sim = dot / (norm_a * norm_b + 1e-8)
    return 1 - cos_sim


def compute_spatial_stats(x: Tensor) -> dict[str, float]:
    """Compute mean/std across spatial dimension, averaged over batch.

    Args:
        x: [B, N, D] tensor (N = spatial tokens)

    Returns:
        Dict with 'mean' and 'std' scalars:
        - mean: average of per-sample spatial means
        - std: average of per-sample spatial stds
    """
    spatial_mean = x.mean(dim=1)
    spatial_std = x.std(dim=1)
    return {
        "mean": spatial_mean.mean().item(),
        "std": spatial_std.mean().item(),
    }
