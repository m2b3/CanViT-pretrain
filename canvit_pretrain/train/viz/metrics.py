"""Numerical metrics for visualization."""

import numpy as np
from numpy.typing import NDArray


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
