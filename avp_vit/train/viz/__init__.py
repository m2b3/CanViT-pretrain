"""Visualization utilities for training.

IMPORTANT: Functions here do NOT move tensors between devices.
Caller is responsible for .cpu(), .detach() etc. as needed.
"""

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from torch import Tensor

from avp_vit.train.data import IMAGENET_MEAN, IMAGENET_STD


def fit_pca(features: NDArray[np.floating]) -> PCA:
    """Fit PCA on features for RGB visualization.

    Args:
        features: [N, D] numpy array of features (already on CPU)
    """
    pca = PCA(n_components=3, whiten=True)
    pca.fit(features)
    return pca


def pca_rgb(pca: PCA, features: NDArray[np.floating], H: int, W: int) -> NDArray[np.floating]:
    """Project features to RGB via PCA, reshape to [H, W, 3].

    Args:
        features: [H*W, D] numpy array (already on CPU)

    Returns:
        [H, W, 3] numpy array with sigmoid-scaled values in [0, 1]
    """
    proj = pca.transform(features)
    # Sigmoid scaling: 2.0 is a heuristic that works well for visualization
    return 1.0 / (1.0 + np.exp(-proj.reshape(H, W, 3) * 2.0))


def imagenet_denormalize(t: Tensor) -> Tensor:
    """Convert [3, H, W] ImageNet-normalized tensor to [H, W, 3] in [0, 1].

    Args:
        t: [3, H, W] tensor (caller handles device placement)

    Returns:
        [H, W, 3] tensor on same device as input
    """
    mean = t.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = t.new_tensor(IMAGENET_STD).view(3, 1, 1)
    return ((t * std + mean).clamp(0, 1)).permute(1, 2, 0)
