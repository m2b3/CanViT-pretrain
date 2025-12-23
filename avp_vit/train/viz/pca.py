"""PCA utilities for feature visualization."""

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA


def _pca_proj_to_rgb(proj: NDArray[np.floating], H: int, W: int) -> NDArray[np.floating]:
    """Convert PCA projection to RGB image via sigmoid. Clips to avoid overflow."""
    x = proj.reshape(H, W, 3) * 2.0
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def fit_pca(features: NDArray[np.floating], n_components: int = 12) -> PCA | None:
    """Fit PCA on features for RGB visualization.

    Args:
        features: [N, D] numpy array of features (already on CPU)
        n_components: Number of components to fit (default 12 to support offset viewing)

    Returns:
        Fitted PCA, or None if features have zero variance (e.g., constant init).
    """
    if features.var(axis=0).max() < 1e-5:
        return None
    n_components = min(n_components, features.shape[0], features.shape[1])
    pca = PCA(n_components=n_components, whiten=True)
    pca.fit(features)
    return pca


def pca_rgb(
    pca: PCA | None,
    features: NDArray[np.floating],
    H: int,
    W: int,
    normalize: bool = False,
    pc_offset: int = 0,
) -> NDArray[np.floating]:
    """Project features to RGB via PCA, reshape to [H, W, 3].

    Args:
        pca: Fitted PCA, or None (returns gray image).
        features: [H*W, D] numpy array (already on CPU)
        normalize: If True, normalize projection to std=1 before sigmoid.
        pc_offset: Which PC to start from (0=PC1-3, 1=PC2-4, etc.)

    Returns:
        [H, W, 3] numpy array with sigmoid-scaled values in [0, 1]
    """
    if pca is None:
        return np.full((H, W, 3), 0.5, dtype=np.float32)
    proj = pca.transform(features)
    end = min(pc_offset + 3, proj.shape[1])
    start = max(0, end - 3)
    proj = proj[:, start:end]
    if proj.shape[1] < 3:
        pad = np.zeros((proj.shape[0], 3 - proj.shape[1]), dtype=proj.dtype)
        proj = np.concatenate([proj, pad], axis=1)
    if normalize:
        proj = proj / (proj.std() + 1e-8)
    return _pca_proj_to_rgb(proj, H, W)
