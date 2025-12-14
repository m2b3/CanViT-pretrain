"""Visualization utilities for training.

IMPORTANT: Functions here take NUMPY ARRAYS, not tensors.
Caller is responsible for .cpu(), .detach(), .numpy() as needed.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from torch import Tensor

from avp_vit.glimpse import PixelBox
from avp_vit.train.data import IMAGENET_MEAN, IMAGENET_STD

# Type alias for RGB tuple
RGBA = tuple[float, float, float, float]


def timestep_colors(n: int) -> list[RGBA]:
    """Get n distinct colors from viridis colormap."""
    cmap = plt.get_cmap("viridis")
    return [cmap(i / max(1, n - 1)) for i in range(n)]


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


def plot_trajectory(
    img: NDArray[np.floating],
    boxes: list[PixelBox],
    names: list[str],
) -> Figure:
    """Plot image with viewpoint boxes overlaid.

    Args:
        img: [H, W, 3] numpy array in [0, 1]
        boxes: List of PixelBox defining regions
        names: List of names for each box (same length as boxes)

    Returns:
        matplotlib Figure
    """
    assert len(boxes) == len(names)
    colors = timestep_colors(len(boxes))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)

    # Draw connecting lines between consecutive centers
    if len(boxes) > 1:
        xs = [b.center_x for b in boxes]
        ys = [b.center_y for b in boxes]
        ax.plot(xs, ys, "-", color="white", linewidth=1.5, alpha=0.7, zorder=1)

    for i, (box, name) in enumerate(zip(boxes, names, strict=True)):
        rect = Rectangle(
            (box.left, box.top),
            box.width,
            box.height,
            linewidth=2,
            edgecolor=colors[i],
            facecolor="none",
            label=f"t={i} ({name}, s={box.width / img.shape[1]:.2f})",
        )
        ax.add_patch(rect)
        ax.plot(box.center_x, box.center_y, "o", color=colors[i], markersize=6, zorder=2)

    ax.set_title("Viewpoint Trajectory")
    ax.legend(loc="upper right", fontsize=8)
    ax.axis("off")
    plt.tight_layout()
    return fig


def plot_pca_grid(
    pca: PCA,
    reference: NDArray[np.floating],
    samples: list[NDArray[np.floating]],
    grid_size: int,
    titles: list[str],
) -> Figure:
    """Plot PCA visualization comparing reference to samples.

    Args:
        pca: Fitted PCA (fit on reference)
        reference: [G*G, D] numpy array (teacher features)
        samples: List of [G*G, D] numpy arrays (model outputs)
        grid_size: G for reshaping to [G, G, 3]
        titles: Titles for each sample (same length as samples)

    Returns:
        matplotlib Figure
    """
    assert len(samples) == len(titles)
    n_cols = 1 + len(samples)

    reference_rgb = pca_rgb(pca, reference, grid_size, grid_size)

    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    axes[0].imshow(reference_rgb)
    axes[0].set_title("Teacher")
    axes[0].axis("off")

    for i, (sample, title) in enumerate(zip(samples, titles, strict=True)):
        sample_rgb = pca_rgb(pca, sample, grid_size, grid_size)
        mse = float(((sample - reference) ** 2).mean())
        axes[i + 1].imshow(sample_rgb)
        axes[i + 1].set_title(f"{title}\nMSE={mse:.4f}")
        axes[i + 1].axis("off")

    plt.tight_layout()
    return fig


def plot_multistep_pca(
    full_img: NDArray[np.floating],
    teacher: NDArray[np.floating],
    scenes: list[NDArray[np.floating]],
    locals_: list[NDArray[np.floating]],
    glimpses: list[NDArray[np.floating]],
    boxes: list[PixelBox],
    names: list[str],
    scene_grid_size: int,
    glimpse_grid_size: int,
    initial_scene: NDArray[np.floating],
) -> Figure:
    """Full multi-row visualization with all diagnostic columns.

    Columns: Trajectory | Glimpse | Teacher | Scene | Local | Delta | Error

    - Trajectory: Full image with cumulative viewpoint boxes up to time t
    - Glimpse: Actual glimpse image at time t
    - Teacher: PCA of teacher features (reference)
    - Scene: PCA of scene at time t
    - Local: PCA of glimpse backbone features at time t
    - Delta: Spatial MSE of (scene_t - scene_{t-1}), or (scene_0 - initial) for t=0
    - Error: Spatial MSE of (scene_t - teacher)

    Args:
        full_img: [H, W, 3] full image in [0, 1]
        teacher: [S*S, D] teacher features
        scenes: List of [S*S, D] scene features per timestep
        locals_: List of [G*G, D] glimpse backbone features per timestep
        glimpses: List of [H', W', 3] glimpse images per timestep
        boxes: List of PixelBox in pixel coords per timestep
        names: List of viewpoint names per timestep
        scene_grid_size: S
        glimpse_grid_size: G
        initial_scene: [S*S, D] projected initial hidden (for t=0 delta)

    Returns:
        matplotlib Figure
    """
    n_views = len(scenes)
    assert len(locals_) == n_views
    assert len(glimpses) == n_views
    assert len(boxes) == n_views
    assert len(names) == n_views

    S, G = scene_grid_size, glimpse_grid_size
    colors = timestep_colors(n_views)

    # Fit PCA on teacher
    pca = fit_pca(teacher)
    teacher_rgb = pca_rgb(pca, teacher, S, S)

    # Precompute error maps for consistent colorbar
    error_maps = [((s - teacher) ** 2).mean(axis=-1).reshape(S, S) for s in scenes]
    error_vmax = max(e.max() for e in error_maps)

    # Compute delta maps (scene change from previous timestep)
    delta_maps: list[NDArray[np.floating]] = []
    prev = initial_scene
    for scene in scenes:
        delta = ((scene - prev) ** 2).mean(axis=-1).reshape(S, S)
        delta_maps.append(delta)
        prev = scene

    fig, axes = plt.subplots(n_views, 7, figsize=(28, 4 * n_views))
    if n_views == 1:
        axes = axes.reshape(1, -1)

    for row in range(n_views):
        scene_rgb = pca_rgb(pca, scenes[row], S, S)
        local_rgb = pca_rgb(pca, locals_[row], G, G)

        # Col 0: Trajectory - cumulative boxes up to row
        ax = axes[row, 0]
        ax.imshow(full_img)
        for t in range(row + 1):
            box = boxes[t]
            alpha = 1.0 if t == row else 0.4
            rect = Rectangle(
                (box.left, box.top), box.width, box.height,
                linewidth=2, edgecolor=colors[t], facecolor="none", alpha=alpha,
            )
            ax.add_patch(rect)
            ax.plot(box.center_x, box.center_y, "o", color=colors[t], markersize=5, alpha=alpha)
        # Draw path
        for t in range(1, row + 1):
            ax.plot(
                [boxes[t - 1].center_x, boxes[t].center_x],
                [boxes[t - 1].center_y, boxes[t].center_y],
                "-", color=colors[t], linewidth=1.5, alpha=0.7,
            )
        ax.set_title(f"t={row}")
        ax.axis("off")

        # Col 1: Glimpse
        axes[row, 1].imshow(glimpses[row])
        axes[row, 1].set_title(f"Glimpse ({names[row]})")
        axes[row, 1].axis("off")

        # Col 2: Teacher PCA
        axes[row, 2].imshow(teacher_rgb)
        axes[row, 2].set_title("Teacher" if row == 0 else "")
        axes[row, 2].axis("off")

        # Col 3: Scene PCA
        axes[row, 3].imshow(scene_rgb)
        axes[row, 3].set_title(f"Scene t={row}")
        axes[row, 3].axis("off")

        # Col 4: Local PCA
        axes[row, 4].imshow(local_rgb)
        axes[row, 4].set_title(f"Local t={row}")
        axes[row, 4].axis("off")

        # Col 5: Delta
        im_delta = axes[row, 5].imshow(delta_maps[row], cmap="viridis")
        axes[row, 5].set_title(f"Δ t={row}")
        axes[row, 5].axis("off")
        fig.colorbar(im_delta, ax=axes[row, 5], fraction=0.046, pad=0.04)

        # Col 6: Error
        im_err = axes[row, 6].imshow(error_maps[row], cmap="hot", vmin=0, vmax=error_vmax)
        mse = float(error_maps[row].mean())
        axes[row, 6].set_title(f"Err ({mse:.4f})")
        axes[row, 6].axis("off")
        fig.colorbar(im_err, ax=axes[row, 6], fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig
