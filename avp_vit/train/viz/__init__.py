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
    locals_avp: list[NDArray[np.floating]],
    locals_teacher: list[NDArray[np.floating]],
    glimpses: list[NDArray[np.floating]],
    boxes: list[PixelBox],
    names: list[str],
    scene_grid_size: int,
    glimpse_grid_size: int,
    initial_scene: NDArray[np.floating],
) -> Figure:
    """Full multi-row visualization with all diagnostic columns.

    Row 0 = "init": learned hidden_tokens projected through output_proj, BEFORE any glimpses
    Row 1+ = "t=0, t=1, ...": scene state AFTER processing each glimpse

    Columns: Trajectory | Glimpse | Teacher | Scene | Local (AVP) | Local (Teacher) | Delta | Error

    - Trajectory: Full image with cumulative viewpoint boxes
    - Glimpse: Actual glimpse image (empty for init row)
    - Teacher: PCA of teacher features (reference)
    - Scene: PCA of scene representation
    - Local (AVP): PCA of glimpse features from AVP's TRAINABLE backbone
    - Local (Teacher): PCA of glimpse features from FROZEN teacher backbone
    - Delta: Spatial MSE change from previous row
    - Error: Spatial MSE vs teacher

    IMPORTANT: locals_avp vs locals_teacher distinction is CRITICAL!
    - locals_avp: normalized by AVP's trainable backbone (shows what AVP "sees")
    - locals_teacher: normalized by frozen teacher (shows ground truth for glimpse region)
    Comparing these reveals how AVP's internal representation diverges from teacher.

    Args:
        full_img: [H, W, 3] full image in [0, 1]
        teacher: [S*S, D] teacher features
        scenes: List of [S*S, D] scene features per timestep (AFTER each glimpse)
        locals_avp: List of [G*G, D] glimpse features from AVP's TRAINABLE backbone
        locals_teacher: List of [G*G, D] glimpse features from FROZEN teacher backbone
        glimpses: List of [H', W', 3] glimpse images per timestep
        boxes: List of PixelBox in pixel coords per timestep
        names: List of viewpoint names per timestep
        scene_grid_size: S
        glimpse_grid_size: G
        initial_scene: [S*S, D] projected initial hidden (BEFORE any glimpses)

    Returns:
        matplotlib Figure
    """
    n_views = len(scenes)
    assert len(locals_avp) == n_views
    assert len(locals_teacher) == n_views
    assert len(glimpses) == n_views
    assert len(boxes) == n_views
    assert len(names) == n_views

    S, G = scene_grid_size, glimpse_grid_size
    n_rows = n_views + 1  # +1 for init row
    colors = timestep_colors(n_views)

    # Fit PCA on teacher
    pca = fit_pca(teacher)
    teacher_rgb = pca_rgb(pca, teacher, S, S)
    initial_rgb = pca_rgb(pca, initial_scene, S, S)

    # Precompute error maps (including initial)
    initial_error = ((initial_scene - teacher) ** 2).mean(axis=-1).reshape(S, S)
    error_maps = [initial_error] + [((s - teacher) ** 2).mean(axis=-1).reshape(S, S) for s in scenes]
    error_vmax = max(e.max() for e in error_maps)

    # Compute delta maps
    delta_maps: list[NDArray[np.floating]] = []
    # Init row has no delta (or we could show zeros)
    delta_maps.append(np.zeros((S, S)))
    prev = initial_scene
    for scene in scenes:
        delta = ((scene - prev) ** 2).mean(axis=-1).reshape(S, S)
        delta_maps.append(delta)
        prev = scene

    n_cols = 8  # Trajectory | Glimpse | Teacher | Scene | Local(AVP) | Local(Teacher) | Delta | Error
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    # Row 0: Initial state (before any glimpses)
    row = 0
    axes[row, 0].imshow(full_img)
    axes[row, 0].set_title("init")
    axes[row, 0].axis("off")

    axes[row, 1].axis("off")  # No glimpse yet
    axes[row, 1].set_title("(no glimpse)")

    axes[row, 2].imshow(teacher_rgb)
    axes[row, 2].set_title("Teacher")
    axes[row, 2].axis("off")

    axes[row, 3].imshow(initial_rgb)
    axes[row, 3].set_title("Scene (init)")
    axes[row, 3].axis("off")

    axes[row, 4].axis("off")  # No local yet
    axes[row, 5].axis("off")  # No local yet

    axes[row, 6].axis("off")  # No delta for init
    axes[row, 6].set_title("(no Δ)")

    im_err = axes[row, 7].imshow(error_maps[0], cmap="hot", vmin=0, vmax=error_vmax)
    mse = float(error_maps[0].mean())
    axes[row, 7].set_title(f"Err ({mse:.4f})")
    axes[row, 7].axis("off")
    fig.colorbar(im_err, ax=axes[row, 7], fraction=0.046, pad=0.04)

    # Rows 1+: After each glimpse (t=0, t=1, ...)
    for t in range(n_views):
        row = t + 1
        scene_rgb = pca_rgb(pca, scenes[t], S, S)
        local_avp_rgb = pca_rgb(pca, locals_avp[t], G, G)
        local_teacher_rgb = pca_rgb(pca, locals_teacher[t], G, G)

        # Col 0: Trajectory - cumulative boxes up to t
        ax = axes[row, 0]
        ax.imshow(full_img)
        for i in range(t + 1):
            box = boxes[i]
            alpha = 1.0 if i == t else 0.4
            rect = Rectangle(
                (box.left, box.top), box.width, box.height,
                linewidth=2, edgecolor=colors[i], facecolor="none", alpha=alpha,
            )
            ax.add_patch(rect)
            ax.plot(box.center_x, box.center_y, "o", color=colors[i], markersize=5, alpha=alpha)
        for i in range(1, t + 1):
            ax.plot(
                [boxes[i - 1].center_x, boxes[i].center_x],
                [boxes[i - 1].center_y, boxes[i].center_y],
                "-", color=colors[i], linewidth=1.5, alpha=0.7,
            )
        ax.set_title(f"t={t}")
        ax.axis("off")

        # Col 1: Glimpse
        axes[row, 1].imshow(glimpses[t])
        axes[row, 1].set_title(f"Glimpse ({names[t]})")
        axes[row, 1].axis("off")

        # Col 2: Teacher PCA
        axes[row, 2].imshow(teacher_rgb)
        axes[row, 2].axis("off")

        # Col 3: Scene PCA
        axes[row, 3].imshow(scene_rgb)
        axes[row, 3].set_title(f"Scene t={t}")
        axes[row, 3].axis("off")

        # Col 4: Local (AVP) - from TRAINABLE backbone
        axes[row, 4].imshow(local_avp_rgb)
        axes[row, 4].set_title(f"Local AVP t={t}" if t == 0 else "")
        axes[row, 4].axis("off")

        # Col 5: Local (Teacher) - from FROZEN teacher
        axes[row, 5].imshow(local_teacher_rgb)
        axes[row, 5].set_title(f"Local Teacher t={t}" if t == 0 else "")
        axes[row, 5].axis("off")

        # Col 6: Delta
        im_delta = axes[row, 6].imshow(delta_maps[row], cmap="viridis")
        axes[row, 6].set_title(f"Δ t={t}")
        axes[row, 6].axis("off")
        fig.colorbar(im_delta, ax=axes[row, 6], fraction=0.046, pad=0.04)

        # Col 7: Error
        im_err = axes[row, 7].imshow(error_maps[row], cmap="hot", vmin=0, vmax=error_vmax)
        mse = float(error_maps[row].mean())
        axes[row, 7].set_title(f"Err ({mse:.4f})")
        axes[row, 7].axis("off")
        fig.colorbar(im_err, ax=axes[row, 7], fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig
