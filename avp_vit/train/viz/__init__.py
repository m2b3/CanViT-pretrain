"""Visualization utilities for training.

IMPORTANT: Functions here take NUMPY ARRAYS, not tensors.
Caller is responsible for .cpu(), .detach(), .numpy() as needed.
"""

from typing import NamedTuple

import matplotlib
# Use non-interactive backend BEFORE importing pyplot to prevent GUI resource leaks
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from torch import Tensor

from avp_vit.train.data import IMAGENET_MEAN, IMAGENET_STD
from avp_vit.train.probe import TopKPrediction


class PixelBox(NamedTuple):
    """Axis-aligned bounding box in pixel coordinates."""

    left: float
    top: float
    width: float
    height: float
    center_x: float
    center_y: float


def viewpoint_to_pixel_box(
    centers: Tensor, scales: Tensor, batch_idx: int, H: int, W: int
) -> PixelBox:
    """Convert viewpoint geometry to pixel coordinates for visualization."""
    cy, cx = centers[batch_idx].tolist()
    scale = scales[batch_idx].item()
    center_x = (cx + 1) / 2 * W
    center_y = (cy + 1) / 2 * H
    width = scale * W
    height = scale * H
    return PixelBox(
        left=center_x - width / 2,
        top=center_y - height / 2,
        width=width,
        height=height,
        center_x=center_x,
        center_y=center_y,
    )

# Type alias for RGB tuple
RGBA = tuple[float, float, float, float]


def _cosine_dissimilarity(a: NDArray, b: NDArray) -> NDArray:
    """Compute 1 - cosine_similarity along last axis. Returns [N] for [N, D] inputs."""
    dot = (a * b).sum(axis=-1)
    norm_a = np.linalg.norm(a, axis=-1)
    norm_b = np.linalg.norm(b, axis=-1)
    cos_sim = dot / (norm_a * norm_b + 1e-8)
    return 1 - cos_sim


def timestep_colors(n: int) -> list[RGBA]:
    """Get n distinct colors from viridis colormap."""
    cmap = plt.get_cmap("viridis")
    return [cmap(i / max(1, n - 1)) for i in range(n)]


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
    # Check variance along each spatial position - if too uniform, skip
    # Early training has ~1e-8 variance which still breaks PCA
    if features.var(axis=0).max() < 1e-5:
        return None
    # Clamp to available dimensions
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
            Use for data with different variance than PCA was fit on.
        pc_offset: Which PC to start from (0=PC1-3, 1=PC2-4, etc.)

    Returns:
        [H, W, 3] numpy array with sigmoid-scaled values in [0, 1]
    """
    if pca is None:
        return np.full((H, W, 3), 0.5, dtype=np.float32)
    proj = pca.transform(features)
    # Select 3 components starting at offset
    end = min(pc_offset + 3, proj.shape[1])
    start = max(0, end - 3)  # Ensure we always get 3 if possible
    proj = proj[:, start:end]
    # Pad if fewer than 3 components available
    if proj.shape[1] < 3:
        pad = np.zeros((proj.shape[0], 3 - proj.shape[1]), dtype=proj.dtype)
        proj = np.concatenate([proj, pad], axis=1)
    if normalize:
        proj = proj / (proj.std() + 1e-8)
    return _pca_proj_to_rgb(proj, H, W)


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


def plot_norm_stats(
    mean: NDArray[np.floating],
    std: NDArray[np.floating],
    grid_size: int,
) -> Figure:
    """Plot normalizer running stats (mean and std amplitude heatmaps).

    Args:
        mean: [G*G, D] running mean per position
        std: [G*G, D] running std per position
        grid_size: G (spatial grid size)

    Returns:
        matplotlib Figure with two heatmaps (mean L2 norm, std L2 norm)
    """
    G = grid_size
    mean_2d = mean.reshape(G, G, -1)
    std_2d = std.reshape(G, G, -1)

    mean_amp = np.linalg.norm(mean_2d, axis=-1)
    std_amp = np.linalg.norm(std_2d, axis=-1)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    im0 = axes[0].imshow(mean_amp, cmap="viridis")
    axes[0].set_title(f"Mean L2\n[{mean_amp.min():.2f}, {mean_amp.max():.2f}]")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(std_amp, cmap="viridis")
    axes[1].set_title(f"Std L2\n[{std_amp.min():.2f}, {std_amp.max():.2f}]")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


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
    if boxes:
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


class TimestepPredictions(NamedTuple):
    """Top-k predictions at a single timestep for a single sample."""
    predictions: list[TopKPrediction]  # Top-k predictions (e.g., k=5)
    gt_idx: int  # Ground truth class index
    gt_name: str  # Ground truth class name


def plot_multistep_pca(
    full_img: NDArray[np.floating],
    teacher: NDArray[np.floating],
    scenes: list[NDArray[np.floating]],
    locals_avp: list[NDArray[np.floating]] | None,
    locals_teacher: list[NDArray[np.floating]] | None,
    glimpses: list[NDArray[np.floating]],
    boxes: list[PixelBox],
    names: list[str],
    scene_grid_size: int,
    glimpse_grid_size: int,
    initial_scene: NDArray[np.floating],
    hidden_spatials: list[NDArray[np.floating]] | None = None,
    initial_hidden_spatial: NDArray[np.floating] | None = None,
    locals_teacher_cropped: list[NDArray[np.floating]] | None = None,
    show_locals: bool = False,
    timestep_predictions: list[TimestepPredictions] | None = None,
) -> Figure:
    """Full multi-row visualization with all diagnostic columns.

    Row 0 = "init": learned spatial_hidden_init projected through scene_proj, BEFORE any glimpses
    Row 1+ = "t=0, t=1, ...": scene state AFTER processing each glimpse

    Columns: [Preds] | Trajectory | Glimpse | Teacher | Scene | [Hidden] | ... | Δ Scene | Error

    Args:
        full_img: [H, W, 3] full image in [0, 1]
        teacher: [S*S, D] teacher features
        scenes: List of [S*S, D] scene features per timestep (AFTER each glimpse)
        locals_avp: List of [G*G, D] glimpse features from AVP's TRAINABLE backbone
        locals_teacher: List of [G*G, D] glimpse features from FROZEN teacher backbone (local context only)
        glimpses: List of [H', W', 3] glimpse images per timestep
        boxes: List of PixelBox in pixel coords per timestep
        names: List of viewpoint names per timestep
        scene_grid_size: S
        glimpse_grid_size: G
        initial_scene: [S*S, D] projected initial hidden (BEFORE any glimpses)
        hidden_spatials: Optional list of [S*S, D] raw hidden spatial per timestep (before scene_proj)
        initial_hidden_spatial: Optional [S*S, D] raw initial hidden spatial
        locals_teacher_cropped: Optional list of [G*G, D] teacher features cropped from full image (full context)
        timestep_predictions: Optional predictions per timestep (for barplot column). Length = n_views.

    Returns:
        matplotlib Figure
    """
    n_views = len(scenes)
    assert len(glimpses) == n_views
    assert len(boxes) == n_views
    assert len(names) == n_views
    if show_locals:
        assert locals_avp is not None and len(locals_avp) == n_views
        assert locals_teacher is not None and len(locals_teacher) == n_views

    show_hidden = hidden_spatials is not None
    if show_hidden:
        assert len(hidden_spatials) == n_views
        assert initial_hidden_spatial is not None

    show_cropped = show_locals and locals_teacher_cropped is not None
    if show_cropped:
        assert locals_teacher_cropped is not None
        assert len(locals_teacher_cropped) == n_views

    S, G = scene_grid_size, glimpse_grid_size
    n_rows = n_views + 1  # +1 for init row
    colors = timestep_colors(n_views)

    # Fit PCA on teacher (used for scene comparisons)
    pca_teacher = fit_pca(teacher)
    teacher_rgb = pca_rgb(pca_teacher, teacher, S, S)
    initial_rgb = pca_rgb(pca_teacher, initial_scene, S, S, normalize=True)

    # Scene with own PCA (initial)
    pca_init_scene = fit_pca(initial_scene)
    initial_scene_own_rgb = pca_rgb(pca_init_scene, initial_scene, S, S)

    # Hidden PCA for init (own basis)
    if show_hidden:
        assert initial_hidden_spatial is not None
        pca_init_hidden = fit_pca(initial_hidden_spatial)
        initial_hidden_rgb = pca_rgb(pca_init_hidden, initial_hidden_spatial, S, S)
    else:
        initial_hidden_rgb = None

    # Precompute error maps (cosine dissimilarity, per-row scaling)
    initial_error = _cosine_dissimilarity(initial_scene, teacher).reshape(S, S)
    error_maps = [initial_error] + [_cosine_dissimilarity(s, teacher).reshape(S, S) for s in scenes]

    # Compute delta maps for scene predictions (cosine dissimilarity from prev)
    delta_scene_maps: list[NDArray[np.floating]] = [np.zeros((S, S))]
    prev_scene = initial_scene
    for scene in scenes:
        delta_scene_maps.append(_cosine_dissimilarity(scene, prev_scene).reshape(S, S))
        prev_scene = scene

    # Compute delta maps for hidden states (if available)
    delta_hidden_maps: list[NDArray[np.floating]] | None = None
    if show_hidden:
        assert hidden_spatials is not None and initial_hidden_spatial is not None
        delta_hidden_maps = [np.zeros((S, S))]
        prev_hidden = initial_hidden_spatial
        for h in hidden_spatials:
            delta_hidden_maps.append(_cosine_dissimilarity(h, prev_hidden).reshape(S, S))
            prev_hidden = h

    # Column indices
    # Trajectory | Glimpse | Teacher | Scene | Scene (own) | [Hidden] | [Local AVP] | [Local Teacher] | [Cropped Teacher] | [Δ Hidden] | Δ Scene | Error
    C_TRAJ, C_GLIMPSE, C_TEACHER, C_SCENE, C_SCENE_OWN = 0, 1, 2, 3, 4
    c = 5
    C_HIDDEN = c if show_hidden else None
    if show_hidden:
        c += 1
    C_LOCAL_AVP = c if show_locals else None
    C_LOCAL_TEACHER = (c + 1) if show_locals else None
    if show_locals:
        c += 2
    C_CROPPED_TEACHER = c if show_cropped else None
    if show_cropped:
        c += 1
    C_DELTA_HIDDEN = c if show_hidden else None
    if show_hidden:
        c += 1
    C_DELTA_SCENE, C_ERROR = c, c + 1
    n_cols = c + 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    # Row 0: Initial state (before any glimpses)
    row = 0
    axes[row, C_TRAJ].imshow(full_img)
    axes[row, C_TRAJ].set_title("init")
    axes[row, C_TRAJ].axis("off")

    axes[row, C_GLIMPSE].axis("off")
    axes[row, C_GLIMPSE].set_title("(no glimpse)")

    axes[row, C_TEACHER].imshow(teacher_rgb)
    axes[row, C_TEACHER].set_title("Teacher")
    axes[row, C_TEACHER].axis("off")

    axes[row, C_SCENE].imshow(initial_rgb)
    axes[row, C_SCENE].set_title("Scene (init)")
    axes[row, C_SCENE].axis("off")

    axes[row, C_SCENE_OWN].imshow(initial_scene_own_rgb)
    axes[row, C_SCENE_OWN].set_title("Scene own (init)")
    axes[row, C_SCENE_OWN].axis("off")

    if show_hidden:
        assert C_HIDDEN is not None and initial_hidden_rgb is not None
        axes[row, C_HIDDEN].imshow(initial_hidden_rgb)
        axes[row, C_HIDDEN].set_title("Hidden (init)")
        axes[row, C_HIDDEN].axis("off")

    if show_locals:
        assert C_LOCAL_AVP is not None and C_LOCAL_TEACHER is not None
        axes[row, C_LOCAL_AVP].axis("off")
        axes[row, C_LOCAL_TEACHER].axis("off")

    if show_cropped:
        assert C_CROPPED_TEACHER is not None
        axes[row, C_CROPPED_TEACHER].axis("off")

    if show_hidden:
        assert C_DELTA_HIDDEN is not None
        axes[row, C_DELTA_HIDDEN].axis("off")
        axes[row, C_DELTA_HIDDEN].set_title("Δ Hidden")

    axes[row, C_DELTA_SCENE].axis("off")
    axes[row, C_DELTA_SCENE].set_title("Δ Scene")

    im_err = axes[row, C_ERROR].imshow(error_maps[0], cmap="hot")
    cos_dis = float(error_maps[0].mean())
    axes[row, C_ERROR].set_title(f"CosDis ({cos_dis:.4f})")
    axes[row, C_ERROR].axis("off")
    fig.colorbar(im_err, ax=axes[row, C_ERROR], fraction=0.046, pad=0.04)

    # Rows 1+: After each glimpse (t=0, t=1, ...)
    for t in range(n_views):
        row = t + 1
        scene_rgb = pca_rgb(pca_teacher, scenes[t], S, S, normalize=True)

        # Scene with own PCA
        pca_scene_t = fit_pca(scenes[t])
        scene_own_rgb = pca_rgb(pca_scene_t, scenes[t], S, S)

        # Hidden: own PCA per timestep
        if show_hidden:
            assert hidden_spatials is not None
            pca_hidden_t = fit_pca(hidden_spatials[t])
            hidden_rgb = pca_rgb(pca_hidden_t, hidden_spatials[t], S, S)
        else:
            hidden_rgb = None

        # Local columns (only computed if show_locals)
        local_avp_rgb = None
        local_teacher_rgb = None
        pca_cropped: PCA | None = None
        if show_locals:
            assert locals_avp is not None and locals_teacher is not None
            pca_local_teacher = fit_pca(locals_teacher[t])
            local_teacher_rgb = pca_rgb(pca_local_teacher, locals_teacher[t], G, G)
            pca_local_avp = fit_pca(locals_avp[t])
            local_avp_rgb = pca_rgb(pca_local_avp, locals_avp[t], G, G)
            if show_cropped:
                assert locals_teacher_cropped is not None
                pca_cropped = fit_pca(locals_teacher_cropped[t])

        # Col: Trajectory - cumulative boxes up to t
        ax = axes[row, C_TRAJ]
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

        # Col: Glimpse
        axes[row, C_GLIMPSE].imshow(glimpses[t])
        axes[row, C_GLIMPSE].set_title(f"Glimpse ({names[t]})")
        axes[row, C_GLIMPSE].axis("off")

        # Col: Teacher PCA
        axes[row, C_TEACHER].imshow(teacher_rgb)
        axes[row, C_TEACHER].axis("off")

        # Col: Scene (teacher basis)
        axes[row, C_SCENE].imshow(scene_rgb)
        axes[row, C_SCENE].set_title(f"Scene t={t}")
        axes[row, C_SCENE].axis("off")

        # Col: Scene (own PCA)
        axes[row, C_SCENE_OWN].imshow(scene_own_rgb)
        axes[row, C_SCENE_OWN].set_title("Scene own" if t == 0 else "")
        axes[row, C_SCENE_OWN].axis("off")

        # Col: Hidden (own PCA per timestep)
        if show_hidden:
            assert C_HIDDEN is not None and hidden_rgb is not None
            axes[row, C_HIDDEN].imshow(hidden_rgb)
            axes[row, C_HIDDEN].set_title("Hidden" if t == 0 else "")
            axes[row, C_HIDDEN].axis("off")

        # Col: Local AVP
        if show_locals:
            assert C_LOCAL_AVP is not None and C_LOCAL_TEACHER is not None
            assert local_avp_rgb is not None and local_teacher_rgb is not None
            axes[row, C_LOCAL_AVP].imshow(local_avp_rgb)
            if t == 0:
                axes[row, C_LOCAL_AVP].set_title("Local AVP")
            axes[row, C_LOCAL_AVP].axis("off")

            # Col: Local Teacher (own PCA - internal structure)
            axes[row, C_LOCAL_TEACHER].imshow(local_teacher_rgb)
            axes[row, C_LOCAL_TEACHER].set_title("Local Teacher" if t == 0 else "")
            axes[row, C_LOCAL_TEACHER].axis("off")

        # Col: Cropped Teacher
        if show_cropped:
            assert C_CROPPED_TEACHER is not None and locals_teacher_cropped is not None and pca_cropped is not None
            cropped_rgb = pca_rgb(pca_cropped, locals_teacher_cropped[t], G, G)
            axes[row, C_CROPPED_TEACHER].imshow(cropped_rgb)
            if t == 0:
                axes[row, C_CROPPED_TEACHER].set_title("Cropped Teacher")
            axes[row, C_CROPPED_TEACHER].axis("off")

        # Col: Δ Hidden (if available)
        if show_hidden:
            assert C_DELTA_HIDDEN is not None and delta_hidden_maps is not None
            im_dh = axes[row, C_DELTA_HIDDEN].imshow(delta_hidden_maps[row], cmap="viridis")
            axes[row, C_DELTA_HIDDEN].set_title(f"Δh t={t}" if t == 0 else f"t={t}")
            axes[row, C_DELTA_HIDDEN].axis("off")
            fig.colorbar(im_dh, ax=axes[row, C_DELTA_HIDDEN], fraction=0.046, pad=0.04)

        # Col: Δ Scene
        im_ds = axes[row, C_DELTA_SCENE].imshow(delta_scene_maps[row], cmap="viridis")
        axes[row, C_DELTA_SCENE].set_title(f"Δs t={t}" if t == 0 else f"t={t}")
        axes[row, C_DELTA_SCENE].axis("off")
        fig.colorbar(im_ds, ax=axes[row, C_DELTA_SCENE], fraction=0.046, pad=0.04)

        # Col: Error (cosine dissimilarity, per-row scaling)
        im_err = axes[row, C_ERROR].imshow(error_maps[row], cmap="hot")
        cos_dis = float(error_maps[row].mean())
        axes[row, C_ERROR].set_title(f"CosDis ({cos_dis:.4f})")
        axes[row, C_ERROR].axis("off")
        fig.colorbar(im_err, ax=axes[row, C_ERROR], fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig
