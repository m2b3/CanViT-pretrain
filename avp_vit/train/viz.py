"""Visualization utilities for training.

This module combines:
- Matplotlib primitives (PCA visualization, plotting)
- Comet ML logging utilities for training metrics
"""

import gc
import io
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

import comet_ml
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from canvit.backbone.dinov3 import DINOv3Backbone, NormFeatures
from canvit.model.active.base import GlimpseOutput
from canvit.viewpoint import Viewpoint as CanvitViewpoint
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from torch import Tensor

from dinov3_probes import DINOv3LinearClassificationHead

from avp_vit import ActiveCanViT
from .data import IMAGENET_MEAN, IMAGENET_STD
from .norm import PositionAwareNorm
from .probe import TopKPrediction, compute_in1k_top1, get_imagenet_class_names, get_probe_resolution, get_top_k_predictions, labels_are_in1k
from .viewpoint import PixelBox, Viewpoint, make_eval_viewpoints

log = logging.getLogger(__name__)

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

    show_preds = timestep_predictions is not None
    if show_preds:
        assert len(timestep_predictions) == n_views

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
    # [Preds] | Trajectory | Glimpse | Teacher | Scene | Scene (own) | [Hidden] | [Local AVP] | [Local Teacher] | [Cropped Teacher] | [Δ Hidden] | Δ Scene | Error
    c = 0
    C_PREDS = c if show_preds else None
    if show_preds:
        c += 1
    C_TRAJ, C_GLIMPSE, C_TEACHER, C_SCENE, C_SCENE_OWN = c, c + 1, c + 2, c + 3, c + 4
    c += 5
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
    if show_preds:
        assert C_PREDS is not None
        axes[row, C_PREDS].axis("off")
        axes[row, C_PREDS].set_title("Predictions")
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

        # Col: Predictions barplot
        if show_preds:
            assert C_PREDS is not None and timestep_predictions is not None
            preds_t = timestep_predictions[t]
            ax = axes[row, C_PREDS]
            k = len(preds_t.predictions)
            # Horizontal barplot: class names on y-axis, probabilities on x-axis
            y_pos = np.arange(k)
            probs = [p.probability for p in preds_t.predictions]
            names_short = [p.class_name[:20] for p in preds_t.predictions]  # Truncate long names
            bar_colors = ["green" if p.class_idx == preds_t.gt_idx else "steelblue" for p in preds_t.predictions]
            ax.barh(y_pos, probs, color=bar_colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names_short, fontsize=8)
            ax.set_xlim(0, 1)
            ax.invert_yaxis()  # Top prediction at top
            ax.set_xlabel("prob", fontsize=8)
            # Add GT label if not in top-k
            gt_in_topk = any(p.class_idx == preds_t.gt_idx for p in preds_t.predictions)
            if not gt_in_topk:
                ax.set_title(f"GT: {preds_t.gt_name[:15]}", fontsize=8, color="green")

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

# === Comet Logging Utilities ===

log = logging.getLogger(__name__)

# Comet curve budget - enforced at logging point, not upfront
_curve_count = 0
_CURVE_BUDGET = 900


def _log_curve(exp: comet_ml.Experiment, name: str, **kwargs) -> None:
    """Log curve with budget enforcement. Skips silently once exhausted."""
    global _curve_count
    if _curve_count >= _CURVE_BUDGET:
        if _curve_count == _CURVE_BUDGET:
            log.warning(f"Curve budget exhausted ({_CURVE_BUDGET}), skipping further curves")
            _curve_count += 1  # only warn once
        return
    exp.log_curve(name, **kwargs)
    _curve_count += 1


@dataclass
class VizSampleData:
    """Viz data extracted for a single sample during streaming validation.

    Shape annotations use:
        G = canvas grid size (e.g., 32)
        g = glimpse grid size (e.g., 3)
        D = teacher feature dim (e.g., 768)
        C = canvas hidden dim
    """

    glimpse: np.ndarray  # [g, g, 3] denormalized RGB
    predicted_scene: np.ndarray  # [G², D] teacher-space prediction
    canvas_spatial: np.ndarray  # [G², C] raw hidden state


@dataclass
class ValAccumulator:
    """Accumulator for streaming validation metrics.

    MEMORY OPTIMIZATION:
    - Validation metrics (scene_cos_sim, cls_cos_sim, IN1k acc) require the FULL BATCH
      to compute batch-averaged values. These are computed in step_fn, producing scalars,
      then the full-batch tensors are discarded (go out of scope).
    - PCA visualization only needs ONE sample. We extract sample 0's data in step_fn,
      move to CPU as numpy, and store in viz_samples. This is O(T) for one sample,
      not O(B×T) for all samples.

    Result: memory footprint is O(1) for metrics + O(T × single_sample) for viz,
    instead of O(B × T × tensor_size) if we kept all intermediate batch tensors.
    """

    # Per-timestep metrics (batch-averaged scalars)
    scene_cos_sims: list[float] = field(default_factory=list)
    cls_cos_sims: list[float] = field(default_factory=list)
    in1k_accs: list[float] = field(default_factory=list)
    pca_predictions: list[TimestepPredictions] = field(default_factory=list)
    # Viz data for sample 0 only (not full batch!)
    viz_samples: list[VizSampleData] = field(default_factory=list)
    # Initial state (before any glimpses), sample 0 only
    initial_scene: np.ndarray | None = None
    initial_canvas_spatial: np.ndarray | None = None


def _extract_sample0_viz(
    out: GlimpseOutput,
    predicted_scene: Tensor,
    model: "ActiveCanViT",
) -> VizSampleData:
    """Extract viz data for sample 0, move to CPU as numpy.

    This extracts ONE sample from the batch, allowing the full batch tensors
    to be garbage collected after step_fn returns.
    """
    # Glimpse: [B, C, g*ps, g*ps] -> [g*ps, g*ps, C] for sample 0
    glimpse_cpu = out.glimpse[0].cpu()
    glimpse_np = imagenet_denormalize(glimpse_cpu).numpy()

    # Predicted scene: [B, G², D] -> [G², D] for sample 0
    scene_cpu = predicted_scene[0].cpu().float()
    scene_np = scene_cpu.numpy()

    # Canvas spatial (hidden state): [B, n_prefix + G², C] -> [G², C] for sample 0
    canvas_single = out.canvas[0:1]  # Keep batch dim for get_spatial
    spatial = model.get_spatial(canvas_single)[0]  # [G², C]
    spatial_np = spatial.cpu().float().numpy()

    return VizSampleData(
        glimpse=glimpse_np,
        predicted_scene=scene_np,
        canvas_spatial=spatial_np,
    )




def compute_spatial_stats(x: Tensor) -> dict[str, float]:
    """Compute mean/std across spatial dimension, averaged over batch.

    Args:
        x: [B, N, D] tensor (N = spatial tokens)

    Returns:
        Dict with 'mean' and 'std' scalars:
        - mean: average of per-sample spatial means
        - std: average of per-sample spatial stds
    """
    # Per-sample spatial stats: [B, D]
    spatial_mean = x.mean(dim=1)
    spatial_std = x.std(dim=1)
    # Average across batch and dimensions to get scalars
    return {
        "mean": spatial_mean.mean().item(),
        "std": spatial_std.mean().item(),
    }


def log_figure(exp: comet_ml.Experiment, fig: Figure, name: str, step: int) -> None:
    """Log matplotlib figure to Comet. Aggressively cleans up to prevent memory leaks."""
    with io.BytesIO() as buf:
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        exp.log_image(buf, name=name, step=step)
    # Clear all axes before closing to release colorbar/patch references
    for ax in fig.axes:
        ax.clear()
    fig.clf()
    plt.close(fig)
    # Force garbage collection after complex figures
    gc.collect()


@dataclass
class _TrainVizAccumulator:
    """Accumulator for training viz (uses forward_reduce, sample 0 only)."""

    scene_cos_sims: list[float] = field(default_factory=list)
    cls_cos_sims: list[float] = field(default_factory=list)
    viz_samples: list[VizSampleData] = field(default_factory=list)
    final_predicted_scene: Tensor | None = None  # Keep last for spatial stats


def viz_and_log(
    exp: comet_ml.Experiment,
    step: int,
    prefix: str,
    model: ActiveCanViT,
    teacher: DINOv3Backbone,
    normalizer: PositionAwareNorm,
    images: Tensor,
    viewpoints: list[Viewpoint],
    target: Tensor,
    canvas: Tensor,
    glimpse_size_px: int,
    cls_target: Tensor | None = None,
    log_spatial_stats: bool = True,
    log_curves: bool = True,
) -> None:
    """Run forward pass and log PCA visualization (training viz).

    Uses forward_reduce to avoid O(B×T) memory - only sample 0 viz data is retained.
    """
    assert isinstance(model.backbone, DINOv3Backbone)
    n_spatial = canvas.shape[1] - model.n_canvas_registers
    canvas_grid_size = int(n_spatial**0.5)
    assert canvas_grid_size**2 == n_spatial
    glimpse_grid_size = glimpse_size_px // model.backbone.patch_size_px
    has_cls = cls_target is not None

    with torch.inference_mode():
        # Capture initial state for sample 0
        initial_scene_np = model.predict_teacher_scene(canvas)[0].cpu().float().numpy()
        initial_canvas_spatial_np = model.get_spatial(canvas[0:1])[0].cpu().float().numpy()

        def init_fn(_canvas: Tensor, _cls: Tensor) -> _TrainVizAccumulator:
            return _TrainVizAccumulator()

        def step_fn(acc: _TrainVizAccumulator, out: GlimpseOutput, _vp: CanvitViewpoint) -> _TrainVizAccumulator:
            predicted_scene = model.predict_teacher_scene(out.canvas)

            # Metrics (batch-averaged)
            acc.scene_cos_sims.append(
                F.cosine_similarity(predicted_scene, target, dim=-1).mean().item()
            )
            if has_cls:
                assert cls_target is not None
                predicted_cls = model.predict_teacher_cls(out.cls, out.canvas)
                acc.cls_cos_sims.append(
                    F.cosine_similarity(predicted_cls, cls_target, dim=-1).mean().item()
                )

            # Viz data: sample 0 only
            acc.viz_samples.append(_extract_sample0_viz(out, predicted_scene, model))
            acc.final_predicted_scene = predicted_scene  # Keep for spatial stats
            return acc

        acc, _, _ = model.forward_reduce(
            image=images,
            viewpoints=viewpoints,  # pyright: ignore[reportArgumentType]
            glimpse_size_px=glimpse_size_px,
            canvas_grid_size=canvas_grid_size,
            init_fn=init_fn,
            step_fn=step_fn,
            canvas=canvas,
        )

        # Log curves
        if log_curves:
            _log_curve(exp, f"{prefix}/scene_cos_sim_vs_timestep", x=list(range(len(acc.scene_cos_sims))), y=acc.scene_cos_sims, step=step)
            if acc.cls_cos_sims:
                _log_curve(exp, f"{prefix}/cls_cos_sim_vs_timestep", x=list(range(len(acc.cls_cos_sims))), y=acc.cls_cos_sims, step=step)

        # Spatial stats
        if log_spatial_stats and acc.final_predicted_scene is not None:
            target_stats = compute_spatial_stats(target)
            pred_stats = compute_spatial_stats(acc.final_predicted_scene)
            exp.log_metrics({
                f"{prefix}/target_spatial_mean": target_stats["mean"],
                f"{prefix}/target_spatial_std": target_stats["std"],
                f"{prefix}/pred_spatial_mean": pred_stats["mean"],
                f"{prefix}/pred_spatial_std": pred_stats["std"],
            }, step=step)

        # PCA viz
        H, W = images.shape[-2], images.shape[-1]
        boxes = [vp.to_pixel_box(0, H, W) for vp in viewpoints]
        names = [vp.name for vp in viewpoints]

        fig_pca = plot_multistep_pca(
            imagenet_denormalize(images[0].cpu()).numpy(),
            target[0].cpu().float().numpy(),
            [vs.predicted_scene for vs in acc.viz_samples],
            None, None,  # locals_model, locals_teacher (skip)
            [vs.glimpse for vs in acc.viz_samples],
            boxes, names,
            canvas_grid_size, glimpse_grid_size,
            initial_scene_np,
            hidden_spatials=[vs.canvas_spatial for vs in acc.viz_samples],
            initial_hidden_spatial=initial_canvas_spatial_np,
            locals_teacher_cropped=None,
            show_locals=False,
            timestep_predictions=None,
        )
        log_figure(exp, fig_pca, f"{prefix}/pca", step)


def _log_pca_from_accumulator(
    exp: comet_ml.Experiment,
    step: int,
    prefix: str,
    acc: ValAccumulator,
    full_img: np.ndarray,
    teacher_np: np.ndarray,
    boxes: list,
    names: list[str],
    canvas_grid_size: int,
    glimpse_grid_size: int,
    log_spatial_stats: bool,
    log_curves: bool,
) -> None:
    """Log PCA visualization from pre-computed accumulator data (no forward pass)."""
    assert acc.initial_scene is not None, "initial_scene must be set when log_pca=True"
    scenes = [vs.predicted_scene for vs in acc.viz_samples]
    glimpses = [vs.glimpse for vs in acc.viz_samples]
    canvas_spatials = [vs.canvas_spatial for vs in acc.viz_samples]

    fig_pca = plot_multistep_pca(
        full_img,
        teacher_np,
        scenes,
        None,  # locals_model (expensive, skip)
        None,  # locals_teacher
        glimpses,
        boxes,
        names,
        canvas_grid_size,
        glimpse_grid_size,
        acc.initial_scene,
        hidden_spatials=canvas_spatials if canvas_spatials[0] is not None else None,
        initial_hidden_spatial=acc.initial_canvas_spatial,
        locals_teacher_cropped=None,
        show_locals=False,
        timestep_predictions=acc.pca_predictions if acc.pca_predictions else None,
    )
    log_figure(exp, fig_pca, f"{prefix}/pca", step)

    if log_spatial_stats and acc.viz_samples:
        target_stats = {"mean": float(np.mean(teacher_np)), "std": float(np.std(teacher_np))}
        pred_stats = {"mean": float(np.mean(scenes[-1])), "std": float(np.std(scenes[-1]))}
        exp.log_metrics(
            {
                f"{prefix}/target_spatial_mean": target_stats["mean"],
                f"{prefix}/target_spatial_std": target_stats["std"],
                f"{prefix}/pred_spatial_mean": pred_stats["mean"],
                f"{prefix}/pred_spatial_std": pred_stats["std"],
            },
            step=step,
        )

    if log_curves:
        _log_curve(
            exp,
            f"{prefix}/scene_cos_sim_vs_timestep",
            x=list(range(len(acc.scene_cos_sims))),
            y=acc.scene_cos_sims,
            step=step,
        )
        if acc.cls_cos_sims:
            _log_curve(
                exp,
                f"{prefix}/cls_cos_sim_vs_timestep",
                x=list(range(len(acc.cls_cos_sims))),
                y=acc.cls_cos_sims,
                step=step,
            )


def validate(
    exp: comet_ml.Experiment,
    step: int,
    model: ActiveCanViT,
    compute_raw_targets: Callable[[Tensor, int], "NormFeatures"],
    scene_normalizer: PositionAwareNorm,
    cls_normalizer: PositionAwareNorm,
    images: Tensor,
    canvas_grid_size: int,
    scene_size_px: int,
    glimpse_size_px: int,
    prefix: str = "val",
    probe: DINOv3LinearClassificationHead | None = None,
    labels: Tensor | None = None,
    log_curves: bool = False,
    log_pca: bool = False,
    teacher: DINOv3Backbone | None = None,
    log_spatial_stats: bool = False,
    backbone: str | None = None,
) -> float:
    """Run validation with streaming metrics (no O(B×T) memory).

    Uses forward_reduce to compute metrics on the fly, discarding batch tensors.
    Only sample 0's viz data is retained (if log_pca=True).
    """
    assert not log_pca or teacher is not None, "teacher required for PCA viz"

    if probe is not None and backbone is not None:
        probe_res = get_probe_resolution(backbone)
        if scene_size_px != probe_res:
            log.warning(
                f"Resolution mismatch: model predicts teacher@{scene_size_px}, "
                f"but probe trained on teacher@{probe_res}. IN1k metrics may be unreliable."
            )

    B = images.shape[0]
    viewpoints = make_eval_viewpoints(B, images.device)
    has_cls = model.cls_proj is not None
    # Skip IN1k probe metrics for IN21k training (labels >= 1000)
    has_probe = probe is not None and labels is not None and labels_are_in1k(labels)

    # Freeze normalizers during validation
    scene_was_training = scene_normalizer.training
    cls_was_training = cls_normalizer.training
    scene_normalizer.eval()
    cls_normalizer.eval()

    try:
        with torch.inference_mode():
            # Compute targets
            raw_feats = compute_raw_targets(images, scene_size_px)
            target = scene_normalizer(raw_feats.patches)
            cls_target = cls_normalizer(raw_feats.cls.unsqueeze(1)).squeeze(1) if has_cls else None

            # For viz: extract sample 0's target now (before we start discarding things)
            target_sample0 = target[0].cpu().float().numpy() if log_pca else None

            # Probe setup
            gt_idx = int(labels[0].item()) if has_probe and labels is not None else 0
            gt_name = get_imagenet_class_names()[gt_idx] if has_probe else ""

            # Teacher baseline (if probe available)
            if has_probe and teacher is not None:
                assert backbone is not None and probe is not None
                probe_res = get_probe_resolution(backbone)
                images_at_probe_res = F.interpolate(images, size=(probe_res, probe_res), mode="bilinear", align_corners=False)
                teacher_cls = teacher.forward_norm_features(images_at_probe_res).cls
                teacher_logits = probe(teacher_cls)
                assert labels is not None
                teacher_acc = compute_in1k_top1(teacher_logits, labels)
                exp.log_metric(f"{prefix}/in1k_teacher_top1", teacher_acc, step=step)

            def init_fn(canvas: Tensor, cls: Tensor) -> ValAccumulator:
                acc = ValAccumulator()
                if log_pca:
                    # Capture initial state for sample 0 only
                    acc.initial_scene = model.predict_teacher_scene(canvas)[0].cpu().float().numpy()
                    acc.initial_canvas_spatial = model.get_spatial(canvas[0:1])[0].cpu().float().numpy()
                return acc

            def step_fn(acc: ValAccumulator, out: GlimpseOutput, _vp: CanvitViewpoint) -> ValAccumulator:
                # Compute predictions from full batch (tensors discarded after this function)
                predicted_scene = model.predict_teacher_scene(out.canvas)
                predicted_cls = model.predict_teacher_cls(out.cls, out.canvas) if has_cls else None

                # --- Batch-averaged metrics (full batch -> scalar -> discard) ---
                scene_cos = F.cosine_similarity(predicted_scene, target, dim=-1).mean().item()
                acc.scene_cos_sims.append(scene_cos)

                if has_cls and cls_target is not None and predicted_cls is not None:
                    cls_cos = F.cosine_similarity(predicted_cls, cls_target, dim=-1).mean().item()
                    acc.cls_cos_sims.append(cls_cos)

                    if has_probe:
                        assert probe is not None and labels is not None
                        cls_raw = cls_normalizer.denormalize(predicted_cls)
                        logits = probe(cls_raw)
                        acc.in1k_accs.append(compute_in1k_top1(logits, labels))

                        if log_pca:
                            top_k = get_top_k_predictions(logits[0:1], k=5)[0]
                            acc.pca_predictions.append(TimestepPredictions(
                                predictions=top_k, gt_idx=gt_idx, gt_name=gt_name
                            ))

                # --- Viz data: sample 0 only (O(1) memory, not O(B)) ---
                if log_pca:
                    acc.viz_samples.append(_extract_sample0_viz(out, predicted_scene, model))

                return acc

            # Run forward_reduce: metrics computed on the fly, intermediates discarded
            acc, _final_canvas, _final_cls = model.forward_reduce(
                image=images,
                viewpoints=viewpoints,  # pyright: ignore[reportArgumentType]
                glimpse_size_px=glimpse_size_px,
                canvas_grid_size=canvas_grid_size,
                init_fn=init_fn,
                step_fn=step_fn,
            )

            # Log metrics
            final_cos_sim = acc.scene_cos_sims[-1]
            exp.log_metric(f"{prefix}/scene_cos_sim", final_cos_sim, step=step)

            for t, sc in enumerate(acc.scene_cos_sims):
                exp.log_metric(f"{prefix}/scene_cos_sim_t{t}", sc, step=step)

            if has_cls:
                exp.log_metric(f"{prefix}/cls_cos_sim", acc.cls_cos_sims[-1], step=step)
                for t, cc in enumerate(acc.cls_cos_sims):
                    exp.log_metric(f"{prefix}/cls_cos_sim_t{t}", cc, step=step)

            if has_probe:
                for t, ia in enumerate(acc.in1k_accs):
                    exp.log_metric(f"{prefix}/in1k_tts_top1_t{t}", ia, step=step)
                if log_curves:
                    _log_curve(
                        exp,
                        f"{prefix}/in1k_tts_top1_vs_timestep",
                        x=list(range(len(acc.in1k_accs))),
                        y=acc.in1k_accs,
                        step=step,
                    )

            # PCA visualization (uses pre-collected sample 0 data, no extra forward pass)
            if log_pca:
                assert target_sample0 is not None
                H, W = images.shape[-2], images.shape[-1]
                boxes = [vp.to_pixel_box(0, H, W) for vp in viewpoints]
                names = [vp.name for vp in viewpoints]
                full_img = imagenet_denormalize(images[0].cpu()).numpy()
                glimpse_grid_size = glimpse_size_px // model.backbone.patch_size_px

                _log_pca_from_accumulator(
                    exp=exp,
                    step=step,
                    prefix=prefix,
                    acc=acc,
                    full_img=full_img,
                    teacher_np=target_sample0,
                    boxes=boxes,
                    names=names,
                    canvas_grid_size=canvas_grid_size,
                    glimpse_grid_size=glimpse_grid_size,
                    log_spatial_stats=log_spatial_stats,
                    log_curves=log_curves,
                )

            return final_cos_sim
    finally:
        if scene_was_training:
            scene_normalizer.train()
        if cls_was_training:
            cls_normalizer.train()


