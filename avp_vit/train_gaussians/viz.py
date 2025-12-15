"""Visualization utilities for gaussian blob training."""

import io

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from torch import Tensor

from avp_vit.glimpse import Viewpoint

from .data import IMAGENET_MEAN, IMAGENET_STD


def log_figure(exp, fig: Figure, name: str, step: int) -> None:
    """Log matplotlib figure to Comet experiment."""
    with io.BytesIO() as buf:
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        exp.log_image(buf, name=name, step=step)
    plt.close(fig)


def timestep_colors(n: int) -> list[tuple[float, float, float, float]]:
    """Get distinct colors for n timesteps."""
    cmap = plt.get_cmap("viridis")
    return [cmap(i / max(1, n - 1)) for i in range(n)]


def plot_trajectory_with_glimpses(
    images: Tensor,
    target_colors: Tensor,
    target_centers: Tensor,
    viewpoints: list[Viewpoint],
    glimpses: list[Tensor],
    sample_idx: int = 0,
) -> Figure:
    """Plot trajectory showing image, viewpoint boxes over time, and glimpses.

    Args:
        images: [B, 3, H, W] ImageNet-normalized images
        target_colors: [B, 3] RGB target colors
        target_centers: [B, 2] target centers (y, x) in [-1, 1]
        viewpoints: List of viewpoints at each timestep
        glimpses: List of [B, 3, G, G] glimpse tensors at each timestep
        sample_idx: Which batch sample to visualize
    """
    n_steps = len(viewpoints)
    H, W = images.shape[2], images.shape[3]

    # Denormalize image
    mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
    img = (images[sample_idx : sample_idx + 1] * std + mean).clamp(0, 1)[0]
    img_np = img.permute(1, 2, 0).cpu().numpy()

    # Denormalize glimpses
    glimpse_imgs = []
    for g in glimpses:
        g_denorm = (g[sample_idx : sample_idx + 1] * std + mean).clamp(0, 1)[0]
        glimpse_imgs.append(g_denorm.permute(1, 2, 0).cpu().numpy())

    # Get boxes
    boxes = [vp.to_pixel_box(sample_idx, H, W) for vp in viewpoints]
    colors = timestep_colors(n_steps)

    # Layout: main image + n_steps glimpses
    fig, axes = plt.subplots(1, n_steps + 1, figsize=(4 * (n_steps + 1), 4))

    # Main image with trajectory
    ax = axes[0]
    ax.imshow(img_np)

    # Draw target marker
    ty = (target_centers[sample_idx, 0].item() + 1) / 2 * H
    tx = (target_centers[sample_idx, 1].item() + 1) / 2 * W
    ax.scatter(
        tx, ty, c="white", s=200, marker="*", edgecolors="black",
        linewidths=2, zorder=10, label="Target",
    )

    # Draw connecting lines
    if n_steps > 1:
        xs = [b.center_x for b in boxes]
        ys = [b.center_y for b in boxes]
        ax.plot(xs, ys, "-", color="white", linewidth=1.5, alpha=0.7, zorder=1)

    # Draw boxes
    for i, box in enumerate(boxes):
        rect = mpatches.Rectangle(
            (box.left, box.top), box.width, box.height,
            linewidth=2, edgecolor=colors[i], facecolor="none",
            label=f"t={i} (s={box.width / W:.2f})",
        )
        ax.add_patch(rect)
        ax.plot(box.center_x, box.center_y, "o", color=colors[i], markersize=6, zorder=2)

    # Target color patch
    tgt_color = target_colors[sample_idx].cpu().numpy()
    ax.add_patch(
        mpatches.Rectangle((5, 5), 30, 30, facecolor=tgt_color, edgecolor="white", linewidth=2)
    )

    ax.set_title("Trajectory")
    ax.legend(loc="upper right", fontsize=7)
    ax.axis("off")

    # Glimpses
    for i, g_img in enumerate(glimpse_imgs):
        ax = axes[i + 1]
        ax.imshow(g_img)
        ax.set_title(f"Glimpse t={i}")
        ax.axis("off")

    plt.tight_layout()
    return fig


def plot_scene_pca(
    hiddens: list[Tensor],
    grid_size: int,
    sample_idx: int = 0,
) -> Figure:
    """Plot PCA visualization of scene hidden states across timesteps.

    Args:
        hiddens: List of [B, H*W, D] hidden states at each timestep (including init)
        grid_size: Spatial grid size (H = W)
        sample_idx: Which batch sample to visualize
    """
    n_steps = len(hiddens)

    # Extract single sample and reshape to [H*W, D]
    feats = [h[sample_idx].cpu().numpy() for h in hiddens]

    # Fit PCA on concatenated features for consistent coloring
    all_feats = np.concatenate(feats, axis=0)
    pca = PCA(n_components=3, whiten=True)
    pca.fit(all_feats)

    def pca_rgb(f: np.ndarray) -> np.ndarray:
        proj = pca.transform(f)
        return 1.0 / (1.0 + np.exp(-proj.reshape(grid_size, grid_size, 3) * 2.0))

    fig, axes = plt.subplots(1, n_steps, figsize=(4 * n_steps, 4))
    if n_steps == 1:
        axes = [axes]

    for t, (ax, f) in enumerate(zip(axes, feats)):
        rgb = pca_rgb(f)
        ax.imshow(rgb)
        ax.set_title(f"t={t}" if t > 0 else "init")
        ax.axis("off")

    plt.suptitle("Scene Hidden State (PCA -> RGB)")
    plt.tight_layout()
    return fig


def plot_policy_scatter(
    pred_centers: Tensor,
    target_centers: Tensor,
    target_colors: Tensor,
    title: str,
    n_samples: int = 200,
) -> Figure:
    """Scatter plot of predicted vs target viewpoint centers.

    Args:
        pred_centers: [B, 2] predicted centers (y, x)
        target_centers: [B, 2] target centers (y, x)
        target_colors: [B, 3] RGB colors for each point
        title: Plot title
        n_samples: Max samples to plot
    """
    n = min(n_samples, pred_centers.shape[0])

    pred = pred_centers[:n].cpu().numpy()
    tgt = target_centers[:n].cpu().numpy()
    colors = target_colors[:n].cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot targets as circles
    ax.scatter(
        tgt[:, 1], tgt[:, 0], c=colors, s=100, marker="o",
        edgecolors="black", linewidths=1, label="Target", alpha=0.7,
    )

    # Plot predictions as X marks
    ax.scatter(
        pred[:, 1], pred[:, 0], c=colors, s=60, marker="x",
        linewidths=2, label="Predicted", alpha=0.9,
    )

    # Draw lines connecting each pred-target pair
    for i in range(n):
        ax.plot([tgt[i, 1], pred[i, 1]], [tgt[i, 0], pred[i, 0]], c=colors[i], alpha=0.3, linewidth=1)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.invert_yaxis()  # match image coords (Y=0 at top)
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    return fig


def plot_scale_distribution(
    scales_det: Tensor,
    scales_noised: Tensor,
    title: str = "Scale Distribution",
) -> Figure:
    """Plot histogram of scales (deterministic vs with noise)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    s_det = scales_det.cpu().numpy()
    s_noised = scales_noised.cpu().numpy()

    ax.hist(s_det, bins=30, alpha=0.7, label=f"Deterministic (u={s_det.mean():.3f})", density=True)
    ax.hist(s_noised, bins=30, alpha=0.7, label=f"With noise (u={s_noised.mean():.3f})", density=True)

    ax.set_xlabel("Scale")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
