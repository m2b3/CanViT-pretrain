"""Visualization for ADE20K probe training."""

from pathlib import Path
from typing import Literal, Protocol

import comet_ml
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure
from torch import Tensor

from canvit_eval.ade20k.dataset import IGNORE_LABEL, NUM_CLASSES
from canvit_pretrain.train.viz.image import imagenet_denormalize
from canvit_pretrain.train.viz.pca import fit_pca, pca_rgb

FeatureType = Literal["hidden", "predicted_norm", "teacher_glimpse"]
STATIC_FEATURES: frozenset[FeatureType] = frozenset({"teacher_glimpse"})


class ProbeStateLike(Protocol):
    """Protocol for probe state - avoids circular import."""

    @property
    def head(self) -> torch.nn.Module: ...

# Deterministic palette for segmentation masks
_PALETTE = np.random.RandomState(42).randint(0, 255, (NUM_CLASSES + 1, 3), dtype=np.uint8)
_PALETTE[NUM_CLASSES] = 0


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Colorize segmentation mask with palette."""
    return _PALETTE[np.where(mask == IGNORE_LABEL, NUM_CLASSES, mask)]


def correctness_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Create correctness heatmap: green=correct, red=wrong, gray=ignore."""
    H, W = pred.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    valid = gt != IGNORE_LABEL
    out[(pred == gt) & valid] = [0, 200, 0]
    out[(pred != gt) & valid] = [200, 0, 0]
    out[~valid] = [128, 128, 128]
    return out


def make_viz_figure(
    probes: dict[FeatureType, ProbeStateLike],
    feats: dict[FeatureType, list[Tensor]],
    images: Tensor,
    masks: Tensor,
    n_samples: int,
    n_timesteps: int,
) -> Figure:
    """Create visualization figure with predictions and PCA."""
    n_samples = min(n_samples, images.shape[0])
    feat_types: list[FeatureType] = [f for f in probes.keys() if f not in STATIC_FEATURES]
    # Columns: Image, GT, then per feat_type: (pred t0, corr t0, PCA t0, pred t-1, corr t-1, PCA t-1)
    n_cols = 2 + len(feat_types) * 6

    fig, axes = plt.subplots(n_samples, n_cols, figsize=(2.5 * n_cols, 2.5 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    # Fit PCA on first sample's features (shared across samples for consistent colors)
    pca_models: dict[FeatureType, dict[int, object]] = {}
    for feat_type in feat_types:
        pca_models[feat_type] = {}
        for t in [0, n_timesteps - 1]:
            feat_np = feats[feat_type][t][0].cpu().float().numpy()
            H, W, D = feat_np.shape
            pca_models[feat_type][t] = fit_pca(feat_np.reshape(-1, D))

    for i in range(n_samples):
        col = 0
        img_np = (imagenet_denormalize(images[i]).cpu().numpy() * 255).astype(np.uint8)
        axes[i, col].imshow(img_np)
        axes[i, col].set_title("Image" if i == 0 else "")
        axes[i, col].axis("off")
        col += 1

        gt = masks[i].cpu().numpy()
        axes[i, col].imshow(colorize_mask(gt))
        axes[i, col].set_title("GT" if i == 0 else "")
        axes[i, col].axis("off")
        col += 1

        for feat_type in feat_types:
            for t, t_name in [(0, "t0"), (n_timesteps - 1, "t-1")]:
                feat_i = feats[feat_type][t][i]
                H, W, D = feat_i.shape

                # Prediction
                with torch.no_grad():
                    logits = probes[feat_type].head(feat_i.unsqueeze(0).float())
                    pred = logits[0].argmax(0).cpu().numpy()

                pred_up = F.interpolate(
                    torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float(),
                    size=gt.shape,
                    mode="nearest",
                ).squeeze().numpy().astype(np.int64)

                axes[i, col].imshow(colorize_mask(pred_up))
                axes[i, col].set_title(f"{feat_type[:6]} {t_name}" if i == 0 else "")
                axes[i, col].axis("off")
                col += 1

                # Correctness
                axes[i, col].imshow(correctness_map(pred_up, gt))
                axes[i, col].set_title(f"corr {t_name}" if i == 0 else "")
                axes[i, col].axis("off")
                col += 1

                # PCA
                feat_np = feat_i.cpu().float().numpy()
                pca = pca_models[feat_type][t]
                pca_img = pca_rgb(pca, feat_np.reshape(-1, D), H, W)
                # Upsample PCA to image size for better visibility
                pca_up = F.interpolate(
                    torch.from_numpy(pca_img).permute(2, 0, 1).unsqueeze(0),
                    size=gt.shape,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze().permute(1, 2, 0).numpy()
                axes[i, col].imshow(pca_up)
                axes[i, col].set_title(f"PCA {t_name}" if i == 0 else "")
                axes[i, col].axis("off")
                col += 1

    plt.tight_layout()
    return fig


def log_viz(
    exp: comet_ml.Experiment,
    step: int,
    probes: dict[FeatureType, ProbeStateLike],
    feats: dict[FeatureType, list[Tensor]],
    images: Tensor,
    masks: Tensor,
    n_samples: int,
    n_timesteps: int,
) -> None:
    """Log visualization to Comet."""
    fig = make_viz_figure(probes, feats, images, masks, n_samples, n_timesteps)
    exp.log_figure(figure_name=f"viz_{step}", figure=fig, step=step)
    plt.close(fig)


def save_viz(path: Path, fig: Figure) -> None:
    """Save figure to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)
