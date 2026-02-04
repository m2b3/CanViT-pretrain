"""Visualization for ADE20K probe training."""

from pathlib import Path
from typing import Literal, Protocol

import comet_ml
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from torch import Tensor

from canvit_eval.ade20k.dataset import IGNORE_LABEL, NUM_CLASSES
from canvit_pretrain.train.viz.image import imagenet_denormalize

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


def _fit_pca(feats: np.ndarray, n_components: int = 3) -> PCA | None:
    """Fit PCA on [N, D] features. Returns None if variance too low."""
    if feats.var(axis=0).max() < 1e-5:
        return None
    n_components = min(n_components, feats.shape[0], feats.shape[1])
    pca = PCA(n_components=n_components, whiten=True)
    pca.fit(feats)
    return pca


def _pca_to_rgb(pca: PCA | None, feats: np.ndarray, H: int, W: int) -> np.ndarray:
    """Project features to RGB using PCA with local percentile normalization.

    Uses 2nd-98th percentile per channel so each frame uses full dynamic range
    while preserving semantic directions from the shared PCA.
    """
    if pca is None:
        return np.full((H, W, 3), 0.5, dtype=np.float32)
    proj = pca.transform(feats)[:, :3]
    # Local percentile normalization: each frame uses its own dynamic range
    lo = np.percentile(proj, 2, axis=0, keepdims=True)
    hi = np.percentile(proj, 98, axis=0, keepdims=True)
    rgb = np.clip((proj - lo) / (hi - lo + 1e-8), 0, 1)
    return rgb.reshape(H, W, 3).astype(np.float32)


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
    """Create visualization figure with predictions and PCA.

    PCA approach (critical for interpretability):
    - PCA is NEVER shared across images - each image gets its own PCA
    - For each image: fit PCA on t=-1 (final timestep, richest features)
    - Apply that same PCA to all timesteps of that image
    - This keeps semantic directions consistent across timesteps
    - Local percentile normalization ensures t=0 isn't washed out
    """
    n_samples = min(n_samples, images.shape[0])
    feat_types: list[FeatureType] = [f for f in probes.keys() if f not in STATIC_FEATURES]
    # Columns: Image, GT, then per feat_type: (pred t0, corr t0, PCA t0, pred t-1, corr t-1, PCA t-1)
    n_cols = 2 + len(feat_types) * 6

    fig, axes = plt.subplots(n_samples, n_cols, figsize=(2.5 * n_cols, 2.5 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    t_final = n_timesteps - 1

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

        # Per-image PCA: fit on t=-1 (final timestep), apply to all timesteps
        # This ensures consistent semantic directions while avoiding washed-out t=0
        pca_per_feat: dict[FeatureType, PCA | None] = {}
        for feat_type in feat_types:
            feat_final = feats[feat_type][t_final][i].cpu().float().numpy()
            H, W, D = feat_final.shape
            pca_per_feat[feat_type] = _fit_pca(feat_final.reshape(-1, D))

        for feat_type in feat_types:
            pca = pca_per_feat[feat_type]

            for t, t_name in [(0, "t0"), (t_final, "t-1")]:
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

                # PCA: use per-image PCA fit on t=-1, with local percentile normalization
                feat_np = feat_i.cpu().float().numpy()
                pca_img = _pca_to_rgb(pca, feat_np.reshape(-1, D), H, W)
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
