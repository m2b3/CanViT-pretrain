"""Visualization for ADE20K probe training."""

from pathlib import Path

import comet_ml
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure
from torch import Tensor

from canvit_eval.ade20k.dataset import IGNORE_LABEL, NUM_CLASSES
from canvit_eval.ade20k.train_probe.config import STATIC_FEATURES, FeatureType
from canvit_eval.ade20k.train_probe.state import ProbeState
from canvit_pretrain.train.viz.image import imagenet_denormalize

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
    probes: dict[FeatureType, ProbeState],
    feats: dict[FeatureType, list[Tensor]],
    images: Tensor,
    masks: Tensor,
    n_samples: int,
    n_timesteps: int,
) -> Figure:
    """Create visualization figure."""
    n_samples = min(n_samples, images.shape[0])
    feat_types: list[FeatureType] = [f for f in probes.keys() if f not in STATIC_FEATURES]
    n_cols = 2 + len(feat_types) * 4

    fig, axes = plt.subplots(n_samples, n_cols, figsize=(3 * n_cols, 3 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

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
                with torch.no_grad():
                    logits = probes[feat_type].head(feats[feat_type][t][i : i + 1].float())
                    pred = logits[0].argmax(0).cpu().numpy()

                pred_up = F.interpolate(
                    torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float(),
                    size=gt.shape, mode="nearest",
                ).squeeze().numpy().astype(np.int64)

                axes[i, col].imshow(colorize_mask(pred_up))
                axes[i, col].set_title(f"{feat_type[:6]} {t_name}" if i == 0 else "")
                axes[i, col].axis("off")
                col += 1

                axes[i, col].imshow(correctness_map(pred_up, gt))
                axes[i, col].set_title(f"corr {t_name}" if i == 0 else "")
                axes[i, col].axis("off")
                col += 1

    plt.tight_layout()
    return fig


def log_viz(
    exp: comet_ml.Experiment,
    step: int,
    probes: dict[FeatureType, ProbeState],
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
