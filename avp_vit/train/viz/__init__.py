"""Visualization utilities for training."""

from .comet import log_curve, log_figure
from .image import imagenet_denormalize
from .metrics import compute_spatial_stats, cosine_dissimilarity
from .pca import fit_pca, pca_rgb
from .plot import (
    RGBA,
    TimestepPredictions,
    plot_multistep_pca,
    plot_pca_grid,
    plot_trajectory,
    timestep_colors,
)
from .sample import VizSampleData, extract_sample0_viz
from .train_viz import viz_and_log
from .validate import ValAccumulator, validate

__all__ = [
    # comet
    "log_curve",
    "log_figure",
    # image
    "imagenet_denormalize",
    # metrics
    "compute_spatial_stats",
    "cosine_dissimilarity",
    # pca
    "fit_pca",
    "pca_rgb",
    # plot
    "RGBA",
    "TimestepPredictions",
    "plot_multistep_pca",
    "plot_pca_grid",
    "plot_trajectory",
    "timestep_colors",
    # sample
    "VizSampleData",
    "extract_sample0_viz",
    # train_viz
    "viz_and_log",
    # validate
    "ValAccumulator",
    "validate",
]
