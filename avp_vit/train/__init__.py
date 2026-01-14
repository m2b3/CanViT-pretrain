"""Training utilities for AVP-ViT.

This module provides the main training loop and supporting utilities.
Entry point: python -m avp_vit.train
"""

from avp_vit.train.data import (
    Batch,
    InfiniteLoader,
    Loaders,
    create_loaders,
    scene_size_px,
)
from avp_vit.train.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    imagenet_normalize,
    train_transform,
    val_transform,
)
from avp_vit.train.norm import PositionAwareNorm
from avp_vit.train.probe import (
    IN1K_NUM_CLASSES,
    PROBE_REGISTRY,
    ProbeInfo,
    TopKPrediction,
    compute_in1k_top1,
    get_imagenet_class_names,
    get_probe_resolution,
    get_top_k_predictions,
    labels_are_in1k,
    load_probe,
)
from avp_vit.train.scheduler import warmup_constant_scheduler
from avp_vit.train.viewpoint import (
    PixelBox,
    Viewpoint,
    make_eval_viewpoints,
    random_viewpoint,
    viewpoint_to_pixel_box,
)
from avp_vit.train.viz import (
    TimestepPredictions,
    fit_pca,
    imagenet_denormalize,
    pca_rgb,
    plot_multistep_pca,
    plot_pca_grid,
    plot_trajectory,
    timestep_colors,
    validate,
)

__all__ = [
    # Data
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "Batch",
    "InfiniteLoader",
    "Loaders",
    "create_loaders",
    "imagenet_normalize",
    "scene_size_px",
    "train_transform",
    "val_transform",
    # Norm
    "PositionAwareNorm",
    # Probe
    "IN1K_NUM_CLASSES",
    "PROBE_REGISTRY",
    "ProbeInfo",
    "TopKPrediction",
    "compute_in1k_top1",
    "get_imagenet_class_names",
    "get_probe_resolution",
    "get_top_k_predictions",
    "labels_are_in1k",
    "load_probe",
    # Scheduler
    "warmup_constant_scheduler",
    # Viewpoint
    "PixelBox",
    "Viewpoint",
    "make_eval_viewpoints",
    "random_viewpoint",
    "viewpoint_to_pixel_box",
    # Viz
    "TimestepPredictions",
    "fit_pca",
    "imagenet_denormalize",
    "pca_rgb",
    "plot_multistep_pca",
    "plot_pca_grid",
    "plot_trajectory",
    "timestep_colors",
    "validate",
]
