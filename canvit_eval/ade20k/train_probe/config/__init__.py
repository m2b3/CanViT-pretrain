"""ADE20K probe training configuration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

FeatureType = Literal["hidden", "predicted_norm", "teacher_glimpse"]
STATIC_FEATURES: frozenset[FeatureType] = frozenset({"teacher_glimpse"})


@dataclass
class Config:
    """ADE20K probe training configuration."""

    ade20k_root: Path
    model_repo: str = "canvit/canvit-vitb16-pretrain-512px-in21k"
    features: list[FeatureType] = field(default_factory=lambda: ["hidden", "predicted_norm", "teacher_glimpse"])
    n_timesteps: int = 10
    image_size: int = 512
    glimpse_px: int = 128

    # Viewpoint policy for TRAINING: pure IID random by default
    min_vp_scale: float = 0.05
    max_vp_scale: float = 1.0
    train_start_full: bool = False  # If True, t=0 is full scene. Val ALWAYS starts full.

    batch_size: int = 64
    eval_batch_size: int = 32
    num_workers: int = 4
    peak_lr: float = 1e-4
    min_lr: float = 1e-7
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.1
    max_steps: int = 5000
    grad_clip: float = 1.0
    focal_gamma: float = 2.0
    log_every: int = 20
    val_every: int = 500
    viz_every: int = 500
    viz_samples: int = 4
    comet_project: str = "canvit-ade20k-probe"
    comet_workspace: str = "m2b3-ava"
    device: str = "cuda"
    amp: bool = True
    probe_ckpt_dir: Path | None = None
