"""Configuration for scene matching training."""

from dataclasses import dataclass, field
from pathlib import Path

import torch
from ytch.device import get_sensible_device

from avp_vit import ActiveCanViTConfig
from avp_vit.train import LossType


@dataclass
class Config:
    # Teacher
    teacher_model: str = "dinov3_vits16"
    teacher_ckpt: Path = Path("dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
    # Student
    student_model: str = "dinov3_vitl16"
    student_ckpt: Path | None = None  # None = random init
    freeze_student_backbone: bool = False
    # Model
    model: ActiveCanViTConfig = field(default_factory=ActiveCanViTConfig)
    grid_size: int = 16
    batch_size: int = 128
    p_reset: float = 0.5  # Probability of resetting canvas each step
    ref_lr: float = 5e-7
    weight_decay: float = 0.05
    n_viewpoints_per_step: int = (
        2  # Inner loop viewpoints (>=2 for length generalization)
    )
    warmup_steps: int = 100_000
    grad_clip: float = 1.0
    # at 128 fresh images per optimizer step, that's
    # ~100 IN1k epochs
    # do take p_reset into account!
    n_steps: int = 2_000_000
    # Target normalization
    norm_warmup_images: int = 4096  # Images to warm up running stats before training
    norm_momentum: float = 0.1  # Momentum for running mean/var updates
    # Data
    train_dir: Path = Path("/datasets/ILSVRC/Data/CLS-LOC/train")
    val_dir: Path = Path("/datasets/ILSVRC/Data/CLS-LOC/val")
    index_dir: Path | None = (
        None  # If set, use IndexedImageFolder for train (needed for IN21k)
    )
    ckpt_dir: Path = Path("checkpoints")
    resume_ckpt: Path | None = None  # AVP checkpoint to resume from
    # Training
    num_workers: int = 8
    crop_scale_min: float = 0.8
    image_resolution: int = 512  # Source image size (independent of grid size)
    loss: LossType = "mse"
    # Logging
    log_every: int = 20
    val_every: int = 250
    total_viz: int = 1000  # PCA viz count (log-spaced, denser early)
    total_curves: int = 50  # Curve count (log-spaced)
    ckpt_every: int = 20_000
    log_spatial_stats: bool = True  # Log target/pred spatial mean/std
    # Compilation and precision
    compile: bool = True
    amp: bool = True  # bfloat16 automatic mixed precision
    # Optuna
    n_trials: int = 100
    # Runtime
    device: torch.device = field(default_factory=get_sensible_device)
