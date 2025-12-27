"""Configuration for scene matching training."""

from dataclasses import dataclass, field
from pathlib import Path

import torch
from ytch.device import get_sensible_device

from avp_vit import ActiveCanViTConfig


@dataclass
class Config:
    # Teacher
    teacher_model: str = "dinov3_vitb16"
    teacher_ckpt: Path = Path("dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
    # Student
    student_model: str = "dinov3_vitb16"
    student_ckpt: Path | None = None  # None = random init
    # Model config (PretrainingConfig via alias)
    # teacher_dim placeholder - overridden by create_model based on actual teacher
    model: ActiveCanViTConfig = field(
        default_factory=lambda: ActiveCanViTConfig(teacher_dim=768)
    )
    # Glimpse/canvas sizes (runtime, not in model config)
    gram_loss_weight: float = 0
    glimpse_grid_size: int = 8  # tokens per glimpse side
    grid_size: int = 32  # canvas grid size
    # Training
    batch_size: int = 64
    peak_lr: float = 1e-4
    start_lr: float | None = 1e-8  # None = peak_lr / warmup_steps (old behavior)
    end_lr: float | None = 1e-6  # None = 0 (old behavior)
    # weight_decay: float = 0.05  # standard in ViTs
    # we can use a much lower weight decay due to the richness of our training signal
    # and we *should*, due to the use of small batches
    # 1e-3 has proven to be safe and work well in our early experiments in this project
    # 1e-4 was used by the AdaGlimpse authors
    weight_decay: float = 1e-4
    min_viewpoint_scale: float = 0.05  # Minimum scale for random viewpoints
    enable_policy: bool = True  # Enable policy branch (t=1 POLICY viewpoint type)
    ema_alpha: float = 0.1  # EMA smoothing for metrics
    warmup_steps: int = 100_000
    grad_clip: float = 1.0
    n_steps: int = 500_000
    # Target normalization
    norm_warmup_images: int = 4096
    norm_momentum: float = 0.1
    # Data
    train_dir: Path = Path("/datasets/ILSVRC/Data/CLS-LOC/train")
    val_dir: Path = Path("/datasets/ILSVRC/Data/CLS-LOC/val")
    index_dir: Path | None = None
    ckpt_dir: Path = Path("checkpoints")
    resume_ckpt: Path | None = None
    reset_policy: bool = False  # Reinitialize policy weights when resuming
    # Training
    num_workers: int = 8
    crop_scale_min: float = 0.8
    image_resolution: int = 512
    # Logging
    log_every: int = 20
    val_every: int = 250
    n_eval_viewpoints: int = 10  # Number of viewpoints in validation (quadtree)
    total_viz: int = 1000
    total_curves: int = 300  # ~3 curves/event, budget 900
    ckpt_every: int = 10_000
    log_spatial_stats: bool = True
    # Compilation and precision
    compile: bool = True
    amp: bool = True
    # Optuna
    n_trials: int = 100
    # Runtime
    device: torch.device = field(default_factory=get_sensible_device)
