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
    freeze_student_backbone: bool = False
    # Model config (PretrainingConfig via alias)
    # teacher_dim placeholder - overridden by create_model based on actual teacher
    model: ActiveCanViTConfig = field(
        default_factory=lambda: ActiveCanViTConfig(teacher_dim=768)
    )
    # Glimpse/canvas sizes (runtime, not in model config)
    gram_loss_weight: float = 0
    include_init: bool = False  # include initial canvas in scene/cls loss
    glimpse_grid_size: int = 3  # tokens per glimpse side
    grid_size: int = 32  # canvas grid size
    # Training
    batch_size: int = 128
    peak_lr: float = 5e-4
    weight_decay: float = 0.05  # standard in ViTs
    n_viewpoints_per_step: int = 2  # Inner loop viewpoints
    min_viewpoint_scale: float = 0.05  # Minimum scale for random viewpoints
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
    # Training
    num_workers: int = 8
    crop_scale_min: float = 0.8
    image_resolution: int = 512
    # Logging
    log_every: int = 20
    val_every: int = 250
    total_viz: int = 1000
    total_curves: int = 300  # ~3 curves/event, budget 900
    ckpt_every: int = 20_000
    log_spatial_stats: bool = True
    # Compilation and precision
    compile: bool = True
    amp: bool = True
    # Optuna
    n_trials: int = 100
    # Runtime
    device: torch.device = field(default_factory=get_sensible_device)
