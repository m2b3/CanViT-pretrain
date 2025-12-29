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
        default_factory=lambda: ActiveCanViTConfig(teacher_dim=768, canvas_num_heads=8)
    )
    # Glimpse/canvas sizes (runtime, not in model config)
    glimpse_grid_size: int = 8  # tokens per glimpse side
    use_checkpointing: bool = True  # checkpoint odd steps in TBPTT chunks
    grid_size: int = 32  # canvas grid size
    # Training
    batch_size: int = 64
    warmup_steps: int = 100_000
    start_lr: float | None = 1e-7  # None = peak_lr / warmup_steps (old behavior)
    peak_lr: float = 5e-4
    end_lr: float | None = 1e-6  # None = 0 (old behavior)
    # weight_decay: float = 0.05  # standard in ViTs
    # we can use a much lower weight decay due to the richness of our training signal
    # and we *should*, due to the use of small batches
    # 1e-3 has proven to be safe and work well in our early experiments in this project
    # 1e-4 was used by the AdaGlimpse authors
    weight_decay: float = 1e-3
    min_viewpoint_scale: float = 0.05  # Minimum scale for random viewpoints
    n_branches: int = (
        2  # must be >= 2 and even; K/2 RANDOM/FULL at t0, K/2 RANDOM/POLICY at t>=1
    )
    min_glimpses: int = 2  # minimum trajectory length (>= 2)
    continue_prob: float = 0.5  # peak prob of continuing past min_glimpses
    continue_prob_warmup_steps: int = (
        100_000  # ramp 0 → continue_prob over this many steps
    )
    enable_policy: bool = False  # Enable policy branch (t=1 POLICY viewpoint type)
    enable_glimpse_losses: bool = (
        True  # Enable glimpse losses (direct backbone gradient)
    )
    ema_alpha: float = 0.1  # EMA smoothing for metrics
    grad_clip: float = 1.0
    policy_grad_clip: float = 1.0  # Separate clip for policy (applied first)
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
    reset_opt_and_sched: bool = (
        True  # Reset both optimizer and scheduler (tied together)
    )
    reset_normalizer: bool = False  # Re-warmup normalizer stats when resuming
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
