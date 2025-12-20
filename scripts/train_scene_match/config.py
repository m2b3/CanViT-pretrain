"""Configuration for scene matching training."""

from dataclasses import dataclass, field
from pathlib import Path

import torch
from ytch.device import get_sensible_device

from avp_vit import ActiveCanViTConfig
from avp_vit.train import LossType
from avp_vit.train.viewpoint import ViewpointScaleConfig


@dataclass
class Config:
    # Teacher
    teacher_model: str = "dinov3_vits16"
    teacher_ckpt: Path = Path("dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
    # Student
    student_model: str = "dinov3_vits16"
    student_ckpt: Path | None = None  # None = random init
    freeze_student_backbone: bool = False
    # Model
    model: ActiveCanViTConfig = field(default_factory=ActiveCanViTConfig)
    grid_sizes: tuple[int, ...] = (16,)
    batch_size: int = 128  # Batch size at max grid size
    batch_size_at_min_grid: int | None = (
        None  # If set, linearly interpolate BS between grids
    )
    # Fraction of batch replaced each optimizer step
    # Having this NOT be 1.0 makes optimization a lot harder...
    # It could make sense to first pretrain with this at 1.0, maybe adjust it later...
    # There's an enough complexity
    fresh_ratio: float = 1.0
    ref_lr: float = 1e-5
    weight_decay: float = 1e-5
    n_viewpoints_per_step: int = (
        2  # Inner loop viewpoints (>=2 for length generalization)
    )
    viewpoint_scale: ViewpointScaleConfig = field(default_factory=ViewpointScaleConfig)
    warmup_steps: int = 100_000
    grad_clip: float = 1.0
    # at 128 fresh images per optimizer step, that's
    # ~100 IN1k epochs
    n_steps: int = 1_000_000
    # Target normalization
    norm_warmup_images: int = 256  # Images to warm up running stats before training
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
    loss: LossType = "cos"
    # Logging
    log_every: int = 20
    val_every: int = 250
    viz_every: int = 500  # PCA viz (expensive) less often than val
    curve_every: int = 5000  # Curves less often than val (Comet limit: 1000/experiment)
    ckpt_every: int = 5000
    log_spatial_stats: bool = True  # Log target/pred spatial mean/std
    # Compilation
    compile: bool = True
    # Optuna
    n_trials: int = 100
    # Runtime
    device: torch.device = field(default_factory=get_sensible_device)

    @property
    def min_grid_size(self) -> int:
        return min(self.grid_sizes)

    @property
    def max_grid_size(self) -> int:
        return max(self.grid_sizes)
