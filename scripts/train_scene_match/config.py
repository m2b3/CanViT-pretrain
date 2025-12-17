"""Configuration for scene matching training."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
from ytch.device import get_sensible_device

from avp_vit import AVPConfig


@dataclass
class Config:
    # Teacher
    teacher_model: str = "dinov3_vits16"
    teacher_ckpt: Path = Path("dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
    # Student
    student_model: str = "dinov3_vits16"
    student_ckpt: Path | None = None  # None = random init
    freeze_student_backbone: bool = False
    # AVP
    avp: AVPConfig = field(default_factory=AVPConfig)
    grid_sizes: tuple[int, ...] = (16, 32, 64)
    # Data
    train_dir: Path = Path("/datasets/ILSVRC/Data/CLS-LOC/train")
    val_dir: Path = Path("/datasets/ILSVRC/Data/CLS-LOC/val")
    ckpt_dir: Path = Path("checkpoints")
    # Training
    n_viewpoints_per_step: int = (
        2  # Inner loop viewpoints (>=2 for length generalization)
    )
    n_steps: int = 200000
    batch_size: int = 16  # Max batch size (at max grid size)
    num_workers: int = 8
    ref_lr: float = 1e-5
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    crop_scale_min: float = 0.4
    loss: Literal["l1", "mse"] = "mse"
    # Logging
    log_every: int = 20
    val_every: int = 50
    curve_every: int = 1000  # Curves less often than val (Comet limit: 1000/experiment)
    ckpt_every: int = 500
    log_spatial_stats: bool = True  # Log target/pred spatial mean/std
    # Compilation
    compile: bool = True
    # Optuna
    n_trials: int = 100
    # Debug
    debug_train_on_single_batch: bool = False  # Train on single repeated batch (for overfitting test)
    # Runtime
    device: torch.device = field(default_factory=get_sensible_device)

    @property
    def max_grid_size(self) -> int:
        return max(self.grid_sizes)
