"""Configuration for scene matching training."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
from ytch.device import get_sensible_device

from avp_vit import AVPConfig
from avp_vit.attention import AttentionConfig


@dataclass
class Config:
    # Paths
    teacher_ckpt: Path = Path("dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
    train_dir: Path = Path("/datasets/ILSVRC/Data/CLS-LOC/train")
    val_dir: Path = Path("/datasets/ILSVRC/Data/CLS-LOC/val")
    ckpt_dir: Path = Path("checkpoints")
    # Model
    avp: AVPConfig = field(
        default_factory=lambda: AVPConfig(
            scene_grid_size=64,
            glimpse_grid_size=7,
            layer_scale_init=1e-2,
            temporal_gate_init=1e-3,
            use_output_proj=False,
            use_output_proj_norm=False,
            n_scene_registers=32,
            gradient_checkpointing=True,
            use_convex_gating=False,
            use_local_temporal=True,
            attention=AttentionConfig(write_v_expansion=2),
        )
    )
    freeze_inner_backbone: bool = True
    # Grid sizes (randomly sampled each step)
    grid_sizes: tuple[int, ...] = (16, 32, 64)
    # Training
    n_viewpoints_per_step: int = (
        2  # Inner loop viewpoints (>=2 for length generalization)
    )
    n_steps: int = 200000
    batch_size: int = 32  # Max batch size (at max grid size)
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
    ckpt_every: int = 500
    log_spatial_stats: bool = True  # Log target/pred spatial mean/std
    # Compilation
    compile: bool = True
    # Optuna
    n_trials: int = 100
    # Runtime
    device: torch.device = field(default_factory=get_sensible_device)

    @property
    def max_grid_size(self) -> int:
        return max(self.grid_sizes)
