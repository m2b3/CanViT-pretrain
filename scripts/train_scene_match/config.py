"""Configuration for scene matching training."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
from ytch.device import get_sensible_device

from avp_vit import AVPConfig


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
            scene_grid_size=64,  # Max grid size for curriculum
            glimpse_grid_size=7,
            layer_scale_init=1e-2,
            use_output_proj=True,
            n_scene_registers=32,
            gradient_checkpointing=True,
            use_convex_gating=True,
            use_local_temporal=True,
        )
    )
    freeze_inner_backbone: bool = False
    # Curriculum (small → large for main training)
    grid_sizes: tuple[int, ...] = (16, 32, 64)
    # Training
    n_viewpoints_per_step: int = 2  # Inner loop viewpoints (>=2 for length generalization)
    n_steps: int = 200000
    batch_size: int = 32  # Max batch size (at max grid size)
    num_workers: int = 8
    ref_lr: float = 1e-5
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.01  # Fraction of n_steps for warmup phase
    grad_clip: float = 1.0
    crop_scale_min: float = 0.4
    loss: Literal["l1", "mse"] = "mse"
    # Logging
    log_every: int = 20
    val_every: int = 50
    ckpt_every: int = 500
    # Compilation
    compile: bool = True
    # Optuna
    n_trials: int = 100
    # Runtime
    device: torch.device = field(default_factory=get_sensible_device)

    @property
    def max_grid_size(self) -> int:
        return max(self.grid_sizes)

    @property
    def warmup_steps(self) -> int:
        """Total warmup steps (cycles through all sizes in reverse order)."""
        return int(self.n_steps * self.warmup_ratio)

    @property
    def warmup_steps_per_size(self) -> int:
        """Steps per grid size during warmup phase."""
        return self.warmup_steps // len(self.grid_sizes)

    @property
    def main_training_steps(self) -> int:
        """Steps for main curriculum training (after warmup)."""
        return self.n_steps - self.warmup_steps

    @property
    def main_steps_per_stage(self) -> int:
        """Steps per curriculum stage during main training."""
        return self.main_training_steps // len(self.grid_sizes)

    @property
    def warmup_grid_sizes(self) -> tuple[int, ...]:
        """Grid sizes for warmup phase (largest first for OOM detection)."""
        return tuple(reversed(self.grid_sizes))

    def get_schedule(self) -> list[tuple[str, int, int, int]]:
        """Return full training schedule as list of (phase, grid_size, start_step, end_step).

        Warmup phase cycles through all sizes (largest first) with mini LR cycles.
        Main training follows curriculum order (smallest to largest).
        """
        schedule: list[tuple[str, int, int, int]] = []

        # Warmup phase (largest → smallest)
        step = 0
        for G in self.warmup_grid_sizes:
            end = step + self.warmup_steps_per_size - 1
            schedule.append(("warmup", G, step, end))
            step = end + 1

        # Main training (smallest → largest)
        for i, G in enumerate(self.grid_sizes):
            start = self.warmup_steps + i * self.main_steps_per_stage
            end = start + self.main_steps_per_stage - 1
            if i == len(self.grid_sizes) - 1:
                end = self.n_steps - 1  # Last stage gets remaining steps
            schedule.append(("main", G, start, end))

        return schedule

    def get_phase_and_grid(self, step: int) -> tuple[str, int]:
        """Return (phase, grid_size) for a given step."""
        for phase, G, start, end in self.get_schedule():
            if start <= step <= end:
                return phase, G
        return "main", self.grid_sizes[-1]  # Fallback
