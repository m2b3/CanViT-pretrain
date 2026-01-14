"""Configuration for scene matching training."""

from dataclasses import dataclass, field
from pathlib import Path

import torch
from ytch.device import get_sensible_device

from avp_vit import ActiveCanViTConfig
from .loss import LossType


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
    glimpse_grid_size: int = 8  # tokens per glimpse side
    use_checkpointing: bool = True  # checkpoint odd steps in TBPTT chunks
    grid_size: int = 32  # canvas grid size
    # Training
    batch_size: int = 64
    warmup_steps: int = 100_000
    start_lr: float | None = 1e-7  # None = peak_lr / warmup_steps
    peak_lr: float = 4e-4
    # weight_decay: float = 0.05  # standard in ViTs
    # we can use a much lower weight decay due to the richness of our training signal
    # and we *should*, due to the use of small batches
    # 1e-3 has proven to be safe and work well in our early experiments in this project
    # 1e-4 was used by the AdaGlimpse authors
    weight_decay: float = 1e-3
    min_viewpoint_scale: float = 0.05  # Minimum scale for random viewpoints
    n_full_start_branches: int = 1  # branches starting with FULL viewpoint at t0
    n_random_start_branches: int = 1  # branches starting with RANDOM viewpoint at t0
    chunk_size: int = 2  # BPTT chunk size (glimpses per chunk, gradient flows within)
    continue_prob: float = 0.5  # peak prob of adding another chunk to trajectory
    continue_prob_warmup_steps: int = (
        100_000  # ramp 0 → continue_prob over this many steps
    )
    enable_policy: bool = False  # Enable policy branch (t=1 POLICY viewpoint type)
    enable_scene_patches_loss: bool = True  # Scene (canvas) patch reconstruction loss
    enable_scene_cls_loss: bool = True  # Scene (global) CLS reconstruction loss
    enable_glimpse_patches_loss: bool = False  # Glimpse patch reconstruction loss
    enable_glimpse_cls_loss: bool = False  # Glimpse CLS reconstruction loss
    scene_loss_type: LossType = LossType.MSE  # MSE or COSINE for scene losses
    ema_alpha: float = 0.1  # EMA smoothing for metrics
    grad_clip: float = 1.0
    policy_grad_clip: float = 1.0  # Separate clip for policy (applied first)
    # 4992 = 78 × 64 (batches_per_shard) - clean shard boundary alignment
    steps_per_job: int = 4_992  # Steps this job does before exiting (for SLURM arrays)
    # Data
    train_dir: Path = Path("/datasets/ILSVRC/Data/CLS-LOC/train")
    val_dir: Path = Path("/datasets/ILSVRC/Data/CLS-LOC/val")
    train_index_dir: Path | None = None  # Required for raw image training
    val_index_dir: Path | None = None  # Required for validation
    # Precomputed features (skips teacher inference on train images)
    # If feature_base_dir is set, shards path is auto-constructed:
    #   {feature_base_dir}/{teacher_model}/{image_resolution}/shards/
    feature_base_dir: Path | None = None
    feature_image_root: Path | None = None  # Required with feature_base_dir
    # Run identification and checkpointing
    run_name: str | None = None  # Auto-generated if None: YYYY-MM-DD_HH-MM
    ckpt_dir: Path = Path("checkpoints")
    resume_ckpt: Path | None = None  # Explicit checkpoint override (ignores run_name)
    force_new_experiment: bool = False  # Force new Comet experiment instead of continuing
    reset_policy: bool = False  # Reinitialize policy weights when resuming
    reset_opt_and_sched: bool = False  # Reset optimizer and scheduler on resume
    reset_normalizer: bool = False  # Re-warmup normalizer stats when resuming
    # Training
    num_workers: int = 16
    crop_scale_min: float = 0.8
    image_resolution: int = 512
    # Logging
    comet_project: str = "avp-vit-scene-match"
    log_every: int = 20
    val_every: int = 250
    n_eval_viewpoints: int = 10  # Number of viewpoints in validation (quadtree)
    viz_every_n_vals: int = 5  # Log viz every N validation runs
    curve_every_n_vals: int = 5  # Log curves every N validation runs
    ckpt_every: int = 4_992  # Match steps_per_job for clean shard boundaries
    log_spatial_stats: bool = True
    # Compilation and precision
    compile: bool = True
    amp: bool = True
    non_blocking_transfer: bool = True  # Ablation: async CPU→GPU transfers
    # Optuna
    n_trials: int = 100
    # Runtime
    device: torch.device = field(default_factory=get_sensible_device)
