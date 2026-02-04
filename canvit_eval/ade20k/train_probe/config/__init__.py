"""ADE20K probe training configuration.

Defaults aligned with DINOv3's linear probing protocol (Appendix D.1).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

FeatureType = Literal["hidden", "predicted_norm", "teacher_glimpse", "teacher_full"]
STATIC_FEATURES: frozenset[FeatureType] = frozenset({"teacher_glimpse", "teacher_full"})
LossType = Literal["ce", "focal"]


@dataclass
class Config:
    """ADE20K probe training configuration."""

    ade20k_root: Path
    model_repo: str = "canvit/canvit-vitb16-pretrain-512px-in21k"
    features: list[FeatureType] = field(default_factory=lambda: ["hidden", "predicted_norm", "teacher_glimpse"])
    compute_teacher_full: bool = False  # Expensive: runs teacher on full 512px image
    n_timesteps: int = 10
    image_size: int = 512
    glimpse_px: int = 128

    # Viewpoint policy for TRAINING: pure IID random by default
    min_vp_scale: float = 0.05
    max_vp_scale: float = 1.0
    train_start_full: bool = False  # If True, t=0 is full scene. Val ALWAYS starts full.

    # Training hyperparameters (defaults match DINOv3 linear probing protocol)
    batch_size: int = 16  # DINOv3: 2 * 8 GPUs = 16
    eval_batch_size: int = 32
    num_workers: int = 4
    peak_lr: float = 1e-3  # DINOv3: 1e-3
    weight_decay: float = 1e-3  # DINOv3: 1e-3
    warmup_steps: int = 1500  # DINOv3: 1500
    warmup_lr_ratio: float = 1e-6  # DINOv3: start LR = peak_lr * 1e-6
    max_steps: int = 40000  # DINOv3: 40k iterations
    grad_clip: float = float("inf")  # DINOv3: no gradient clipping

    # Loss
    loss_type: LossType = "ce"  # DINOv3 uses CE; "focal" available for comparison
    focal_gamma: float = 2.0  # Only used if loss_type="focal"

    # Probe head
    dropout: float = 0.1  # DINOv3: 0.1 dropout in linear head

    # Data augmentation (DINOv3 defaults)
    aug_scale_range: tuple[float, float] = (0.5, 2.0)
    aug_flip_prob: float = 0.5

    # Logging
    log_every: int = 50
    val_every: int = 5000  # DINOv3: eval_interval=5000
    viz_every: int = 5000
    viz_samples: int = 4
    comet_project: str = "canvit-ade20k-probe"
    comet_workspace: str = "m2b3-ava"
    device: str = "cuda"
    amp: bool = True
    probe_ckpt_dir: Path | None = None
