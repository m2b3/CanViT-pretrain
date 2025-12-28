"""Shared dataclasses for inference app. CPU-only - no torch tensors."""

from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA


@dataclass
class StepResult:
    """Result of one model inference step. All numpy, no torch."""
    hidden: np.ndarray  # [N, D] hidden features
    projected: np.ndarray  # [N, D] projected to teacher space
    glimpse: np.ndarray  # [H, W, 3] uint8
    scene_cos: float | None = None
    cls_cos: float | None = None
    ms: float = 0.0
    top5: list[tuple[str, float]] = field(default_factory=list)
    policy_center: tuple[float, float] | None = None  # (cy, cx) normalized
    policy_scale: float | None = None


@dataclass
class TeacherFeatures:
    """Teacher output. CPU numpy only."""
    scene: np.ndarray | None  # [N, D]
    cls_features: np.ndarray | None  # [D]
    top5: list[tuple[str, float]]
    ms: float
    grid: int


@dataclass
class ImageContext:
    """Per-image context. CPU only."""
    pil: Image.Image  # display image
    H: int
    W: int
    teacher_full: TeacherFeatures
    teacher_glimpse: TeacherFeatures | None
    pca_full: PCA | None
    pca_glimpse: PCA | None


@dataclass
class Viewpoint:
    """Viewpoint for a step. Simple Python types."""
    cy: float  # center y, normalized [-1, 1]
    cx: float  # center x, normalized [-1, 1]
    scale: float  # scale in (0, 1]
    name: str = ""


@dataclass
class SequenceState:
    """Current sequence state."""
    viewpoints: list[Viewpoint]
    results: list[StepResult]
    step_count: int = 0
