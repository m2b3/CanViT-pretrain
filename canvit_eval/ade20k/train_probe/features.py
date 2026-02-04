"""Extracted features dataclass."""

from dataclasses import dataclass

from torch import Tensor

from .config import FeatureType


@dataclass
class ExtractedFeatures:
    """Features extracted from CanViT for probe training.

    Dynamic features (vary per timestep):
        hidden: raw canvas features [n_timesteps] x [B, H, W, D_canvas]
        predicted_norm: CanViT's teacher prediction [n_timesteps] x [B, H, W, D_teacher]

    Static features (same for all timesteps):
        teacher_glimpse: teacher on downscaled image [B, Hg, Wg, D_teacher]
        teacher_full: teacher on full image [B, H, W, D_teacher], None if disabled
    """

    hidden: list[Tensor]
    predicted_norm: list[Tensor]
    teacher_glimpse: Tensor
    teacher_full: Tensor | None

    def get(self, feat_type: FeatureType, t: int) -> Tensor | None:
        """Get feature at timestep t. Static features return same tensor for all t."""
        if feat_type == "hidden":
            return self.hidden[t]
        if feat_type == "predicted_norm":
            return self.predicted_norm[t]
        if feat_type == "teacher_glimpse":
            return self.teacher_glimpse
        if feat_type == "teacher_full":
            return self.teacher_full
        raise ValueError(f"Unknown feature type: {feat_type}")

    def is_available(self, feat_type: FeatureType) -> bool:
        """Check if a feature is available (teacher_full may be None)."""
        if feat_type == "teacher_full":
            return self.teacher_full is not None
        return True
