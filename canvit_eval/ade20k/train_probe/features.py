"""Feature extraction for ADE20K probes."""

from dataclasses import dataclass

import torch.nn.functional as F
from canvit import CanViTForPretrainingHFHub, Viewpoint, sample_at_viewpoint
from canvit.backbone.dinov3 import DINOv3Backbone
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


def extract_features(
    *,
    model: CanViTForPretrainingHFHub,
    teacher: DINOv3Backbone,
    images: Tensor,
    canvas_grid: int,
    glimpse_px: int,
    viewpoints: list[Viewpoint] | None,
    compute_teacher_full: bool,
) -> ExtractedFeatures:
    """Extract features from CanViT and teacher for probe training/evaluation.

    Args:
        model: CanViT model (frozen)
        teacher: Teacher backbone (frozen)
        images: Input images [B, 3, H, W]
        canvas_grid: Canvas grid size (H/patch_size)
        glimpse_px: Glimpse size in pixels
        viewpoints: List of viewpoints for CanViT rollout. None = skip CanViT (static features only).
        compute_teacher_full: Whether to compute teacher features at full resolution.
    """
    B = images.shape[0]
    hidden_list: list[Tensor] = []
    predicted_list: list[Tensor] = []

    # Static: teacher on downscaled image (glimpse resolution)
    glimpse_grid = glimpse_px // teacher.patch_size_px
    sz = glimpse_grid * teacher.patch_size_px
    small = F.interpolate(images, size=(sz, sz), mode="bilinear", align_corners=False)
    teacher_glimpse = teacher.forward_norm_features(small).patches.view(B, glimpse_grid, glimpse_grid, -1)

    # Static: teacher on full image (expensive, optional)
    teacher_full: Tensor | None = None
    if compute_teacher_full:
        teacher_full = teacher.forward_norm_features(images).patches.view(B, canvas_grid, canvas_grid, -1)

    # Dynamic: CanViT rollout (skip if only static features needed)
    if viewpoints is not None:
        state = model.init_state(batch_size=B, canvas_grid_size=canvas_grid)
        for vp in viewpoints:
            glimpse = sample_at_viewpoint(spatial=images, viewpoint=vp, glimpse_size_px=glimpse_px)
            out = model(glimpse=glimpse, state=state, viewpoint=vp)
            state = out.state

            hidden = model.get_spatial(state.canvas).view(B, canvas_grid, canvas_grid, -1)
            predicted = model.predict_teacher_scene(state.canvas).view(B, canvas_grid, canvas_grid, -1)

            hidden_list.append(hidden)
            predicted_list.append(predicted)

    return ExtractedFeatures(
        hidden=hidden_list,
        predicted_norm=predicted_list,
        teacher_glimpse=teacher_glimpse,
        teacher_full=teacher_full,
    )
