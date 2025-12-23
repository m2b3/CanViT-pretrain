"""Sample extraction for visualization (shared by train and validation)."""

from dataclasses import dataclass

import numpy as np
from canvit.model.active.base import GlimpseOutput
from torch import Tensor

from avp_vit import ActiveCanViT
from .image import imagenet_denormalize


@dataclass
class VizSampleData:
    """Viz data extracted for a single sample.

    Shape annotations:
        G = canvas grid size (e.g., 32)
        g = glimpse grid size (e.g., 3)
        D = teacher feature dim (e.g., 768)
        C = canvas hidden dim
    """

    glimpse: np.ndarray  # [g, g, 3] denormalized RGB
    predicted_scene: np.ndarray  # [G², D] teacher-space prediction
    canvas_spatial: np.ndarray  # [G², C] raw hidden state


def extract_sample0_viz(
    out: GlimpseOutput,
    predicted_scene: Tensor,
    model: ActiveCanViT,
) -> VizSampleData:
    """Extract viz data for sample 0, move to CPU as numpy."""
    glimpse_cpu = out.glimpse[0].cpu()
    glimpse_np = imagenet_denormalize(glimpse_cpu).numpy()

    scene_cpu = predicted_scene[0].cpu().float()
    scene_np = scene_cpu.numpy()

    canvas_single = out.canvas[0:1]
    spatial = model.get_spatial(canvas_single)[0]
    spatial_np = spatial.cpu().float().numpy()

    return VizSampleData(
        glimpse=glimpse_np,
        predicted_scene=scene_np,
        canvas_spatial=spatial_np,
    )
