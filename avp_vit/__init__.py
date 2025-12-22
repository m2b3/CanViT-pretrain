"""avp_vit: Active Vision Pretraining with ViT.

Re-exports from canvit for backward compatibility.
"""

from canvit import CanViT, CanViTConfig
from canvit.attention import CanvasAttentionConfig
from canvit.gram import gram_mse, spatial_gram
from canvit.model.active.pretraining import (
    ActiveCanViTForReconstructivePretraining as ActiveCanViT,
    ActiveCanViTForReconstructivePretrainingConfig as ActiveCanViTConfig,
    LossOutputs,
    PretrainingGlimpseOutput as StepOutput,
)

__all__ = [
    "ActiveCanViT",
    "ActiveCanViTConfig",
    "CanViT",
    "CanViTConfig",
    "CanvasAttentionConfig",
    "LossOutputs",
    "StepOutput",
    "gram_mse",
    "spatial_gram",
]
