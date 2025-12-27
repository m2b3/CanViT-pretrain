"""avp_vit: Active Vision Pretraining with ViT.

Re-exports from canvit for backward compatibility.
"""

from canvit import CanViT, CanViTConfig, GlimpseOutput, gram_mse, spatial_gram
from canvit.attention import CanvasAttentionConfig
from canvit.model.active.pretraining import (
    ActiveCanViTForReconstructivePretraining as ActiveCanViT,
    ActiveCanViTForReconstructivePretrainingConfig as ActiveCanViTConfig,
)

__all__ = [
    "ActiveCanViT",
    "ActiveCanViTConfig",
    "CanViT",
    "CanViTConfig",
    "CanvasAttentionConfig",
    "GlimpseOutput",
    "gram_mse",
    "spatial_gram",
]
