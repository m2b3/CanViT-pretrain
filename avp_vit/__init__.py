"""avp_vit: Active Vision Pretraining with ViT."""

from canvit.model.pretraining import CanViTForPretraining, CanViTForPretrainingConfig

# Short aliases for the pretraining model
ACVFRP = CanViTForPretraining
ACVFRPConfig = CanViTForPretrainingConfig

__all__ = [
    "ACVFRP",
    "ACVFRPConfig",
    "CanViTForPretraining",
    "CanViTForPretrainingConfig",
]
