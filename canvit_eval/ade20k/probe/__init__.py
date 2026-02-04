"""Segmentation probe head for ADE20K.

Architecture: LN → BN → Dropout → Conv1x1

- LN: normalizes raw canvas features (DINOv3 uses backbone's final LN via use_backbone_norm=True)
- BN + Dropout + Conv1x1: matches DINOv3's LinearHead

Shape flow for 512x512 input with patch_size=16:
  - Input image: [B, 3, 512, 512]
  - Backbone patches: [B, 32, 32, D]  (512/16 = 32 patches per side)
  - Probe output: [B, 150, 32, 32]    (150 = NUM_CLASSES for ADE20k)
  - After rescale: [B, 150, H, W]     (H, W = original image size for metric computation)

The rescale is needed because:
  1. Ground truth masks are at full image resolution (e.g., 512x512)
  2. Probe outputs are at patch resolution (e.g., 32x32)
  3. For mIoU computation, predictions must match GT resolution
"""

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from canvit_eval.ade20k.dataset import NUM_CLASSES


class ProbeHead(nn.Module):
    """Linear probe: optional LN + DINOv3-style head (BN + Dropout + Conv1x1).

    Args:
        embed_dim: Feature dimension
        dropout: Dropout rate (0.1 matches DINOv3)
        use_ln: Whether to apply LayerNorm before BN. Enable for raw CanViT canvas
                features, disable for teacher features (already normalized).
    """

    def __init__(self, embed_dim: int, dropout: float, use_ln: bool) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ln = nn.LayerNorm(embed_dim) if use_ln else nn.Identity()
        self.bn = nn.BatchNorm2d(embed_dim)
        self.dropout = nn.Dropout2d(dropout)
        self.conv = nn.Conv2d(embed_dim, NUM_CLASSES, kernel_size=1)
        nn.init.normal_(self.conv.weight, mean=0, std=0.01)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass. Dropout active only in train mode (use model.eval() to disable).

        Args:
            x: [B, H_patches, W_patches, D] - spatial features in BHWD format

        Returns:
            [B, NUM_CLASSES, H_patches, W_patches] - class logits at patch resolution
        """
        B, Hp, Wp, D = x.shape
        assert D == self.embed_dim, f"Expected embed_dim={self.embed_dim}, got {D}"

        x = self.ln(x)                    # [B, Hp, Wp, D]
        x = x.permute(0, 3, 1, 2)         # [B, D, Hp, Wp]
        x = self.dropout(x)               # no-op in eval mode
        x = self.bn(x)                    # [B, D, Hp, Wp]
        x = self.conv(x)                  # [B, NUM_CLASSES, Hp, Wp]

        assert x.shape == (B, NUM_CLASSES, Hp, Wp)
        return x

    def predict(self, x: Tensor, rescale_to: tuple[int, int]) -> Tensor:
        """Forward + upsample. For DINOv3's slide_inference compatibility."""
        out = self(x)
        return F.interpolate(out, size=rescale_to, mode="bilinear", align_corners=False)
