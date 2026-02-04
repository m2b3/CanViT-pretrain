"""Loss functions for ADE20K probe training."""

import torch.nn.functional as F
from torch import Tensor

from canvit_eval.ade20k.dataset import IGNORE_LABEL


def ce_loss(logits: Tensor, masks: Tensor) -> Tensor:
    """Cross-entropy loss for semantic segmentation (matches DINOv3)."""
    if masks.shape[1:] != logits.shape[2:]:
        masks = F.interpolate(masks.unsqueeze(1).float(), logits.shape[2:], mode="nearest").squeeze(1).long()
    return F.cross_entropy(logits, masks, ignore_index=IGNORE_LABEL)


def focal_loss(logits: Tensor, masks: Tensor, gamma: float) -> Tensor:
    """Focal loss for semantic segmentation."""
    B, C, Hl, Wl = logits.shape
    if masks.shape[1:] != (Hl, Wl):
        masks = F.interpolate(masks.unsqueeze(1).float(), (Hl, Wl), mode="nearest").squeeze(1).long()
    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp()
    valid = masks != IGNORE_LABEL
    targets = masks.clamp(0, C - 1)
    log_p = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    p = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    return -((1 - p) ** gamma * log_p * valid).sum() / valid.sum().clamp(min=1)


def upsample_preds(preds: Tensor, H: int, W: int) -> Tensor:
    """Upsample predictions to target resolution."""
    if preds.shape[1:] == (H, W):
        return preds
    return F.interpolate(preds.unsqueeze(1).float(), (H, W), mode="nearest").squeeze(1).long()
