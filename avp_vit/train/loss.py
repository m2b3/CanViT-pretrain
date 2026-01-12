"""Loss functions for training.

Provides LossType enum and dispatch for scene reconstruction losses.
"""

from enum import Enum

import torch.nn.functional as F
from torch import Tensor


class LossType(Enum):
    """Type of reconstruction loss."""

    MSE = "mse"
    COSINE = "cosine"


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Mean squared error loss."""
    return F.mse_loss(pred, target)


def cosine_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Cosine similarity loss: 1 - mean(cos_sim)."""
    return 1 - F.cosine_similarity(pred, target, dim=-1).mean()


# Dispatch table - clean, extensible, no inline ifs
_LOSS_FN = {
    LossType.MSE: mse_loss,
    LossType.COSINE: cosine_loss,
}


def reconstruction_loss(pred: Tensor, target: Tensor, loss_type: LossType) -> Tensor:
    """Compute reconstruction loss based on type."""
    return _LOSS_FN[loss_type](pred, target)
