"""Loss functions for training."""

import torch.nn.functional as F
from torch import Tensor


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Mean squared error loss."""
    return F.mse_loss(pred, target)
