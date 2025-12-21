"""Loss functions for scene matching training."""

from collections.abc import Callable
from typing import Literal

import torch
from torch import Tensor
from torch.nn.functional import cosine_similarity, l1_loss, mse_loss


def cos_dissim(pred: Tensor, target: Tensor) -> Tensor:
    """Cosine dissimilarity: 1 - cosine_similarity, averaged over batch."""
    return (1 - cosine_similarity(pred, target, dim=-1)).mean()


def spatial_gram(x: Tensor) -> Tensor:
    """Compute spatial Gram matrix: position-to-position correlations.

    Args:
        x: [B, N, D] tensor (N = spatial positions, D = features)

    Returns:
        [B, N, N] Gram matrix where entry [i,j] = dot(x[:,i,:], x[:,j,:]) / D
    """
    return torch.bmm(x, x.transpose(1, 2)) / x.size(-1)


def gram_mse(pred: Tensor, target: Tensor) -> Tensor:
    """MSE between spatial Gram matrices. Captures spatial covariance structure."""
    return mse_loss(spatial_gram(pred), spatial_gram(target))


LossFn = Callable[[Tensor, Tensor], Tensor]
LossType = Literal["l1", "mse", "cos", "gram"]

LOSS_FNS: dict[LossType, LossFn] = {
    "l1": l1_loss,
    "mse": mse_loss,
    "cos": cos_dissim,
    "gram": gram_mse,
}


def get_loss_fn(loss_type: LossType) -> LossFn:
    """Get loss function by name."""
    return LOSS_FNS[loss_type]
