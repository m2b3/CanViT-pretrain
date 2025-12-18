"""Loss functions for scene matching training."""

from collections.abc import Callable
from typing import Literal

from torch import Tensor
from torch.nn.functional import cosine_similarity, l1_loss, mse_loss


def cos_dissim(pred: Tensor, target: Tensor) -> Tensor:
    """Cosine dissimilarity: 1 - cosine_similarity, averaged over batch."""
    return (1 - cosine_similarity(pred, target, dim=-1)).mean()


LossFn = Callable[[Tensor, Tensor], Tensor]
LossType = Literal["l1", "mse", "cos"]

LOSS_FNS: dict[LossType, LossFn] = {
    "l1": l1_loss,
    "mse": mse_loss,
    "cos": cos_dissim,
}


def get_loss_fn(loss_type: LossType) -> LossFn:
    """Get loss function by name."""
    return LOSS_FNS[loss_type]
