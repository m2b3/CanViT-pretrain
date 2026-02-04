"""Image transformation utilities for visualization."""

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import Tensor


def imagenet_denormalize(t: Tensor) -> Tensor:
    """Convert [3, H, W] ImageNet-normalized tensor to [H, W, 3] in [0, 1].

    Args:
        t: [3, H, W] tensor (caller handles device placement)

    Returns:
        [H, W, 3] tensor on same device as input
    """
    mean = t.new_tensor(IMAGENET_DEFAULT_MEAN).view(3, 1, 1)
    std = t.new_tensor(IMAGENET_DEFAULT_STD).view(3, 1, 1)
    return ((t * std + mean).clamp(0, 1)).permute(1, 2, 0)
