"""ImageNet transforms for pretraining."""

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms


def imagenet_normalize() -> transforms.Normalize:
    """ImageNet normalization transform."""
    return transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)


def train_transform(size: int, crop_scale: tuple[float, float]) -> transforms.Compose:
    """Training transform: random crop + flip + normalize."""
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=crop_scale),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        imagenet_normalize(),
    ])
