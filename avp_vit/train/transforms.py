"""ImageNet normalization and transforms.

Single source of truth for normalization constants used in training and validation.
"""

from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def imagenet_normalize() -> transforms.Normalize:
    """ImageNet normalization transform."""
    return transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def train_transform(size: int, crop_scale: tuple[float, float]) -> transforms.Compose:
    """Training transform: random crop + flip + normalize."""
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=crop_scale),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        imagenet_normalize(),
    ])


def val_transform(size: int) -> transforms.Compose:
    """Validation transform: center crop + normalize."""
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        imagenet_normalize(),
    ])
