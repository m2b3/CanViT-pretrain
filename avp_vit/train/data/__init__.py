"""Data loading utilities for AVP training.

Single source of truth for ImageNet normalization constants.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import TypeAlias

from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ImageBatch: TypeAlias = tuple[Tensor, Tensor]  # (images, labels) batched


def imagenet_normalize() -> transforms.Normalize:
    """ImageNet normalization transform."""
    return transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def train_transform(size: int, crop_scale: tuple[float, float]) -> transforms.Compose:
    """Training transform: random crop + flip + normalize.

    Args:
        size: Output image size (square).
        crop_scale: (min, max) scale range for RandomResizedCrop.
    """
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


def make_loader(
    root: Path,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader[ImageBatch]:
    """Create a DataLoader for ImageFolder dataset."""
    dataset = ImageFolder(str(root), transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


class InfiniteLoader:
    """Infinite iterator over a DataLoader, yields images only."""

    def __init__(self, loader: DataLoader[ImageBatch]) -> None:
        self._gen = self._infinite(loader)

    def _infinite(self, loader: DataLoader[ImageBatch]) -> Iterator[ImageBatch]:
        while True:
            yield from loader

    def next_batch(self) -> Tensor:
        """Get next batch of images (discards labels)."""
        imgs, _ = next(self._gen)
        return imgs

    def next_batch_with_labels(self) -> tuple[Tensor, Tensor]:
        """Get next batch of (images, labels)."""
        return next(self._gen)
