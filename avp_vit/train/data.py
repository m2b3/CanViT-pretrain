"""Data loading utilities for AVP training.

Single source of truth for ImageNet normalization constants.
"""

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, NamedTuple, TypeAlias

from torch import Tensor

if TYPE_CHECKING:
    from .config import Config
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from drac_imagenet import IndexedImageFolder

log = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ImageBatch: TypeAlias = tuple[Tensor, Tensor]  # (images, labels) batched


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


# IN21k contains some corrupt images that cause DataLoader workers to fail.
# Skip failed batches up to this limit before raising.
MAX_CONSECUTIVE_FAILURES = 10


class InfiniteLoader:
    """Infinite iterator over a DataLoader, yields images only."""

    def __init__(self, loader: DataLoader[ImageBatch]) -> None:
        self._gen = self._infinite(loader)

    def _infinite(self, loader: DataLoader[ImageBatch]) -> Iterator[ImageBatch]:
        while True:
            yield from loader

    def _next_with_retry(self) -> ImageBatch:
        failures = 0
        while True:
            try:
                batch = next(self._gen)
                return batch
            except StopIteration:
                raise
            except Exception as e:
                failures += 1
                log.warning(f"Batch failed ({failures}/{MAX_CONSECUTIVE_FAILURES}): {e}")
                if failures >= MAX_CONSECUTIVE_FAILURES:
                    raise RuntimeError(f"{MAX_CONSECUTIVE_FAILURES} consecutive batch failures") from e

    def next_batch(self) -> Tensor:
        """Get next batch of images (discards labels)."""
        imgs, _ = self._next_with_retry()
        return imgs

    def next_batch_with_labels(self) -> tuple[Tensor, Tensor]:
        """Get next batch of (images, labels)."""
        return self._next_with_retry()


class Loaders(NamedTuple):
    """Train and validation data loaders."""

    train: InfiniteLoader
    val: InfiniteLoader


def scene_size_px(grid_size: int, patch_size: int) -> int:
    return grid_size * patch_size


def create_loaders(cfg: "Config") -> Loaders:
    """Create train and validation data loaders."""
    from .config import Config
    assert isinstance(cfg, Config)

    train_dir, val_dir, index_dir = cfg.train_dir, cfg.val_dir, cfg.index_dir
    assert train_dir.is_dir(), f"train_dir not found: {train_dir}"
    assert val_dir.is_dir(), f"val_dir not found: {val_dir}"

    use_indexed = index_dir is not None
    if use_indexed:
        log.info(f"Using IndexedImageFolder for train (index_dir={index_dir})")

    sz = cfg.image_resolution
    train_tf = train_transform(sz, (cfg.crop_scale_min, 1.0))
    val_tf = val_transform(sz)
    log.info(f"Image resolution: {sz}px")

    if use_indexed:
        assert index_dir is not None
        train_ds: Dataset[tuple] = IndexedImageFolder(train_dir, index_dir, train_tf)
    else:
        train_ds = ImageFolder(str(train_dir), train_tf)
    val_ds: Dataset[tuple] = ImageFolder(str(val_dir), val_tf)

    assert len(train_ds) > 0, "train dataset empty"
    assert len(val_ds) > 0, "val dataset empty"
    log.info(f"Datasets: train={len(train_ds):,}, val={len(val_ds):,}")

    persistent = cfg.num_workers > 0
    train_loader = InfiniteLoader(DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True, persistent_workers=persistent,
    ))
    # CRITICAL: shuffle=True required for validation! Without it, batches are sequential
    # (all tench, then all goldfish, etc.) which gives misleading metrics due to class bias.
    val_loader = InfiniteLoader(DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True, persistent_workers=persistent,
    ))

    return Loaders(train=train_loader, val=val_loader)
