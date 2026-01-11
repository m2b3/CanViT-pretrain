"""Data loading utilities for AVP training.

Single source of truth for ImageNet normalization constants and transforms.
"""

import logging
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, TypeAlias

from torch import Tensor

if TYPE_CHECKING:
    from .config import Config
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from drac_imagenet import IndexedImageFolder

log = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

Batch: TypeAlias = tuple[Tensor, ...]  # Generic batch (images, labels, ...)


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
    """Infinite iterator over a DataLoader with retry on worker errors.

    Note: We use explicit iterator management instead of a generator because
    when an exception propagates out of a Python generator, the generator is
    finalized (gi_frame=None) and subsequent next() calls raise StopIteration.
    """

    def __init__(self, loader: DataLoader) -> None:
        self._loader = loader
        self._iter: Iterator[Batch] | None = None

    def _next_with_retry(self) -> Batch:
        failures = 0
        while True:
            if self._iter is None:
                self._iter = iter(self._loader)
            try:
                return next(self._iter)
            except StopIteration:
                # End of epoch - start new one
                self._iter = iter(self._loader)
            except Exception as e:
                failures += 1
                log.warning(f"Batch failed ({failures}/{MAX_CONSECUTIVE_FAILURES}): {e}")
                if failures >= MAX_CONSECUTIVE_FAILURES:
                    raise RuntimeError(f"{MAX_CONSECUTIVE_FAILURES} consecutive batch failures") from e
                # Worker error corrupts iterator state - reset it
                self._iter = None

    def next(self) -> Batch:
        """Get next batch (raw tuple from DataLoader)."""
        return self._next_with_retry()

    def next_batch(self) -> Tensor:
        """Get images only (first element of batch)."""
        images, *_ = self._next_with_retry()
        return images

    def next_batch_with_labels(self) -> tuple[Tensor, Tensor]:
        """Get (images, labels) - for raw image loaders."""
        batch = self._next_with_retry()
        return batch[0], batch[1]


class Loaders(NamedTuple):
    """Train and validation data loaders."""

    train: InfiniteLoader
    val: InfiniteLoader


def scene_size_px(grid_size: int, patch_size: int) -> int:
    return grid_size * patch_size


def create_loaders(cfg: "Config") -> Loaders:
    """Create train and validation data loaders.

    If cfg.feature_base_dir is set, train loader uses precomputed features.
    Val loader always uses raw images.
    """
    from .config import Config
    assert isinstance(cfg, Config)

    val_dir = cfg.val_dir
    assert val_dir.is_dir(), f"val_dir not found: {val_dir}"

    sz = cfg.image_resolution
    persistent = cfg.num_workers > 0

    # Train loader: features or raw images
    if cfg.feature_base_dir is not None:
        log.info("Train: using PRECOMPUTED FEATURES")
        assert cfg.feature_image_root is not None, "feature_image_root required with feature_base_dir"
        # Construct shards path from config: {base}/{teacher_model}/{resolution}/shards/
        shards_dir = cfg.feature_base_dir / cfg.teacher_model / str(sz) / "shards"
        log.info(f"  feature_base_dir: {cfg.feature_base_dir}")
        log.info(f"  teacher_model: {cfg.teacher_model}")
        log.info(f"  resolution: {sz}")
        log.info(f"  → shards_dir: {shards_dir}")
        log.info(f"  image_root: {cfg.feature_image_root}")
        assert shards_dir.is_dir(), f"shards_dir not found: {shards_dir}"
        from .feature_dataset import FeatureIterableDataset
        train_ds = FeatureIterableDataset(shards_dir, cfg.feature_image_root)
        log.info(f"  {len(train_ds.shard_files)} shards")
        train_loader = InfiniteLoader(DataLoader(
            train_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
            pin_memory=True, drop_last=True, persistent_workers=persistent,
        ))
    else:
        log.info("Train: using RAW IMAGES (teacher inference at runtime)")
        train_dir, train_index_dir = cfg.train_dir, cfg.train_index_dir
        assert train_dir.is_dir(), f"train_dir not found: {train_dir}"
        assert train_index_dir is not None, "train_index_dir required for raw image training"
        log.info(f"  train_dir: {train_dir}")
        log.info(f"  index_dir: {train_index_dir}")
        train_tf = train_transform(sz, (cfg.crop_scale_min, 1.0))
        train_ds_img: Dataset[tuple] = IndexedImageFolder(train_dir, train_index_dir, train_tf)
        assert len(train_ds_img) > 0, "train dataset empty"
        log.info(f"  {len(train_ds_img):,} images")
        train_loader = InfiniteLoader(DataLoader(
            train_ds_img, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True, drop_last=True, persistent_workers=persistent,
        ))

    # Val loader
    val_tf = val_transform(sz)
    if cfg.val_index_dir is not None:
        val_index_dir = cfg.val_index_dir
        log.info(f"Val: using provided val_index_dir={val_index_dir}")
    elif cfg.train_index_dir is not None:
        val_index_dir = cfg.train_index_dir
        log.info(f"Val: val_index_dir not set, using train_index_dir={val_index_dir}")
    else:
        val_index_dir = Path(tempfile.mkdtemp(prefix="avp_val_index_"))
        log.info(f"Val: no index_dir available, using temp dir: {val_index_dir}")
    val_ds: Dataset[tuple] = IndexedImageFolder(val_dir, val_index_dir, val_tf)
    assert len(val_ds) > 0, "val dataset empty"
    log.info(f"Val dataset: {len(val_ds):,} images, resolution: {sz}px")
    # CRITICAL: shuffle=True required! Without it, batches are sequential
    # (all tench, then all goldfish, etc.) which gives misleading metrics.
    val_loader = InfiniteLoader(DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True, persistent_workers=persistent,
    ))

    return Loaders(train=train_loader, val=val_loader)
