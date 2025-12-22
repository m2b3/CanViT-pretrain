"""Data loading."""

import logging

from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from avp_vit.train import InfiniteLoader, train_transform, val_transform
from drac_imagenet import IndexedImageFolder

from .config import Config

log = logging.getLogger(__name__)


def scene_size_px(grid_size: int, patch_size: int) -> int:
    return grid_size * patch_size


def create_loaders(cfg: Config) -> tuple[InfiniteLoader, InfiniteLoader]:
    assert cfg.train_dir.is_dir(), f"train_dir not found: {cfg.train_dir}"
    assert cfg.val_dir.is_dir(), f"val_dir not found: {cfg.val_dir}"

    use_indexed = cfg.index_dir is not None
    if use_indexed:
        log.info(f"Using IndexedImageFolder for train (index_dir={cfg.index_dir})")

    sz = cfg.image_resolution
    train_tf = train_transform(sz, (cfg.crop_scale_min, 1.0))
    val_tf = val_transform(sz)
    log.info(f"Image resolution: {sz}px")

    if use_indexed:
        assert cfg.index_dir is not None
        train_ds: Dataset[tuple] = IndexedImageFolder(cfg.train_dir, cfg.index_dir, train_tf)
    else:
        train_ds = ImageFolder(str(cfg.train_dir), train_tf)
    val_ds: Dataset[tuple] = ImageFolder(str(cfg.val_dir), val_tf)

    assert len(train_ds) > 0, "train dataset empty"
    assert len(val_ds) > 0, "val dataset empty"
    log.info(f"Datasets: train={len(train_ds):,}, val={len(val_ds):,}")

    persistent = cfg.num_workers > 0
    train_loader = InfiniteLoader(DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True, persistent_workers=persistent,
    ))
    val_loader = InfiniteLoader(DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True, persistent_workers=persistent,
    ))

    return train_loader, val_loader
