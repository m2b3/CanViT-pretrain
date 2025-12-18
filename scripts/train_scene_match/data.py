"""Data loading with multi-resolution support."""

import logging
from dataclasses import dataclass

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from avp_vit.train import InfiniteLoader, train_transform, val_transform

from .config import Config

log = logging.getLogger(__name__)


@dataclass
class ResolutionStage:
    scene_grid_size: int
    glimpse_grid_size: int
    patch_size: int
    batch_size: int
    fresh_count: int
    n_viewpoints_per_step: int

    @property
    def scene_size_px(self) -> int:
        return self.scene_grid_size * self.patch_size

    @property
    def min_viewpoint_scale(self) -> float:
        return self.glimpse_grid_size / self.scene_grid_size

    @property
    def fresh_ratio(self) -> float:
        return self.fresh_count / self.batch_size


def create_resolution_stages(cfg: Config, patch_size: int) -> dict[int, ResolutionStage]:
    assert len(cfg.grid_sizes) > 0
    assert cfg.avp.glimpse_grid_size <= cfg.min_grid_size

    ratio = cfg.max_grid_size / cfg.min_grid_size
    bs_at_min = cfg.batch_size_at_min_grid or round(cfg.batch_size * ratio * ratio)

    log.info(f"Batch sizes: G={cfg.min_grid_size} -> {bs_at_min}, G={cfg.max_grid_size} -> {cfg.batch_size}")

    stages: dict[int, ResolutionStage] = {}
    for G in cfg.grid_sizes:
        # Linear interpolation in token count (G²)
        if cfg.min_grid_size == cfg.max_grid_size:
            bs = cfg.batch_size
        else:
            t = (G**2 - cfg.min_grid_size**2) / (cfg.max_grid_size**2 - cfg.min_grid_size**2)
            bs = max(2, round(bs_at_min + t * (cfg.batch_size - bs_at_min)))

        fresh_count = max(1, min(bs - 1, round(cfg.fresh_ratio * bs)))

        stages[G] = ResolutionStage(
            scene_grid_size=G,
            glimpse_grid_size=cfg.avp.glimpse_grid_size,
            patch_size=patch_size,
            batch_size=bs,
            fresh_count=fresh_count,
            n_viewpoints_per_step=cfg.n_viewpoints_per_step,
        )
        log.info(f"  G={G}: {stages[G].scene_size_px}px, batch={bs}, fresh={fresh_count}")

    return stages


def create_loaders(
    cfg: Config, stages: dict[int, ResolutionStage]
) -> tuple[dict[int, InfiniteLoader], dict[int, InfiniteLoader]]:
    assert cfg.train_dir.is_dir(), f"train_dir not found: {cfg.train_dir}"
    assert cfg.val_dir.is_dir(), f"val_dir not found: {cfg.val_dir}"

    train_loaders: dict[int, InfiniteLoader] = {}
    val_loaders: dict[int, InfiniteLoader] = {}

    for G, stage in stages.items():
        sz = stage.scene_size_px

        train_ds = ImageFolder(str(cfg.train_dir), train_transform(sz, (cfg.crop_scale_min, 1.0)))
        val_ds = ImageFolder(str(cfg.val_dir), val_transform(sz))

        assert len(train_ds) > 0, "train dataset empty"
        assert len(val_ds) > 0, "val dataset empty"
        log.info(f"  G={G}: train={len(train_ds):,}, val={len(val_ds):,}")

        persistent = cfg.num_workers > 0
        train_loaders[G] = InfiniteLoader(DataLoader(
            train_ds, batch_size=stage.fresh_count, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True, drop_last=True, persistent_workers=persistent,
        ))
        val_loaders[G] = InfiniteLoader(DataLoader(
            val_ds, batch_size=stage.fresh_count, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True, drop_last=True, persistent_workers=persistent,
        ))

    return train_loaders, val_loaders
