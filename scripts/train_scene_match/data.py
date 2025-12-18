"""Data loading with multi-resolution support."""

import logging
from dataclasses import dataclass
from typing import Any

from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from avp_vit.train import InfiniteLoader, train_transform, val_transform

from .config import Config

log = logging.getLogger(__name__)


@dataclass
class ResolutionStage:
    """Configuration for one resolution stage."""

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
    def n_scene_tokens(self) -> int:
        return self.scene_grid_size**2

    @property
    def min_viewpoint_scale(self) -> float:
        return self.glimpse_grid_size / self.scene_grid_size

    @property
    def n_eval(self) -> int:
        return _n_eval_viewpoints(self.scene_grid_size)

    @property
    def fresh_ratio_desired(self) -> float:
        return self.n_viewpoints_per_step / self.n_eval

    @property
    def fresh_ratio_actual(self) -> float:
        return self.fresh_count / self.batch_size

    @property
    def e_glimpses_desired(self) -> float:
        return self.n_viewpoints_per_step / self.fresh_ratio_desired

    @property
    def e_glimpses_actual(self) -> float:
        return self.n_viewpoints_per_step / self.fresh_ratio_actual


def _n_eval_viewpoints(grid_size: int) -> int:
    """Expected viewpoints for evaluation. Linear in G since glimpses reuse hidden state."""
    return max(4, round(grid_size / 16))


def _batch_size_for_grid(
    grid_size: int,
    min_grid_size: int,
    max_grid_size: int,
    bs_at_min: int,
    bs_at_max: int,
) -> int:
    """Linear interpolation in token count (G²), i.e., quadratic in G."""
    if min_grid_size == max_grid_size:
        return bs_at_max
    T = grid_size ** 2
    T_min, T_max = min_grid_size ** 2, max_grid_size ** 2
    t = (T - T_min) / (T_max - T_min)
    return max(1, round(bs_at_min + t * (bs_at_max - bs_at_min)))


def _fresh_ratio(grid_size: int, n_viewpoints_per_step: int) -> float:
    """Fresh ratio to achieve expected viewpoint count: E[glimpses] = n_vp / fresh_ratio."""
    return n_viewpoints_per_step / _n_eval_viewpoints(grid_size)


def create_resolution_stage(
    scene_grid_size: int,
    glimpse_grid_size: int,
    patch_size: int,
    batch_size: int,
    n_viewpoints_per_step: int,
) -> ResolutionStage:
    """Create a resolution stage with given batch size."""
    fresh_count = max(
        1, round(_fresh_ratio(scene_grid_size, n_viewpoints_per_step) * batch_size)
    )
    return ResolutionStage(
        scene_grid_size=scene_grid_size,
        glimpse_grid_size=glimpse_grid_size,
        patch_size=patch_size,
        batch_size=batch_size,
        fresh_count=fresh_count,
        n_viewpoints_per_step=n_viewpoints_per_step,
    )


class SingleBatchDataset(Dataset[tuple[Tensor, int]]):
    """Wraps a dataset, caches first N items and cycles through them.

    This ensures diverse samples within each batch (for correct variance estimation
    in batch normalization) while keeping the same batch every time (for debugging).
    """

    def __init__(self, dataset: Dataset[Any], n: int) -> None:
        self._items = [dataset[i] for i in range(n)]
        log.warning(f"Cached {n} images for single-batch training")

    def __len__(self) -> int:
        return 1_000_000

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        return self._items[idx % len(self._items)]


def create_resolution_stages(
    cfg: Config, patch_size: int
) -> dict[int, ResolutionStage]:
    """Create resolution stages for each grid size."""
    # Default: old quadratic behavior (bs ∝ 1/G²)
    ratio = cfg.max_grid_size / cfg.min_grid_size
    bs_at_min = cfg.batch_size_at_min_grid or round(cfg.batch_size * ratio * ratio)

    stages: dict[int, ResolutionStage] = {}
    for G in cfg.grid_sizes:
        bs = _batch_size_for_grid(G, cfg.min_grid_size, cfg.max_grid_size, bs_at_min, cfg.batch_size)
        stages[G] = create_resolution_stage(
            scene_grid_size=G,
            glimpse_grid_size=cfg.avp.glimpse_grid_size,
            patch_size=patch_size,
            batch_size=bs,
            n_viewpoints_per_step=cfg.n_viewpoints_per_step,
        )
    return stages


def create_loaders(
    cfg: Config, stages: dict[int, ResolutionStage]
) -> tuple[dict[int, InfiniteLoader], dict[int, InfiniteLoader]]:
    """Create train/val loaders for each resolution stage."""
    if cfg.debug_train_on_single_batch:
        log.warning("=" * 60)
        log.warning("DEBUG MODE: Training on single repeated batch")
        log.warning("=" * 60)

    train_loaders: dict[int, InfiniteLoader] = {}
    val_loaders: dict[int, InfiniteLoader] = {}

    for G, stage in stages.items():
        scene_size_px = stage.scene_size_px
        fresh_count = stage.fresh_count

        log.info(
            f"Creating loaders for G={G}: scene_size={scene_size_px}px, "
            f"batch={stage.batch_size}, fresh={fresh_count}"
        )

        train_dataset: Dataset[Any] = ImageFolder(
            str(cfg.train_dir),
            train_transform(scene_size_px, (cfg.crop_scale_min, 1.0)),
        )
        val_dataset: Dataset[Any] = ImageFolder(
            str(cfg.val_dir), val_transform(scene_size_px)
        )

        if cfg.debug_train_on_single_batch:
            train_dataset = SingleBatchDataset(train_dataset, fresh_count)
            val_dataset = SingleBatchDataset(val_dataset, fresh_count)

        train_loader: DataLoader[Any] = DataLoader(
            train_dataset,
            batch_size=fresh_count,
            shuffle=not cfg.debug_train_on_single_batch,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_loader: DataLoader[Any] = DataLoader(
            val_dataset,
            batch_size=fresh_count,
            shuffle=not cfg.debug_train_on_single_batch,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        train_loaders[G] = InfiniteLoader(train_loader)
        val_loaders[G] = InfiniteLoader(val_loader)

    return train_loaders, val_loaders
