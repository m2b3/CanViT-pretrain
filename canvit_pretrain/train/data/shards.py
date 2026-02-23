"""Shard-based feature loading with deterministic resume.

Design:
- Single IterableDataset that iterates over ALL shards sequentially
- Workers split SAMPLES within each shard (not shards across workers)
- Deterministic order: shard 0, 1, 2, ..., n-1, 0, 1, ... (no shuffling)
- Resume via start_shard = start_step // batches_per_shard
- Persistent workers work naturally (dataset controls iteration)

Image sources:
- image_root (IN21k): images on filesystem, loaded via PIL
- tar_dir (SA-1B): images read directly from mmap'd tar files, no extraction
"""

import logging
import time
from collections.abc import Iterator
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from canvit_utils.transforms import preprocess

from .tar_images import TarImageReader

log = logging.getLogger(__name__)


class AllShardsDataset(IterableDataset[tuple[Tensor, Tensor, Tensor, int]]):
    """IterableDataset that iterates over all shards sequentially, forever.

    Workers split samples within each shard via `sample_idx % num_workers == worker_id`.
    Resume is handled via `start_shard` parameter.
    """

    def __init__(
        self,
        shard_files: list[Path],
        image_size: int,
        start_shard: int = 0,
        expected_samples_per_shard: int | None = None,
        *,
        image_root: Path | None = None,
        tar_dir: Path | None = None,
    ) -> None:
        assert (image_root is None) != (tar_dir is None), \
            "Exactly one of image_root or tar_dir must be set"
        self.shard_files = shard_files
        self.image_root = Path(image_root) if image_root is not None else None
        self.tar_dir = Path(tar_dir) if tar_dir is not None else None
        self.image_size = image_size
        self.start_shard = start_shard
        self.expected_samples_per_shard = expected_samples_per_shard
        self.transform = preprocess(image_size)

    def _tar_path_for_shard(self, shard_path: Path) -> Path:
        assert self.tar_dir is not None
        return self.tar_dir / f"{shard_path.stem}.tar"

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor, Tensor, int]]:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        n_shards = len(self.shard_files)
        shard_counter = self.start_shard
        tar_reader: TarImageReader | None = None

        while True:
            shard_idx = shard_counter % n_shards
            shard_path = self.shard_files[shard_idx]

            # All workers load same shard (mmap=cheap), but each processes different samples:
            # worker i gets samples i, i+num_workers, i+2*num_workers, ...
            t0 = time.perf_counter()
            shard = torch.load(shard_path, map_location="cpu", weights_only=False, mmap=True)
            t_load = time.perf_counter() - t0

            # Open tar for this shard (mmap'd, shared across workers via fork COW)
            if self.tar_dir is not None:
                if tar_reader is not None:
                    tar_reader.close()
                tar_reader = TarImageReader(self._tar_path_for_shard(shard_path))

            n_samples = len(shard["paths"])
            failed_indices = set(shard.get("failed_indices", []))

            if self.expected_samples_per_shard is not None and n_samples != self.expected_samples_per_shard:
                log.warning(
                    f"Shard {shard_path.name}: {n_samples} samples, expected {self.expected_samples_per_shard}. "
                    f"Resume calculation may be off."
                )

            if worker_id == 0:
                log.info(f"Shard {shard_idx}/{n_shards} ({shard_path.name}): {n_samples} samples, loaded in {t_load:.3f}s")
                if failed_indices:
                    log.warning(f"  {len(failed_indices)} pre-marked failures")

            yielded = 0
            skipped = 0
            t_read = t_transform = t_patches = 0.0

            for i in range(worker_id, n_samples, num_workers):
                if i in failed_indices:
                    skipped += 1
                    continue

                rel_path = shard["paths"][i]
                try:
                    t0 = time.perf_counter()
                    if tar_reader is not None:
                        img = tar_reader.read_image(rel_path)
                    else:
                        assert self.image_root is not None
                        with Image.open(self.image_root / rel_path) as f:
                            img = f.convert("RGB")
                    t1 = time.perf_counter()
                    img_tensor = self.transform(img)
                    t2 = time.perf_counter()
                except Exception as e:
                    log.warning(f"Worker {worker_id}: RUNTIME FAILURE {rel_path}: {e}")
                    skipped += 1
                    continue

                t_read += t1 - t0
                t_transform += t2 - t1

                assert isinstance(img_tensor, Tensor)
                t0 = time.perf_counter()
                patches = shard["patches"][i].clone()
                cls = shard["cls"][i].clone()
                t_patches += time.perf_counter() - t0

                yield (img_tensor, patches, cls, int(shard["class_idxs"][i]))
                yielded += 1

            if worker_id == 0:
                total = t_read + t_transform + t_patches
                if yielded > 0 and total > 0:
                    log.info(
                        f"Shard {shard_idx} done: {yielded} imgs, "
                        f"read={t_read/yielded*1e3:.1f}ms "
                        f"transform={t_transform/yielded*1e3:.1f}ms "
                        f"patches={t_patches/yielded*1e3:.1f}ms "
                        f"total={total/yielded*1e3:.1f}ms/img ({yielded/total:.0f} img/s) "
                        f"skipped={skipped}"
                    )
                else:
                    log.info(f"Shard {shard_idx} done: yielded={yielded}, skipped={skipped}")

            shard_counter += 1


class ShardedFeatureLoader:
    """Infinite loader over shards with checkpoint/resume support.

    Resume is exact: jumps to the right shard, then skips partial-shard batches.
    """

    def __init__(
        self,
        shards_dir: Path,
        image_size: int,
        batch_size: int,
        num_workers: int,
        start_step: int,
        *,
        image_root: Path | None = None,
        tar_dir: Path | None = None,
    ) -> None:
        assert (image_root is None) != (tar_dir is None), \
            "Exactly one of image_root or tar_dir must be set"
        self.shards_dir = Path(shards_dir)
        self.image_root = image_root
        self.tar_dir = tar_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        log.info(f"Globbing shards in {self.shards_dir}...")
        t0 = time.perf_counter()
        self.shard_files = sorted(self.shards_dir.glob("*.pt"))
        log.info(f"  Found {len(self.shard_files)} shards in {time.perf_counter() - t0:.2f}s")
        assert self.shard_files, f"No shards found in {shards_dir}"

        # Read first shard to get samples_per_shard (all shards same size)
        # mmap=True: SA-1B shards are ~70 GB, don't load into RAM just to count samples
        first_shard = torch.load(self.shard_files[0], map_location="cpu", weights_only=False, mmap=True)
        self.samples_per_shard = len(first_shard["paths"])
        del first_shard
        self.batches_per_shard = self.samples_per_shard // batch_size
        self.start_shard = start_step // self.batches_per_shard
        self._skip_batches = start_step % self.batches_per_shard
        log.info(
            f"  {self.samples_per_shard} samples/shard, {self.batches_per_shard} batches/shard, "
            f"start_shard={self.start_shard}, skip={self._skip_batches}"
        )

        # Will be created lazily on first iteration
        self.loader: DataLoader | None = None
        self.loader_iter: Iterator | None = None

    def _create_loader(self) -> DataLoader:
        """Create DataLoader with dataset starting at start_shard."""
        dataset = AllShardsDataset(
            shard_files=self.shard_files,
            image_size=self.image_size,
            start_shard=self.start_shard,
            expected_samples_per_shard=self.samples_per_shard,
            image_root=self.image_root,
            tar_dir=self.tar_dir,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def next(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Get next batch."""
        if self.loader is None:
            log.info(f"Creating DataLoader with {self.num_workers} workers, persistent={self.num_workers > 0}")
            self.loader = self._create_loader()
            self.loader_iter = iter(self.loader)
            if self._skip_batches > 0:
                log.info(f"Skipping {self._skip_batches} batches for exact resume")
                for _ in range(self._skip_batches):
                    next(self.loader_iter)
                self._skip_batches = 0

        assert self.loader_iter is not None
        return next(self.loader_iter)
