"""Shard-based feature loading with deterministic resume.

Design:
- One shard at a time, fully exhausted before moving to next
- Workers split SAMPLES within a shard (not shards across workers)
- Deterministic order: shard 0, 1, 2, ..., n-1, 0, 1, ... (no shuffling)
- Progress tracked via `shards_completed` for clean checkpoint/resume
- Runtime failures: log and skip (batches stay full via IterableDataset pattern)
"""

import logging
import time
from pathlib import Path
from typing import Iterator, TypedDict

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .transforms import val_transform

log = logging.getLogger(__name__)


class SingleShardDataset(IterableDataset[tuple[Tensor, Tensor, Tensor, int]]):
    """IterableDataset for a single shard. Workers split samples internally.

    Workers iterate over samples where `sample_idx % num_workers == worker_id`.
    This ensures all samples are covered exactly once across workers.
    """

    def __init__(self, shard_path: Path, image_root: Path, image_size: int) -> None:
        self.shard_path = Path(shard_path)
        self.image_root = Path(image_root)
        self.image_size = image_size
        self._transform = val_transform(image_size)

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor, Tensor, int]]:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        t0 = time.perf_counter()
        shard = torch.load(self.shard_path, map_location="cpu", weights_only=False, mmap=True)
        load_time = time.perf_counter() - t0

        n_samples = len(shard["paths"])
        failed_indices = set(shard.get("failed_indices", []))

        if worker_id == 0:
            log.info(f"Loaded shard {self.shard_path.name}: {n_samples} samples in {load_time:.2f}s")
            if failed_indices:
                log.warning(f"  {len(failed_indices)} pre-marked failures will be skipped")

        yielded = 0
        skipped_failed = 0
        skipped_runtime = 0

        # Workers split by sample index: worker i gets samples i, i+nw, i+2*nw, ...
        for i in range(worker_id, n_samples, num_workers):
            if i in failed_indices:
                skipped_failed += 1
                continue

            rel_path = shard["paths"][i]
            try:
                with Image.open(self.image_root / rel_path) as f:
                    img = f.convert("RGB")
                img_tensor = self._transform(img)
            except Exception as e:
                # Vanishingly rare runtime failure - log and skip
                log.warning(f"Worker {worker_id}: RUNTIME FAILURE {rel_path}: {type(e).__name__}: {e}")
                skipped_runtime += 1
                continue

            assert isinstance(img_tensor, Tensor)
            yield (
                img_tensor,
                shard["patches"][i].clone(),
                shard["cls"][i].clone(),
                int(shard["class_idxs"][i]),
            )
            yielded += 1

        log.debug(
            f"Worker {worker_id}: shard done - yielded={yielded}, "
            f"skipped_failed={skipped_failed}, skipped_runtime={skipped_runtime}"
        )


class LoaderState(TypedDict):
    """Checkpoint state for ShardedFeatureLoader."""
    shards_completed: int


class ShardedFeatureLoader:
    """Infinite loader over shards with checkpoint/resume support.

    Iterates shards in deterministic order: 0, 1, 2, ..., n-1, 0, 1, ...
    Each shard is fully exhausted before moving to the next.
    Progress is tracked via `shards_completed` for clean resume.
    """

    def __init__(
        self,
        shards_dir: Path,
        image_root: Path,
        image_size: int,
        batch_size: int,
        num_workers: int,
    ) -> None:
        self.shards_dir = Path(shards_dir)
        self.image_root = Path(image_root)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        log.info(f"Globbing shards in {self.shards_dir}...")
        t0 = time.perf_counter()
        self.shard_files = sorted(self.shards_dir.glob("*.pt"))
        log.info(f"  Found {len(self.shard_files)} shards in {time.perf_counter() - t0:.2f}s")
        assert self.shard_files, f"No shards found in {shards_dir}"

        self.shards_completed = 0

    def state_dict(self) -> LoaderState:
        """Return checkpoint state."""
        return {"shards_completed": self.shards_completed}

    def load_state_dict(self, state: LoaderState) -> None:
        """Restore from checkpoint."""
        self.shards_completed = state["shards_completed"]
        log.info(f"Resumed ShardedFeatureLoader at shard {self.shards_completed}")

    def _create_dataloader(self, shard_path: Path) -> DataLoader:
        """Create DataLoader for a single shard."""
        ds = SingleShardDataset(shard_path, self.image_root, self.image_size)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=False,  # New loader per shard, no persistence
        )

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor, Tensor, Tensor]]:
        """Infinite iteration over shards.

        Yields batched (images, patches, cls, class_idxs).
        Never stops - loops back to shard 0 after exhausting all shards.
        """
        n_shards = len(self.shard_files)

        while True:
            shard_idx = self.shards_completed % n_shards
            shard_path = self.shard_files[shard_idx]

            log.info(f"Starting shard {shard_idx}/{n_shards} ({shard_path.name}), "
                     f"total shards_completed={self.shards_completed}")

            loader = self._create_dataloader(shard_path)
            batches_this_shard = 0

            for batch in loader:
                batches_this_shard += 1
                yield batch

            log.info(f"Finished shard {shard_idx}: {batches_this_shard} batches")
            self.shards_completed += 1

    def next(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Get next batch (images, patches, cls, class_idxs). Creates iterator on first call."""
        if not hasattr(self, "_iter"):
            self._iter = iter(self)
        return next(self._iter)
