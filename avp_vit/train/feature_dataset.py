"""IterableDataset for precomputed teacher features + images.

Design choices:
- IterableDataset, not map-style: each worker loads its own shards independently,
  no shared file handles, no forking issues.
- Sequential iteration within shards: shards are pre-shuffled, no need to shuffle
  within. Shuffle shard order between epochs if desired.
- Workers split shards via get_worker_info(): worker i gets shards i, i+nw, i+2*nw, ...
- No __init__ file I/O beyond glob: avoids creating handles that get forked.
- Metadata (image_size, etc.) read lazily from first shard in each worker.
"""

import logging
import time
from pathlib import Path
from typing import Iterator

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info

from .data import val_transform

log = logging.getLogger(__name__)


class FeatureIterableDataset(IterableDataset):
    """Iterable dataset for precomputed features + corresponding images.

    Args:
        shards_dir: Directory containing shard .pt files
        image_root: Root directory for images (paths in shards are relative to this)
        epoch: Current epoch (for shard order shuffling between epochs)
    """

    def __init__(self, shards_dir: Path, image_root: Path, epoch: int = 0):
        t0 = time.perf_counter()
        self.shards_dir = Path(shards_dir)
        self.image_root = Path(image_root)
        self.epoch = epoch

        # Only glob here - no torch.load, no file handles to fork
        log.info(f"Globbing shards in {self.shards_dir}...")
        t_glob = time.perf_counter()
        self.shard_files = sorted(self.shards_dir.glob("*.pt"))
        log.info(f"  Found {len(self.shard_files)} shards in {time.perf_counter() - t_glob:.2f}s")

        assert self.shard_files, f"No shards found in {shards_dir}"

        # Transform built lazily in __iter__ (needs image_size from shard metadata)
        self._transform = None

        log.info(f"FeatureIterableDataset init: {time.perf_counter() - t0:.2f}s")

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic shard order shuffling."""
        self.epoch = epoch

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor, Tensor, int]]:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        # Shuffle shard order deterministically per epoch
        import random
        rng = random.Random(self.epoch)
        shards = list(self.shard_files)
        rng.shuffle(shards)

        # Each worker takes every num_workers-th shard
        shards = shards[worker_id::num_workers]

        log.debug(f"Worker {worker_id}/{num_workers}: processing {len(shards)} shards")

        for shard_idx, shard_path in enumerate(shards):
            t0 = time.perf_counter()
            shard = torch.load(shard_path, map_location="cpu", weights_only=False, mmap=True)
            load_time = time.perf_counter() - t0

            # Build transform lazily from first shard's metadata
            if self._transform is None:
                image_size = shard["image_size"]
                self._transform = val_transform(image_size)
                log.debug(f"Worker {worker_id}: built transform for {image_size}px")

            n_samples = len(shard["paths"])
            log.debug(f"Worker {worker_id}: shard {shard_idx}/{len(shards)} "
                     f"({shard_path.name}, {n_samples} samples, loaded in {load_time:.2f}s)")

            # Sequential iteration - shards are pre-shuffled
            # .clone() required: mmap'd tensor views serialize poorly across DataLoader workers
            for i in range(n_samples):
                rel_path = shard["paths"][i]
                with Image.open(self.image_root / rel_path) as f:
                    img = f.convert("RGB")
                img_tensor = self._transform(img)
                assert isinstance(img_tensor, Tensor)

                yield (
                    img_tensor,
                    shard["patches"][i].clone(),
                    shard["cls"][i].clone(),
                    int(shard["class_idxs"][i]),
                )
