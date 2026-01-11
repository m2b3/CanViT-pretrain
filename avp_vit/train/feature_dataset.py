"""Dataset for loading precomputed teacher features from shards.

Assumes shards are pre-shuffled (random class distribution per shard).
Use ShardSampler for efficient shard-sequential access with within-shard shuffle.
"""

import bisect
import logging
import random
from pathlib import Path
from typing import Iterator, NamedTuple

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, Sampler
from torchvision import transforms

from .data import imagenet_normalize

log = logging.getLogger(__name__)


class FeatureSample(NamedTuple):
    """Single sample from feature dataset."""
    image: Tensor    # [3, H, W] normalized
    patches: Tensor  # [n_patches, embed_dim]
    cls: Tensor      # [embed_dim]
    class_idx: int


class FeatureDataset(Dataset[FeatureSample]):
    """Dataset for precomputed teacher features + corresponding images."""

    def __init__(self, shards_dir: Path, image_root: Path):
        self.shards_dir = Path(shards_dir)
        self.image_root = Path(image_root)

        # Discover shards
        shard_files = sorted(self.shards_dir.glob("*.pt"))
        assert shard_files, f"No shards found in {shards_dir}"
        self._shard_ids = [int(f.stem) for f in shard_files]
        self.n_shards = len(self._shard_ids)

        # Load metadata from first shard
        first = self._load_shard(0)
        self.image_size = first["image_size"]
        self.embed_dim = first["embed_dim"]
        self.n_patches = first["n_patches"]

        # Build shard offsets
        self._shard_sizes: list[int] = []
        self._shard_offsets: list[int] = []
        offset = 0
        for i in range(self.n_shards):
            self._shard_offsets.append(offset)
            size = len(self._load_shard(i)["paths"])
            self._shard_sizes.append(size)
            offset += size
        self.n_images = offset

        # Transform matching export
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            imagenet_normalize(),
        ])

        # Current shard cache (single shard, no LRU)
        self._cached_shard_idx: int | None = None
        self._cached_shard: dict | None = None

        log.info(f"FeatureDataset: {self.n_images:,} images, {self.n_shards} shards")

    def _load_shard(self, logical_idx: int) -> dict:
        shard_id = self._shard_ids[logical_idx]
        path = self.shards_dir / f"{shard_id:05d}.pt"
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)

    def _get_shard(self, logical_idx: int) -> dict:
        """Get shard, using single-shard cache."""
        if self._cached_shard_idx != logical_idx:
            self._cached_shard = self._load_shard(logical_idx)
            self._cached_shard_idx = logical_idx
        assert self._cached_shard is not None
        return self._cached_shard

    def _find_shard(self, idx: int) -> tuple[int, int]:
        """Find (logical_shard_idx, local_idx) for global index."""
        logical_idx = bisect.bisect_right(self._shard_offsets, idx) - 1
        local_idx = idx - self._shard_offsets[logical_idx]
        return logical_idx, local_idx

    def __len__(self) -> int:
        return self.n_images

    def __getitem__(self, idx: int) -> FeatureSample:
        logical_shard_idx, local_idx = self._find_shard(idx)
        shard = self._get_shard(logical_shard_idx)

        rel_path = shard["paths"][local_idx]
        img = Image.open(self.image_root / rel_path).convert("RGB")
        img_tensor = self.transform(img)
        assert isinstance(img_tensor, Tensor)

        return FeatureSample(
            image=img_tensor,
            patches=shard["patches"][local_idx],
            cls=shard["cls"][local_idx],
            class_idx=int(shard["class_idxs"][local_idx]),
        )

    def get_path(self, idx: int) -> str:
        logical_shard_idx, local_idx = self._find_shard(idx)
        return self._get_shard(logical_shard_idx)["paths"][local_idx]

    def get_metadata(self) -> dict:
        shard = self._get_shard(0)
        return {
            "teacher_model": shard["teacher_model"],
            "image_size": shard["image_size"],
            "embed_dim": shard["embed_dim"],
            "n_patches": shard["n_patches"],
            "dtype": shard["dtype"],
            "parquet_sha256": shard["parquet_sha256"],
        }


class ShardSampler(Sampler[int]):
    """Sampler for shard-sequential access with within-shard shuffle.

    Each epoch: shuffle shard order, shuffle within each shard.
    Yields all indices from shard N before moving to shard N+1.
    """

    def __init__(self, dataset: FeatureDataset, seed: int = 0):
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.dataset.n_images

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)

        # Shuffle shard order
        shard_order = list(range(self.dataset.n_shards))
        rng.shuffle(shard_order)

        for shard_idx in shard_order:
            offset = self.dataset._shard_offsets[shard_idx]
            size = self.dataset._shard_sizes[shard_idx]
            # Shuffle within shard
            local_indices = list(range(offset, offset + size))
            rng.shuffle(local_indices)
            yield from local_indices
