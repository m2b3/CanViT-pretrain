"""Dataset for loading precomputed teacher features from shards.

Each shard contains features for `shard_size` images (except last shard).
Handles gaps in shard numbering (e.g., during partial export).
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from .data import imagenet_normalize

log = logging.getLogger(__name__)

# Keep this many shards open simultaneously (mmap file handles)
MAX_OPEN_SHARDS = 64


class FeatureSample(NamedTuple):
    """Single sample from feature dataset."""
    image: Tensor    # [3, H, W] normalized
    patches: Tensor  # [n_patches, embed_dim]
    cls: Tensor      # [embed_dim]
    class_idx: int


class FeatureDataset(Dataset[FeatureSample]):
    """Dataset for precomputed teacher features + corresponding images.

    Args:
        shards_dir: Directory containing shard .pt files (00000.pt, 00001.pt, ...)
        image_root: Root directory for images (paths in shards are relative to this)
        expected_shard_size: Expected images per shard (for index math). Must match export.
    """

    def __init__(self, shards_dir: Path, image_root: Path, expected_shard_size: int = 4096):
        self.shards_dir = Path(shards_dir)
        self.image_root = Path(image_root)
        self.shard_size = expected_shard_size

        # Discover shards (handles gaps in numbering)
        shard_files = sorted(self.shards_dir.glob("*.pt"))
        assert len(shard_files) > 0, f"No shards found in {shards_dir}"

        # Extract actual shard IDs from filenames
        self._shard_ids = [int(f.stem) for f in shard_files]
        self.n_shards = len(self._shard_ids)

        if self._shard_ids != list(range(self.n_shards)):
            log.warning(f"Gaps in shard numbering: have {self.n_shards} shards, IDs {self._shard_ids[0]}-{self._shard_ids[-1]}")

        # Load metadata from first shard
        first_shard = self._load_shard_by_idx(0)
        self.embed_dim = first_shard["embed_dim"]
        self.n_patches = first_shard["n_patches"]
        self.dtype = first_shard["dtype"]
        self.image_size = first_shard["image_size"]

        # Transform must match export: resize + center crop + normalize
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            imagenet_normalize(),
        ])

        # Calculate total images across all available shards
        self.n_images = 0
        self._shard_offsets = []  # global index where each shard starts
        for i in range(self.n_shards):
            self._shard_offsets.append(self.n_images)
            shard = self._load_shard_by_idx(i)
            self.n_images += len(shard["paths"])

        log.info(
            f"FeatureDataset: {self.n_images:,} images, {self.n_shards} shards, "
            f"{self.n_patches} patches, {self.embed_dim}d, {self.dtype}"
        )

    def _load_shard_by_idx(self, idx: int) -> dict:
        """Load shard by logical index (0 to n_shards-1)."""
        return self._load_shard(self._shard_ids[idx])

    @lru_cache(maxsize=MAX_OPEN_SHARDS)
    def _load_shard(self, shard_id: int) -> dict:
        """Load shard with mmap. LRU cached."""
        path = self.shards_dir / f"{shard_id:05d}.pt"
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)

    def _find_shard(self, idx: int) -> tuple[int, int]:
        """Find (logical_shard_idx, local_idx) for global index using binary search."""
        import bisect
        # bisect_right gives insertion point; subtract 1 to get shard containing idx
        logical_idx = bisect.bisect_right(self._shard_offsets, idx) - 1
        local_idx = idx - self._shard_offsets[logical_idx]
        return logical_idx, local_idx

    def __len__(self) -> int:
        return self.n_images

    def __getitem__(self, idx: int) -> FeatureSample:
        if idx < 0 or idx >= self.n_images:
            raise IndexError(f"Index {idx} out of range [0, {self.n_images})")

        logical_shard_idx, local_idx = self._find_shard(idx)
        shard = self._load_shard_by_idx(logical_shard_idx)

        # Load and transform image
        rel_path = shard["paths"][local_idx]
        img = Image.open(self.image_root / rel_path).convert("RGB")
        img_tensor = self.transform(img)
        assert isinstance(img_tensor, Tensor)

        return FeatureSample(
            image=img_tensor,
            patches=shard["patches"][local_idx],
            cls=shard["cls"][local_idx],
            class_idx=shard["class_idxs"][local_idx].item(),
        )

    def get_path(self, idx: int) -> str:
        """Get relative image path for index."""
        logical_shard_idx, local_idx = self._find_shard(idx)
        shard = self._load_shard_by_idx(logical_shard_idx)
        return shard["paths"][local_idx]

    def get_metadata(self) -> dict:
        """Get metadata from first shard."""
        shard = self._load_shard_by_idx(0)
        return {
            "teacher_model": shard["teacher_model"],
            "image_size": shard["image_size"],
            "shard_size": shard["shard_size"],
            "embed_dim": shard["embed_dim"],
            "n_patches": shard["n_patches"],
            "dtype": shard["dtype"],
            "batch_size": shard.get("batch_size"),
            "parquet_sha256": shard["parquet_sha256"],
        }
