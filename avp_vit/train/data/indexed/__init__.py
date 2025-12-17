"""Indexed ImageFolder: fast loading for large datasets without directory-based splits."""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class IndexMetadata:
    schema_version: int
    root_name: str
    split: Literal["train", "val"]
    val_ratio: float
    seed: int
    n_samples: int
    n_classes: int
    generated_at: str


@dataclass(frozen=True)
class SplitIndex:
    train: Path
    val: Path


def parse_index_metadata(raw: dict[bytes, bytes]) -> IndexMetadata:
    """Parse raw parquet metadata. Raises ValueError on schema mismatch."""
    schema_version = int(raw.get(b"schema_version", b"-1").decode())
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"Schema version mismatch: got {schema_version}, expected {SCHEMA_VERSION}")

    def get(key: str) -> str:
        return raw.get(key.encode(), b"").decode()

    split = get("split")
    assert split in ("train", "val"), f"Invalid split: {split}"

    return IndexMetadata(
        schema_version=schema_version,
        root_name=get("root_name"),
        split=split,  # type: ignore[arg-type]
        val_ratio=float(get("val_ratio") or 0),
        seed=int(get("seed") or 0),
        n_samples=int(get("n_samples") or 0),
        n_classes=int(get("n_classes") or 0),
        generated_at=get("generated_at"),
    )


def split_by_class(
    class_to_images: dict[str, list[str]],
    val_ratio: float,
    seed: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Split images into train/val. Returns (train, val) as [(path, class_name), ...]."""
    assert 0 < val_ratio < 1
    rng = random.Random(seed)
    train, val = [], []

    for cls in sorted(class_to_images):
        imgs = list(class_to_images[cls])
        rng.shuffle(imgs)
        n_val = max(1, int(len(imgs) * val_ratio))
        for img in imgs[:n_val]:
            val.append((f"{cls}/{img}", cls))
        for img in imgs[n_val:]:
            train.append((f"{cls}/{img}", cls))

    return train, val
