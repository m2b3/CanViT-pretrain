#!/usr/bin/env python
"""Export teacher features for IN21k.

Self-contained script. Only external deps: torch, torchvision, pyarrow, tyro, canvit.

Usage:
    uv run python scripts/export_features.py --shard 0
    uv run python scripts/export_features.py --start-shard 0 --end-shard 5
"""

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import NamedTuple

import pyarrow.parquet as pq
import torch
import tyro
from canvit.hub import create_backbone
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ImageNet normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# =============================================================================
# Config
# =============================================================================


@dataclass
class ExportConfig:
    """Feature export configuration."""

    # Shard selection
    shard: int | None = None
    """Single shard to export."""
    start_shard: int | None = None
    """Start shard (inclusive)."""
    end_shard: int | None = None
    """End shard (exclusive)."""

    # Paths (defaults from env)
    parquet: Path | None = None
    """Parquet index. Default: $AVP_INDEX_DIR/{image_root.name}.parquet"""
    image_root: Path | None = None
    """Image root. Default: $AVP_TRAIN_DIR"""
    out_dir: Path | None = None
    """Output directory. Default: $AVP_FEATURES_DIR"""
    teacher_ckpt: Path | None = None
    """Teacher checkpoint. Default: $AVP_TEACHER_CKPT"""

    # Teacher
    teacher_model: str = "dinov3_vitb16"
    """Teacher model name."""

    # Export parameters
    shard_size: int = 4096
    """Images per shard."""
    batch_size: int = 64
    """Inference batch size."""
    num_workers: int = 8
    """DataLoader workers."""
    image_size: int = 512
    """Image resolution."""

    # Runtime
    compile: bool = False
    """torch.compile the teacher."""
    amp: bool = True
    """Use bfloat16 AMP."""


# =============================================================================
# Types
# =============================================================================


class ShardData(NamedTuple):
    patches: Tensor
    cls: Tensor
    paths: list[str]
    class_idxs: list[int]
    shard_id: int
    start_idx: int
    failed_indices: list[int]


# =============================================================================
# Helpers
# =============================================================================


def val_transform(size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_teacher(model: str, ckpt: Path, device: torch.device):
    """Load frozen teacher backbone."""
    backbone = create_backbone(model, weights=str(ckpt))
    backbone.vit.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone.to(device)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def resolve_paths(cfg: ExportConfig) -> tuple[Path, Path, Path, Path]:
    """Resolve paths from config + env. Returns (parquet, image_root, out_dir, teacher_ckpt)."""
    image_root = cfg.image_root or Path(os.environ["AVP_TRAIN_DIR"])
    if cfg.parquet is not None:
        parquet = cfg.parquet
    else:
        index_dir = Path(os.environ["AVP_INDEX_DIR"])
        parquet = index_dir / f"{image_root.name}.parquet"
    out_dir = cfg.out_dir or Path(os.environ["AVP_FEATURES_DIR"])
    teacher_ckpt = cfg.teacher_ckpt or Path(os.path.expanduser(os.environ["AVP_TEACHER_CKPT"]))
    return parquet, image_root, out_dir, teacher_ckpt


def resolve_shard_range(cfg: ExportConfig, n_shards: int) -> tuple[int, int]:
    """Return (start, end) shard range."""
    if cfg.shard is not None:
        return cfg.shard, cfg.shard + 1
    if cfg.start_shard is not None and cfg.end_shard is not None:
        return cfg.start_shard, min(cfg.end_shard, n_shards)
    raise ValueError("Specify --shard or --start-shard/--end-shard")


def verify_meta(meta_path: Path, shard_size: int, n_images: int, parquet_hash: str) -> None:
    """Verify meta.json matches current config, or create it."""
    expected = {
        "schema_version": 1,
        "shard_size": shard_size,
        "n_images": n_images,
        "parquet_sha256": parquet_hash,
    }

    if meta_path.exists():
        with open(meta_path) as f:
            existing = json.load(f)
        for key in expected:
            if existing.get(key) != expected[key]:
                raise ValueError(
                    f"meta.json mismatch: {key}={existing.get(key)}, expected {expected[key]}. "
                    "Config changed? Delete output dir to restart fresh."
                )
        log.info("Verified meta.json")
    else:
        full_meta = {
            **expected,
            "n_shards": ceil(n_images / shard_size),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(meta_path, "w") as f:
            json.dump(full_meta, f, indent=2)
        log.info("Created meta.json")


def verify_existing_shard(path: Path, expected_parquet_hash: str, expected_shard_size: int) -> bool:
    """Check if existing shard matches current config. Returns True if valid."""
    data = torch.load(path, weights_only=False)

    assert isinstance(data, dict), f"{path.name}: not a dict"
    assert "parquet_sha256" in data, f"{path.name}: missing parquet_sha256"
    assert "paths" in data, f"{path.name}: missing paths"

    if data["parquet_sha256"] != expected_parquet_hash:
        log.warning(f"{path.name}: parquet_sha256 mismatch, will re-export")
        return False

    if len(data["paths"]) > expected_shard_size:
        log.warning(f"{path.name}: size mismatch, will re-export")
        return False

    return True


# =============================================================================
# Dataset
# =============================================================================


class ImagePathDataset(Dataset):
    def __init__(self, root: Path, paths: list[str], transform, image_size: int):
        self.root = root
        self.paths = paths
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[Tensor, int, bool]:
        path = self.root / self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img), idx, True
        except Exception as e:
            log.warning(f"Bad image {path}: {e}")
            return torch.full((3, self.image_size, self.image_size), float("nan")), idx, False


# =============================================================================
# Export
# =============================================================================


def writer_thread(q: Queue, shards_dir: Path, parquet_hash: str, teacher_model: str, image_size: int):
    """Async writer. Receives ShardData, writes atomically."""
    while True:
        item = q.get()
        if item is None:
            break
        shard_data: ShardData = item

        path = shards_dir / f"{shard_data.shard_id:05d}.pt"
        tmp = path.with_suffix(".tmp")

        torch.save({
            "patches": shard_data.patches,
            "cls": shard_data.cls,
            "paths": shard_data.paths,
            "class_idxs": torch.tensor(shard_data.class_idxs, dtype=torch.int32),
            "shard_id": shard_data.shard_id,
            "start_idx": shard_data.start_idx,
            "end_idx": shard_data.start_idx + len(shard_data.paths),
            "failed_indices": shard_data.failed_indices,
            "parquet_sha256": parquet_hash,
            "teacher": teacher_model,
            "image_size": image_size,
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }, tmp)
        tmp.rename(path)

        size_mb = path.stat().st_size / 1e6
        log.info(f"Wrote {path.name}: {len(shard_data.paths)} images, {size_mb:.1f} MB")


def export_shard(
    shard_id: int,
    paths: list[str],
    class_idxs: list[int],
    start_idx: int,
    image_root: Path,
    teacher,
    device: torch.device,
    cfg: ExportConfig,
    embed_dim: int,
    n_patches: int,
) -> ShardData:
    """Run teacher on images, return ShardData."""
    transform = val_transform(cfg.image_size)
    dataset = ImagePathDataset(image_root, paths, transform, cfg.image_size)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    patches_list: list[Tensor] = []
    cls_list: list[Tensor] = []
    failed_indices: list[int] = []

    amp_dtype = torch.bfloat16 if cfg.amp else None
    bytes_per_img = (n_patches + 1) * embed_dim * 2

    pbar = tqdm(
        loader,
        desc=f"Shard {shard_id}",
        leave=False,
        unit="batch",
        postfix={"img/s": 0, "MB/s": 0},
    )

    t0 = time.perf_counter()
    images_done = 0

    with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype, enabled=cfg.amp):
        for images, indices, success in pbar:
            for idx, ok in zip(indices.tolist(), success.tolist()):
                if not ok:
                    failed_indices.append(idx)

            images = images.to(device, non_blocking=True)
            feats = teacher.forward_norm_features(images)
            patches_list.append(feats.patches.half().cpu())
            cls_list.append(feats.cls.half().cpu())

            images_done += images.shape[0]
            elapsed = time.perf_counter() - t0
            img_s = images_done / elapsed
            mb_s = (images_done * bytes_per_img) / elapsed / 1e6
            pbar.set_postfix({"img/s": f"{img_s:.0f}", "MB/s": f"{mb_s:.0f}"})

    if failed_indices:
        log.warning(f"Shard {shard_id}: {len(failed_indices)} failed images: {failed_indices}")

    return ShardData(
        patches=torch.cat(patches_list),
        cls=torch.cat(cls_list),
        paths=paths,
        class_idxs=class_idxs,
        shard_id=shard_id,
        start_idx=start_idx,
        failed_indices=failed_indices,
    )


# =============================================================================
# Main
# =============================================================================


def main(cfg: ExportConfig) -> None:
    t_start = time.perf_counter()
    log.info("=" * 60)
    log.info("EXPORT FEATURES")
    log.info("=" * 60)

    parquet_path, image_root, out_dir, teacher_ckpt = resolve_paths(cfg)
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    log.info(f"parquet:      {parquet_path}")
    log.info(f"image_root:   {image_root}")
    log.info(f"out_dir:      {out_dir}")
    log.info(f"teacher:      {cfg.teacher_model} @ {teacher_ckpt.name}")
    log.info(f"shard_size:   {cfg.shard_size}")
    log.info(f"batch_size:   {cfg.batch_size}")
    log.info(f"num_workers:  {cfg.num_workers}")
    log.info(f"image_size:   {cfg.image_size}")
    log.info(f"amp:          {cfg.amp}")
    log.info(f"compile:      {cfg.compile}")

    # Load parquet
    log.info("-" * 60)
    t0 = time.perf_counter()
    table = pq.read_table(parquet_path)
    n_images = len(table)
    n_shards = ceil(n_images / cfg.shard_size)
    parquet_hash = sha256_file(parquet_path)
    log.info(f"Parquet: {n_images:,} images, {n_shards} shards, hash={parquet_hash} ({time.perf_counter()-t0:.1f}s)")

    start_shard, end_shard = resolve_shard_range(cfg, n_shards)
    log.info(f"Shard range: [{start_shard}, {end_shard}) = {end_shard - start_shard} shards")

    verify_meta(out_dir / "meta.json", cfg.shard_size, n_images, parquet_hash)

    # Load teacher
    log.info("-" * 60)
    t0 = time.perf_counter()
    teacher = load_teacher(cfg.teacher_model, teacher_ckpt, device)
    patch_size = teacher.patch_size_px
    embed_dim = teacher.embed_dim
    grid_size = cfg.image_size // patch_size
    n_patches = grid_size * grid_size
    assert cfg.image_size % patch_size == 0, f"image_size={cfg.image_size} not divisible by patch_size={patch_size}"
    log.info(f"Teacher: {embed_dim}d, patch={patch_size}px, grid={grid_size}x{grid_size}, n_patches={n_patches} ({time.perf_counter()-t0:.1f}s)")

    if cfg.compile:
        t0 = time.perf_counter()
        teacher.compile()
        log.info(f"Compiled teacher ({time.perf_counter()-t0:.1f}s)")

    # Warmup
    log.info("-" * 60)
    t0 = time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.amp):
        teacher.forward_norm_features(torch.randn(1, 3, cfg.image_size, cfg.image_size, device=device))
    torch.cuda.synchronize()
    log.info(f"Warmup done ({time.perf_counter()-t0:.1f}s)")

    # Writer thread
    write_queue: Queue = Queue(maxsize=2)
    writer = Thread(
        target=writer_thread,
        args=(write_queue, shards_dir, parquet_hash, cfg.teacher_model, cfg.image_size),
        daemon=True,
    )
    writer.start()

    # Export loop
    log.info("-" * 60)
    log.info("Exporting...")
    total_images = 0
    total_time = 0.0
    skipped = 0

    for shard_id in range(start_shard, end_shard):
        shard_path = shards_dir / f"{shard_id:05d}.pt"

        if shard_path.exists():
            if verify_existing_shard(shard_path, parquet_hash, cfg.shard_size):
                log.info(f"Shard {shard_id}: exists, valid, skipping")
                skipped += 1
                continue
            else:
                shard_path.unlink()

        start_idx = shard_id * cfg.shard_size
        end_idx = min(start_idx + cfg.shard_size, n_images)
        n = end_idx - start_idx
        slice_table = table.slice(start_idx, n)
        paths = slice_table.column("path").to_pylist()
        class_idxs = slice_table.column("class_idx").to_pylist()

        log.info(f"Shard {shard_id}: [{start_idx}, {end_idx}) = {n} images")

        t0 = time.perf_counter()
        shard_data = export_shard(
            shard_id=shard_id,
            paths=paths,
            class_idxs=class_idxs,
            start_idx=start_idx,
            image_root=image_root,
            teacher=teacher,
            device=device,
            cfg=cfg,
            embed_dim=embed_dim,
            n_patches=n_patches,
        )
        elapsed = time.perf_counter() - t0
        total_images += n
        total_time += elapsed
        log.info(f"  -> {elapsed:.1f}s, {n/elapsed:.0f} img/s")

        write_queue.put(shard_data)

    write_queue.put(None)
    writer.join()

    log.info("=" * 60)
    log.info("DONE")
    log.info(f"  Processed: {end_shard - start_shard - skipped} shards, {total_images:,} images")
    log.info(f"  Skipped:   {skipped} shards")
    if total_time > 0:
        log.info(f"  Throughput: {total_images/total_time:.0f} img/s")
    log.info(f"  Total time: {time.perf_counter()-t_start:.1f}s")


if __name__ == "__main__":
    main(tyro.cli(ExportConfig))
