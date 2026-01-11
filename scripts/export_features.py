"""Export teacher features for IN21k.

Self-contained. External deps: torch, torchvision, pyarrow, tyro, canvit.

Usage:
    uv run python scripts/export_features.py --shard 0
    uv run python scripts/export_features.py --start-shard 0 --end-shard 5
"""

import hashlib
import json
import logging
import os
import time
import warnings
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
from PIL import Image, ImageFile
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

# =============================================================================
# Constants
# =============================================================================

# Refuse to load truncated images - fail loud, not silent garbage
ImageFile.LOAD_TRUNCATED_IMAGES = False

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FILE_READ_CHUNK = 64 * 1024  # 64KB


# =============================================================================
# Config
# =============================================================================


@dataclass
class ExportConfig:
    """Feature export configuration."""

    # Shard selection (mutually exclusive)
    shard: int | None = None
    """Single shard to export."""
    start_shard: int | None = None
    """Start shard (inclusive)."""
    end_shard: int | None = None
    """End shard (exclusive)."""

    # Paths (defaults from env vars)
    parquet: Path | None = None
    """Parquet index. Default: $AVP_INDEX_DIR/{image_root.name}.parquet"""
    image_root: Path | None = None
    """Image root. Default: $AVP_TRAIN_DIR"""
    out_dir: Path | None = None
    """Output directory. Default: $AVP_FEATURES_DIR"""
    teacher_ckpt: Path | None = None
    """Teacher checkpoint. Default: $AVP_TEACHER_CKPT"""

    # Model
    teacher_model: str = "dinov3_vitb16"
    """Teacher model name."""

    # Export
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


# =============================================================================
# Types
# =============================================================================


class ShardData(NamedTuple):
    """Data for one shard, ready to be written."""

    patches: Tensor  # [N, n_patches, embed_dim] bfloat16
    cls: Tensor  # [N, embed_dim] bfloat16
    paths: list[str]
    class_idxs: list[int]
    shard_id: int
    start_idx: int
    failed_indices: list[int]  # Indices with NaN features (bad images)


# =============================================================================
# Helpers
# =============================================================================


def log_gpu_memory(label: str) -> None:
    """Log GPU memory stats."""
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    log.info(f"GPU [{label}]: {allocated:.2f}GB alloc, {reserved:.2f}GB reserved, {peak:.2f}GB peak")


def sha256_file(path: Path) -> str:
    """Return first 16 chars of SHA256 hash of file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(FILE_READ_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def val_transform(size: int) -> transforms.Compose:
    """Validation transform: resize, center crop, normalize."""
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


def resolve_paths(cfg: ExportConfig) -> tuple[Path, Path, Path, Path]:
    """Resolve paths from config + env vars."""
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
    """Return (start, end) shard range from config."""
    if cfg.shard is not None:
        return cfg.shard, cfg.shard + 1
    if cfg.start_shard is not None and cfg.end_shard is not None:
        return cfg.start_shard, min(cfg.end_shard, n_shards)
    raise ValueError("Specify --shard or --start-shard/--end-shard")


# =============================================================================
# Metadata
# =============================================================================


def verify_or_create_meta(
    meta_path: Path,
    *,
    shard_size: int,
    n_images: int,
    parquet_sha256: str,
    teacher_model: str,
    image_size: int,
    parquet_path: Path,
    image_root: Path,
    teacher_ckpt: Path,
) -> None:
    """Verify meta.json matches config, or create it.

    Verified fields (must match for shard compatibility):
        schema_version, shard_size, n_images, parquet_sha256, teacher_model, image_size

    Provenance fields (logged but not verified - paths may differ across nodes):
        parquet_path, image_root, teacher_ckpt, created_at, n_shards
    """
    verified = {
        "schema_version": 1,
        "shard_size": shard_size,
        "n_images": n_images,
        "parquet_sha256": parquet_sha256,
        "teacher_model": teacher_model,
        "image_size": image_size,
    }

    if meta_path.exists():
        with open(meta_path) as f:
            existing = json.load(f)
        for key, expected in verified.items():
            actual = existing.get(key)
            if actual != expected:
                raise ValueError(
                    f"meta.json mismatch: {key}={actual!r}, expected {expected!r}. "
                    "Config changed? Delete output dir to restart fresh."
                )
        log.info("Verified meta.json")
    else:
        meta = {
            **verified,
            "n_shards": ceil(n_images / shard_size),
            "parquet_path": str(parquet_path),
            "image_root": str(image_root),
            "teacher_ckpt": str(teacher_ckpt),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        log.info("Created meta.json")


def verify_existing_shard(path: Path, expected_hash: str, max_size: int) -> bool:
    """Check if existing shard is valid. Uses mmap to avoid loading 6GB into RAM."""
    data = torch.load(path, weights_only=False, mmap=True)

    assert isinstance(data, dict), f"{path.name}: not a dict"
    assert "parquet_sha256" in data, f"{path.name}: missing parquet_sha256"
    assert "paths" in data, f"{path.name}: missing paths"

    if data["parquet_sha256"] != expected_hash:
        log.warning(f"{path.name}: hash mismatch, will re-export")
        return False

    if len(data["paths"]) > max_size:
        log.warning(f"{path.name}: size {len(data['paths'])} > {max_size}, will re-export")
        return False

    return True


# =============================================================================
# Dataset
# =============================================================================


class ImagePathDataset(Dataset):
    """Dataset that loads images by path, returns (tensor, idx, success)."""

    def __init__(self, root: Path, paths: list[str], transform, image_size: int):
        self.root = root
        self.paths = paths
        self.transform = transform
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[Tensor, int, bool]:
        path = self.root / self.paths[idx]
        try:
            # Make PIL warnings (e.g., "Truncated File Read") raise exceptions
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                img = Image.open(path).convert("RGB")
                # Force full decode (open() is lazy, load() forces read)
                img.load()
            return self.transform(img), idx, True
        except Exception as e:
            log.warning(f"Bad image {path}: {e}")
            # NaN tensor - will propagate and explode if accidentally used
            return torch.full((3, self.image_size, self.image_size), float("nan")), idx, False


# =============================================================================
# Writer
# =============================================================================


def writer_thread(
    queue: Queue,
    shards_dir: Path,
    parquet_sha256: str,
    teacher_model: str,
    image_size: int,
) -> None:
    """Async writer. Receives ShardData from queue, writes atomically (.tmp → .pt)."""
    while True:
        item = queue.get()
        if item is None:
            break

        shard: ShardData = item
        final_path = shards_dir / f"{shard.shard_id:05d}.pt"
        tmp_path = final_path.with_suffix(".tmp")

        # Write to .tmp first
        torch.save(
            {
                "patches": shard.patches,
                "cls": shard.cls,
                "paths": shard.paths,
                "class_idxs": torch.tensor(shard.class_idxs, dtype=torch.int32),
                "shard_id": shard.shard_id,
                "start_idx": shard.start_idx,
                "end_idx": shard.start_idx + len(shard.paths),
                "failed_indices": shard.failed_indices,
                "parquet_sha256": parquet_sha256,
                "teacher_model": teacher_model,
                "image_size": image_size,
                "dtype": str(shard.patches.dtype),
                "exported_at": datetime.now(timezone.utc).isoformat(),
            },
            tmp_path,
        )

        # Atomic rename
        tmp_path.rename(final_path)

        size_mb = final_path.stat().st_size / 1e6
        log.info(f"Wrote {final_path.name}: {len(shard.paths)} images, {size_mb:.1f} MB, dtype={shard.patches.dtype}")


# =============================================================================
# Export
# =============================================================================


def export_shard(
    *,
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
    """Run teacher inference on images, return ShardData."""
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

    # Throughput tracking
    bytes_per_img = (n_patches + 1) * embed_dim * 2  # bfloat16 = 2 bytes
    t0 = time.perf_counter()
    images_done = 0

    pbar = tqdm(loader, desc=f"Shard {shard_id}", leave=False, unit="batch")

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for images, indices, success in pbar:
            # Track failed images
            for idx, ok in zip(indices.tolist(), success.tolist()):
                if not ok:
                    failed_indices.append(idx)

            images = images.to(device, non_blocking=True)
            feats = teacher.forward_norm_features(images)

            # Verify dtype
            assert feats.patches.dtype == torch.bfloat16, f"Expected bfloat16, got {feats.patches.dtype}"
            assert feats.cls.dtype == torch.bfloat16, f"Expected bfloat16, got {feats.cls.dtype}"

            # Move to CPU (GPU memory is precious, CPU has more headroom)
            patches_list.append(feats.patches.cpu())
            cls_list.append(feats.cls.cpu())

            # Update progress
            images_done += images.shape[0]
            elapsed = time.perf_counter() - t0
            pbar.set_postfix({
                "img/s": f"{images_done / elapsed:.0f}",
                "MB/s": f"{images_done * bytes_per_img / elapsed / 1e6:.0f}",
            })

    if failed_indices:
        log.warning(f"Shard {shard_id}: {len(failed_indices)} failed: {failed_indices}")

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
    device = torch.device("cuda")

    log.info("=" * 60)
    log.info("EXPORT FEATURES")
    log.info("=" * 60)

    # Resolve paths
    parquet_path, image_root, out_dir, teacher_ckpt = resolve_paths(cfg)
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"parquet:       {parquet_path}")
    log.info(f"image_root:    {image_root}")
    log.info(f"out_dir:       {out_dir}")
    log.info(f"teacher:       {cfg.teacher_model} @ {teacher_ckpt.name}")
    log.info(f"shard_size:    {cfg.shard_size}")
    log.info(f"batch_size:    {cfg.batch_size}")
    log.info(f"num_workers:   {cfg.num_workers}")
    log.info(f"image_size:    {cfg.image_size}")
    log.info(f"compile:       {cfg.compile}")

    # Load parquet index
    log.info("-" * 60)
    t0 = time.perf_counter()
    table = pq.read_table(parquet_path)
    n_images = len(table)
    n_shards = ceil(n_images / cfg.shard_size)
    parquet_sha256 = sha256_file(parquet_path)
    log.info(f"Parquet: {n_images:,} images, {n_shards} shards, hash={parquet_sha256} ({time.perf_counter()-t0:.1f}s)")

    # Shard range
    start_shard, end_shard = resolve_shard_range(cfg, n_shards)
    log.info(f"Shard range: [{start_shard}, {end_shard}) = {end_shard - start_shard} shards")

    # Verify/create meta.json
    verify_or_create_meta(
        out_dir / "meta.json",
        shard_size=cfg.shard_size,
        n_images=n_images,
        parquet_sha256=parquet_sha256,
        teacher_model=cfg.teacher_model,
        image_size=cfg.image_size,
        parquet_path=parquet_path,
        image_root=image_root,
        teacher_ckpt=teacher_ckpt,
    )

    # Load teacher
    log.info("-" * 60)
    t0 = time.perf_counter()
    teacher = load_teacher(cfg.teacher_model, teacher_ckpt, device)
    patch_size = teacher.patch_size_px
    embed_dim = teacher.embed_dim
    grid_size = cfg.image_size // patch_size
    n_patches = grid_size * grid_size
    assert cfg.image_size % patch_size == 0, f"{cfg.image_size} not divisible by {patch_size}"
    log.info(f"Teacher: {embed_dim}d, patch={patch_size}px, grid={grid_size}x{grid_size} ({time.perf_counter()-t0:.1f}s)")
    log_gpu_memory("after teacher load")

    if cfg.compile:
        t0 = time.perf_counter()
        teacher.compile()
        log.info(f"Compiled ({time.perf_counter()-t0:.1f}s)")

    # Warmup
    log.info("-" * 60)
    t0 = time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        teacher.forward_norm_features(torch.randn(1, 3, cfg.image_size, cfg.image_size, device=device))
    torch.cuda.synchronize()
    log.info(f"Warmup ({time.perf_counter()-t0:.1f}s)")
    log_gpu_memory("after warmup")

    # Start writer thread
    write_queue: Queue = Queue(maxsize=1)
    writer = Thread(
        target=writer_thread,
        args=(write_queue, shards_dir, parquet_sha256, cfg.teacher_model, cfg.image_size),
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

        # Skip if valid shard exists
        if shard_path.exists():
            if verify_existing_shard(shard_path, parquet_sha256, cfg.shard_size):
                log.info(f"Shard {shard_id}: valid, skipping")
                skipped += 1
                continue
            shard_path.unlink()  # Remove invalid

        # Get paths and labels for this shard
        start_idx = shard_id * cfg.shard_size
        end_idx = min(start_idx + cfg.shard_size, n_images)
        n = end_idx - start_idx
        slice_table = table.slice(start_idx, n)
        paths = slice_table.column("path").to_pylist()
        class_idxs = slice_table.column("class_idx").to_pylist()

        log.info(f"Shard {shard_id}: [{start_idx}, {end_idx}) = {n} images")

        # Export
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
        log.info(f"  -> {elapsed:.1f}s, {n / elapsed:.0f} img/s")

        # Queue for async write
        write_queue.put(shard_data)
        del shard_data

    # Wait for writer
    write_queue.put(None)
    writer.join()

    # Summary
    log.info("=" * 60)
    log.info("DONE")
    log.info(f"  Processed: {end_shard - start_shard - skipped} shards, {total_images:,} images")
    log.info(f"  Skipped:   {skipped} shards")
    if total_time > 0:
        log.info(f"  Throughput: {total_images / total_time:.0f} img/s")
    log.info(f"  Total: {time.perf_counter() - t_start:.1f}s")
    log_gpu_memory("final")


if __name__ == "__main__":
    main(tyro.cli(ExportConfig))
