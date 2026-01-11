"""Export teacher features for IN21k.

Precomputes DINOv3 features for all images, stores as sharded .pt files.
This eliminates expensive 512px teacher inference during training.

USAGE:
    uv run python scripts/export_features.py \
        --parquet /path/to/index.parquet \
        --image-root /path/to/images \
        --out-dir /path/to/output \
        --teacher-ckpt /path/to/weights.pth \
        --shard 0

    # Range of shards (for SLURM array jobs)
    uv run python scripts/export_features.py ... --start-shard 0 --end-shard 100

DESIGN DECISIONS:
    - Shards are self-describing: each .pt contains all metadata needed to verify compatibility
    - Atomic writes: .tmp → .pt rename prevents partial/corrupt shards on crash
    - Resume-friendly: existing shards are skipped automatically
    - GPU buffers preallocated: no per-batch .cpu() calls, no memory fragmentation

SHARD SCHEMA:
    # Data
    patches: [N, n_patches, embed_dim] STORAGE_DTYPE  - patch features (L2-normalized by teacher)
    cls: [N, embed_dim] STORAGE_DTYPE                 - CLS token (L2-normalized by teacher)
    paths: list[str]                                  - relative paths within image_root
    class_idxs: [N] int32                             - class indices from parquet
    failed_indices: list[int]                         - indices where image load failed (features are NaN)

    # Position (row indices into parquet table)
    shard_id: int                                     - which shard (0-indexed)
    start_idx: int                                    - first parquet row (inclusive)
    end_idx: int                                      - last parquet row (exclusive)

    # Compatibility (must match across all shards for a dataset)
    parquet_path: str                                 - path to index file used
    parquet_sha256: str                               - 16-char hash ensures same image ordering
    teacher_model: str                                - e.g. "dinov3_vitb16"
    teacher_ckpt: str                                 - path to weights file used
    image_size: int                                   - input resolution (e.g. 512)
    shard_size: int                                   - max images per shard (e.g. 4096)
    dtype: str                                        - e.g. "torch.bfloat16"
    embed_dim: int                                    - feature dimension (e.g. 768)
    n_patches: int                                    - patches per image (e.g. 1024 for 512px/16px)

    # Provenance
    created_at: str                                   - ISO 8601 UTC timestamp
    git_commit: str                                   - commit hash of export script

VERIFYING SHARD COMPATIBILITY:
    Two shards are compatible if these fields match:
    - parquet_sha256 (same image ordering)
    - teacher_model, image_size, shard_size, dtype, embed_dim, n_patches

MEMORY MODEL:
    - Teacher model: ~2GB VRAM (ViT-B/16)
    - Per-shard buffers: ~6GB VRAM for 4096 images at 1024 patches × 768 dims × 2 bytes
    - Buffers freed after each shard via gc.collect() + empty_cache()
    - Total: ~8-10GB VRAM, stable across shards (no leaks)

THROUGHPUT:
    - Measured as wall time per shard (inference + save combined)
    - No artificial torch.cuda.synchronize() - torch.save syncs implicitly
    - tqdm shows img/s and MB/s per shard
"""

import gc
import hashlib
import logging
import subprocess
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from math import ceil
from pathlib import Path

import pyarrow.parquet as pq
import torch
import tyro
from canvit.hub import create_backbone
from PIL import Image, ImageFile
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# Single source of truth for storage format. Change this to change output dtype.
STORAGE_DTYPE = torch.bfloat16
# Derived: bytes per element. Used for size estimation. Adapts if STORAGE_DTYPE changes.
STORAGE_BYTES = torch.tensor([], dtype=STORAGE_DTYPE).element_size()

# Reject truncated images rather than silently loading garbage.
ImageFile.LOAD_TRUNCATED_IMAGES = False
# Standard ImageNet normalization (teacher was trained with this).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass
class Config:
    """CLI arguments. Parsed by tyro."""

    # Required paths
    parquet: Path  # Index file (columns: path, class_idx, class_name)
    image_root: Path  # Root directory containing images
    out_dir: Path  # Output directory (shards/ subdirectory created)
    teacher_ckpt: Path  # Teacher model weights

    # Shard selection: use --shard for single, or --start-shard/--end-shard for range
    shard: int | None = None
    start_shard: int | None = None
    end_shard: int | None = None

    # Export settings
    teacher_model: str = "dinov3_vitb16"  # Model architecture name
    shard_size: int = 4096  # Images per shard (~6GB at 512px)
    batch_size: int = 64  # GPU batch size
    num_workers: int = 8  # DataLoader workers
    image_size: int = 512  # Input resolution (must be divisible by patch size)


# -----------------------------------------------------------------------------
# Preflight Checks
# -----------------------------------------------------------------------------


def preflight_checks(
    parquet_path: Path,
    image_root: Path,
    teacher_ckpt: Path,
    cfg: Config,
    n_images: int,
    n_shards: int,
    start: int,
    end: int,
) -> None:
    """Validate config and paths before any GPU work. Fail fast on issues."""
    # Check paths exist
    assert parquet_path.exists(), f"Parquet not found: {parquet_path}"
    assert image_root.is_dir(), f"Image root not a directory: {image_root}"
    assert teacher_ckpt.exists(), f"Teacher ckpt not found: {teacher_ckpt}"

    # Check parquet has required columns (cheap: only reads schema, not data)
    schema = pq.read_schema(parquet_path)
    required = {"path", "class_idx"}
    missing = required - set(schema.names)
    assert not missing, f"Parquet missing columns: {missing}"

    # Check shard range is valid
    assert 0 <= start < end <= n_shards, (
        f"Invalid range [{start}, {end}) for {n_shards} shards"
    )

    # Check config values are sane
    assert cfg.image_size > 0
    assert cfg.shard_size > 0
    assert cfg.batch_size > 0
    assert cfg.num_workers >= 0


def estimate_bytes(n_images: int, n_patches: int, embed_dim: int) -> int:
    """Estimate shard size in bytes (features only, metadata negligible)."""
    # patches: [n_images, n_patches, embed_dim] + cls: [n_images, embed_dim]
    # Both stored as STORAGE_DTYPE
    return n_images * (n_patches + 1) * embed_dim * STORAGE_BYTES


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def log_gpu(label: str) -> None:
    """Log current GPU memory allocation. Useful for detecting leaks."""
    log.info(f"GPU [{label}]: {torch.cuda.memory_allocated() / 1e9:.2f}GB")


def get_git_commit() -> str:
    """Get current git commit hash. Returns 'unknown' if not in a git repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    """Compute SHA256 hash of file, return first 16 hex chars."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()[:16]


def get_shard_range(cfg: Config, n_shards: int) -> tuple[int, int]:
    """Parse shard selection from config. Returns [start, end) range."""
    if cfg.shard is not None:
        return cfg.shard, cfg.shard + 1
    if cfg.start_shard is not None and cfg.end_shard is not None:
        return cfg.start_shard, min(cfg.end_shard, n_shards)
    raise ValueError("Specify --shard or --start-shard/--end-shard")


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


class ImageDataset(Dataset):
    """Load and transform images for teacher inference.

    Returns (image_tensor, index, success) tuples.
    On load failure: returns NaN tensor and success=False.
    """

    def __init__(self, root: Path, paths: list[str], size: int):
        self.root = root
        self.paths = paths
        self.size = size
        self.transform = transforms.Compose(
            [
                transforms.Resize(size),  # Resize shortest edge
                transforms.CenterCrop(size),  # Square crop
                transforms.ToTensor(),  # [0,255] uint8 → [0,1] float32
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[Tensor, int, bool]:
        path = self.root / self.paths[idx]
        try:
            # Treat warnings as errors to catch truncated images
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                img = Image.open(path).convert("RGB")
                img.load()  # Force decode now, not lazily
            return self.transform(img), idx, True
        except Exception as e:
            log.warning(f"Bad image {path}: {e}")
            # Return NaN tensor so batch size stays consistent
            return torch.full((3, self.size, self.size), float("nan")), idx, False


# -----------------------------------------------------------------------------
# Export
# -----------------------------------------------------------------------------


def export_shard(
    *,
    shard_id: int,
    paths: list[str],
    class_idxs: list[int],
    start_idx: int,
    image_root: Path,
    shards_dir: Path,
    teacher,
    device: torch.device,
    cfg: Config,
    n_patches: int,
    embed_dim: int,
    parquet_path: Path,
    parquet_hash: str,
    teacher_ckpt: Path,
    pbar_global: tqdm,
) -> tuple[int, int, int]:
    """Export one shard: load images → teacher inference → save .pt.

    Returns (n_images, n_failed, shard_bytes).
    """
    n = len(paths)
    shard_path = shards_dir / f"{shard_id:05d}.pt"

    # Preallocate GPU buffers for the entire shard.
    # Filled in-place during inference loop. No per-batch .cpu() calls.
    patches_buf = torch.empty(
        n, n_patches, embed_dim, dtype=STORAGE_DTYPE, device=device
    )
    cls_buf = torch.empty(n, embed_dim, dtype=STORAGE_DTYPE, device=device)
    failed: list[int] = []

    loader = DataLoader(
        ImageDataset(image_root, paths, cfg.image_size),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,  # Faster CPU→GPU transfer
    )

    t0 = time.perf_counter()
    write_idx = 0

    # Inference loop: fill buffers in-place
    with torch.no_grad(), torch.autocast("cuda", dtype=STORAGE_DTYPE):
        for imgs, indices, ok in tqdm(loader, desc=f"Shard {shard_id}", leave=False):
            # Track failed images (their features will be NaN)
            for i, success in zip(indices.tolist(), ok.tolist()):
                if not success:
                    failed.append(i)

            imgs = imgs.to(device)
            feats = teacher.forward_norm_features(
                imgs
            )  # Returns L2-normalized features

            # Write directly into preallocated buffer
            bs = imgs.shape[0]
            patches_buf[write_idx : write_idx + bs] = feats.patches.to(STORAGE_DTYPE)
            cls_buf[write_idx : write_idx + bs] = feats.cls.to(STORAGE_DTYPE)
            write_idx += bs

    assert write_idx == n, f"Expected {n}, wrote {write_idx}"

    # Atomic save: write to .tmp, rename to .pt
    # Prevents partial files if process is killed mid-write.
    # Note: torch.save implicitly syncs GPU when serializing GPU tensors.
    tmp = shard_path.with_suffix(".tmp")
    torch.save(
        {
            # Data
            "patches": patches_buf,
            "cls": cls_buf,
            "paths": paths,
            "class_idxs": torch.tensor(class_idxs, dtype=torch.int32),
            "failed_indices": failed,
            # Position
            "shard_id": shard_id,
            "start_idx": start_idx,
            "end_idx": start_idx + n,
            # Compatibility
            "parquet_path": str(parquet_path),
            "parquet_sha256": parquet_hash,
            "teacher_model": cfg.teacher_model,
            "teacher_ckpt": str(teacher_ckpt),
            "image_size": cfg.image_size,
            "shard_size": cfg.shard_size,
            "dtype": str(STORAGE_DTYPE),
            "embed_dim": embed_dim,
            "n_patches": n_patches,
            # Provenance
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": get_git_commit(),
        },
        tmp,
    )
    tmp.rename(shard_path)

    elapsed = time.perf_counter() - t0
    shard_bytes = shard_path.stat().st_size

    # Update global progress bar
    pbar_global.update(n)
    pbar_global.set_postfix(
        {
            "img/s": f"{n / elapsed:.0f}",
            "MB/s": f"{shard_bytes / elapsed / 1e6:.0f}",
            "fail": len(failed),
        }
    )

    # Free GPU memory before next shard.
    # Without this, memory accumulates across shards.
    del patches_buf, cls_buf, loader
    gc.collect()
    torch.cuda.empty_cache()

    return n, len(failed), shard_bytes


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main(cfg: Config) -> None:
    """Main entry point. Run with: uv run python scripts/export_features.py --help"""
    device = torch.device("cuda")

    shards_dir = cfg.out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"out_dir: {cfg.out_dir}")
    log.info(f"dtype: {STORAGE_DTYPE} ({STORAGE_BYTES} bytes)")

    # --- Load parquet metadata (cheap: doesn't load data) ---
    n_images = pq.read_metadata(cfg.parquet).num_rows
    n_shards = ceil(n_images / cfg.shard_size)
    log.info(f"Parquet: {n_images:,} images → {n_shards} shards")

    # --- Parse shard range ---
    start, end = get_shard_range(cfg, n_shards)

    # --- Preflight checks (before any GPU work) ---
    preflight_checks(
        cfg.parquet,
        cfg.image_root,
        cfg.teacher_ckpt,
        cfg,
        n_images,
        n_shards,
        start,
        end,
    )
    log.info("Preflight OK")

    # --- Compute parquet hash (ensures same image ordering across runs) ---
    parquet_hash = sha256_file(cfg.parquet)
    log.info(f"Parquet hash: {parquet_hash}")

    # --- Load teacher model ---
    teacher = (
        create_backbone(cfg.teacher_model, weights=str(cfg.teacher_ckpt))
        .to(device)
        .eval()
    )
    for p in teacher.parameters():
        p.requires_grad = False  # Inference only

    patch_size = teacher.patch_size_px
    embed_dim = teacher.embed_dim
    n_patches = (cfg.image_size // patch_size) ** 2
    assert cfg.image_size % patch_size == 0, (
        f"{cfg.image_size} not divisible by {patch_size}"
    )
    log.info(f"Teacher: {cfg.teacher_model}, {embed_dim}d, {n_patches} patches")
    log_gpu("after teacher")

    # --- Determine shards to process (skip existing) ---
    shards_todo = []
    images_todo = 0
    for sid in range(start, end):
        if (shards_dir / f"{sid:05d}.pt").exists():
            continue
        shard_start = sid * cfg.shard_size
        shard_end = min(shard_start + cfg.shard_size, n_images)
        shards_todo.append(sid)
        images_todo += shard_end - shard_start

    if not shards_todo:
        log.info("All shards exist")
        return

    est_bytes = estimate_bytes(cfg.shard_size, n_patches, embed_dim)
    est_total_gb = len(shards_todo) * est_bytes / 1e9
    log.info(
        f"Exporting: {len(shards_todo)} shards, {images_todo:,} images, ~{est_total_gb:.1f} GB"
    )

    # --- Load full parquet table for slicing ---
    table = pq.read_table(cfg.parquet)

    # --- Export loop ---
    t0_total = time.perf_counter()
    total_bytes = 0
    total_failed = 0

    pbar = tqdm(total=images_todo, unit="img", desc="Export")

    for shard_id in shards_todo:
        # Compute row range for this shard
        start_idx = shard_id * cfg.shard_size
        end_idx = min(start_idx + cfg.shard_size, n_images)
        n = end_idx - start_idx

        # Slice parquet table (fast: zero-copy view)
        slice_table = table.slice(start_idx, n)
        paths = slice_table.column("path").to_pylist()
        class_idxs = slice_table.column("class_idx").to_pylist()

        _, n_failed, shard_bytes = export_shard(
            shard_id=shard_id,
            paths=paths,
            class_idxs=class_idxs,
            start_idx=start_idx,
            image_root=cfg.image_root,
            shards_dir=shards_dir,
            teacher=teacher,
            device=device,
            cfg=cfg,
            n_patches=n_patches,
            embed_dim=embed_dim,
            parquet_path=cfg.parquet,
            parquet_hash=parquet_hash,
            teacher_ckpt=cfg.teacher_ckpt,
            pbar_global=pbar,
        )
        total_bytes += shard_bytes
        total_failed += n_failed

    pbar.close()
    elapsed = time.perf_counter() - t0_total

    # --- Summary ---
    log.info(
        f"Done: {images_todo:,} images, {total_bytes / 1e9:.2f} GB in {elapsed:.1f}s"
    )
    log.info(
        f"  {images_todo / elapsed:.0f} img/s, {total_bytes / elapsed / 1e6:.0f} MB/s"
    )
    if total_failed:
        log.warning(f"  {total_failed} failed images")
    log_gpu("final")


if __name__ == "__main__":
    main(tyro.cli(Config))
