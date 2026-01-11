"""Export teacher features for IN21k.

Self-contained. External deps: torch, torchvision, pyarrow, tyro, canvit.

Usage:
    uv run python scripts/export_features.py --shard 0
    uv run python scripts/export_features.py --start-shard 0 --end-shard 5
"""

import gc
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

ImageFile.LOAD_TRUNCATED_IMAGES = False

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FILE_READ_CHUNK = 64 * 1024


# =============================================================================
# Config
# =============================================================================


@dataclass
class ExportConfig:
    """Feature export configuration."""

    shard: int | None = None
    start_shard: int | None = None
    end_shard: int | None = None

    parquet: Path | None = None
    image_root: Path | None = None
    out_dir: Path | None = None
    teacher_ckpt: Path | None = None

    teacher_model: str = "dinov3_vitb16"
    shard_size: int = 4096
    batch_size: int = 64
    num_workers: int = 8
    image_size: int = 512
    compile: bool = False
    verify_existing: bool = False
    """Verify existing shards (slow on network storage). If False, just check file exists."""


# =============================================================================
# Helpers
# =============================================================================


def log_gpu_memory(label: str) -> None:
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    log.info(f"GPU [{label}]: {allocated:.2f}GB alloc, {reserved:.2f}GB reserved, {peak:.2f}GB peak")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(FILE_READ_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def val_transform(size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_teacher(model: str, ckpt: Path, device: torch.device):
    backbone = create_backbone(model, weights=str(ckpt))
    backbone.vit.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone.to(device)


def resolve_paths(cfg: ExportConfig) -> tuple[Path, Path, Path, Path]:
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
                    "Delete output dir to restart fresh."
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


def verify_existing_shard(
    path: Path,
    *,
    expected_hash: str,
    max_size: int,
    teacher_model: str,
    image_size: int,
) -> bool:
    """Check if existing shard is valid. Uses mmap to avoid loading 6GB into RAM."""
    data = torch.load(path, weights_only=False, mmap=True)

    assert isinstance(data, dict), f"{path.name}: not a dict"
    assert "parquet_sha256" in data, f"{path.name}: missing parquet_sha256"
    assert "paths" in data, f"{path.name}: missing paths"

    if data["parquet_sha256"] != expected_hash:
        log.warning(f"{path.name}: hash mismatch")
        return False
    if len(data["paths"]) > max_size:
        log.warning(f"{path.name}: size mismatch")
        return False
    if data.get("teacher_model") != teacher_model:
        log.warning(f"{path.name}: teacher_model mismatch")
        return False
    if data.get("image_size") != image_size:
        log.warning(f"{path.name}: image_size mismatch")
        return False
    if data.get("dtype") != "torch.bfloat16":
        log.warning(f"{path.name}: dtype mismatch")
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

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[Tensor, int, bool]:
        path = self.root / self.paths[idx]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                img = Image.open(path).convert("RGB")
                img.load()
            return self.transform(img), idx, True
        except Exception as e:
            log.warning(f"Bad image {path}: {e}")
            return torch.full((3, self.image_size, self.image_size), float("nan")), idx, False


# =============================================================================
# Export
# =============================================================================


def export_and_save_shard(
    *,
    shard_id: int,
    paths: list[str],
    class_idxs: list[int],
    start_idx: int,
    image_root: Path,
    shards_dir: Path,
    teacher,
    device: torch.device,
    cfg: ExportConfig,
    embed_dim: int,
    n_patches: int,
    parquet_sha256: str,
) -> None:
    """Run teacher inference and save shard. No async, no queues."""
    n_images = len(paths)

    # Preallocate on GPU
    patches = torch.empty(n_images, n_patches, embed_dim, dtype=torch.bfloat16, device=device)
    cls = torch.empty(n_images, embed_dim, dtype=torch.bfloat16, device=device)
    # Also keep fp32 for error analysis
    patches_fp32 = torch.empty(n_images, n_patches, embed_dim, dtype=torch.float32, device=device)
    cls_fp32 = torch.empty(n_images, embed_dim, dtype=torch.float32, device=device)
    failed_indices: list[int] = []

    transform = val_transform(cfg.image_size)
    dataset = ImagePathDataset(image_root, paths, transform, cfg.image_size)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    bytes_per_img = (n_patches + 1) * embed_dim * 2
    t0 = time.perf_counter()
    write_idx = 0

    pbar = tqdm(loader, desc=f"Shard {shard_id}", leave=False, unit="batch")

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for images, indices, success in pbar:
            for idx, ok in zip(indices.tolist(), success.tolist()):
                if not ok:
                    failed_indices.append(idx)

            images = images.to(device)
            feats = teacher.forward_norm_features(images)

            batch_size = images.shape[0]
            # Store fp32 original
            patches_fp32[write_idx : write_idx + batch_size] = feats.patches.float()
            cls_fp32[write_idx : write_idx + batch_size] = feats.cls.float()
            # Store bfloat16
            patches[write_idx : write_idx + batch_size] = feats.patches.to(torch.bfloat16)
            cls[write_idx : write_idx + batch_size] = feats.cls.to(torch.bfloat16)
            write_idx += batch_size

            elapsed = time.perf_counter() - t0
            pbar.set_postfix({
                "img/s": f"{write_idx / elapsed:.0f}",
                "MB/s": f"{write_idx * bytes_per_img / elapsed / 1e6:.0f}",
            })

    assert write_idx == n_images, f"Wrote {write_idx}, expected {n_images}"

    # Compute conversion errors (only on valid images)
    valid_mask = torch.ones(n_images, dtype=torch.bool, device=device)
    for idx in failed_indices:
        valid_mask[idx] = False

    if valid_mask.any():
        p_valid = patches_fp32[valid_mask]
        c_valid = cls_fp32[valid_mask]

        # bfloat16 error
        p_bf16 = patches[valid_mask].float()
        c_bf16 = cls[valid_mask].float()
        bf16_patches_err = (p_valid - p_bf16).norm() / p_valid.norm()
        bf16_cls_err = (c_valid - c_bf16).norm() / c_valid.norm()

        # float16 error (for comparison)
        p_fp16 = p_valid.half().float()
        c_fp16 = c_valid.half().float()
        fp16_patches_err = (p_valid - p_fp16).norm() / p_valid.norm()
        fp16_cls_err = (c_valid - c_fp16).norm() / c_valid.norm()

        log.info(f"  bf16 rel err: patches={bf16_patches_err:.2e} cls={bf16_cls_err:.2e}")
        log.info(f"  fp16 rel err: patches={fp16_patches_err:.2e} cls={fp16_cls_err:.2e}")

    # Stats on bfloat16 (skip NaN)
    if valid_mask.any():
        p_valid_bf16 = patches[valid_mask]
        c_valid_bf16 = cls[valid_mask]
        log.info(
            f"  patches: min={p_valid_bf16.min().item():.3f} max={p_valid_bf16.max().item():.3f} "
            f"mean={p_valid_bf16.mean().item():.3f} std={p_valid_bf16.std().item():.3f}"
        )
        log.info(
            f"  cls:     min={c_valid_bf16.min().item():.3f} max={c_valid_bf16.max().item():.3f} "
            f"mean={c_valid_bf16.mean().item():.3f} std={c_valid_bf16.std().item():.3f}"
        )

    log_gpu_memory(f"shard {shard_id} before save")

    if failed_indices:
        log.warning(f"Shard {shard_id}: {len(failed_indices)} failed: {failed_indices}")

    # Save directly from GPU - torch.save handles it
    final_path = shards_dir / f"{shard_id:05d}.pt"
    tmp_path = final_path.with_suffix(".tmp")

    torch.save(
        {
            "patches": patches,
            "cls": cls,
            "paths": paths,
            "class_idxs": torch.tensor(class_idxs, dtype=torch.int32),
            "shard_id": shard_id,
            "start_idx": start_idx,
            "end_idx": start_idx + len(paths),
            "failed_indices": failed_indices,
            "parquet_sha256": parquet_sha256,
            "teacher_model": cfg.teacher_model,
            "image_size": cfg.image_size,
            "dtype": str(patches.dtype),
            "exported_at": datetime.now(timezone.utc).isoformat(),
        },
        tmp_path,
    )
    tmp_path.rename(final_path)

    size_mb = final_path.stat().st_size / 1e6
    log.info(f"  Wrote {final_path.name}: {size_mb:.1f} MB")

    # Explicit cleanup
    del patches, cls, patches_fp32, cls_fp32
    gc.collect()
    torch.cuda.empty_cache()

    log_gpu_memory(f"shard {shard_id} after cleanup")


# =============================================================================
# Main
# =============================================================================


def main(cfg: ExportConfig) -> None:
    t_start = time.perf_counter()
    device = torch.device("cuda")

    log.info("=" * 60)
    log.info("EXPORT FEATURES")
    log.info("=" * 60)

    parquet_path, image_root, out_dir, teacher_ckpt = resolve_paths(cfg)
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"parquet:       {parquet_path}")
    log.info(f"image_root:    {image_root}")
    log.info(f"out_dir:       {out_dir}")
    log.info(f"teacher:       {cfg.teacher_model} @ {teacher_ckpt.name}")
    log.info(f"shard_size:    {cfg.shard_size}")
    log.info(f"batch_size:    {cfg.batch_size}")
    log.info(f"image_size:    {cfg.image_size}")

    # Load parquet
    log.info("-" * 60)
    t0 = time.perf_counter()
    table = pq.read_table(parquet_path)
    n_images = len(table)
    n_shards = ceil(n_images / cfg.shard_size)
    parquet_sha256 = sha256_file(parquet_path)
    log.info(f"Parquet: {n_images:,} images, {n_shards} shards, hash={parquet_sha256} ({time.perf_counter()-t0:.1f}s)")

    start_shard, end_shard = resolve_shard_range(cfg, n_shards)
    log.info(f"Shard range: [{start_shard}, {end_shard})")

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
    assert cfg.image_size % patch_size == 0
    log.info(f"Teacher: {embed_dim}d, patch={patch_size}px, grid={grid_size}x{grid_size} ({time.perf_counter()-t0:.1f}s)")
    log_gpu_memory("after teacher load")

    if cfg.compile:
        t0 = time.perf_counter()
        teacher.compile()
        log.info(f"Compiled ({time.perf_counter()-t0:.1f}s)")

    # Warmup
    log.info("-" * 60)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        teacher.forward_norm_features(torch.randn(1, 3, cfg.image_size, cfg.image_size, device=device))
    torch.cuda.synchronize()
    log.info("Warmup done")
    log_gpu_memory("after warmup")

    # Export loop
    log.info("-" * 60)
    log.info("Exporting...")
    total_images = 0
    total_time = 0.0
    skipped = 0

    for shard_id in range(start_shard, end_shard):
        shard_path = shards_dir / f"{shard_id:05d}.pt"

        if shard_path.exists():
            if cfg.verify_existing:
                if verify_existing_shard(
                    shard_path,
                    expected_hash=parquet_sha256,
                    max_size=cfg.shard_size,
                    teacher_model=cfg.teacher_model,
                    image_size=cfg.image_size,
                ):
                    log.info(f"Shard {shard_id}: valid, skipping")
                    skipped += 1
                    continue
                shard_path.unlink()
            else:
                log.info(f"Shard {shard_id}: exists, skipping")
                skipped += 1
                continue

        start_idx = shard_id * cfg.shard_size
        end_idx = min(start_idx + cfg.shard_size, n_images)
        n = end_idx - start_idx
        slice_table = table.slice(start_idx, n)
        paths = slice_table.column("path").to_pylist()
        class_idxs = slice_table.column("class_idx").to_pylist()

        log.info(f"Shard {shard_id}: [{start_idx}, {end_idx}) = {n} images")

        t0 = time.perf_counter()
        export_and_save_shard(
            shard_id=shard_id,
            paths=paths,
            class_idxs=class_idxs,
            start_idx=start_idx,
            image_root=image_root,
            shards_dir=shards_dir,
            teacher=teacher,
            device=device,
            cfg=cfg,
            embed_dim=embed_dim,
            n_patches=n_patches,
            parquet_sha256=parquet_sha256,
        )
        elapsed = time.perf_counter() - t0
        total_images += n
        total_time += elapsed
        log.info(f"  -> {elapsed:.1f}s, {n / elapsed:.0f} img/s")

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
