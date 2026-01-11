"""Export teacher features for IN21k.

Precomputes DINOv3 features for all images, stores as sharded .pt files.
Each shard: {patches: [N, n_patches, D], cls: [N, D], paths, class_idxs, metadata}.
"""

import gc
import hashlib
import logging
import os
import time
import warnings
from dataclasses import dataclass
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
# Config
# -----------------------------------------------------------------------------

STORAGE_DTYPE = torch.bfloat16  # Single source of truth

ImageFile.LOAD_TRUNCATED_IMAGES = False
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


@dataclass
class Config:
    """Feature export configuration."""

    # Shard selection (mutually exclusive)
    shard: int | None = None
    start_shard: int | None = None
    end_shard: int | None = None

    # Paths (defaults from env vars)
    parquet: Path | None = None
    image_root: Path | None = None
    out_dir: Path | None = None
    teacher_ckpt: Path | None = None

    # Model & export
    teacher_model: str = "dinov3_vitb16"
    shard_size: int = 4096
    batch_size: int = 64
    num_workers: int = 8
    image_size: int = 512


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def log_gpu(label: str) -> None:
    log.info(f"GPU [{label}]: {torch.cuda.memory_allocated() / 1e9:.2f}GB")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()[:16]


def get_shard_range(cfg: Config, n_shards: int) -> tuple[int, int]:
    if cfg.shard is not None:
        return cfg.shard, cfg.shard + 1
    if cfg.start_shard is not None and cfg.end_shard is not None:
        return cfg.start_shard, min(cfg.end_shard, n_shards)
    raise ValueError("Specify --shard or --start-shard/--end-shard")


def resolve_paths(cfg: Config) -> tuple[Path, Path, Path, Path]:
    image_root = cfg.image_root or Path(os.environ["AVP_TRAIN_DIR"])
    parquet = (
        cfg.parquet or Path(os.environ["AVP_INDEX_DIR"]) / f"{image_root.name}.parquet"
    )
    out_dir = cfg.out_dir or Path(os.environ["AVP_FEATURES_DIR"])
    teacher_ckpt = cfg.teacher_ckpt or Path(
        os.path.expanduser(os.environ["AVP_TEACHER_CKPT"])
    )
    return parquet, image_root, out_dir, teacher_ckpt


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


class ImageDataset(Dataset):
    def __init__(self, root: Path, paths: list[str], size: int):
        self.root = root
        self.paths = paths
        self.size = size
        self.transform = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

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
    parquet_hash: str,
) -> None:
    """Export one shard: inference → save → cleanup."""
    n = len(paths)
    shard_path = shards_dir / f"{shard_id:05d}.pt"

    # Preallocate
    patches = torch.empty(n, n_patches, embed_dim, dtype=STORAGE_DTYPE, device=device)
    cls = torch.empty(n, embed_dim, dtype=STORAGE_DTYPE, device=device)
    failed: list[int] = []

    # Inference
    loader = DataLoader(
        ImageDataset(image_root, paths, cfg.image_size),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    t0 = time.perf_counter()
    write_idx = 0

    with torch.no_grad(), torch.autocast("cuda", dtype=STORAGE_DTYPE):
        for imgs, indices, ok in tqdm(loader, desc=f"Shard {shard_id}", leave=False):
            for i, success in zip(indices.tolist(), ok.tolist()):
                if not success:
                    failed.append(i)

            imgs = imgs.to(device)
            feats = teacher.forward_norm_features(imgs)

            bs = imgs.shape[0]
            patches[write_idx : write_idx + bs] = feats.patches.to(STORAGE_DTYPE)
            cls[write_idx : write_idx + bs] = feats.cls.to(STORAGE_DTYPE)
            write_idx += bs

    assert write_idx == n, f"Expected {n}, wrote {write_idx}"

    elapsed = time.perf_counter() - t0
    log.info(f"  {n / elapsed:.0f} img/s, {len(failed)} failed")
    log_gpu("before save")

    # Save atomically
    tmp = shard_path.with_suffix(".tmp")
    torch.save(
        {
            "patches": patches,
            "cls": cls,
            "paths": paths,
            "class_idxs": torch.tensor(class_idxs, dtype=torch.int32),
            "shard_id": shard_id,
            "start_idx": start_idx,
            "end_idx": start_idx + n,
            "failed_indices": failed,
            "parquet_sha256": parquet_hash,
            "teacher_model": cfg.teacher_model,
            "image_size": cfg.image_size,
            "dtype": str(STORAGE_DTYPE),
        },
        tmp,
    )
    tmp.rename(shard_path)
    log.info(f"  Wrote {shard_path.name} ({shard_path.stat().st_size / 1e6:.1f} MB)")

    # Cleanup
    del patches, cls, loader
    gc.collect()
    torch.cuda.empty_cache()
    log_gpu("after cleanup")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main(cfg: Config) -> None:
    device = torch.device("cuda")

    # Paths
    parquet_path, image_root, out_dir, teacher_ckpt = resolve_paths(cfg)
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"out_dir: {out_dir}")
    log.info(f"dtype: {STORAGE_DTYPE}")

    # Parquet
    table = pq.read_table(parquet_path)
    n_images = len(table)
    n_shards = ceil(n_images / cfg.shard_size)
    parquet_hash = sha256_file(parquet_path)
    log.info(f"Parquet: {n_images:,} images, {n_shards} shards, hash={parquet_hash}")

    # Shard range
    start, end = get_shard_range(cfg, n_shards)
    log.info(f"Shards: [{start}, {end})")

    # Teacher
    teacher = (
        create_backbone(cfg.teacher_model, weights=str(teacher_ckpt)).to(device).eval()
    )
    for p in teacher.parameters():
        p.requires_grad = False

    patch_size = teacher.patch_size_px
    embed_dim = teacher.embed_dim
    n_patches = (cfg.image_size // patch_size) ** 2
    assert cfg.image_size % patch_size == 0, (
        f"{cfg.image_size} not divisible by {patch_size}"
    )
    log.info(f"Teacher: {cfg.teacher_model}, {embed_dim}d, {n_patches} patches")
    log_gpu("after teacher")

    # Warmup
    with torch.no_grad(), torch.autocast("cuda", dtype=STORAGE_DTYPE):
        teacher.forward_norm_features(
            torch.randn(1, 3, cfg.image_size, cfg.image_size, device=device)
        )
    torch.cuda.synchronize()
    log_gpu("after warmup")

    # Export loop
    for shard_id in range(start, end):
        shard_path = shards_dir / f"{shard_id:05d}.pt"
        if shard_path.exists():
            log.info(f"Shard {shard_id}: exists, skip")
            continue

        start_idx = shard_id * cfg.shard_size
        end_idx = min(start_idx + cfg.shard_size, n_images)
        n = end_idx - start_idx
        slice_table = table.slice(start_idx, n)
        paths = slice_table.column("path").to_pylist()
        class_idxs = slice_table.column("class_idx").to_pylist()

        log.info(f"Shard {shard_id}: [{start_idx}, {end_idx}) = {n} images")

        export_shard(
            shard_id=shard_id,
            paths=paths,
            class_idxs=class_idxs,
            start_idx=start_idx,
            image_root=image_root,
            shards_dir=shards_dir,
            teacher=teacher,
            device=device,
            cfg=cfg,
            n_patches=n_patches,
            embed_dim=embed_dim,
            parquet_hash=parquet_hash,
        )

    log.info("Done")


if __name__ == "__main__":
    main(tyro.cli(Config))
