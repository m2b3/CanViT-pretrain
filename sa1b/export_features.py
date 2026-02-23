"""Export DINOv3 features for one SA-1B tar.

Designed for SLURM array jobs: 1 task = 1 tar = 1 shard.

Each invocation:
  1. Extracts one tar to a temp dir (SLURM_TMPDIR or specified)
  2. Loads frozen DINOv3 teacher from HuggingFace Hub
  3. Runs batched inference on all JPEGs
  4. Saves one .pt shard (named after tar: sa_NNNNNN.pt)

Shard format matches training loader expectations (shards.py):
  patches:        [N, n_patches, embed_dim] float16
  cls:            [N, embed_dim] float16
  paths:          list[str] — filenames relative to image dir
  class_idxs:     [N] int32 (all 0 for SA-1B)
  failed_indices: list[int]
  image_hashes:   list[str] — xxh64 of decoded pixels

Usage:
  # Single tar (interactive)
  uv run python sa1b/export_features.py \
      --tar /path/to/sa_000020.tar \
      --out-dir /path/to/shards \
      --extract-dir $SLURM_TMPDIR/sa1b_images

  # SLURM array job (see sa1b/export_features.sh)
  sbatch --array=0-999 sa1b/export_features.sh
"""

import logging
import subprocess
import time
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import tyro
import xxhash
from canvit_utils.teacher import load_teacher
from PIL import Image, ImageFile
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from canvit_utils.transforms import preprocess

STORAGE_DTYPE = torch.float16
NUMPY_DTYPE = np.float16  # Must match STORAGE_DTYPE
ImageFile.LOAD_TRUNCATED_IMAGES = False

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


@dataclass
class Config:
    tar: Path
    out_dir: Path
    extract_dir: Path
    image_size: int = 1024
    teacher_repo_id: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    batch_size: int = 32
    num_workers: int = 8


class ImageDataset(Dataset[tuple[Tensor, int, bool, str]]):
    def __init__(self, paths: list[Path], size: int) -> None:
        self.paths = paths
        self.transform = preprocess(size)
        self.size = size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[Tensor, int, bool, str]:
        path = self.paths[idx]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                with Image.open(path) as f:
                    img = f.convert("RGB")
                    img.load()
            img_hash = xxhash.xxh64(img.tobytes()).hexdigest()
            tensor = self.transform(img)
            assert isinstance(tensor, Tensor)
            return tensor, idx, True, img_hash
        except Exception as e:
            log.warning(f"Bad image {path}: {e}")
            return torch.full((3, self.size, self.size), float("nan")), idx, False, ""


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def extract_tar(tar: Path, dest: Path) -> None:
    """Extract only JPEGs from tar to dest."""
    dest.mkdir(parents=True, exist_ok=True)
    # List tar contents, filter to .jpg, extract only those.
    # This avoids extracting ~11k JSON mask files we don't need.
    listing = subprocess.check_output(
        ["tar", "tf", str(tar)], text=True
    )
    jpg_members = [line for line in listing.splitlines() if line.endswith(".jpg")]
    assert jpg_members, f"No .jpg entries in {tar.name}"

    subprocess.run(
        ["tar", "xf", str(tar), "-C", str(dest)] + jpg_members,
        check=True,
    )


def main(cfg: Config) -> None:
    t_start = time.perf_counter()
    device = torch.device("cuda")
    tar_stem = cfg.tar.stem  # e.g. "sa_000020"
    shard_path = cfg.out_dir / f"{tar_stem}.pt"

    # Skip if already exported
    if shard_path.exists():
        log.info(f"Shard already exists: {shard_path}")
        return

    assert cfg.tar.exists(), f"Tar not found: {cfg.tar}"

    log.info(f"tar: {cfg.tar}")
    log.info(f"out_dir: {cfg.out_dir}")
    log.info(f"extract_dir: {cfg.extract_dir}")
    log.info(f"image_size: {cfg.image_size}")
    log.info(f"teacher_repo_id: {cfg.teacher_repo_id}")

    # --- Phase 1: Extract ---
    log.info(f"Extracting JPEGs from {cfg.tar.name}...")
    t0 = time.perf_counter()
    extract_tar(cfg.tar, cfg.extract_dir)
    jpg_paths = sorted(cfg.extract_dir.glob("*.jpg"))
    n = len(jpg_paths)
    assert n > 0, f"No JPEGs found in {cfg.extract_dir}"
    t_extract = time.perf_counter() - t0
    log.info(f"Extract: {n} JPEGs in {t_extract:.1f}s")

    # --- Phase 2: Load teacher ---
    t0 = time.perf_counter()
    teacher = load_teacher(cfg.teacher_repo_id, device)
    patch_size = teacher.model.config.patch_size
    embed_dim = teacher.embed_dim
    n_patches = (cfg.image_size // patch_size) ** 2
    assert cfg.image_size % patch_size == 0
    t_teacher = time.perf_counter() - t0
    log.info(f"Teacher: {embed_dim}d, {n_patches} patches, patch_size={patch_size} (loaded in {t_teacher:.1f}s)")
    log.info(f"GPU after teacher: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # --- Phase 3: Inference ---
    # Accumulate into mmap'd files on SLURM_TMPDIR — the full shard is ~66 GB
    # in fp16, which won't fit in RAM on most nodes. mmap lets the OS page
    # data to/from the local SSD as needed, keeping RSS at a few GB.
    patches_mmap_path = cfg.extract_dir / f"{tar_stem}_patches.mmap"
    cls_mmap_path = cfg.extract_dir / f"{tar_stem}_cls.mmap"
    patches_buf = np.memmap(patches_mmap_path, dtype=NUMPY_DTYPE, mode="w+", shape=(n, n_patches, embed_dim))
    cls_buf = np.memmap(cls_mmap_path, dtype=np.float32, mode="w+", shape=(n, embed_dim))
    log.info(f"Mmap buffers: patches={patches_buf.nbytes/1e9:.1f}GB cls={cls_buf.nbytes/1e6:.0f}MB")

    hashes: list[str] = [""] * n
    failed: list[int] = []

    loader = DataLoader(
        ImageDataset(jpg_paths, cfg.image_size),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    t0 = time.perf_counter()
    n_batches = 0
    t_data_total = 0.0
    t_gpu_total = 0.0
    write_idx = 0
    pbar = tqdm(loader, desc=tar_stem)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        t_batch_end = time.perf_counter()
        for imgs, indices, ok, batch_hashes in pbar:
            t_data = time.perf_counter() - t_batch_end
            t_data_total += t_data

            for i, success, h in zip(indices.tolist(), ok.tolist(), batch_hashes):
                hashes[i] = h
                if not success:
                    failed.append(i)

            t_gpu_start = time.perf_counter()
            imgs = imgs.to(device, non_blocking=True)
            feats = teacher.forward_norm_features(imgs)
            bs = imgs.shape[0]
            # GPU → CPU → mmap (OS handles write-back to SSD asynchronously)
            patches_buf[write_idx : write_idx + bs] = feats.patches.to(STORAGE_DTYPE).cpu().numpy()
            cls_buf[write_idx : write_idx + bs] = feats.cls.float().cpu().numpy()
            t_gpu = time.perf_counter() - t_gpu_start
            t_gpu_total += t_gpu

            write_idx += bs
            n_batches += 1
            pbar.set_postfix_str(f"data={t_data:.2f}s gpu={t_gpu:.2f}s img/s={write_idx/(time.perf_counter()-t0):.0f}")
            t_batch_end = time.perf_counter()

    # Sync before asserting — GPU ops may still be in flight
    torch.cuda.synchronize()
    assert write_idx == n, f"Expected {n}, wrote {write_idx}"
    patches_buf.flush()
    cls_buf.flush()
    t_inference = time.perf_counter() - t0

    log.info(
        f"Inference: {n} images in {t_inference:.1f}s ({n / t_inference:.0f} img/s) | "
        f"data: {t_data_total:.1f}s ({t_data_total / t_inference * 100:.0f}%) | "
        f"gpu: {t_gpu_total:.1f}s ({t_gpu_total / t_inference * 100:.0f}%)"
    )
    if failed:
        log.warning(f"{len(failed)} failed images")

    # --- Phase 4: Save ---
    # torch.from_numpy shares the mmap pointer — PyTorch's serializer streams
    # through the storage linearly, OS pages in/out as needed. Peak RSS stays low.
    # (Raw numpy mmap would be pickled → materializes entire array. Don't do that.)
    shard_mb_est = (patches_buf.nbytes + cls_buf.nbytes) / 1e6
    log.info(f"Saving shard to {shard_path} (via .tmp, ~{shard_mb_est:.0f} MB)...")
    t0 = time.perf_counter()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    filenames = [p.name for p in jpg_paths]

    tmp = shard_path.with_suffix(".tmp")
    torch.save(
        {
            "patches": torch.from_numpy(patches_buf),
            "cls": torch.from_numpy(cls_buf),
            "paths": filenames,
            "class_idxs": torch.zeros(n, dtype=torch.int32),
            "image_hashes": hashes,
            "failed_indices": failed,
            # Metadata
            "tar_name": cfg.tar.name,
            "image_size": cfg.image_size,
            "teacher_repo_id": cfg.teacher_repo_id,
            "dtype": str(STORAGE_DTYPE),
            "embed_dim": embed_dim,
            "n_patches": n_patches,
            "n_images": n,
            "created_at": datetime.now(UTC).isoformat(),
            "git_commit": get_git_commit(),
        },
        tmp,
    )
    tmp.rename(shard_path)
    t_save = time.perf_counter() - t0

    # Clean up mmap temp files (data is now in the .pt shard)
    del patches_buf, cls_buf
    patches_mmap_path.unlink(missing_ok=True)
    cls_mmap_path.unlink(missing_ok=True)

    shard_mb = shard_path.stat().st_size / 1e6
    t_total = time.perf_counter() - t_start

    log.info(f"Saved {shard_path.name} ({shard_mb:.0f} MB, {n} images)")
    log.info(
        f"Timing: extract={t_extract:.0f}s teacher={t_teacher:.0f}s "
        f"inference={t_inference:.0f}s save={t_save:.0f}s total={t_total:.0f}s"
    )
    try:
        import resource
        peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KB → MB
        peak_gpu_mb = torch.cuda.max_memory_allocated() / 1e6
        log.info(f"Peak memory: GPU={peak_gpu_mb:.0f}MB CPU_RSS={peak_rss_mb:.0f}MB")
    except Exception:
        pass


if __name__ == "__main__":
    main(tyro.cli(Config))
