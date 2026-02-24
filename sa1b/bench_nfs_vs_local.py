"""Benchmark NFS vs local NVMe for shard access patterns.

Tests:
  1. Raw sequential read speed: NFS vs $SLURM_TMPDIR (dd-equivalent)
  2. mmap page fault speed: NFS vs local for patches-sized random reads
  3. Full per-op serial breakdown with shard on local NVMe
  4. DataLoader pipelined throughput with local shard at various worker counts

Usage:
  uv run python sa1b/bench_nfs_vs_local.py \
      --shard-path /path/to/shard.pt --tar-path /path/to/tar
"""

import logging
import mmap
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import tyro
from torch.utils.data import DataLoader

from canvit_pretrain.train.data.shards import AllShardsDataset
from canvit_pretrain.train.data.tar_images import TarImageReader, scan_tar_headers
from canvit_utils.transforms import preprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


@dataclass
class Config:
    shard_path: Path
    tar_path: Path
    tmpdir: Path = Path(os.environ.get("SLURM_TMPDIR", "/tmp"))
    workers: list[int] = field(default_factory=lambda: [0, 1, 2, 4, 8])
    n_serial: int = 500
    n_batches: int = 40
    batch_size: int = 64
    image_size: int = 1024


def bench_sequential_read(path: Path, label: str, read_gb: float = 2.0) -> float:
    """Raw sequential read speed (dd-equivalent). Returns MB/s."""
    size = path.stat().st_size
    to_read = min(int(read_gb * 1e9), size)
    buf_size = 4 * 1024 * 1024  # 4 MB blocks

    t0 = time.perf_counter()
    with open(path, "rb") as f:
        read_total = 0
        while read_total < to_read:
            chunk = f.read(min(buf_size, to_read - read_total))
            if not chunk:
                break
            read_total += len(chunk)
    elapsed = time.perf_counter() - t0
    mb_s = read_total / elapsed / 1e6
    log.info(f"  {label}: {read_total/1e9:.1f} GB in {elapsed:.1f}s → {mb_s:.0f} MB/s")
    return mb_s


def bench_mmap_sequential(path: Path, label: str, chunk_bytes: int = 6_291_456, n_chunks: int = 200) -> float:
    """Sequential mmap access in patches-sized chunks. Returns MB/s."""
    fd = open(path, "rb")
    mm = mmap.mmap(fd.fileno(), 0, access=mmap.ACCESS_READ)
    file_size = mm.size()

    # Sequential access
    total_bytes = 0
    t0 = time.perf_counter()
    offset = 0
    for _ in range(n_chunks):
        if offset + chunk_bytes > file_size:
            offset = 0
        _ = bytes(mm[offset:offset + chunk_bytes])  # force page faults + copy
        total_bytes += chunk_bytes
        offset += chunk_bytes
    elapsed = time.perf_counter() - t0
    mm.close()
    fd.close()
    mb_s = total_bytes / elapsed / 1e6
    log.info(f"  {label}: {n_chunks} × {chunk_bytes/1e6:.1f}MB = {total_bytes/1e6:.0f}MB in {elapsed:.2f}s → {mb_s:.0f} MB/s")
    return mb_s


def bench_mmap_with_madvise(path: Path, label: str, chunk_bytes: int = 6_291_456, n_chunks: int = 200) -> float:
    """mmap with MADV_SEQUENTIAL hint."""
    fd = open(path, "rb")
    mm = mmap.mmap(fd.fileno(), 0, access=mmap.ACCESS_READ)
    mm.madvise(mmap.MADV_SEQUENTIAL)
    file_size = mm.size()

    total_bytes = 0
    t0 = time.perf_counter()
    offset = 0
    for _ in range(n_chunks):
        if offset + chunk_bytes > file_size:
            offset = 0
        _ = bytes(mm[offset:offset + chunk_bytes])
        total_bytes += chunk_bytes
        offset += chunk_bytes
    elapsed = time.perf_counter() - t0
    mm.close()
    fd.close()
    mb_s = total_bytes / elapsed / 1e6
    log.info(f"  {label}: {n_chunks} × {chunk_bytes/1e6:.1f}MB → {mb_s:.0f} MB/s (MADV_SEQUENTIAL)")
    return mb_s


def bench_serial_breakdown(shard_path: Path, tar_path: Path, size: int, n: int, label: str) -> None:
    """Per-operation serial breakdown."""
    shard = torch.load(shard_path, map_location="cpu", weights_only=False, mmap=True)
    reader = TarImageReader(tar_path, index=scan_tar_headers(tar_path))
    transform = preprocess(size)

    paths = [shard["paths"][i] for i in range(n)]
    jpeg_bytes = sum(reader.index[p][1] for p in paths)
    patch_bytes_each = shard["patches"][0].nelement() * shard["patches"][0].element_size()

    t_read = t_transform = t_patches = 0.0
    for i, path in enumerate(paths):
        t0 = time.perf_counter()
        img = reader.read_image(path)
        t_read += time.perf_counter() - t0

        t0 = time.perf_counter()
        transform(img)
        t_transform += time.perf_counter() - t0

        t0 = time.perf_counter()
        shard["patches"][i].clone()
        t_patches += time.perf_counter() - t0

    reader.close()
    del shard

    total = t_read + t_transform + t_patches
    log.info(f"  Serial [{label}] ({n} imgs @ {size}px):")
    log.info(f"    read_image:    {t_read/n*1000:.1f}ms/img  {t_read/total*100:.0f}%  {jpeg_bytes/t_read/1e6:.0f} MB/s")
    log.info(f"    transform:     {t_transform/n*1000:.1f}ms/img  {t_transform/total*100:.0f}%")
    log.info(f"    patches clone: {t_patches/n*1000:.1f}ms/img  {t_patches/total*100:.0f}%  {n*patch_bytes_each/t_patches/1e6:.0f} MB/s")
    log.info(f"    TOTAL: {n/total:.1f} img/s serial")


def bench_dataloader(shard_files: list[Path], tar_dir: Path, size: int, nw: int,
                     n_batches: int, batch_size: int, label: str) -> None:
    """Pipelined DataLoader throughput."""
    dataset = AllShardsDataset(
        shard_files=shard_files,
        image_size=size,
        start_shard=0,
        tar_dir=tar_dir,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=nw,
        pin_memory=False,
        drop_last=True,
        persistent_workers=nw > 0,
    )
    it = iter(loader)

    # First batch (cold)
    t0 = time.perf_counter()
    next(it)
    t_first = time.perf_counter() - t0

    # Steady-state
    t0 = time.perf_counter()
    for _ in range(n_batches - 1):
        next(it)
    t_steady = time.perf_counter() - t0

    del loader

    steady_imgs = (n_batches - 1) * batch_size
    log.info(
        f"  DL [{label}, {nw}w]: first={t_first:.2f}s, "
        f"steady={steady_imgs} imgs in {t_steady:.2f}s → {steady_imgs/t_steady:.1f} img/s"
    )


def main(cfg: Config) -> None:
    cpus = len(os.sched_getaffinity(0))
    log.info("=== NFS vs Local NVMe Benchmark ===")
    log.info(f"Shard: {cfg.shard_path} ({cfg.shard_path.stat().st_size/1e9:.1f} GB)")
    log.info(f"Tar:   {cfg.tar_path} ({cfg.tar_path.stat().st_size/1e9:.1f} GB)")
    log.info(f"TMPDIR: {cfg.tmpdir}")
    log.info(f"CPUs: {cpus}, workers: {cfg.workers}")

    # ========== Phase 1: Raw sequential read speed ==========
    log.info("\n--- Sequential Read Speed ---")
    bench_sequential_read(cfg.shard_path, "NFS shard (2 GB)")
    bench_sequential_read(cfg.tar_path, "NFS tar (2 GB)")

    # ========== Phase 2: mmap access patterns on NFS ==========
    log.info("\n--- mmap Patterns (NFS, 6.3MB chunks = 1 patches sample) ---")
    bench_mmap_sequential(cfg.shard_path, "NFS mmap sequential")
    bench_mmap_with_madvise(cfg.shard_path, "NFS mmap MADV_SEQUENTIAL")

    # ========== Phase 3: Copy shard to local NVMe ==========
    local_shard = cfg.tmpdir / cfg.shard_path.name
    log.info(f"\n--- Copy shard to local ({cfg.tmpdir}) ---")
    t0 = time.perf_counter()
    shutil.copy2(cfg.shard_path, local_shard)
    t_copy = time.perf_counter() - t0
    shard_gb = cfg.shard_path.stat().st_size / 1e9
    log.info(f"  Copied {shard_gb:.1f} GB in {t_copy:.1f}s → {shard_gb/t_copy*1000:.0f} MB/s")

    # ========== Phase 4: Local NVMe sequential + mmap ==========
    log.info("\n--- Local NVMe Read Speed ---")
    bench_sequential_read(local_shard, "Local shard (2 GB)")
    bench_mmap_sequential(local_shard, "Local mmap sequential")

    # ========== Phase 5: Serial breakdown — NFS shard vs local shard ==========
    log.info(f"\n--- Serial Breakdown @ {cfg.image_size}px ---")
    bench_serial_breakdown(cfg.shard_path, cfg.tar_path, cfg.image_size, cfg.n_serial, "NFS shard")
    bench_serial_breakdown(local_shard, cfg.tar_path, cfg.image_size, cfg.n_serial, "local shard")

    # ========== Phase 6: DataLoader — NFS vs local, varying workers ==========
    log.info(f"\n--- DataLoader Throughput @ {cfg.image_size}px ---")
    tar_dir = cfg.tar_path.parent
    for nw in cfg.workers:
        bench_dataloader(
            [cfg.shard_path], tar_dir, cfg.image_size, nw,
            cfg.n_batches, cfg.batch_size, "NFS",
        )
    for nw in cfg.workers:
        bench_dataloader(
            [local_shard], tar_dir, cfg.image_size, nw,
            cfg.n_batches, cfg.batch_size, "local",
        )

    # Cleanup
    local_shard.unlink(missing_ok=True)
    log.info("\nDone.")


if __name__ == "__main__":
    main(tyro.cli(Config))
