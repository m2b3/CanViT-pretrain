"""Benchmark SA-1B dataloader: per-op breakdown + pipelined throughput.

CPU-only. Measures:
  1. Serial per-op breakdown mimicking AllShardsDataset.__iter__ with MB/s
  2. Pipelined DataLoader: creation time, first-batch latency, steady-state throughput
  3. Per-process RSS (main + worker children)

Usage:
  uv run python sa1b/bench_dataloader.py \
      --shard-dir /path/to/shards --tar-dir /path/to/tars
"""

import logging
import os
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
    shard_dir: Path
    tar_dir: Path
    image_sizes: list[int] = field(default_factory=lambda: [1024, 1500])
    workers: list[int] = field(default_factory=lambda: [0, 1, 2, 4])
    n_serial: int = 200
    n_batches: int = 30
    batch_size: int = 64


def _rss_kb(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError):
        pass
    return 0


def get_total_rss_mb() -> float:
    """RSS of main process + all direct children (DataLoader workers)."""
    pid = os.getpid()
    total = _rss_kb(pid)
    try:
        with open(f"/proc/{pid}/task/{pid}/children") as f:
            for child_pid in f.read().split():
                total += _rss_kb(int(child_pid))
    except FileNotFoundError:
        pass
    return total / 1024


def bench_serial(shard_path: Path, tar_path: Path, size: int, n: int) -> None:
    """Per-operation breakdown mimicking AllShardsDataset.__iter__, no pipelining."""
    shard = torch.load(shard_path, map_location="cpu", weights_only=False, mmap=True)
    reader = TarImageReader(tar_path, index=scan_tar_headers(tar_path))
    transform = preprocess(size)

    paths = [shard["paths"][i] for i in range(n)]
    jpeg_bytes_total = sum(reader.index[p][1] for p in paths)
    patch_elem_bytes = shard["patches"][0].nelement() * shard["patches"][0].element_size()
    patches_bytes_total = n * patch_elem_bytes

    t_read = 0.0
    t_transform = 0.0
    t_patches = 0.0
    t_cls = 0.0

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

        t0 = time.perf_counter()
        shard["cls"][i].clone()
        t_cls += time.perf_counter() - t0

    reader.close()
    del shard

    total = t_read + t_transform + t_patches + t_cls
    log.info(f"  Serial breakdown ({n} imgs @ {size}px):")
    log.info(f"    read_image (mmap+decode): {t_read:.2f}s  {t_read/n*1000:.1f}ms/img  {t_read/total*100:.0f}%  {jpeg_bytes_total/t_read/1e6:.0f} MB/s")
    log.info(f"    transform (resize+crop):  {t_transform:.2f}s  {t_transform/n*1000:.1f}ms/img  {t_transform/total*100:.0f}%")
    log.info(f"    patches clone (mmap→RAM): {t_patches:.2f}s  {t_patches/n*1000:.1f}ms/img  {t_patches/total*100:.0f}%  {patches_bytes_total/t_patches/1e6:.0f} MB/s")
    log.info(f"    cls clone:               {t_cls:.2f}s  {t_cls/n*1000:.1f}ms/img")
    log.info(f"    TOTAL:                   {total:.2f}s  {n/total:.1f} img/s serial")
    log.info(f"    JPEG input: {jpeg_bytes_total/1e6:.0f} MB total, avg {jpeg_bytes_total/n/1e3:.0f} KB/img")
    log.info(f"    Patches:    {patches_bytes_total/1e6:.0f} MB total, {patch_elem_bytes/1e6:.1f} MB/img")


def bench_dataloader(cfg: Config, size: int, num_workers: int, shard_files: list[Path]) -> None:
    """Pipelined DataLoader: creation, first batch, steady-state."""
    use_pin_memory = torch.cuda.is_available()

    # --- Creation (includes worker spawn + tar index in workers) ---
    t0 = time.perf_counter()
    dataset = AllShardsDataset(
        shard_files=shard_files,
        image_size=size,
        start_shard=0,
        tar_dir=cfg.tar_dir,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    it = iter(loader)
    t_create = time.perf_counter() - t0

    # --- First batch (cold: workers build tar index + load shard + process) ---
    t0 = time.perf_counter()
    batch = next(it)
    t_first = time.perf_counter() - t0

    img_t, patches_t, cls_t, _ = batch
    bytes_per_batch = (
        img_t.nelement() * img_t.element_size()
        + patches_t.nelement() * patches_t.element_size()
        + cls_t.nelement() * cls_t.element_size()
    )

    # --- Steady-state (remaining batches, workers already warm) ---
    t0 = time.perf_counter()
    for _ in range(cfg.n_batches - 1):
        next(it)
    t_steady = time.perf_counter() - t0

    rss = get_total_rss_mb()
    del loader

    steady_batches = cfg.n_batches - 1
    steady_imgs = steady_batches * cfg.batch_size
    steady_mb = steady_batches * bytes_per_batch / 1e6

    log.info(f"  [{num_workers}w, {size}px, bs={cfg.batch_size}, pin_memory={use_pin_memory}]")
    log.info(f"    Creation:    {t_create:.2f}s")
    log.info(f"    First batch: {t_first:.2f}s ({cfg.batch_size} imgs)")
    if t_steady > 0:
        log.info(f"    Steady-state: {steady_imgs} imgs in {t_steady:.2f}s → {steady_imgs/t_steady:.1f} img/s, {steady_mb/t_steady:.0f} MB/s output")
    log.info(f"    Shapes: img={list(img_t.shape)}, patches={list(patches_t.shape)}, cls={list(cls_t.shape)}")
    log.info(f"    RSS (main+workers): {rss:.0f} MB")


def main(cfg: Config) -> None:
    # sched_getaffinity respects cgroups (SLURM); os.cpu_count() reports entire node
    cpus = len(os.sched_getaffinity(0))
    log.info("=== SA-1B Dataloader Benchmark ===")
    log.info(f"shard_dir: {cfg.shard_dir}")
    log.info(f"tar_dir:   {cfg.tar_dir}")
    log.info(f"CPUs: {cpus} (allocated), CUDA: {torch.cuda.is_available()}")
    log.info(f"Serial: {cfg.n_serial} imgs | DataLoader: {cfg.n_batches} batches × {cfg.batch_size}")

    shard_files = sorted(Path(cfg.shard_dir).glob("*.pt"))
    assert shard_files, f"No shards in {cfg.shard_dir}"
    shard_path = shard_files[0]
    tar_path = cfg.tar_dir / f"{shard_path.stem}.tar"
    assert tar_path.exists(), f"Tar not found: {tar_path}"
    log.info(f"Shard: {shard_path.name}, Tar: {tar_path.name}")

    # --- Serial per-op breakdown ---
    for size in cfg.image_sizes:
        log.info(f"\n--- Serial @ {size}px ---")
        bench_serial(shard_path, tar_path, size, cfg.n_serial)

    # --- Pipelined DataLoader ---
    log.info(f"\n--- DataLoader (pipelined) ---")
    for size in cfg.image_sizes:
        for nw in cfg.workers:
            bench_dataloader(cfg, size, nw, shard_files)


if __name__ == "__main__":
    main(tyro.cli(Config))
