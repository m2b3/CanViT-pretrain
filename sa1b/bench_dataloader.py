"""Benchmark SA-1B dataloader: real code path, component breakdown.

CPU-only — no GPU needed. Uses the actual AllShardsDataset + DataLoader
with the same settings as training (pin_memory=cuda_available, drop_last=True,
persistent_workers). Measures throughput at different image sizes and worker counts.

Usage:
  uv run python sa1b/bench_dataloader.py \
      --shard-dir /path/to/shards --tar-dir /path/to/tars
"""

import logging
import os
import resource
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import torch
import tyro
from PIL import Image
from torch.utils.data import DataLoader

from canvit_pretrain.train.data.shards import AllShardsDataset
from canvit_pretrain.train.data.tar_images import TarImageReader
from canvit_utils.transforms import preprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


@dataclass
class Config:
    shard_dir: Path
    tar_dir: Path
    image_sizes: list[int] = field(default_factory=lambda: [1024, 1500])
    workers: list[int] = field(default_factory=lambda: [0, 1, 2, 4, 8])
    n_images: int = 200
    batch_size: int = 64  # same as training


def bench_components(tar_path: Path, names: list[str], size: int, n: int) -> None:
    """Time individual components: mmap read, PIL decode, transform."""
    reader = TarImageReader(tar_path)
    transform = preprocess(size)
    names = names[:n]

    # 1) Raw mmap read (bytes from tar)
    t0 = time.perf_counter()
    raw_buffers = []
    for name in names:
        offset, length = reader.index[name]
        raw_buffers.append(reader._mmap[offset : offset + length])
    t_mmap = time.perf_counter() - t0

    # 2) PIL decode (bytes → Image)
    t0 = time.perf_counter()
    images = []
    for buf in raw_buffers:
        img = Image.open(BytesIO(buf)).convert("RGB")
        img.load()
        images.append(img)
    t_decode = time.perf_counter() - t0

    # 3) Transform (Image → Tensor at target size)
    t0 = time.perf_counter()
    for img in images:
        transform(img)
    t_transform = time.perf_counter() - t0

    reader.close()

    total = t_mmap + t_decode + t_transform
    log.info(f"  Components ({n} images @ {size}px):")
    log.info(f"    mmap read:  {t_mmap:.3f}s ({t_mmap/n*1000:.1f}ms/img, {t_mmap/total*100:.0f}%)")
    log.info(f"    PIL decode: {t_decode:.3f}s ({t_decode/n*1000:.1f}ms/img, {t_decode/total*100:.0f}%)")
    log.info(f"    transform:  {t_transform:.3f}s ({t_transform/n*1000:.1f}ms/img, {t_transform/total*100:.0f}%)")
    log.info(f"    total:      {total:.3f}s ({n/total:.1f} img/s)")


def bench_dataloader(cfg: Config, size: int, num_workers: int) -> None:
    """Measure end-to-end DataLoader throughput using the real AllShardsDataset."""
    shard_files = sorted(Path(cfg.shard_dir).glob("*.pt"))
    assert shard_files, f"No shards in {cfg.shard_dir}"

    use_pin_memory = torch.cuda.is_available()
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

    n_batches = 0
    n_samples = 0
    t0 = time.perf_counter()
    for batch in loader:
        n_batches += 1
        n_samples += batch[0].shape[0]
        if n_samples >= cfg.n_images:
            break
    elapsed = time.perf_counter() - t0

    # Clean up persistent workers
    del loader

    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # Linux: KB
    rss_mb = rss_kb / 1024
    log.info(
        f"  DataLoader: {num_workers}w, {size}px, bs={cfg.batch_size}, "
        f"{n_samples} imgs in {elapsed:.2f}s → {n_samples/elapsed:.1f} img/s, "
        f"pin_memory={use_pin_memory}, RSS={rss_mb:.0f}MB"
    )


def bench_shard_load(shard_path: Path) -> None:
    """Time shard loading with mmap."""
    t0 = time.perf_counter()
    shard = torch.load(shard_path, map_location="cpu", weights_only=False, mmap=True)
    n = len(shard["paths"])
    t_load = time.perf_counter() - t0
    log.info(f"  Shard load (mmap): {t_load:.2f}s, {n} samples")

    # Access patches[0] to measure page fault cost
    t0 = time.perf_counter()
    _ = shard["patches"][0].clone()
    t_access = time.perf_counter() - t0
    log.info(f"  First patch access: {t_access*1000:.1f}ms")
    del shard


def main(cfg: Config) -> None:
    log.info("=== Dataloader Benchmark ===")
    log.info(f"shard_dir: {cfg.shard_dir}")
    log.info(f"tar_dir: {cfg.tar_dir}")
    log.info(f"image_sizes: {cfg.image_sizes}")
    log.info(f"workers: {cfg.workers}")
    log.info(f"n_images: {cfg.n_images}, batch_size: {cfg.batch_size}")
    log.info(f"PID: {os.getpid()}, CPUs: {os.cpu_count()}, CUDA: {torch.cuda.is_available()}")

    shard_files = sorted(Path(cfg.shard_dir).glob("*.pt"))
    assert shard_files, f"No shards in {cfg.shard_dir}"
    log.info(f"Shards: {len(shard_files)}")

    # Find a tar for component benchmarks
    tar_path = cfg.tar_dir / f"{shard_files[0].stem}.tar"
    assert tar_path.exists(), f"Tar not found: {tar_path}"

    # 1) Shard load timing
    log.info("\n--- Shard Load ---")
    bench_shard_load(shard_files[0])

    # 2) Tar index timing
    log.info("\n--- Tar Index ---")
    t0 = time.perf_counter()
    reader = TarImageReader(tar_path)
    t_index = time.perf_counter() - t0
    names = list(reader.index.keys())
    log.info(f"  Index: {len(names)} JPEGs in {t_index:.1f}s")
    reader.close()

    # 3) Component breakdown per image size
    for size in cfg.image_sizes:
        log.info(f"\n--- Components @ {size}px ---")
        bench_components(tar_path, names, size, cfg.n_images)

    # 4) DataLoader throughput: size × workers grid
    log.info(f"\n--- DataLoader Throughput (real AllShardsDataset) ---")
    for size in cfg.image_sizes:
        for nw in cfg.workers:
            bench_dataloader(cfg, size, nw)


if __name__ == "__main__":
    main(tyro.cli(Config))
