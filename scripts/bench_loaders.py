#!/usr/bin/env python
"""Benchmark data loader throughput.

Usage:
    # Raw images (IndexedImageFolder)
    uv run python scripts/bench_loaders.py --train-dir $IN21K_DIR --index-dir $INDEX_DIR --num-workers 16

    # Features + images (FeatureIterableDataset)
    uv run python scripts/bench_loaders.py --shards-dir $FEATURES_DIR/in21k/dinov3_vitb16/512/shards --image-root $IN21K_DIR --num-workers 16

    # Sweep worker counts
    uv run python scripts/bench_loaders.py --train-dir $IN21K_DIR --index-dir $INDEX_DIR --sweep
"""

import argparse
import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from drac_imagenet import IndexedImageFolder

from avp_vit.train.data import train_transform
from avp_vit.train.feature_dataset import FeatureIterableDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 8
DEFAULT_IMAGE_SIZE = 512
DEFAULT_NUM_BATCHES = 100


def _run_benchmark(loader, num_batches: int, batch_size: int) -> float:
    """Common benchmark loop. Returns samples/sec."""
    log.info("Warming up (1 batch)...")
    it = iter(loader)
    next(it)

    log.info(f"Timing {num_batches} batches...")
    t0 = time.perf_counter()
    for i, _ in enumerate(tqdm(it, total=num_batches, desc="Loading")):
        if i >= num_batches - 1:
            break
    elapsed = time.perf_counter() - t0

    total = num_batches * batch_size
    rate = total / elapsed
    log.info(f"  elapsed: {elapsed:.2f}s")
    log.info(f"  throughput: {rate:.1f} img/sec")
    log.info(f"  per batch: {elapsed / num_batches * 1000:.1f}ms")
    return rate


def bench_image_loader(
    train_dir: Path,
    index_dir: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    image_size: int = DEFAULT_IMAGE_SIZE,
    num_batches: int = DEFAULT_NUM_BATCHES,
) -> float:
    """Benchmark raw image loading (IndexedImageFolder)."""
    log.info("=== Image Loader Benchmark ===")
    log.info(f"  train_dir: {train_dir}")
    log.info(f"  index_dir: {index_dir}")
    log.info(f"  batch_size: {batch_size}, num_workers: {num_workers}, image_size: {image_size}")

    tf = train_transform(image_size, (0.8, 1.0))
    ds = IndexedImageFolder(train_dir, index_dir, tf)
    log.info(f"  dataset: {len(ds):,} images")

    persistent = num_workers > 0
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True, persistent_workers=persistent,
    )
    return _run_benchmark(loader, num_batches, batch_size)


def bench_feature_loader(
    shards_dir: Path,
    image_root: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    num_batches: int = DEFAULT_NUM_BATCHES,
) -> float:
    """Benchmark feature loading (FeatureIterableDataset)."""
    log.info("=== Feature Loader Benchmark ===")
    log.info(f"  shards_dir: {shards_dir}")
    log.info(f"  image_root: {image_root}")
    log.info(f"  batch_size: {batch_size}, num_workers: {num_workers}")

    ds = FeatureIterableDataset(shards_dir, image_root)
    log.info(f"  shards: {len(ds.shard_files)}")

    persistent = num_workers > 0
    loader = DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers,
        pin_memory=True, drop_last=True, persistent_workers=persistent,
    )

    # Log shapes from first batch
    it = iter(loader)
    batch = next(it)
    images, patches, cls_tokens, labels = batch
    log.info(f"  shapes: images={tuple(images.shape)}, patches={tuple(patches.shape)}, cls={tuple(cls_tokens.shape)}")

    t0 = time.perf_counter()
    for i, _ in enumerate(tqdm(it, total=num_batches, desc="Loading")):
        if i >= num_batches - 1:
            break
    elapsed = time.perf_counter() - t0

    total = num_batches * batch_size
    rate = total / elapsed
    log.info(f"  elapsed: {elapsed:.2f}s")
    log.info(f"  throughput: {rate:.1f} img/sec")
    log.info(f"  per batch: {elapsed / num_batches * 1000:.1f}ms")
    return rate


def sweep_workers(bench_fn, worker_counts: list[int] = [0, 1, 2, 4, 8, 12, 16], **kwargs) -> dict[int, float]:
    """Sweep num_workers and return throughput for each."""
    results: dict[int, float] = {}
    for nw in worker_counts:
        log.info(f"\n{'='*60}\nnum_workers = {nw}\n{'='*60}")
        results[nw] = bench_fn(num_workers=nw, **kwargs)

    log.info(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for nw, throughput in results.items():
        log.info(f"  {nw} workers: {throughput:.1f} img/sec")
    best = max(results, key=results.get)
    log.info(f"Best: {best} workers @ {results[best]:.1f} img/sec")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark data loaders")
    parser.add_argument("--train-dir", type=Path, help="Path to training images")
    parser.add_argument("--index-dir", type=Path, help="Path to index")
    parser.add_argument("--shards-dir", type=Path, help="Path to feature shards")
    parser.add_argument("--image-root", type=Path, help="Path to images for feature loader")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--num-batches", type=int, default=DEFAULT_NUM_BATCHES)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--sweep", action="store_true", help="Sweep worker counts")
    args = parser.parse_args()

    log.info(f"PyTorch {torch.__version__}")
    log.info(f"Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")

    if args.train_dir:
        assert args.index_dir, "--index-dir required with --train-dir"
        if args.sweep:
            sweep_workers(bench_image_loader, train_dir=args.train_dir, index_dir=args.index_dir,
                          batch_size=args.batch_size, image_size=args.image_size, num_batches=args.num_batches)
        else:
            bench_image_loader(args.train_dir, args.index_dir, batch_size=args.batch_size,
                               num_workers=args.num_workers, image_size=args.image_size, num_batches=args.num_batches)

    if args.shards_dir:
        assert args.image_root, "--image-root required with --shards-dir"
        if args.sweep:
            sweep_workers(bench_feature_loader, shards_dir=args.shards_dir, image_root=args.image_root,
                          batch_size=args.batch_size, num_batches=args.num_batches)
        else:
            bench_feature_loader(args.shards_dir, args.image_root, batch_size=args.batch_size,
                                 num_workers=args.num_workers, num_batches=args.num_batches)
