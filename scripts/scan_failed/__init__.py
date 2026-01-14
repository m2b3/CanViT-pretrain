"""Scan feature shards for failed indices, build master index.

Efficiently scans shards in parallel using mmap (tensor data never paged in).
Outputs a parquet with global failed indices for downstream filtering.

Usage:
    uv run python -m scripts.scan_failed /path/to/shards -o failed.parquet -w 32
"""

import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import torch
import tyro
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def extract_failed(path: Path) -> tuple[int, int, int, list[int]]:
    """Extract (shard_id, start_idx, shard_size, local_failed). mmap avoids loading tensors."""
    data = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    shard_size = data["end_idx"] - data["start_idx"]
    return data["shard_id"], data["start_idx"], shard_size, data["failed_indices"]


@dataclass
class Config:
    shards_dir: Path
    out: Path | None = None
    workers: int = 32


def main(cfg: Config) -> None:
    log.info("=" * 60)
    log.info("Scan Failed Indices")
    log.info("=" * 60)
    log.info(f"shards_dir: {cfg.shards_dir}")
    log.info(f"out: {cfg.out or '(auto)'}")
    log.info(f"workers: {cfg.workers}")

    assert cfg.shards_dir.exists(), f"Directory not found: {cfg.shards_dir}"
    paths = sorted(cfg.shards_dir.glob("*.pt"))
    assert paths, f"No .pt files in {cfg.shards_dir}"

    log.info(f"Found {len(paths)} shard files")
    log.info(f"First: {paths[0].name}, Last: {paths[-1].name}")

    records: list[dict] = []
    total_images = 0
    shards_with_failures = 0
    errors = 0
    t0 = time.perf_counter()

    log.info(f"Starting parallel scan with {cfg.workers} workers...")

    with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
        futures = {ex.submit(extract_failed, p): p for p in paths}
        pbar = tqdm(as_completed(futures), total=len(paths), desc="Shards", unit="shard")

        for fut in pbar:
            path = futures[fut]
            try:
                shard_id, start_idx, shard_size, local_failed = fut.result()
                total_images += shard_size

                if local_failed:
                    shards_with_failures += 1
                    log.debug(
                        f"Shard {shard_id:05d}: {len(local_failed)} failed "
                        f"(start={start_idx}, size={shard_size})"
                    )

                for local_idx in local_failed:
                    records.append({
                        "global_idx": start_idx + local_idx,
                        "shard_id": shard_id,
                        "local_idx": local_idx,
                    })

                pbar.set_postfix({
                    "failed": len(records),
                    "errors": errors,
                })

            except Exception as e:
                errors += 1
                log.error(f"FAILED to read {path.name}: {e}")

    elapsed = time.perf_counter() - t0
    log.info(f"Scan complete in {elapsed:.1f}s ({len(paths) / elapsed:.1f} shards/s)")
    log.info(f"Total images scanned: {total_images:,}")
    log.info(f"Total failed: {len(records):,} ({100 * len(records) / max(1, total_images):.4f}%)")
    log.info(f"Shards with failures: {shards_with_failures}/{len(paths)}")
    log.info(f"Read errors: {errors}")

    df = pl.DataFrame(records).sort("global_idx")
    out = cfg.out or cfg.shards_dir / "failed_index.parquet"

    log.info(f"Writing {len(df)} records to {out}")
    df.write_parquet(out)
    log.info(f"Output size: {out.stat().st_size / 1024:.1f} KB")

    if len(df) > 0:
        log.info(f"Shard distribution:\n{df.group_by('shard_id').len().sort('shard_id')}")
        log.info(f"\nFirst 20 failed:\n{df.head(20)}")

    log.info("Done.")


if __name__ == "__main__":
    main(tyro.cli(Config))
