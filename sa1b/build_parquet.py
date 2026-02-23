"""Build a parquet index from a flat directory of images.

Output schema matches what export_features.py expects:
  - path: str (relative to --image-dir)
  - class_idx: int (always 0 — no class structure)

Usage:
    uv run python scripts/build_parquet.py --image-dir /path/to/images --output index.parquet
    uv run python scripts/build_parquet.py --image-dir /path/to/images --output index.parquet --glob '*.png'
"""

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(description="Build parquet index from image directory")
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--glob", type=str, default="*.jpg")
    args = parser.parse_args()

    assert args.image_dir.is_dir(), f"Not a directory: {args.image_dir}"

    paths = sorted(p.relative_to(args.image_dir).as_posix() for p in args.image_dir.glob(args.glob))
    assert len(paths) > 0, f"No files matching {args.glob!r} in {args.image_dir}"

    table = pa.table({"path": paths, "class_idx": [0] * len(paths)})
    pq.write_table(table, args.output)
    print(f"Wrote {len(paths)} entries to {args.output}")


if __name__ == "__main__":
    main()
