"""Download SA-1B tarballs from Meta CDN.

Idempotent: skips tars that already exist. Safe to ctrl-C and rerun.
Partial downloads use a .tmp suffix and are resumed on next run.

Each tar is ~10.5 GB (gzipped, despite .tar extension), ~11k JPEGs + masks.
~50 MB/s on Nibi login nodes → ~3.5 min per tar.

Usage (from repo root):
    uv run python scripts/sa1b/download.py                # all 1000 tars
    uv run python scripts/sa1b/download.py --limit 3      # first 3 only
"""

import csv
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Download SA-1B tarballs from Meta CDN")
    parser.add_argument("--limit", type=int, default=0, help="Max tars to process (0 = all)")
    parser.add_argument("--tar-dir", type=Path, default=None, help="Override $SA1B_TAR_DIR")
    parser.add_argument("--links", type=Path, default=None, help="Override $SA1B_LINKS")
    args = parser.parse_args()

    tar_dir = args.tar_dir or Path(os.environ.get("SA1B_TAR_DIR", ""))
    links = args.links or Path(os.environ.get("SA1B_LINKS", ""))
    assert tar_dir != Path(""), "Set --tar-dir or $SA1B_TAR_DIR"
    assert links != Path(""), "Set --links or $SA1B_LINKS"
    assert links.is_file(), f"Links file not found: {links}"

    tar_dir.mkdir(parents=True, exist_ok=True)

    # sa1b_links.tsv: TSV with header "file_name\tcdn_link", 1000 data rows
    with open(links) as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    assert len(rows) > 0, f"No rows in {links}"
    assert "file_name" in rows[0] and "cdn_link" in rows[0], (
        f"Expected columns 'file_name' and 'cdn_link', got {list(rows[0].keys())}"
    )

    if args.limit > 0:
        rows = rows[: args.limit]

    print(f"=== SA-1B Download ===")
    print(f"TAR_DIR: {tar_dir}")
    print(f"LINKS:   {links} ({len(rows)} tars to process)")
    print(f"======================")

    downloaded = 0
    skipped = 0

    for row in rows:
        filename = row["file_name"]
        url = row["cdn_link"]
        dest = tar_dir / filename
        tmp = dest.with_suffix(dest.suffix + ".tmp")

        if dest.exists():
            print(f"[skip] {filename}")
            skipped += 1
            continue

        print(f"[download] {filename}")
        result = subprocess.run(
            ["wget", "-c", "-q", "--show-progress", "-O", str(tmp), url],
            check=False,
        )
        if result.returncode != 0:
            print(f"[FAILED] {filename} (wget exit {result.returncode})", file=sys.stderr)
            # .tmp stays for resume on next run
            break

        tmp.rename(dest)
        size_mb = dest.stat().st_size / (1024 * 1024)
        downloaded += 1
        print(f"[done] {filename} ({size_mb:.0f} MB)")

    print(f"=== Complete: {downloaded} downloaded, {skipped} skipped ===")


if __name__ == "__main__":
    main()
