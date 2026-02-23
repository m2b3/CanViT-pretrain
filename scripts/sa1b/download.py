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
import re
import subprocess
import sys
from pathlib import Path

# sa_NNNNNN.tar — the actual image tars. The TSV also contains
# sa_images_ids.txt (metadata) and sa_co_gold.tar (benchmark), which we skip.
IMAGE_TAR_RE = re.compile(r"^sa_\d{6}\.tar$")


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

    # sa1b_links.tsv: TSV with header "file_name\tcdn_link"
    # Contains 1000 image tars + 2 non-image files (sa_images_ids.txt, sa_co_gold.tar)
    with open(links) as f:
        reader = csv.DictReader(f, delimiter="\t")
        assert reader.fieldnames == ["file_name", "cdn_link"], f"Unexpected columns: {reader.fieldnames}"
        rows = [r for r in reader if IMAGE_TAR_RE.match(r["file_name"])]

    assert len(rows) == 1000, f"Expected 1000 image tars, got {len(rows)}"

    if args.limit > 0:
        rows = rows[: args.limit]

    print(f"=== SA-1B Download ===")
    print(f"TAR_DIR: {tar_dir}")
    print(f"LINKS:   {links} ({len(rows)} tars to process)")
    print(f"======================")

    # Also download sa_images_ids.txt (canonical list of all image IDs) if not present
    with open(links) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            if r["file_name"] == "sa_images_ids.txt":
                ids_dest = tar_dir.parent / "sa_images_ids.txt"
                if not ids_dest.exists():
                    print(f"[download] sa_images_ids.txt -> {ids_dest}")
                    subprocess.run(
                        ["wget", "-c", "-q", "--show-progress", "-O", str(ids_dest), r["cdn_link"]],
                        check=True,
                    )
                else:
                    print(f"[skip] sa_images_ids.txt (already exists)")
                break

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
