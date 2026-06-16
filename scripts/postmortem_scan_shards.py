"""Read-only scan of precomputed teacher feature shards for bad values.

Tests the "bad data" hypothesis: do the stored DINOv3 teacher targets
(`patches`, `cls`) contain non-finite or extreme (fp16-overflow-adjacent)
values that would produce a NaN MSE target?

Scans selected shards (by index) fully, reporting dtype, finite min/max,
non-finite counts, and the worst per-sample magnitudes. mmap load + chunked
reduction to bound RAM.
"""

import argparse
from pathlib import Path

import torch


def scan_tensor(name: str, t: torch.Tensor, *, chunk: int = 256) -> dict:
    n = t.shape[0]
    nonfinite = 0
    gmin = float("inf")
    gmax = float("-inf")
    worst_absmax = 0.0
    worst_idx = -1
    for s in range(0, n, chunk):
        blk = t[s : s + chunk].float()
        finite_mask = torch.isfinite(blk)
        nf = int((~finite_mask).sum().item())
        nonfinite += nf
        fin = blk[finite_mask]
        if fin.numel():
            gmin = min(gmin, float(fin.min().item()))
            gmax = max(gmax, float(fin.max().item()))
        # per-sample max-abs over the block (finite only)
        blk_safe = torch.where(finite_mask, blk.abs(), torch.zeros_like(blk))
        per_sample = blk_safe.reshape(blk.shape[0], -1).amax(dim=1)
        bidx = int(per_sample.argmax().item())
        if float(per_sample[bidx].item()) > worst_absmax:
            worst_absmax = float(per_sample[bidx].item())
            worst_idx = s + bidx
    return {
        "name": name,
        "dtype": str(t.dtype),
        "shape": tuple(t.shape),
        "nonfinite": nonfinite,
        "finite_min": gmin,
        "finite_max": gmax,
        "worst_absmax": worst_absmax,
        "worst_sample_idx": worst_idx,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("shards_dir", type=Path)
    ap.add_argument("--indices", type=int, nargs="+", required=True, help="shard indices to scan (sorted glob order)")
    args = ap.parse_args()

    shard_files = sorted(args.shards_dir.glob("*.pt"))
    print(f"shards_dir={args.shards_dir}  total_shards={len(shard_files)}  scanning indices={args.indices}\n")
    for idx in args.indices:
        assert 0 <= idx < len(shard_files), f"index {idx} out of range (0..{len(shard_files)-1})"
        sf = shard_files[idx]
        shard = torch.load(sf, map_location="cpu", weights_only=False, mmap=True)
        n = len(shard["paths"])
        failed = shard.get("failed_indices", [])
        print(f"=== shard[{idx}] {sf.name}  n_samples={n}  failed_indices={len(failed)} ===")
        for key in ("patches", "cls"):
            if key not in shard:
                print(f"  {key}: <absent>")
                continue
            r = scan_tensor(key, shard[key])
            flag = "  <-- NONFINITE" if r["nonfinite"] else ""
            print(f"  {key:8s} dtype={r['dtype']} shape={r['shape']}")
            print(f"           finite_min={r['finite_min']:.4e} finite_max={r['finite_max']:.4e} "
                  f"nonfinite={r['nonfinite']}{flag}")
            print(f"           worst |x|={r['worst_absmax']:.4e} at sample {r['worst_sample_idx']} "
                  f"(path={shard['paths'][r['worst_sample_idx']] if r['worst_sample_idx']>=0 else 'n/a'})")
        print()
        del shard


if __name__ == "__main__":
    main()
