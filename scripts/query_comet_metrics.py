"""Pull metric time-series from a Comet experiment around a step window.

Read-only forensic helper: list metric names, then dump the requested metrics
(default: grad-norm + loss + val accuracy) filtered to a step window, to inspect
precursors (grad spikes, val drops) before a training NaN.
"""

import argparse
import os
from pathlib import Path

from comet_ml.api import API


def get_key() -> str:
    k = os.environ.get("COMET_API_KEY")
    if k:
        return k
    p = Path.home() / "comet_api_key.txt"
    assert p.exists(), "no COMET_API_KEY env and no ~/comet_api_key.txt"
    return p.read_text().strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True, help="Comet experiment key")
    ap.add_argument("--min-step", type=int, default=0)
    ap.add_argument("--max-step", type=int, default=10**9)
    ap.add_argument("--metrics", nargs="*", default=None,
                    help="metric names to dump; default = grad/loss/val matches")
    args = ap.parse_args()

    api = API(api_key=get_key())
    exp = api.get_experiment_by_key(args.key)
    assert exp is not None, f"no experiment {args.key}"
    names = sorted(m["name"] for m in exp.get_metrics_summary())
    print(f"experiment {args.key}  ({len(names)} metrics)")
    print("ALL METRIC NAMES:")
    for n in names:
        print(f"  {n}")

    if args.metrics:
        wanted = args.metrics
    else:
        wanted = [n for n in names if any(k in n.lower() for k in
                  ("grad_norm", "total_loss", "top1", "acc", "clf", "loss"))]
    print(f"\n=== DUMPING {len(wanted)} metrics in step window [{args.min_step}, {args.max_step}] ===")
    for name in wanted:
        pts = exp.get_metrics(name)
        rows = []
        for p in pts:
            s = p.get("step")
            if s is None:
                continue
            s = int(s)
            if args.min_step <= s <= args.max_step:
                rows.append((s, float(p["metricValue"])))
        rows.sort()
        if not rows:
            continue
        vals = [v for _, v in rows]
        print(f"\n--- {name}  (n={len(rows)} in window) min={min(vals):.4g} max={max(vals):.4g} ---")
        for s, v in rows:
            print(f"    step {s}: {v:.6g}")


if __name__ == "__main__":
    main()
