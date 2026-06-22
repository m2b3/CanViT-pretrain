"""WeightWatcher (data-free) spectral analysis of one CanViT pretraining checkpoint.

Loads the model, runs ww.analyze(), writes per-layer details (CSV) + a summary
(JSON) keyed by training step. Swept across the divergence and correlated against
the known val-quality curve, this tells us which WW metrics (if any) track the
failure for OUR model — and whether any move at the step-1,719,760 spike, which
predates the loss degradation by ~160k steps. CPU only, no data, no GPU.
"""

import argparse
import json
import re
from pathlib import Path

import weightwatcher as ww

from canvit_pretrain.checkpoint import load_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--fix-fingers", action="store_true",
                    help="use fix_fingers='clip_xmax' for reliable alpha (removes spurious >8 'fingers')")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    m = re.search(r"step-(\d+)", args.ckpt.name)
    assert m is not None, f"no step in {args.ckpt.name}"
    step = int(m.group(1))

    model, ckpt = load_model(args.ckpt, "cpu")
    watcher = ww.WeightWatcher(model=model)
    details = watcher.analyze(fix_fingers="clip_xmax") if args.fix_fingers else watcher.analyze()
    summary = watcher.get_summary(details)

    details["step"] = step
    details.to_csv(args.out_dir / f"ww_step-{step}.csv", index=False)
    out = {
        "step": step,
        "train_loss": ckpt.get("train_loss"),
        "summary": {k: float(v) for k, v in summary.items()},
    }
    (args.out_dir / f"ww_step-{step}.json").write_text(json.dumps(out, indent=2))
    print(f"step {step}: {out['summary']}")


if __name__ == "__main__":
    main()
