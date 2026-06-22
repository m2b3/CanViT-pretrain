"""Calibrate WeightWatcher metrics against our own ground truth.

Reads a sweep of per-checkpoint WW outputs (ww_step-*.json + .csv) and the known
val-quality curve (Comet val/scene_cos), then:
  1. correlates each WW summary metric against val quality across the sweep,
  2. compares the pre-spike (step-1717248) vs post-spike (step-1722240) checkpoints
     per layer, to see whether the step-1,719,760 spike left a spectral fingerprint,
  3. ranks layers by how much they move across plateau -> divergence.
WW's published alpha thresholds are NOT assumed; correlation with val IS the test.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from comet_ml.api import API
from scipy.stats import spearmanr

SUMMARY_COLS = ["alpha", "alpha_weighted", "log_norm", "log_spectral_norm",
                "log_alpha_norm", "stable_rank", "train_loss"]


def val_quality(key: str, steps: np.ndarray) -> np.ndarray:
    exp = API().get(key)

    def series(name: str):
        d = exp.get_metrics(name)
        xs = np.array([int(p["step"]) for p in d if p["step"] is not None])
        ys = np.array([float(p["metricValue"]) for p in d if p["step"] is not None])
        o = np.argsort(xs)
        return xs[o], ys[o]

    acc, X = None, None
    for t in (5, 6, 7, 8, 9):  # average t5-9 to cut per-val noise
        x, y = series(f"val/scene_cos_norm_t{t}")
        if X is None:
            X, acc = x, y.astype(float).copy()
        elif len(y) == len(acc):
            acc += y
    scene = acc / 5
    return np.array([scene[np.argmin(np.abs(X - s))] for s in steps])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--key", default="m2b3-ava/canvit-pretrain/5d501de5f1ea4d0daccf10ddeb1ce3b7")
    args = ap.parse_args()

    rows = []
    for j in sorted(glob.glob(str(args.results / "ww_step-*.json"))):
        d = json.loads(Path(j).read_text())
        rows.append({"step": d["step"], "train_loss": d.get("train_loss"), **d["summary"]})
    df = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
    df["val_scene_cos"] = val_quality(args.key, df["step"].values)

    pd.set_option("display.width", 200, "display.max_rows", 200)
    print(df.round(4).to_string())

    print("\n=== Spearman corr of WW metric vs val_scene_cos (n={}) ===".format(len(df)))
    print("   (val higher = better; so a metric that DROPS as quality drops -> +corr)")
    for c in SUMMARY_COLS:
        if c in df and df[c].notna().all():
            print(f"  {c:20s} rho={spearmanr(df[c], df['val_scene_cos']).correlation:+.3f}")

    # spike fingerprint: per-layer pre (1717248) vs post (1722240)
    def layers(step):
        p = args.results / f"ww_step-{step}.csv"
        return pd.read_csv(p) if p.exists() else None
    pre, post = layers(1717248), layers(1722240)
    if pre is not None and post is not None:
        m = pre.merge(post, on="layer_id", suffixes=("_pre", "_post"))
        m["dalpha"] = m["alpha_post"] - m["alpha_pre"]
        m["dsr"] = m["stable_rank_post"] - m["stable_rank_pre"]
        nm = "name_pre" if "name_pre" in m else "name"
        print("\n=== SPIKE fingerprint: layers changed most pre(1717248)->post(1722240) ===")
        print(m.reindex(m["dalpha"].abs().sort_values(ascending=False).index)
              [["layer_id", nm, "alpha_pre", "alpha_post", "dalpha", "stable_rank_pre", "stable_rank_post"]].head(8).to_string())


if __name__ == "__main__":
    main()
