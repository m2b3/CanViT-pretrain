"""Per-checkpoint AdamW optimizer-state health signatures (data-free).

Global second-moment health, plus per-layer signatures for the canvas-write key
and value projections: first/second-moment norms, the actual Adam update
magnitude, and — the key probe — the STABLE RANK of the first moment (the smoothed
gradient). If the gradient itself concentrates to rank-1 on the keys (but not the
values), the optimizer is actively driving the W_K rank-1 collapse. Writes JSON
keyed by step. CPU, no data.
"""

import argparse
import json
import re
from pathlib import Path

import torch

from canvit_pretrain.checkpoint import load_model


def stable_rank(t: torch.Tensor) -> float:
    a = t.detach().float()
    if a.ndim < 2:
        return float("nan")
    a = a.reshape(a.shape[0], -1)
    s1 = torch.linalg.svdvals(a)[0]
    return float((a.norm() ** 2) / (s1 ** 2 + 1e-12))


TARGETS = [
    "canvas_write.0.k_proj.weight", "canvas_write.1.k_proj.weight", "canvas_write.2.k_proj.weight",
    "canvas_write.0.v_proj.weight", "canvas_write.1.v_proj.weight", "canvas_write.2.v_proj.weight",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    step = int(re.search(r"step-(\d+)", args.ckpt.name).group(1))

    raw = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    opt = raw["optimizer_state"]
    state = opt["state"]
    assert len(opt["param_groups"]) == 1, "expected a single AdamW param group"
    idxs = opt["param_groups"][0]["params"]

    model, _ = load_model(args.ckpt, "cpu")
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert len(names) == len(idxs), f"param count mismatch {len(names)} vs {len(idxs)}"
    idx_of = {n: idxs[i] for i, n in enumerate(names)}

    allv = torch.cat([state[i]["exp_avg_sq"].flatten().float() for i in idxs])
    glob = {
        "v_min": float(allv.min()), "v_med": float(allv.median()), "v_max": float(allv.max()),
        "eff_lr_amp_max": float((1.0 / (allv.sqrt() + 1e-8)).max()),
        "nonfinite": int((~torch.isfinite(allv)).sum()),
    }

    per = {}
    for tn in TARGETS:
        if tn not in idx_of:
            continue
        st = state[idx_of[tn]]
        m, v = st["exp_avg"].float(), st["exp_avg_sq"].float()
        upd = m / (v.sqrt() + 1e-8)
        per[tn] = {
            "m_norm": float(m.norm()), "m_stable_rank": stable_rank(m),
            "v_mean": float(v.mean()), "update_norm": float(upd.norm()),
            "update_stable_rank": stable_rank(upd),
        }

    (args.out_dir / f"opt_step-{step}.json").write_text(
        json.dumps({"step": step, "global": glob, "per_layer": per}, indent=2))
    k = per.get("canvas_write.2.k_proj.weight", {})
    print(f"step {step}: cw2_k m_stable_rank={k.get('m_stable_rank')}")


if __name__ == "__main__":
    main()
