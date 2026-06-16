"""Read-only forensic scan of a checkpoint sequence.

Postmortem tool: given a run dir, load each `step-*.pt` and report finiteness,
per-module weight norms, stored EMA train_loss, and AdamW second-moment health.

Goal: distinguish a single-step "bad data" spike (norms flat, then one ckpt is
NaN) from a "resonance"/slow-blowup (weight norm or 1/sqrt(exp_avg_sq) drifting
up over many checkpoints before the NaN). No GPU, no model instantiation.
"""

import argparse
import re
from pathlib import Path

import torch

STEP_RE = re.compile(r"step-(\d+)\.pt$")


def top_module(param_name: str) -> str:
    return param_name.split(".")[0]


def nonfinite_count(t: torch.Tensor) -> int:
    return int((~torch.isfinite(t)).sum().item())


def scan_state_dict(sd: dict[str, torch.Tensor]) -> dict:
    groups: dict[str, list[torch.Tensor]] = {}
    total_nonfinite = 0
    for name, t in sd.items():
        if not torch.is_floating_point(t):
            continue
        total_nonfinite += nonfinite_count(t)
        groups.setdefault(top_module(name), []).append(t)
    per_module = {}
    for mod, ts in groups.items():
        flat = torch.cat([t.flatten().float() for t in ts])
        finite = flat[torch.isfinite(flat)]
        per_module[mod] = {
            "l2": float(finite.norm().item()) if finite.numel() else float("nan"),
            "absmax": float(finite.abs().max().item()) if finite.numel() else float("nan"),
            "nonfinite": int((~torch.isfinite(flat)).sum().item()),
        }
    return {"total_nonfinite": total_nonfinite, "per_module": per_module}


def scan_optimizer(opt_state: dict | None) -> dict:
    if opt_state is None or "state" not in opt_state:
        return {"present": False}
    sq_min = float("inf")
    sq_max = 0.0
    avg_absmax = 0.0
    eff_lr_max = 0.0  # max of 1/(sqrt(exp_avg_sq)+eps) over all elements -> effective-LR amplifier
    near_zero_sq = 0  # count of exp_avg_sq < 1e-12
    nonfinite = 0
    eps = 1e-8
    for st in opt_state["state"].values():
        sq = st.get("exp_avg_sq")
        avg = st.get("exp_avg")
        if sq is not None:
            sqf = sq.float()
            nonfinite += nonfinite_count(sqf)
            fin = sqf[torch.isfinite(sqf)]
            if fin.numel():
                sq_min = min(sq_min, float(fin.min().item()))
                sq_max = max(sq_max, float(fin.max().item()))
                amp = (1.0 / (fin.sqrt() + eps)).max().item()
                eff_lr_max = max(eff_lr_max, float(amp))
                near_zero_sq += int((fin < 1e-12).sum().item())
        if avg is not None:
            avgf = avg.float()
            nonfinite += nonfinite_count(avgf)
            fin = avgf[torch.isfinite(avgf)]
            if fin.numel():
                avg_absmax = max(avg_absmax, float(fin.abs().max().item()))
    return {
        "present": True,
        "exp_avg_sq_min": sq_min,
        "exp_avg_sq_max": sq_max,
        "exp_avg_absmax": avg_absmax,
        "eff_lr_amplifier_max": eff_lr_max,  # 1/(sqrt(min_sq)+eps)
        "near_zero_sq_count": near_zero_sq,
        "nonfinite": nonfinite,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--last", type=int, default=10, help="scan the last N step-checkpoints")
    ap.add_argument("--also", type=int, nargs="*", default=[], help="extra step numbers to include (e.g. early ones)")
    args = ap.parse_args()

    ckpts = sorted(
        (p for p in args.run_dir.glob("step-*.pt") if STEP_RE.search(p.name)),
        key=lambda p: int(STEP_RE.search(p.name).group(1)),
    )
    by_step = {int(STEP_RE.search(p.name).group(1)): p for p in ckpts}
    chosen = ckpts[-args.last :]
    for s in args.also:
        if s in by_step and by_step[s] not in chosen:
            chosen = [by_step[s], *chosen]
    chosen = sorted(set(chosen), key=lambda p: int(STEP_RE.search(p.name).group(1)))

    print(f"run_dir={args.run_dir}  scanning {len(chosen)} checkpoints\n")
    for p in chosen:
        raw = torch.load(p, weights_only=False, map_location="cpu")
        sd_scan = scan_state_dict(raw["state_dict"])
        opt_scan = scan_optimizer(raw.get("optimizer_state"))
        step = raw.get("step")
        loss = raw.get("train_loss")
        print(f"=== {p.name}  step={step}  ts={raw.get('timestamp')}  comet={raw.get('comet_id')} ===")
        print(f"  train_loss(EMA stored): {loss}")
        print(f"  state_dict total_nonfinite: {sd_scan['total_nonfinite']}")
        # per-module weight norms, sorted by absmax desc (top offenders first)
        pm = sd_scan["per_module"]
        for mod in sorted(pm, key=lambda m: pm[m]["absmax"], reverse=True):
            d = pm[mod]
            flag = "  <-- NONFINITE" if d["nonfinite"] else ""
            print(f"    {mod:28s} l2={d['l2']:.3e}  absmax={d['absmax']:.3e}  nf={d['nonfinite']}{flag}")
        if opt_scan["present"]:
            print(f"  AdamW: exp_avg_sq[min={opt_scan['exp_avg_sq_min']:.3e} max={opt_scan['exp_avg_sq_max']:.3e}] "
                  f"exp_avg_absmax={opt_scan['exp_avg_absmax']:.3e}")
            print(f"         eff_lr_amplifier_max(1/(sqrt(sq)+eps))={opt_scan['eff_lr_amplifier_max']:.3e}  "
                  f"near_zero_sq={opt_scan['near_zero_sq_count']}  nonfinite={opt_scan['nonfinite']}")
        else:
            print("  AdamW: <no optimizer_state>")
        print()
        del raw


if __name__ == "__main__":
    main()
