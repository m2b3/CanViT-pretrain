"""Cross-run k_proj collapse forensics (read-only, data-free).

For a set of labeled checkpoints, extract every ``canvas_write.{i}.k_proj.weight``
and characterize the rank-1 collapse spectrally:

  - global SVD: sigma1, stable_rank, top right-singular vector v1 (in local_dim space);
  - per-head SVD (num_heads read from model_config): per-head stable_rank + per-head v1.

The question this answers: two INDEPENDENT runs (original diverged vs pre-spike
relaunch, different glimpse RNG, same data+config) both collapse canvas_write.2.k_proj
to rank ~1. Do they collapse onto the SAME input direction and leave the SAME head
uncollapsed?

  |cos(v1_runA, v1_runB)| ~ 1  => collapse direction determined by data/task geometry
                                  => reproducible, not luck.
  |cos(v1_runA, v1_runB)| ~ 0  => independent random drift.

Pure SVD on the saved weights: no model build, no backbone, no data, no GPU.
"""

import argparse
import json
from pathlib import Path

import torch

from canvit_pretrain.checkpoint import load


def discover_num_heads(model_config: dict, out_dim: int) -> int | None:
    """Find the canvas-write head count: a 'head'-keyed config value that divides
    out_dim into a sane head_dim (16..256). Non-fatal: returns None if unresolved
    (global SVD does not need it; only per-head breakdown does).
    """
    found: set[int] = set()

    def walk(d: object) -> None:
        if isinstance(d, dict):
            for k, v in d.items():
                if "head" in k.lower() and isinstance(v, int) and v > 0:
                    found.add(v)
                walk(v)
        elif isinstance(d, (list, tuple)):
            for v in d:
                walk(v)

    walk(model_config)
    candidates = [c for c in found if out_dim % c == 0 and 16 <= out_dim // c <= 256]
    if len(candidates) == 1:
        return candidates[0]
    print(f"  [warn] head count unresolved from {sorted(found)} (out_dim={out_dim}, "
          f"candidates={sorted(candidates)}); skipping per-head analysis")
    return None


def top_singular(weight: torch.Tensor) -> tuple[float, float, torch.Tensor]:
    """Return (sigma1, stable_rank, v1) for a [out, in] matrix.

    v1 is the top RIGHT singular vector (in input space), sign-canonicalized so its
    largest-magnitude entry is positive (so cross-checkpoint cos is sign-stable).
    """
    w = weight.float()
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    sigma1 = float(s[0])
    stable_rank = float((s.pow(2).sum() / s[0].pow(2)).item())
    v1 = vh[0]
    v1 = v1 * torch.sign(v1[v1.abs().argmax()])
    return sigma1, stable_rank, v1


def abscos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.abs(torch.nn.functional.cosine_similarity(a, b, dim=0)).item())


def analyze_layer(weight: torch.Tensor, num_heads: int | None) -> dict:
    out_dim, in_dim = weight.shape
    g_sigma1, g_sr, g_v1 = top_singular(weight)
    heads = []
    head_dim = None
    if num_heads is not None:
        assert out_dim % num_heads == 0, f"{out_dim} not divisible by {num_heads}"
        head_dim = out_dim // num_heads
        for h in range(num_heads):
            wh = weight[h * head_dim : (h + 1) * head_dim, :]
            s1, sr, v1 = top_singular(wh)
            heads.append({"sigma1": s1, "stable_rank": sr, "v1": v1})
    return {
        "out_dim": out_dim,
        "in_dim": in_dim,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "global": {"sigma1": g_sigma1, "stable_rank": g_sr, "v1": g_v1},
        "heads": heads,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, metavar="LABEL=PATH",
                    help="repeatable; e.g. --ckpt orig_diverged=/path/step-2001792.pt")
    ap.add_argument("--out", type=Path, required=True, help="output JSON path")
    ap.add_argument("--collapse-sr", type=float, default=2.0,
                    help="per-head stable_rank below this counts as 'collapsed'")
    args = ap.parse_args()

    pairs = []
    for spec in args.ckpt:
        label, _, path = spec.partition("=")
        assert path, f"bad --ckpt spec (need LABEL=PATH): {spec!r}"
        pairs.append((label, Path(path)))

    # label -> {layer_longname -> analysis}
    results: dict[str, dict] = {}
    for label, path in pairs:
        print(f"\n=== loading {label}: {path}", flush=True)
        ckpt = load(path, "cpu")
        sd = ckpt["state_dict"]
        kproj = {k: v for k, v in sd.items()
                 if "canvas_write" in k and k.endswith("k_proj.weight")}
        assert kproj, f"no canvas_write k_proj weights in {label}"
        out_dim = next(iter(kproj.values())).shape[0]
        num_heads = discover_num_heads(ckpt["model_config"], out_dim)
        print(f"    step={ckpt['step']} num_heads={num_heads} "
              f"k_proj layers={sorted(kproj)}", flush=True)
        results[label] = {"step": ckpt["step"], "num_heads": num_heads,
                          "layers": {name: analyze_layer(w, num_heads)
                                     for name, w in sorted(kproj.items())}}

    labels = [lbl for lbl, _ in pairs]
    layer_names = sorted(next(iter(results.values()))["layers"])

    # ---- Report: per-layer global stable_rank across checkpoints ----
    print("\n########## GLOBAL stable_rank (per k_proj layer x checkpoint) ##########")
    header = "layer".ljust(26) + "".join(f"{lbl[:16]:>18}" for lbl in labels)
    print(header)
    for name in layer_names:
        row = name.ljust(26)
        for lbl in labels:
            g = results[lbl]["layers"][name]["global"]
            row += f"{g['stable_rank']:>9.2f}/s{g['sigma1']:<7.1f}"
        print(row)

    deepest = layer_names[-1]
    nh = results[labels[0]]["num_heads"]

    # ---- KEY TEST 1: cross-checkpoint GLOBAL v1 alignment (no head count needed) ----
    print(f"\n########## CROSS-CHECKPOINT |cos(v1)| for {deepest} (GLOBAL) ##########")
    print("Global v1 = top right-singular vector in k_proj INPUT space (glimpse-feature space):")
    align: dict[str, dict] = {"global": {}, "per_head": {}}
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            va = results[a]["layers"][deepest]["global"]["v1"]
            vb = results[b]["layers"][deepest]["global"]["v1"]
            c = abscos(va, vb)
            align["global"][f"{a}|{b}"] = c
            print(f"  |cos(v1)|  {a:>22} vs {b:<22} = {c:.4f}")

    if nh is None:
        print("\n[per-head analysis skipped: head count unresolved]")
    else:
        print(f"\n########## PER-HEAD stable_rank for {deepest} ##########")
        print("head".ljust(8) + "".join(f"{lbl[:16]:>18}" for lbl in labels))
        for h in range(nh):
            row = f"{h}".ljust(8)
            for lbl in labels:
                row += f"{results[lbl]['layers'][deepest]['heads'][h]['stable_rank']:>18.3f}"
            print(row)

        def collapsed_heads(label: str, name: str) -> list[int]:
            return [h for h, hd in enumerate(results[label]["layers"][name]["heads"])
                    if hd["stable_rank"] < args.collapse_sr]

        print(f"\n########## COLLAPSED heads (stable_rank < {args.collapse_sr}) for {deepest} ##########")
        for lbl in labels:
            print(f"  {lbl}: collapsed={collapsed_heads(lbl, deepest)} "
                  f"surviving={[h for h in range(nh) if h not in collapsed_heads(lbl, deepest)]}")

        print("\nPer-head v1 alignment (only heads collapsed in BOTH are meaningful):")
        for i, a in enumerate(labels):
            for b in labels[i + 1:]:
                ca, cb = collapsed_heads(a, deepest), collapsed_heads(b, deepest)
                shared = [h for h in ca if h in cb]
                per = {}
                for h in range(nh):
                    va = results[a]["layers"][deepest]["heads"][h]["v1"]
                    vb = results[b]["layers"][deepest]["heads"][h]["v1"]
                    per[h] = abscos(va, vb)
                align["per_head"][f"{a}|{b}"] = {"cos": per, "shared_collapsed": shared}
                shared_cos = [f"h{h}:{per[h]:.3f}" for h in shared]
                print(f"  {a:>22} vs {b:<22} shared_collapsed={shared} "
                      f"-> {' '.join(shared_cos) if shared_cos else '(none)'}")

    # ---- Serialize (v1 vectors -> lists so the result is fully reproducible) ----
    def to_serializable(obj: object) -> object:
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        return obj

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"results": to_serializable(results), "alignment": align,
               "collapse_sr_threshold": args.collapse_sr,
               "ckpts": {lbl: str(p) for lbl, p in pairs}}
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
