"""Read-only probe of the TRAINING DATA structure around the deterministic collapse.

The IN1k pretrain data order is fully deterministic (shards.py: shard 0,1,...,n-1,0,1,...
no shuffling; resume = start_step // batches_per_shard). So the images/teacher-features
at any step are a pure function of the step. Both the original run and the pre-spike
relaunch see identical data per step; the collapse reproduces at the same step.

This probe asks: is there something SPECIAL about the fixed teacher-feature distribution
(the DINOv3 targets the canvas reconstructs) that an unregularized attention (no QK-norm)
would collapse onto? Concretely:

  1. Map steps of interest (1.72M spike, 1.897M collapse onset) -> exact shard + sample
     indices + image paths (so the data 'there' can be looked at).
  2. Anisotropy of the teacher CLS + patch feature distribution: covariance spectrum,
     top-eigenvalue share, participation ratio, top PCA directions (saved for comparison
     against the k_proj collapse direction v1).
  3. Outlier samples by feature norm (candidate high-gradient images).

CPU, mmap'd shards, subsampled patches. No model, no training.
"""

import argparse
import json
from pathlib import Path

import torch


def shard_files(shard_dir: Path) -> list[Path]:
    files = sorted(shard_dir.glob("*.pt"))
    assert files, f"no .pt shards in {shard_dir}"
    return files


def participation_ratio(eigvals: torch.Tensor) -> float:
    """(sum lambda)^2 / sum(lambda^2): effective number of dims the variance occupies.

    PR = D for isotropic; PR -> 1 for one dominant direction.
    """
    return float((eigvals.sum() ** 2 / eigvals.pow(2).sum()).item())


def cov_spectrum(x: torch.Tensor) -> dict:
    """x: [N, D] features. Centered covariance eigenspectrum + anisotropy summaries."""
    x = x.float()
    xc = x - x.mean(0, keepdim=True)
    cov = (xc.T @ xc) / (x.shape[0] - 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)  # ascending
    eigvals = eigvals.flip(0).clamp_min(0)
    eigvecs = eigvecs.flip(1)
    total = float(eigvals.sum())
    return {
        "n": int(x.shape[0]),
        "dim": int(x.shape[1]),
        "top1_share": float(eigvals[0] / total),
        "top5_share": float(eigvals[:5].sum() / total),
        "top10_share": float(eigvals[:10].sum() / total),
        "participation_ratio": participation_ratio(eigvals),
        "mean_norm": float(x.norm(dim=1).mean()),
        "top_eigvec": eigvecs[:, 0],           # top PCA direction (for v1 comparison)
        "mean_direction": (x.mean(0) / x.mean(0).norm()),
        "eigvals_top20": eigvals[:20],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--steps", type=int, nargs="*", default=[1_719_780, 1_896_960, 1_976_832])
    ap.add_argument("--n-shards-sample", type=int, default=20, help="shards to sample for anisotropy")
    ap.add_argument("--patches-per-shard", type=int, default=20, help="samples whose patch grid is used")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--copy-images-to", type=Path, default=None,
                    help="if set, write the image RELATIVE paths at each step to <dir>/step_<s>.txt")
    args = ap.parse_args()

    files = shard_files(args.shard_dir)
    n_shards = len(files)
    first = torch.load(files[0], map_location="cpu", weights_only=False, mmap=True)
    samples_per_shard = len(first["paths"])
    batches_per_shard = samples_per_shard // args.batch_size
    cls_dim = first["cls"][0].shape[-1]
    patch_shape = tuple(first["patches"][0].shape)
    print(f"n_shards={n_shards} samples_per_shard={samples_per_shard} "
          f"batches_per_shard={batches_per_shard} cls_dim={cls_dim} patch_shape={patch_shape}")
    steps_per_epoch = n_shards * batches_per_shard
    print(f"steps_per_epoch={steps_per_epoch} ({n_shards} shards x {batches_per_shard} batches)")

    # ---- 1. Map steps of interest -> shard + within-shard sample range + image paths ----
    step_map = {}
    for step in args.steps:
        shard_counter = step // batches_per_shard
        shard_idx = shard_counter % n_shards
        batch_in_shard = step % batches_per_shard
        s0 = batch_in_shard * args.batch_size
        s1 = s0 + args.batch_size
        epoch = step / steps_per_epoch
        sh = torch.load(files[shard_idx], map_location="cpu", weights_only=False, mmap=True)
        paths = [str(sh["paths"][i]) for i in range(s0, min(s1, len(sh["paths"])))]
        classes = [int(sh["class_idxs"][i]) for i in range(s0, min(s1, len(sh["paths"])))]
        step_map[step] = {"shard_idx": shard_idx, "shard_file": files[shard_idx].name,
                          "epoch": epoch, "sample_range": [s0, s1],
                          "image_paths": paths, "class_idxs": classes}
        print(f"\nstep {step}: epoch~{epoch:.2f} shard={shard_idx} ({files[shard_idx].name}) "
              f"samples[{s0}:{s1}]")
        print(f"  first 5 images: {paths[:5]}")
        if args.copy_images_to:
            args.copy_images_to.mkdir(parents=True, exist_ok=True)
            (args.copy_images_to / f"step_{step}_images.txt").write_text("\n".join(paths))

    # ---- 2. Anisotropy of teacher features (CLS over many shards; patches subsampled) ----
    sample_idx = torch.linspace(0, n_shards - 1, args.n_shards_sample).round().long().tolist()
    cls_all, patch_all, cls_norms_by_shard = [], [], []
    for si in sample_idx:
        sh = torch.load(files[si], map_location="cpu", weights_only=False, mmap=True)
        ns = len(sh["paths"])
        cls = torch.stack([sh["cls"][i].float() for i in range(ns)])  # [ns, D]
        cls_all.append(cls)
        cls_norms_by_shard.append((si, files[si].name, float(cls.norm(dim=1).mean()),
                                   float(cls.norm(dim=1).max())))
        psel = torch.linspace(0, ns - 1, args.patches_per_shard).round().long().tolist()
        for i in psel:
            p = sh["patches"][i].float().reshape(-1, cls_dim)  # [num_patches, D]
            patch_all.append(p)
    cls_cat = torch.cat(cls_all)
    patch_cat = torch.cat(patch_all)
    print(f"\nanisotropy sample: CLS {tuple(cls_cat.shape)}  patches {tuple(patch_cat.shape)}")
    cls_spec = cov_spectrum(cls_cat)
    patch_spec = cov_spectrum(patch_cat)
    print(f"CLS:    top1_share={cls_spec['top1_share']:.3f} PR={cls_spec['participation_ratio']:.1f}/{cls_dim} "
          f"top5={cls_spec['top5_share']:.3f}")
    print(f"PATCH:  top1_share={patch_spec['top1_share']:.3f} PR={patch_spec['participation_ratio']:.1f}/{cls_dim} "
          f"top5={patch_spec['top5_share']:.3f}")

    # ---- 3. Outlier shards by CLS norm ----
    cls_norms_by_shard.sort(key=lambda t: t[3], reverse=True)
    print("\ntop-5 shards by max CLS norm:")
    for si, name, mean_n, max_n in cls_norms_by_shard[:5]:
        print(f"  shard {si} ({name}): mean_norm={mean_n:.2f} max_norm={max_n:.2f}")

    def ser(o: object) -> object:
        if isinstance(o, torch.Tensor):
            return o.tolist()
        if isinstance(o, dict):
            return {k: ser(v) for k, v in o.items()}
        if isinstance(o, list):
            return [ser(v) for v in o]
        return o

    payload = {
        "n_shards": n_shards, "samples_per_shard": samples_per_shard,
        "batches_per_shard": batches_per_shard, "steps_per_epoch": steps_per_epoch,
        "cls_dim": cls_dim, "patch_shape": list(patch_shape),
        "step_map": ser(step_map),
        "cls_spectrum": ser(cls_spec), "patch_spectrum": ser(patch_spec),
        "shard_cls_norms_sorted": cls_norms_by_shard,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
