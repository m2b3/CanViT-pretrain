"""Forward-pass probe: does the collapse direction v1 live in the high-variance
subspace of the ACTUAL k_proj inputs? (Settles the teacher-vs-glimpse space caveat.)

The earlier cross-run SVD found canvas_write.2.k_proj collapses to a reproducible
rank-1 direction v1 (input space). Comparing v1 to TEACHER-feature PCA was suggestive
(|cos|~random) but spans two networks. Here we PCA the REAL k_proj inputs:
forward-hook canvas_write.2.k_proj during recurrent rollouts on real images, capture
its input (= kv_norm(glimpse tokens)), and measure where v1 sits in that distribution:

  - |cos(v1, input_top_PCA_1)|, |cos(v1, input_mean_dir)|
  - fraction of v1 energy in the top-k input PCA subspace ||P_k v1||^2
  - the input variance v1 itself carries (rank of v1 in the input spectrum)

If v1 is a HIGH-variance input direction -> collapse amplifies what the data emphasizes
(data-driven). If v1 is a LOW-variance direction -> the collapse direction is selected
by dynamics, not by the input distribution.
"""

import argparse
import json
import math
from pathlib import Path

import torch
from canvit_pretrain.checkpoint import load_model
from canvit_pretrain.train.viewpoint import Viewpoint as NamedViewpoint
from canvit_pytorch import Viewpoint, sample_at_viewpoint
from canvit_pytorch.preprocess import preprocess
from PIL import Image


def find_kproj(model: torch.nn.Module, idx: int) -> tuple[str, torch.nn.Module]:
    for name, m in model.named_modules():
        if name.endswith(f"canvas_write.{idx}.k_proj"):
            return name, m
    raise AssertionError(f"no canvas_write.{idx}.k_proj in model")


def load_images(image_dir: Path, n: int, scene_px: int, device: str) -> torch.Tensor:
    files = sorted(image_dir.rglob("*.JPEG"))[:n]
    assert len(files) >= n, f"only {len(files)} images in {image_dir}"
    tf = preprocess(scene_px)
    imgs = []
    for f in files:
        with Image.open(f) as im:
            imgs.append(tf(im.convert("RGB")))
    return torch.stack(imgs).to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--v1-json", type=Path, required=True, help="kproj_forensics.json")
    ap.add_argument("--v1-label", default="relaunch_diverged_2001792")
    ap.add_argument("--image-dir", type=Path, required=True)
    ap.add_argument("--n-images", type=int, default=64)
    ap.add_argument("--timesteps", type=int, default=10)
    ap.add_argument("--write-idx", type=int, default=2, help="canvas_write block index")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    model, ckpt = load_model(args.ckpt, args.device)
    model.eval()
    scene_px = ckpt["scene_resolution"]
    glimpse_px = ckpt["glimpse_grid_size"] * 16  # DINOv3 patch = 16px
    canvas_grid = ckpt["canvas_patch_grid_sizes"][0]
    print(f"step={ckpt['step']} scene_px={scene_px} glimpse_px={glimpse_px} canvas_grid={canvas_grid}")

    captured: list[torch.Tensor] = []

    def hook(_module: torch.nn.Module, inp: tuple) -> None:
        x = inp[0].detach()
        captured.append(x.reshape(-1, x.shape[-1]).float().cpu())

    name, kp = find_kproj(model, args.write_idx)
    handle = kp.register_forward_pre_hook(hook)
    print(f"hooked {name}  (weight out_dim={kp.weight.shape[0]} in_dim={kp.weight.shape[1]})")

    images = load_images(args.image_dir, args.n_images, scene_px, args.device)
    B = images.shape[0]
    print(f"images {tuple(images.shape)}")

    with torch.inference_mode():
        state = model.init_state(batch_size=B, canvas_grid_size=canvas_grid)
        for t in range(args.timesteps):
            if t == 0:
                vpn = NamedViewpoint.full_scene(batch_size=B, device=images.device)
            else:
                vpn = NamedViewpoint.random(batch_size=B, device=images.device, min_scale=0.05)
            vp = Viewpoint(centers=vpn.centers, scales=vpn.scales)
            glimpse = sample_at_viewpoint(spatial=images, viewpoint=vp, glimpse_size_px=glimpse_px)
            out = model(glimpse=glimpse, state=state, viewpoint=vp)
            state = out.state
    handle.remove()

    X = torch.cat(captured)  # [M, D]
    D = X.shape[1]
    print(f"captured k_proj inputs: {tuple(X.shape)} over {args.timesteps} timesteps")

    Xc = (X - X.mean(0)).double()
    cov = (Xc.T @ Xc) / (X.shape[0] - 1)
    eigval, eigvec = torch.linalg.eigh(cov)  # ascending
    eigval = eigval.flip(0).clamp_min(0)
    eigvec = eigvec.flip(1)  # columns = PCA dirs, descending
    total = float(eigval.sum())
    input_mean_dir = X.mean(0).double()
    input_mean_dir = input_mean_dir / input_mean_dir.norm()

    v1 = torch.tensor(
        json.load(open(args.v1_json))["results"][args.v1_label]["layers"]
        [f"canvas_write.{args.write_idx}.k_proj.weight"]["global"]["v1"], dtype=torch.float64)
    v1 = v1 / v1.norm()

    def acos(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(torch.abs(torch.dot(a / a.norm(), b / b.norm())))

    coords = eigvec.T @ v1  # v1 in PCA basis; coords[k]^2 = energy in dir k
    rand_floor = 1 / math.sqrt(D)
    energy_in_topk = {k: float((coords[:k] ** 2).sum()) for k in [1, 5, 10, 20, 50, 100]}
    # "effective input variance v1 carries" = variance along v1 / top eigval
    var_along_v1 = float((v1 @ cov @ v1))
    print(f"\nrandom |cos| floor ~ {rand_floor:.4f}")
    print(f"input participation ratio = {float(eigval.sum()**2/eigval.pow(2).sum()):.1f}/{D}")
    print(f"input top1 var share = {float(eigval[0]/total):.3f}  top10 = {float(eigval[:10].sum()/total):.3f}")
    print(f"\n|cos(v1, input_top_PCA_1)| = {acos(v1, eigvec[:,0]):.4f}")
    print(f"|cos(v1, input_mean_dir)|  = {acos(v1, input_mean_dir):.4f}")
    print(f"\nfraction of v1 energy in top-k input PCA subspace:")
    for k, e in energy_in_topk.items():
        print(f"  top-{k:>3}: {e:.3f}   (random expectation ~ {k/D:.3f})")
    print(f"\nvariance along v1 = {var_along_v1:.4g}   top eigval = {float(eigval[0]):.4g}   "
          f"ratio = {var_along_v1/float(eigval[0]):.4f}")
    # rank of v1 in input spectrum: how many input dirs have more variance than v1
    rank_v1 = int((eigval > var_along_v1).sum())
    print(f"v1 input-variance rank: {rank_v1}/{D}  (1=top variance dir, D=lowest)")

    payload = {
        "ckpt_step": ckpt["step"], "v1_label": args.v1_label, "n_captured": int(X.shape[0]),
        "dim": D, "random_cos_floor": rand_floor,
        "cos_v1_input_top1": acos(v1, eigvec[:, 0]),
        "cos_v1_input_mean": acos(v1, input_mean_dir),
        "energy_v1_in_topk": energy_in_topk,
        "var_along_v1": var_along_v1, "top_eigval": float(eigval[0]),
        "v1_variance_rank": rank_v1,
        "input_participation_ratio": float(eigval.sum() ** 2 / eigval.pow(2).sum()),
        "input_top1_var_share": float(eigval[0] / total),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
