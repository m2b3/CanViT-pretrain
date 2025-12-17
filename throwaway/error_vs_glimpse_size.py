"""Investigate relationship between error reduction and glimpse size."""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.nn.functional import mse_loss
from tqdm import tqdm
from ytch.device import get_sensible_device

from avp_vit import AVPConfig, AVPViT
from avp_vit.backbone.dinov3 import DINOv3Backbone
from avp_vit.glimpse import Viewpoint
from avp_vit.train.data import val_transform, make_loader

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_avp(ckpt_path: Path, device: torch.device) -> tuple[AVPViT, DINOv3Backbone]:
    """Load AVP model and teacher from checkpoint."""
    from dinov3.hub.backbones import dinov3_vits16

    teacher_raw = dinov3_vits16(pretrained=True, weights="dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
    teacher = DINOv3Backbone(teacher_raw.eval().to(device))
    for p in teacher.parameters():
        p.requires_grad = False

    student_raw = dinov3_vits16(pretrained=False)
    student = DINOv3Backbone(student_raw.to(device))

    avp_cfg = AVPConfig()
    avp = AVPViT(student, avp_cfg, teacher.embed_dim).to(device)

    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    avp.load_state_dict(ckpt["avp"])
    avp.eval()
    log.info(f"Loaded AVP from {ckpt_path}")

    return avp, teacher


def sample_viewpoint(B: int, scale: float, device: torch.device) -> Viewpoint:
    """Create viewpoint with fixed scale, random center (constrained to stay in bounds)."""
    max_offset = 1 - scale
    centers = (torch.rand(B, 2, device=device) * 2 - 1) * max_offset
    scales = torch.full((B,), scale, device=device)
    return Viewpoint(name=f"s{scale:.2f}", centers=centers, scales=scales)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("/datasets/ILSVRC/Data/CLS-LOC/val"))
    parser.add_argument("--n-images", type=int, default=200)
    parser.add_argument("--scales-per-image", type=int, default=10)
    parser.add_argument("--scene-grid-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("outputs/error_vs_glimpse.png"))
    args = parser.parse_args()

    device = get_sensible_device()
    log.info(f"Device: {device}")

    avp, teacher = load_avp(args.ckpt, device)

    img_size = args.scene_grid_size * teacher.patch_size
    transform = val_transform(img_size)
    loader = make_loader(args.data, transform, batch_size=1, num_workers=4, shuffle=True)

    # Collect data: (scale, error_reduction) pairs
    scales_list: list[float] = []
    error_reductions: list[float] = []
    error_reductions_normalized: list[float] = []  # normalized by area

    # Scale range: from glimpse_grid_size/scene_grid_size up to 1.0
    min_scale = avp.cfg.glimpse_grid_size / args.scene_grid_size
    scale_values = torch.linspace(min_scale, 1.0, args.scales_per_image).tolist()

    with torch.inference_mode():
        for i, (img, _) in enumerate(tqdm(loader, total=args.n_images, desc="Images")):
            if i >= args.n_images:
                break

            img = img.to(device)
            B = img.shape[0]

            # Compute teacher target
            feats = teacher.forward_norm_features(img)
            target = feats.patches  # [B, N, D]

            for scale in scale_values:
                # Init hidden and compute initial loss
                hidden = avp.init_hidden(B, args.scene_grid_size)
                hidden_normed = avp._normalize_hidden(hidden)
                initial_scene = avp.compute_scene(hidden_normed)
                initial_loss = mse_loss(initial_scene, target).item()

                # One glimpse at this scale
                vp = sample_viewpoint(B, scale, device)
                out = avp.forward_step(img, vp, hidden)
                post_loss = mse_loss(out.scene, target).item()

                error_reduction = initial_loss - post_loss
                area = scale ** 2
                error_reduction_per_area = error_reduction / area

                scales_list.append(scale)
                error_reductions.append(error_reduction)
                error_reductions_normalized.append(error_reduction_per_area)

    # Convert to tensors for stats
    scales_t = torch.tensor(scales_list)
    reductions_t = torch.tensor(error_reductions)
    reductions_norm_t = torch.tensor(error_reductions_normalized)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Raw error reduction vs scale
    ax = axes[0]
    ax.scatter(scales_t, reductions_t, alpha=0.3, s=5)
    ax.set_xlabel("Glimpse scale")
    ax.set_ylabel("Error reduction (initial_loss - post_loss)")
    ax.set_title("Error reduction vs glimpse scale")
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)

    # Error reduction vs area
    ax = axes[1]
    areas = scales_t ** 2
    ax.scatter(areas, reductions_t, alpha=0.3, s=5)
    ax.set_xlabel("Glimpse area (scale²)")
    ax.set_ylabel("Error reduction")
    ax.set_title("Error reduction vs glimpse area")
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)

    # Normalized error reduction (per unit area) vs scale
    ax = axes[2]
    ax.scatter(scales_t, reductions_norm_t, alpha=0.3, s=5)
    ax.set_xlabel("Glimpse scale")
    ax.set_ylabel("Error reduction / area")
    ax.set_title("Error reduction per unit area vs scale")
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    log.info(f"Saved plot to {args.output}")

    # Print summary stats
    log.info(f"Total samples: {len(scales_list)}")
    log.info(f"Error reduction range: [{reductions_t.min():.4f}, {reductions_t.max():.4f}]")
    log.info(f"Mean error reduction: {reductions_t.mean():.4f}")

    # Correlation
    corr_scale = torch.corrcoef(torch.stack([scales_t, reductions_t]))[0, 1].item()
    corr_area = torch.corrcoef(torch.stack([areas, reductions_t]))[0, 1].item()
    log.info(f"Correlation (scale vs reduction): {corr_scale:.3f}")
    log.info(f"Correlation (area vs reduction): {corr_area:.3f}")


if __name__ == "__main__":
    main()
