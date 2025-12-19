#!/usr/bin/env python3
"""Multi-grid inference: run model at various scene grid sizes with PCA viz."""

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from PIL import Image
from torchvision import transforms

from avp_vit import AVPConfig, AVPViT
from avp_vit.backbone.dinov3 import DINOv3Backbone
from avp_vit.checkpoint import load as load_checkpoint
from avp_vit.train.data import imagenet_normalize
from avp_vit.train.viewpoint import make_eval_viewpoints
from avp_vit.train.viz import fit_pca, pca_rgb, imagenet_denormalize, timestep_colors
from scripts.train_scene_match.model import MODEL_REGISTRY

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


@dataclass
class Args:
    checkpoint: Path
    image: Path
    output: Path = Path("outputs/multi_grid_pca.png")
    grid_sizes: tuple[int, ...] = (8, 16, 32, 64, 128)
    backbone_weights: Path | None = None
    device: str = "mps"


def load_model(
    ckpt_path: Path, backbone_weights: Path | None, device: torch.device
) -> tuple[AVPViT, DINOv3Backbone]:
    """Load AVP model and teacher backbone from checkpoint."""
    ckpt = load_checkpoint(ckpt_path, device)
    cfg = AVPConfig(**ckpt["avp_config"])
    backbone_slug = ckpt["backbone"]
    teacher_dim = ckpt["teacher_dim"]

    factory = MODEL_REGISTRY[backbone_slug]
    if backbone_weights is not None:
        log.info(f"Loading backbone {backbone_slug} from {backbone_weights}")
        raw_backbone = factory(pretrained=True, weights=str(backbone_weights))
    else:
        log.info(f"Loading backbone {backbone_slug} with default pretrained weights")
        raw_backbone = factory(pretrained=True)

    backbone = DINOv3Backbone(raw_backbone.to(device).eval())
    for p in backbone.parameters():
        p.requires_grad = False

    avp = AVPViT(backbone, cfg, teacher_dim).to(device)
    avp.load_state_dict(ckpt["state_dict"])
    avp.eval()

    # Load TEACHER backbone separately - must use same weights as training!
    # The teacher during training uses explicit checkpoint path, not default hub weights
    teacher_ckpt = backbone_weights if backbone_weights is not None else None
    if teacher_ckpt is not None:
        log.info(f"Loading TEACHER backbone {backbone_slug} from {teacher_ckpt}")
        raw_teacher = factory(pretrained=True, weights=str(teacher_ckpt))
    else:
        log.info(f"Loading TEACHER backbone {backbone_slug} with default pretrained weights")
        raw_teacher = factory(pretrained=True)
    teacher = DINOv3Backbone(raw_teacher.to(device).eval())
    for p in teacher.parameters():
        p.requires_grad = False

    log.info(f"Model loaded: {backbone_slug}, teacher_dim={teacher_dim}")
    log.info(f"  glimpse_grid_size={cfg.glimpse_grid_size}, registers={cfg.n_scene_registers}")
    return avp, teacher


def load_image(path: Path, size: int, device: torch.device) -> torch.Tensor:
    """Load and preprocess image."""
    transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        imagenet_normalize(),
    ])
    pil_img = Image.open(path).convert("RGB")
    tensor = transform(pil_img)
    assert isinstance(tensor, torch.Tensor)
    return tensor.unsqueeze(0).to(device)


def main(args: Args) -> None:
    from matplotlib.patches import Rectangle

    device = torch.device(args.device)
    log.info(f"Device: {device}")

    avp, teacher = load_model(args.checkpoint, args.backbone_weights, device)
    patch_size = teacher.patch_size

    max_grid = max(args.grid_sizes)
    img_size = max_grid * patch_size
    log.info(f"Image size: {img_size}px (for max grid {max_grid}, patch_size={patch_size})")

    image = load_image(args.image, img_size, device)
    log.info(f"Loaded image: {args.image} -> {image.shape}")

    # Eval viewpoints: full scene + 4 quadrants (coarse to fine)
    viewpoints = make_eval_viewpoints(1, device)
    log.info(f"Viewpoints: {[vp.name for vp in viewpoints]}")

    # Collect: AVP scenes + teacher at G res upsampled to max_grid
    all_scenes: dict[int, list[np.ndarray]] = {}
    teacher_upsampled: dict[int, np.ndarray] = {}  # teacher at G res, latents upsampled to max_grid

    with torch.no_grad():
        # Teacher at max resolution (ground truth + PCA fitting)
        teacher_patches = teacher.forward_norm_features(image).patches
        assert teacher_patches.shape == (1, max_grid * max_grid, teacher_patches.shape[-1])
        teacher_dim = teacher_patches.shape[-1]
        teacher_full_flat = teacher_patches[0].cpu().numpy()
        log.info(f"Teacher at full res: {max_grid}×{max_grid}")

        # Teacher at FIXED 16×16 resolution (256px input)
        teacher_base_grid = 16
        teacher_base_px = teacher_base_grid * patch_size  # 256px
        img_at_base = F.interpolate(image, size=(teacher_base_px, teacher_base_px), mode="bilinear", align_corners=False)
        teacher_base = teacher.forward_norm_features(img_at_base).patches  # [1, 16*16, D]
        assert teacher_base.shape == (1, teacher_base_grid * teacher_base_grid, teacher_dim)
        teacher_base_spatial = teacher_base.view(1, teacher_base_grid, teacher_base_grid, teacher_dim).permute(0, 3, 1, 2)  # [1, D, 16, 16]
        log.info(f"Teacher baseline: {teacher_base_px}px → {teacher_base_grid}×{teacher_base_grid} tokens")

        for G in args.grid_sizes:
            # Upsample the SAME 16×16 teacher tokens to G×G
            teacher_at_G = F.interpolate(teacher_base_spatial, size=(G, G), mode="bilinear", align_corners=False)
            teacher_upsampled[G] = teacher_at_G[0].permute(1, 2, 0).cpu().numpy().reshape(-1, teacher_dim)
            log.info(f"G={G}: teacher 16×16 → upsample to {G}×{G}")

            # AVP at this grid size
            hidden = avp.init_hidden(1, G)
            outputs, _ = avp.forward_trajectory_full(image, viewpoints, hidden)

            # AVP scene at G×G
            scenes_list = []
            for out in outputs:
                scene = out.scene[0].cpu().numpy()  # [G*G, D]
                assert scene.shape == (G * G, teacher_dim)
                scenes_list.append(scene)
            all_scenes[G] = scenes_list

    # Create figure: 1 row teacher full + 2 rows per grid size (AVP, Teacher@G↑)
    n_viewpoints = len(viewpoints)
    n_rows = 1 + 2 * len(args.grid_sizes)
    n_cols = n_viewpoints + 1  # +1 for input image column
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))

    # Input image (denormalized)
    img_np = imagenet_denormalize(image[0].cpu()).numpy()
    H, W = img_np.shape[:2]

    # Get viewpoint boxes and colors
    boxes = [vp.to_pixel_box(0, H, W) for vp in viewpoints]
    colors = timestep_colors(n_viewpoints)

    # Fit PCA on teacher full features (ground truth basis)
    pca = fit_pca(teacher_full_flat)

    # Row 0: Teacher at full resolution (ground truth)
    teacher_full_rgb = pca_rgb(pca, teacher_full_flat, max_grid, max_grid)
    ax = axes[0, 0]
    ax.imshow(img_np)
    for box, color in zip(boxes, colors, strict=True):
        rect = Rectangle(
            (box.left, box.top), box.width, box.height,
            linewidth=2, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)
    ax.set_ylabel("Teacher full")
    ax.set_title("Input")
    ax.set_xticks([])
    ax.set_yticks([])
    for t in range(n_viewpoints):
        ax = axes[0, t + 1]
        ax.imshow(teacher_full_rgb)
        ax.set_title(f"t={t} ({viewpoints[t].name})")
        ax.set_xticks([])
        ax.set_yticks([])

    # Per grid size: AVP and Teacher 16×16→G upsampled
    for g_idx, G in enumerate(args.grid_sizes):
        scenes = all_scenes[G]
        teacher_rgb = pca_rgb(pca, teacher_upsampled[G], G, G)

        # AVP row
        row_avp = 1 + g_idx * 2
        ax = axes[row_avp, 0]
        ax.imshow(img_np)
        ax.set_ylabel(f"G={G} AVP")
        ax.set_xticks([])
        ax.set_yticks([])

        for t, scene in enumerate(scenes):
            ax = axes[row_avp, t + 1]
            rgb = pca_rgb(pca, scene, G, G)
            ax.imshow(rgb)
            ax.set_xticks([])
            ax.set_yticks([])

        # Teacher 16×16→G row (static)
        row_teacher = 1 + g_idx * 2 + 1
        ax = axes[row_teacher, 0]
        ax.imshow(img_np)
        ax.set_ylabel(f"G={G} T16↑")
        ax.set_xticks([])
        ax.set_yticks([])

        for t in range(n_viewpoints):
            ax = axes[row_teacher, t + 1]
            ax.imshow(teacher_rgb)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    log.info(f"Saved: {args.output}")


if __name__ == "__main__":
    main(tyro.cli(Args))
