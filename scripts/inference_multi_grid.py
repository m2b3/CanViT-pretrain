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


def load_model(ckpt_path: Path, backbone_weights: Path | None, device: torch.device) -> AVPViT:
    """Load AVP model from checkpoint."""
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

    log.info(f"Model loaded: {backbone_slug}, teacher_dim={teacher_dim}")
    log.info(f"  glimpse_grid_size={cfg.glimpse_grid_size}, registers={cfg.n_scene_registers}")
    return avp


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

    avp = load_model(args.checkpoint, args.backbone_weights, device)
    patch_size = avp.backbone.patch_size

    max_grid = max(args.grid_sizes)
    img_size = max_grid * patch_size
    log.info(f"Image size: {img_size}px (for max grid {max_grid}, patch_size={patch_size})")

    image = load_image(args.image, img_size, device)
    log.info(f"Loaded image: {args.image} -> {image.shape}")

    # Eval viewpoints: full scene + 4 quadrants (coarse to fine)
    viewpoints = make_eval_viewpoints(1, device)
    log.info(f"Viewpoints: {[vp.name for vp in viewpoints]}")

    # Collect scenes at each grid size
    all_scenes: dict[int, list[np.ndarray]] = {}

    with torch.no_grad():
        for G in args.grid_sizes:
            log.info(f"Grid size {G}x{G}...")
            hidden = avp.init_hidden(1, G)
            outputs, _ = avp.forward_trajectory_full(image, viewpoints, hidden)
            all_scenes[G] = [out.scene[0].cpu().numpy() for out in outputs]

    # Create figure: rows = grid sizes, cols = input image + timesteps
    n_viewpoints = len(viewpoints)
    n_rows = len(args.grid_sizes)
    n_cols = n_viewpoints + 1  # +1 for input image column
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))

    # Input image (denormalized)
    img_np = imagenet_denormalize(image[0].cpu()).numpy()
    H, W = img_np.shape[:2]

    # Get viewpoint boxes and colors
    boxes = [vp.to_pixel_box(0, H, W) for vp in viewpoints]
    colors = timestep_colors(n_viewpoints)

    for row_idx, G in enumerate(args.grid_sizes):
        scenes = all_scenes[G]
        pca = fit_pca(scenes[-1])

        # First column: input image with viewpoint boxes
        ax = axes[row_idx, 0]
        ax.imshow(img_np)
        for box, color in zip(boxes, colors, strict=True):
            rect = Rectangle(
                (box.left, box.top), box.width, box.height,
                linewidth=2, edgecolor=color, facecolor="none",
            )
            ax.add_patch(rect)
        ax.set_title("Input + viewpoints" if row_idx == 0 else "")
        ax.set_ylabel(f"G={G}")
        ax.set_xticks([])
        ax.set_yticks([])

        # Remaining columns: scene PCA at each timestep
        for t, scene in enumerate(scenes):
            ax = axes[row_idx, t + 1]
            rgb = pca_rgb(pca, scene, G, G)
            ax.imshow(rgb)
            vp_name = viewpoints[t].name
            ax.set_title(f"t={t} ({vp_name})" if row_idx == 0 else "")
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    log.info(f"Saved: {args.output}")


if __name__ == "__main__":
    main(tyro.cli(Args))
