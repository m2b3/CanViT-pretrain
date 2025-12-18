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
from avp_vit.glimpse import Viewpoint
from avp_vit.train.data import imagenet_normalize
from avp_vit.train.viz import fit_pca, pca_rgb, imagenet_denormalize
from scripts.train_scene_match.model import MODEL_REGISTRY

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


@dataclass
class Args:
    checkpoint: Path
    image: Path
    output: Path = Path("outputs/multi_grid_pca.png")
    grid_sizes: tuple[int, ...] = (8, 16, 32, 64, 128)
    n_glimpses: int = 4
    backbone_weights: Path | None = None
    device: str = "mps"
    seed: int = 42


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


def random_viewpoints(n: int, batch_size: int, device: torch.device, seed: int) -> list[Viewpoint]:
    """Generate reproducible random viewpoints."""
    gen = torch.Generator(device=device).manual_seed(seed)
    viewpoints = []
    for _ in range(n):
        centers = torch.rand(batch_size, 2, generator=gen, device=device)
        scales = 0.3 + 0.7 * torch.rand(batch_size, 1, generator=gen, device=device)
        viewpoints.append(Viewpoint(name=f"random_{len(viewpoints)}", centers=centers, scales=scales))
    return viewpoints


def main(args: Args) -> None:
    device = torch.device(args.device)
    log.info(f"Device: {device}")

    avp = load_model(args.checkpoint, args.backbone_weights, device)
    patch_size = avp.backbone.patch_size

    max_grid = max(args.grid_sizes)
    img_size = max_grid * patch_size
    log.info(f"Image size: {img_size}px (for max grid {max_grid}, patch_size={patch_size})")

    image = load_image(args.image, img_size, device)
    log.info(f"Loaded image: {args.image} -> {image.shape}")

    viewpoints = random_viewpoints(args.n_glimpses, 1, device, args.seed)
    log.info(f"Viewpoints: {args.n_glimpses}")

    # Collect scenes at each grid size
    all_scenes: dict[int, list[np.ndarray]] = {}

    with torch.no_grad():
        for G in args.grid_sizes:
            log.info(f"Grid size {G}x{G}...")
            hidden = avp.init_hidden(1, G)
            outputs, _ = avp.forward_trajectory_full(image, viewpoints, hidden)
            # scenes[i] shape: [1, G*G, D] -> [G*G, D]
            all_scenes[G] = [out.scene[0].cpu().numpy() for out in outputs]

    # Create figure: rows = grid sizes, cols = timesteps + input image
    n_rows = len(args.grid_sizes)
    n_cols = args.n_glimpses + 1  # +1 for input image column
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))

    # Input image (denormalized)
    img_np = imagenet_denormalize(image[0].cpu()).numpy()

    for row_idx, G in enumerate(args.grid_sizes):
        scenes = all_scenes[G]

        # Fit PCA independently for each grid size
        pca = fit_pca(scenes[-1])

        # First column: input image (same for all rows)
        ax = axes[row_idx, 0]
        ax.imshow(img_np)
        ax.set_title("Input" if row_idx == 0 else "")
        ax.set_ylabel(f"G={G}")
        ax.set_xticks([])
        ax.set_yticks([])

        # Remaining columns: scene PCA at each timestep
        for t, scene in enumerate(scenes):
            ax = axes[row_idx, t + 1]
            rgb = pca_rgb(pca, scene, G, G)
            ax.imshow(rgb)
            ax.set_title(f"t={t}" if row_idx == 0 else "")
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    log.info(f"Saved: {args.output}")


if __name__ == "__main__":
    main(tyro.cli(Args))
