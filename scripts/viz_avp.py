"""Minimal script to visualize AVP output on a single image."""

import copy
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from dinov3.hub.backbones import dinov3_vits16
from matplotlib.patches import Rectangle
from PIL import Image
from PIL.Image import Resampling
from sklearn.decomposition import PCA
from ytch.device import get_sensible_device

from avp_vit import AVPConfig, AVPViT
from avp_vit.backbone.dinov3 import DINOv3Backbone
from avp_vit.glimpse import Viewpoint
from avp_vit.train.data import IMAGENET_MEAN, IMAGENET_STD


@dataclass
class Args:
    ckpt: Path
    image: Path
    teacher_ckpt: Path = Path("dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
    n_glimpses: int = 5


def load_image(path: Path, size: int, device: torch.device) -> torch.Tensor:
    """Load and preprocess image to [1, 3, H, W]."""
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Resampling.BILINEAR)
    t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return ((t - mean) / std).unsqueeze(0).to(device)


def make_viewpoints(n: int, device: torch.device) -> list[Viewpoint]:
    """Full scene + quadrants."""
    vps = [Viewpoint.full_scene(1, device)]
    quads = [(0, 0), (1, 0), (0, 1), (1, 1)]
    for qx, qy in quads[: n - 1]:
        vps.append(Viewpoint.quadrant(1, device, qx, qy))
    return vps[:n]


def pca_rgb(features: np.ndarray, H: int, W: int) -> np.ndarray:
    """PCA -> RGB, fit on same features."""
    pca = PCA(n_components=3, whiten=True)
    proj = pca.fit_transform(features)
    return 1.0 / (1.0 + np.exp(-proj.reshape(H, W, 3) * 2.0))


def main(args: Args) -> None:
    device = get_sensible_device()

    # Load backbone for architecture
    backbone = DINOv3Backbone(
        dinov3_vits16(weights=str(args.teacher_ckpt), pretrained=True).eval()
    )

    # Load checkpoint to get grid size
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    G = ckpt.get("current_grid_size", 16)

    # Create AVP and load weights
    cfg = AVPConfig(
        scene_grid_size=G,
        glimpse_grid_size=7,
        n_scene_registers=32,
        use_output_proj=True,
        use_output_proj_norm=False,
        gating="full",
    )
    avp = AVPViT(copy.deepcopy(backbone), cfg).to(device)
    avp.load_state_dict(ckpt["avp"])
    avp.eval()

    # Load image
    scene_px = G * backbone.patch_size
    img = load_image(args.image, scene_px, device)

    # Run forward
    vps = make_viewpoints(args.n_glimpses, device)
    with torch.inference_mode():
        outputs, _ = avp.forward_trajectory_full(img, vps)

    # Visualize: image + trajectory + PCA of final scene
    final_scene = outputs[-1].scene[0].cpu().numpy()  # [G*G, D]
    scene_rgb = pca_rgb(final_scene, G, G)

    # Denormalize image for display
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img_np = ((img[0].cpu() * std + mean).clamp(0, 1)).permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Left: image with boxes
    axes[0].imshow(img_np)
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, len(vps) - 1)) for i in range(len(vps))]
    for i, vp in enumerate(vps):
        box = vp.to_pixel_box(0, scene_px, scene_px)
        rect = Rectangle(
            (box.left, box.top), box.width, box.height,
            linewidth=2, edgecolor=colors[i], facecolor="none",
        )
        axes[0].add_patch(rect)
        axes[0].plot(box.center_x, box.center_y, "o", color=colors[i], markersize=6)
    axes[0].set_title("Glimpse trajectory")
    axes[0].axis("off")

    # Right: PCA of final scene
    axes[1].imshow(scene_rgb)
    axes[1].set_title(f"Final scene (PCA, G={G})")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import tyro
    main(tyro.cli(Args))
