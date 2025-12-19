"""Minimal script to visualize model output on a single image."""

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

from avp_vit import ActiveCanViT, ActiveCanViTConfig
from avp_vit.checkpoint import load
from canvit.backbone.dinov3 import DINOv3Backbone
from avp_vit.glimpse import Viewpoint
from avp_vit.train.data import IMAGENET_MEAN, IMAGENET_STD


@dataclass
class Args:
    ckpt: Path
    image: Path
    teacher_ckpt: Path = Path("dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
    n_glimpses: int = 5
    canvas_grid: int = 16


def load_image(path: Path, size: int, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Resampling.BILINEAR)
    t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return ((t - mean) / std).unsqueeze(0).to(device)


def make_viewpoints(n: int, device: torch.device) -> list[Viewpoint]:
    vps = [Viewpoint.full_scene(1, device)]
    quads = [(0, 0), (1, 0), (0, 1), (1, 1)]
    for qx, qy in quads[: n - 1]:
        vps.append(Viewpoint.quadrant(1, device, qx, qy))
    return vps[:n]


def pca_rgb_sigmoid(features: np.ndarray, H: int, W: int) -> np.ndarray:
    pca = PCA(n_components=3, whiten=True)
    proj = pca.fit_transform(features)
    return 1.0 / (1.0 + np.exp(-proj.reshape(H, W, 3) * 2.0))


def pca_rgb_normalized(features: np.ndarray, H: int, W: int) -> np.ndarray:
    pca = PCA(n_components=3, whiten=True)
    proj = pca.fit_transform(features).reshape(H, W, 3)
    for c in range(3):
        cmin, cmax = proj[:, :, c].min(), proj[:, :, c].max()
        if cmax > cmin:
            proj[:, :, c] = (proj[:, :, c] - cmin) / (cmax - cmin)
        else:
            proj[:, :, c] = 0.5
    return proj


def spatial_std(features: np.ndarray, H: int, W: int) -> np.ndarray:
    std = features.std(axis=1).reshape(H, W)
    std = (std - std.min()) / (std.max() - std.min() + 1e-8)
    return std


def main(args: Args) -> None:
    device = get_sensible_device()
    G = args.canvas_grid

    backbone = DINOv3Backbone(
        dinov3_vits16(weights=str(args.teacher_ckpt), pretrained=True).eval()
    )

    ckpt = load(args.ckpt, device)
    cfg = ActiveCanViTConfig(**ckpt["model_config"])
    model = ActiveCanViT(copy.deepcopy(backbone), cfg, ckpt["teacher_dim"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    canvas_px = G * backbone.patch_size_px
    img = load_image(args.image, canvas_px, device)

    vps = make_viewpoints(args.n_glimpses, device)
    with torch.inference_mode():
        canvas = model.init_canvas(1, G)
        outputs, _ = model.forward_trajectory_full(img, vps, canvas)

    final_scene = outputs[-1].scene[0].cpu().numpy()
    scene_sigmoid = pca_rgb_sigmoid(final_scene, G, G)
    scene_norm = pca_rgb_normalized(final_scene, G, G)
    scene_std = spatial_std(final_scene, G, G)

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img_np = ((img[0].cpu() * std + mean).clamp(0, 1)).permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    ax = axes[0, 0]
    ax.imshow(img_np)
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, len(vps) - 1)) for i in range(len(vps))]
    for i, vp in enumerate(vps):
        box = vp.to_pixel_box(0, canvas_px, canvas_px)
        rect = Rectangle(
            (box.left, box.top), box.width, box.height,
            linewidth=2, edgecolor=colors[i], facecolor="none",
        )
        ax.add_patch(rect)
        ax.plot(box.center_x, box.center_y, "o", color=colors[i], markersize=6)
    ax.set_title("Glimpse trajectory")
    ax.axis("off")

    axes[0, 1].imshow(scene_sigmoid)
    axes[0, 1].set_title(f"PCA sigmoid (G={G})")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(scene_norm)
    axes[1, 0].set_title(f"PCA normalized (G={G})")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(scene_std, cmap="inferno")
    axes[1, 1].set_title(f"Spatial std (G={G})")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import tyro
    main(tyro.cli(Args))
