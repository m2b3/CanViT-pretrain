"""DINOv3 inference with PCA visualization."""

import urllib.request
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF
from dinov3.hub.backbones import dinov3_vits16
from PIL import Image
from sklearn.decomposition import PCA
from ytch.device import get_sensible_device

from canvit.backbone.dinov3 import DINOv3Backbone
from canvit.rope import compute_rope, make_grid_positions

matplotlib.use("Agg")

CKPT_PATH = Path("dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
OUT_PATH = Path("outputs/dinov3_pca.png")
IMAGE_URL = "https://dl.fbaipublicfiles.com/dinov3/notebooks/pca/test_image.jpg"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_backbone(device: torch.device) -> DINOv3Backbone:
    model = dinov3_vits16(weights=str(CKPT_PATH), pretrained=True)
    return DINOv3Backbone(model.eval().to(device))


def load_image(url: str, size: int = 448) -> tuple[Image.Image, torch.Tensor]:
    with urllib.request.urlopen(url) as f:
        img = Image.open(f).convert("RGB").resize((size, size))
    x = TF.normalize(TF.to_tensor(img), mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD))
    return img, x.unsqueeze(0)


def extract_features(backbone: DINOv3Backbone, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    with torch.inference_mode():
        tokens, H, W = backbone.prepare_tokens(x)
        pos = make_grid_positions(H, W, tokens.device, dtype=backbone.rope_dtype).unsqueeze(0)
        rope = compute_rope(pos, backbone.rope_periods)
        for i in range(backbone.n_blocks):
            tokens = backbone.forward_block(i, tokens, rope)
        return tokens[:, backbone.n_prefix_tokens:, :], H, W


def pca_rgb(features: torch.Tensor, H: int, W: int) -> torch.Tensor:
    pca = PCA(n_components=3, whiten=True)
    proj = pca.fit_transform(features.squeeze(0).float().cpu().numpy())
    return torch.sigmoid(torch.from_numpy(proj).view(H, W, 3) * 2.0)


def main() -> None:
    device = get_sensible_device()
    print(f"Using device: {device}")
    OUT_PATH.parent.mkdir(exist_ok=True)

    backbone = load_backbone(device)
    img, x = load_image(IMAGE_URL)
    features, H, W = extract_features(backbone, x.to(device))
    rgb = pca_rgb(features, H, W)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    axes[0].set_title("Input")
    axes[0].axis("off")
    axes[1].imshow(rgb.numpy())
    axes[1].set_title("PCA Features")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
