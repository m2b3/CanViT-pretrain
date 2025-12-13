"""Experiment: Can AVPViT's POL token learn to decode position?

Uses real AVPViT with use_policy=True to validate the policy architecture.
Synthetic data: images with gaussian blob, target = blob center.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from dinov3.hub.backbones import dinov3_vits16
from torch import Tensor

from avp_vit.backbone.dinov3 import DINOv3Backbone
from avp_vit.glimpse import Viewpoint
from avp_vit.model import AVPConfig, AVPViT

CKPT_PATH = Path("dinov3_vits16_pretrain_lvd1689m-08c60483.pth")


def make_gaussian_images(
    B: int, size: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Create images with gaussian blob, return (images, centers).

    Images: [B, 3, size, size] with gaussian intensity blob
    Centers: [B, 2] in [-1, 1] (y, x)
    """
    centers = (torch.rand(B, 2, device=device) * 2 - 1) * 0.7

    coords = torch.linspace(-1, 1, size, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    positions = torch.stack([yy, xx], dim=-1)

    diff = positions.unsqueeze(0) - centers.view(B, 1, 1, 2)
    dist_sq = (diff**2).sum(dim=-1)

    sigma = 0.2
    gaussian = torch.exp(-dist_sq / (2 * sigma**2))

    noise = torch.rand(B, 3, size, size, device=device) * 0.1
    images = gaussian.unsqueeze(1).expand(-1, 3, -1, -1) + noise

    return images, centers


def run_experiment(
    scene_size: int = 224,
    scene_grid_size: int = 14,  # 224 / 16 = 14
    glimpse_grid_size: int = 7,
    B: int = 32,
    n_steps: int = 200,
    lr: float = 1e-4,
    device: str = "mps",
    out_dir: str = "outputs/exp_position_decode",
) -> None:
    """Run position decoding experiment with real AVPViT."""
    dev = torch.device(device)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("Loading DINOv3 backbone...")
    model = dinov3_vits16(weights=str(CKPT_PATH), pretrained=True)
    backbone = DINOv3Backbone(model.eval().to(dev))
    for p in backbone.parameters():
        p.requires_grad = False

    cfg = AVPConfig(
        scene_grid_size=scene_grid_size,
        glimpse_grid_size=glimpse_grid_size,
        use_policy=True,
    )
    avp = AVPViT(backbone, cfg).to(dev)

    trainable = [p for p in avp.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    print(f"Trainable params: {n_params:,}")

    optimizer = torch.optim.Adam(trainable, lr=lr)

    losses = []
    maes = []

    print(f"Config: scene_size={scene_size}, scene_grid={scene_grid_size}, glimpse_grid={glimpse_grid_size}, B={B}, lr={lr}")

    for step in range(n_steps):
        images, target = make_gaussian_images(B, scene_size, dev)

        vp = Viewpoint.full_scene(B, dev)
        _, scene, pol_out = avp.forward_step(images, vp, None)

        assert pol_out is not None
        pred = torch.tanh(pol_out)
        loss = F.mse_loss(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        mae = (pred - target).abs().mean().item()
        maes.append(mae)

        if step % 20 == 0 or step == n_steps - 1:
            mae_frac = mae / 2.0
            print(f"step {step:3d}: loss={loss.item():.4e}, mae={mae:.4f} ({mae_frac*100:.1f}% of grid)")

    print("\nGenerating plots...")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(losses)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Training Loss")
    axes[0].set_yscale("log")

    axes[1].plot(maes)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("MAE")
    axes[1].set_title("Mean Absolute Error")
    plt.tight_layout()
    plt.savefig(out_path / "training_curves.png", dpi=150)
    plt.close()

    avp.eval()
    with torch.no_grad():
        images, target = make_gaussian_images(8, scene_size, dev)
        vp = Viewpoint.full_scene(8, dev)
        _, _, pol_out = avp.forward_step(images, vp, None)
        assert pol_out is not None
        pred = torch.tanh(pol_out)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(8):
        ax = axes[i // 4, i % 4]
        ax.imshow(images[i, 0].cpu(), cmap="viridis", extent=[-1, 1, 1, -1])
        ax.scatter(target[i, 1].cpu(), target[i, 0].cpu(), c="red", s=100, marker="x", label="target")
        ax.scatter(pred[i, 1].cpu(), pred[i, 0].cpu(), c="cyan", s=100, marker="+", label="pred")
        err = (pred[i] - target[i]).norm().item()
        ax.set_title(f"err={err:.3f}")
        ax.set_xlim(-1, 1)
        ax.set_ylim(1, -1)
        if i == 0:
            ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path / "samples.png", dpi=150)
    plt.close()

    with torch.no_grad():
        images, target = make_gaussian_images(256, scene_size, dev)
        vp = Viewpoint.full_scene(256, dev)
        _, _, pol_out = avp.forward_step(images, vp, None)
        assert pol_out is not None
        pred = torch.tanh(pol_out)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].scatter(target[:, 0].cpu(), pred[:, 0].cpu(), alpha=0.3, s=10)
    axes[0].plot([-1, 1], [-1, 1], "r--", label="perfect")
    axes[0].set_xlabel("Target Y")
    axes[0].set_ylabel("Predicted Y")
    axes[0].set_title("Y coordinate")
    axes[0].legend()
    axes[0].set_aspect("equal")

    axes[1].scatter(target[:, 1].cpu(), pred[:, 1].cpu(), alpha=0.3, s=10)
    axes[1].plot([-1, 1], [-1, 1], "r--", label="perfect")
    axes[1].set_xlabel("Target X")
    axes[1].set_ylabel("Predicted X")
    axes[1].set_title("X coordinate")
    axes[1].legend()
    axes[1].set_aspect("equal")

    plt.tight_layout()
    plt.savefig(out_path / "pred_vs_target.png", dpi=150)
    plt.close()

    print(f"Plots saved to {out_path}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_size", type=int, default=224)
    parser.add_argument("--scene_grid_size", type=int, default=14)
    parser.add_argument("--glimpse_grid_size", type=int, default=7)
    parser.add_argument("--B", type=int, default=32)
    parser.add_argument("--n_steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--out_dir", type=str, default="outputs/exp_position_decode")
    args = parser.parse_args()

    run_experiment(
        scene_size=args.scene_size,
        scene_grid_size=args.scene_grid_size,
        glimpse_grid_size=args.glimpse_grid_size,
        B=args.B,
        n_steps=args.n_steps,
        lr=args.lr,
        device=args.device,
        out_dir=args.out_dir,
    )
