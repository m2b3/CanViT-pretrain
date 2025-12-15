"""Train AVP scene reconstruction on synthetic gaussian blob images.

Pixel-space reconstruction: glimpses → AVP → scene embeddings → decoder → pixels.
Loss is MSE in pixel space against original image.
"""

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path

import comet_ml
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from dinov3.hub.backbones import dinov3_vits16
from matplotlib.figure import Figure
from torch import Tensor
from tqdm import tqdm
from ytch.device import get_sensible_device
from ytch.model import count_parameters

from avp_vit import AVPConfig, AVPViT
from avp_vit.backbone.dinov3 import DINOv3Backbone
from avp_vit.glimpse import Viewpoint
from avp_vit.train import warmup_cosine_scheduler
from avp_vit.train.viewpoint import random_viewpoint
from avp_vit.train_gaussians import generate_multi_blob_batch, imagenet_denormalize, log_figure

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


class PixelDecoder(nn.Module):
    """Decode scene embeddings to pixel space."""

    def __init__(self, embed_dim: int, patch_size: int, grid_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.proj = nn.Linear(embed_dim, patch_size * patch_size * 3)

    def forward(self, scene: Tensor) -> Tensor:
        """Decode scene to image.

        Args:
            scene: [B, N, D] scene embeddings where N = grid_size^2

        Returns:
            [B, 3, H, W] reconstructed image (ImageNet-normalized)
        """
        B = scene.shape[0]
        patches = self.proj(scene)  # [B, N, P*P*3]
        patches = patches.view(
            B, self.grid_size, self.grid_size, self.patch_size, self.patch_size, 3
        )
        # Rearrange: [B, Gy, Gx, Py, Px, C] -> [B, C, Gy*Py, Gx*Px]
        img = patches.permute(0, 5, 1, 3, 2, 4).contiguous()
        img = img.view(
            B, 3, self.grid_size * self.patch_size, self.grid_size * self.patch_size
        )
        return img


@dataclass
class Config:
    # Paths
    teacher_ckpt: Path = Path("dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
    ckpt_dir: Path = Path("checkpoints")
    # Model
    avp: AVPConfig = field(
        default_factory=lambda: AVPConfig(
            scene_grid_size=16,  # 16*14=224px scene
            glimpse_grid_size=4,  # 4*14=56px glimpse, min_scale=0.25
            gate_init=1e-4,
            use_output_proj=True,
            use_scene_registers=True,
            gradient_checkpointing=False,
        )
    )
    freeze_backbone: bool = True
    # Training
    n_viewpoints_per_step: int = 2
    n_steps: int = 10000
    batch_size: int = 64
    ref_lr: float = 1e-5
    weight_decay: float = 1e-4
    warmup_steps: int = 5000
    grad_clip: float = 1.0
    adam_beta1: float = 0.85
    adam_beta2: float = 0.995
    # Task
    n_blobs: int = 2
    blob_margin: float = 0.3
    blob_sigma_min: float = 0.08
    blob_sigma_max: float = 0.12
    # Logging
    log_every: int = 20
    val_every: int = 100
    # Compilation
    compile: bool = False
    # Runtime
    device: torch.device = field(default_factory=get_sensible_device)

    @property
    def min_viewpoint_scale(self) -> float:
        return self.avp.glimpse_grid_size / self.avp.scene_grid_size

    @property
    def max_viewpoint_scale(self) -> float:
        return 1.0


def load_backbone(cfg: Config) -> DINOv3Backbone:
    """Load backbone (optionally frozen)."""
    model = dinov3_vits16(weights=str(cfg.teacher_ckpt), pretrained=True)
    backbone = DINOv3Backbone(model.eval().to(cfg.device))
    if cfg.freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = False
    return backbone


def create_avp(backbone: DINOv3Backbone, cfg: Config) -> AVPViT:
    """Create AVP model."""
    backbone_copy = copy.deepcopy(backbone)
    if cfg.freeze_backbone:
        for p in backbone_copy.parameters():
            p.requires_grad = False
    return AVPViT(backbone_copy, cfg.avp).to(cfg.device)


def plot_pixel_recon(
    original: Tensor,
    reconstructions: list[Tensor],
    glimpses: list[Tensor],
    viewpoints: list[Viewpoint],
    sample_idx: int = 0,
) -> Figure:
    """Plot original image, glimpses, and reconstructions at each timestep.

    Args:
        original: [B, 3, H, W] original images (ImageNet-normalized)
        reconstructions: List of [B, 3, H, W] reconstructed images per timestep
        glimpses: List of [B, 3, G, G] glimpse images per timestep
        viewpoints: List of viewpoints
        sample_idx: Which batch sample to show
    """
    n_steps = len(reconstructions)
    # Row 1: original + glimpses, Row 2: reconstructions
    fig, axes = plt.subplots(2, n_steps + 1, figsize=(3 * (n_steps + 1), 6))

    H, W = original.shape[-2], original.shape[-1]
    colors = plt.cm.viridis(torch.linspace(0, 1, n_steps).numpy())

    # Row 0, Col 0: Original with viewpoint boxes
    orig_img = imagenet_denormalize(original[sample_idx : sample_idx + 1])
    axes[0, 0].imshow(orig_img[0].permute(1, 2, 0).cpu().numpy())
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    for t, vp in enumerate(viewpoints):
        box = vp.to_pixel_box(sample_idx, H, W)
        rect = plt.Rectangle(
            (box.left, box.top),
            box.width,
            box.height,
            fill=False,
            edgecolor=colors[t],
            linewidth=2,
        )
        axes[0, 0].add_patch(rect)

    # Row 0, Cols 1+: Glimpses
    for t, glimpse in enumerate(glimpses):
        g_img = imagenet_denormalize(glimpse[sample_idx : sample_idx + 1])
        axes[0, t + 1].imshow(g_img[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy())
        axes[0, t + 1].set_title(f"Glimpse t={t}")
        axes[0, t + 1].axis("off")
        # Color border to match viewpoint box
        for spine in axes[0, t + 1].spines.values():
            spine.set_edgecolor(colors[t])
            spine.set_linewidth(3)
            spine.set_visible(True)

    # Row 1, Col 0: empty (or could show something else)
    axes[1, 0].axis("off")

    # Row 1, Cols 1+: Reconstructions
    for t, recon in enumerate(reconstructions):
        recon_img = imagenet_denormalize(recon[sample_idx : sample_idx + 1])
        axes[1, t + 1].imshow(recon_img[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy())
        mse = nn.functional.mse_loss(recon[sample_idx], original[sample_idx]).item()
        axes[1, t + 1].set_title(f"Recon t={t}\nMSE={mse:.4f}")
        axes[1, t + 1].axis("off")

    plt.tight_layout()
    return fig


def forward_recon(
    avp: AVPViT,
    decoder: PixelDecoder,
    images: Tensor,
    viewpoints: list[Viewpoint],
    hidden: Tensor | None,
) -> tuple[Tensor, list[Tensor], list[Tensor], Tensor]:
    """Forward pass with pixel reconstruction.

    Returns:
        loss: Averaged MSE across all timesteps
        reconstructions: List of reconstructed images per timestep
        glimpses: List of extracted glimpse images per timestep
        final_hidden: Hidden state after all viewpoints
    """
    B = images.shape[0]
    if hidden is None:
        hidden = avp._init_hidden(B, None)

    losses = []
    reconstructions = []
    glimpses = []

    for vp in viewpoints:
        out = avp.forward_step(images, vp, hidden, None, None)
        hidden = out.hidden
        scene = out.scene  # [B, N, D]
        recon = decoder(scene)  # [B, 3, H, W]
        reconstructions.append(recon)
        glimpses.append(out.glimpse)  # [B, 3, G, G]
        losses.append(nn.functional.mse_loss(recon, images))

    loss = torch.stack(losses).mean()
    return loss, reconstructions, glimpses, hidden


def eval_and_log(
    exp: comet_ml.Experiment,
    step: int,
    avp: AVPViT,
    decoder: PixelDecoder,
    cfg: Config,
    image_size: int,
) -> float:
    """Evaluate on fresh batch. Returns final MSE."""
    images, _, _, _ = generate_multi_blob_batch(
        cfg.batch_size,
        image_size,
        cfg.n_blobs,
        cfg.device,
        margin=cfg.blob_margin,
        sigma_range=(cfg.blob_sigma_min, cfg.blob_sigma_max),
    )

    viewpoints = [
        random_viewpoint(
            cfg.batch_size, cfg.device, cfg.min_viewpoint_scale, cfg.max_viewpoint_scale
        )
        for _ in range(cfg.n_viewpoints_per_step)
    ]

    with torch.inference_mode():
        _, reconstructions, glimpses, _ = forward_recon(avp, decoder, images, viewpoints, None)

        # Per-timestep MSE
        mses = [
            nn.functional.mse_loss(recon, images).item() for recon in reconstructions
        ]

        for t, mse in enumerate(mses):
            exp.log_metric(f"val/mse_t{t}", mse, step=step)

        val_loss = mses[-1]
        exp.log_metric("val/loss", val_loss, step=step)

        # Visualization
        fig = plot_pixel_recon(images, reconstructions, glimpses, viewpoints, sample_idx=0)
        log_figure(exp, fig, "val/recon", step)

    return val_loss


def train(cfg: Config) -> None:
    """Train AVP + decoder for pixel reconstruction on gaussian blobs."""
    log.info(f"Device: {cfg.device}")

    exp = comet_ml.Experiment(
        project_name="avp-vit-gaussian-recon", auto_metric_logging=False
    )
    exp.log_parameters(
        {
            k: str(v) if isinstance(v, (torch.device, Path)) else v
            for k, v in cfg.__dict__.items()
        }
    )

    log.info("Loading backbone...")
    backbone = load_backbone(cfg)
    log.info(f"Backbone params: {count_parameters(backbone):,}")

    log.info("Creating AVP model...")
    avp = create_avp(backbone, cfg)

    image_size = avp.scene_size
    log.info(
        f"Image size: {image_size}px ({cfg.avp.scene_grid_size}x{avp.backbone.patch_size})"
    )

    log.info("Creating pixel decoder...")
    decoder = PixelDecoder(
        embed_dim=backbone.embed_dim,
        patch_size=avp.backbone.patch_size,
        grid_size=cfg.avp.scene_grid_size,
    ).to(cfg.device)
    log.info(f"Decoder params: {count_parameters(decoder):,}")

    # Collect trainable params from both AVP and decoder
    trainable = [p for p in avp.parameters() if p.requires_grad]
    trainable += list(decoder.parameters())
    n_trainable = sum(p.numel() for p in trainable)
    n_total = count_parameters(avp) + count_parameters(decoder)
    log.info(
        f"Total: {n_total:,}, trainable: {n_trainable:,} ({100 * n_trainable / n_total:.1f}%)"
    )
    exp.log_parameters({"trainable_params": n_trainable, "total_params": n_total})

    peak_lr = cfg.ref_lr * cfg.batch_size
    optimizer = torch.optim.AdamW(
        trainable,
        lr=peak_lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.weight_decay,
    )
    scheduler = warmup_cosine_scheduler(optimizer, cfg.n_steps, cfg.warmup_steps)
    log.info(
        f"Optimizer: peak_lr={peak_lr:.2e}, warmup={cfg.warmup_steps}, betas=({cfg.adam_beta1}, {cfg.adam_beta2})"
    )

    # Initial eval
    log.info("Initial evaluation...")
    val_loss = eval_and_log(exp, 0, avp, decoder, cfg, image_size)
    log.info(f"Init val_loss: {val_loss:.4f}")

    ema_loss = 0.0
    alpha = 2 / (cfg.log_every + 1)

    log.info("Starting training...")
    pbar = tqdm(range(1, cfg.n_steps + 1), desc="Training", unit="step")

    for step in pbar:
        # Generate batch
        images, _, _, _ = generate_multi_blob_batch(
            cfg.batch_size,
            image_size,
            cfg.n_blobs,
            cfg.device,
            margin=cfg.blob_margin,
            sigma_range=(cfg.blob_sigma_min, cfg.blob_sigma_max),
        )

        # Random viewpoints
        viewpoints = [
            random_viewpoint(
                cfg.batch_size,
                cfg.device,
                cfg.min_viewpoint_scale,
                cfg.max_viewpoint_scale,
            )
            for _ in range(cfg.n_viewpoints_per_step)
        ]

        # Forward + loss
        loss, _, _, _ = forward_recon(avp, decoder, images, viewpoints, None)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        ema_loss = alpha * loss.item() + (1 - alpha) * ema_loss if step > 1 else loss.item()

        if step % cfg.log_every == 0:
            lr = scheduler.get_last_lr()[0]
            exp.log_metrics(
                {
                    "train/loss": ema_loss,
                    "train/grad_norm": grad_norm.item(),
                    "train/lr": lr,
                },
                step=step,
            )
            pbar.set_postfix_str(f"loss={ema_loss:.2e} grad={grad_norm:.2e} lr={lr:.2e}")

        if step % cfg.val_every == 0:
            val_loss = eval_and_log(exp, step, avp, decoder, cfg, image_size)

    # Final eval
    val_loss = eval_and_log(exp, cfg.n_steps, avp, decoder, cfg, image_size)
    log.info(f"Final: train_ema={ema_loss:.4f}, val={val_loss:.4f}")

    # Save AVP checkpoint (just AVP, not decoder - for transfer to search)
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = cfg.ckpt_dir / f"avp_recon_{exp.get_key()}.pt"
    torch.save(avp.state_dict(), ckpt_path)
    log.info(f"Saved AVP checkpoint: {ckpt_path}")

    exp.end()


def main() -> None:
    import tyro

    torch.set_float32_matmul_precision("high")
    cfg = tyro.cli(Config)
    log.info(f"Config: {cfg}")
    train(cfg)


if __name__ == "__main__":
    main()
