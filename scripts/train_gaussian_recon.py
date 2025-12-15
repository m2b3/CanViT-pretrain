"""Train AVP scene reconstruction on synthetic gaussian blob images.

Uses random viewpoints and MSE loss on scene output vs teacher patches.
Similar to train_scene_match.py but with synthetic data for faster iteration.
"""

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path

import comet_ml
import torch
import torch.nn as nn
from dinov3.hub.backbones import dinov3_vits16
from torch import Tensor
from tqdm import tqdm
from ytch.device import get_sensible_device
from ytch.model import count_parameters

from avp_vit import AVPConfig, AVPViT
from avp_vit.backbone.dinov3 import DINOv3Backbone
from avp_vit.glimpse import Viewpoint
from avp_vit.train import warmup_cosine_scheduler
from avp_vit.train.viewpoint import random_viewpoint
from avp_vit.train.viz import imagenet_denormalize, plot_multistep_pca, plot_trajectory
from avp_vit.train_gaussians import generate_multi_blob_batch, log_figure

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


@dataclass
class Config:
    # Paths
    teacher_ckpt: Path = Path("dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
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
    freeze_inner_backbone: bool = False
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


def load_teacher(cfg: Config) -> DINOv3Backbone:
    """Load frozen teacher backbone."""
    model = dinov3_vits16(weights=str(cfg.teacher_ckpt), pretrained=True)
    backbone = DINOv3Backbone(model.eval().to(cfg.device))
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone


def create_avp(teacher: DINOv3Backbone, cfg: Config) -> AVPViT:
    """Create AVP model with optionally frozen inner backbone."""
    backbone_copy = copy.deepcopy(teacher)
    for p in backbone_copy.parameters():
        p.requires_grad = not cfg.freeze_inner_backbone
    return AVPViT(backbone_copy, cfg.avp).to(cfg.device)


def viz_and_log(
    exp: comet_ml.Experiment,
    step: int,
    prefix: str,
    avp: AVPViT,
    teacher: DINOv3Backbone,
    images: Tensor,
    viewpoints: list[Viewpoint],
    target: Tensor,
    hidden: Tensor | None,
) -> list[float]:
    """Run forward trajectory and log visualization. Returns per-step MSEs."""
    assert isinstance(avp.backbone, DINOv3Backbone)

    with torch.inference_mode():
        outputs, _, _ = avp.forward_trajectory_full(images, viewpoints, hidden)
        mses = [nn.functional.mse_loss(out.scene, target).item() for out in outputs]

        # Initial scene
        if hidden is not None:
            initial_scene = avp.compute_scene(hidden[0:1])[0]
        else:
            initial_scene = avp.output_proj(avp.spatial_init)[0]

        # Prepare viz data
        sample_idx = 0
        n_prefix = teacher.n_prefix_tokens
        H, W = avp.scene_size, avp.scene_size

        full_img = imagenet_denormalize(images[sample_idx].cpu()).numpy()
        teacher_np = target[sample_idx].cpu().float().numpy()
        initial_np = initial_scene.cpu().float().numpy()

        scenes = [out.scene[sample_idx].cpu().float().numpy() for out in outputs]
        locals_avp = [
            avp.backbone.output_norm(out.local[sample_idx : sample_idx + 1, n_prefix:])
            .squeeze(0).cpu().float().numpy()
            for out in outputs
        ]
        locals_teacher = [
            teacher.forward_norm_patches(out.glimpse[sample_idx : sample_idx + 1])
            .squeeze(0).cpu().float().numpy()
            for out in outputs
        ]
        glimpses = [
            imagenet_denormalize(out.glimpse[sample_idx].cpu()).numpy()
            for out in outputs
        ]
        boxes = [vp.to_pixel_box(sample_idx, H, W) for vp in viewpoints]
        names = [vp.name for vp in viewpoints]

    fig_pca = plot_multistep_pca(
        full_img, teacher_np, scenes, locals_avp, locals_teacher,
        glimpses, boxes, names, avp.cfg.scene_grid_size,
        avp.cfg.glimpse_grid_size, initial_np,
    )
    log_figure(exp, fig_pca, f"{prefix}/pca", step)

    fig_traj = plot_trajectory(full_img, boxes, names)
    log_figure(exp, fig_traj, f"{prefix}/trajectory", step)

    return mses


def eval_and_log(
    exp: comet_ml.Experiment,
    step: int,
    avp: AVPViT,
    teacher: DINOv3Backbone,
    cfg: Config,
    image_size: int,
) -> float:
    """Evaluate on fresh batch with random viewpoints. Returns final MSE."""
    images, _, _, _ = generate_multi_blob_batch(
        cfg.batch_size, image_size, cfg.n_blobs, cfg.device,
        margin=cfg.blob_margin, sigma_range=(cfg.blob_sigma_min, cfg.blob_sigma_max),
    )

    with torch.inference_mode():
        target = teacher.forward_norm_patches(images)

    viewpoints = [
        random_viewpoint(cfg.batch_size, cfg.device, cfg.min_viewpoint_scale, cfg.max_viewpoint_scale)
        for _ in range(cfg.n_viewpoints_per_step)
    ]

    mses = viz_and_log(exp, step, "val", avp, teacher, images, viewpoints, target, None)

    for t, mse in enumerate(mses):
        exp.log_metric(f"val/mse_t{t}", mse, step=step)

    val_loss = mses[-1]
    exp.log_metric("val/loss", val_loss, step=step)
    return val_loss


def train(cfg: Config) -> None:
    """Train AVP for scene reconstruction on gaussian blobs."""
    log.info(f"Device: {cfg.device}")

    exp = comet_ml.Experiment(project_name="avp-vit-gaussian-recon", auto_metric_logging=False)
    exp.log_parameters({
        k: str(v) if isinstance(v, (torch.device, Path)) else v
        for k, v in cfg.__dict__.items()
    })

    log.info("Loading teacher...")
    teacher = load_teacher(cfg)
    log.info(f"Teacher params: {count_parameters(teacher):,}")

    log.info("Creating AVP model...")
    avp = create_avp(teacher, cfg)

    image_size = avp.scene_size
    log.info(f"Image size: {image_size}px ({cfg.avp.scene_grid_size}x{avp.backbone.patch_size})")

    trainable = [p for p in avp.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = count_parameters(avp)
    log.info(f"AVP total: {n_total:,}, trainable: {n_trainable:,} ({100 * n_trainable / n_total:.1f}%)")
    exp.log_parameters({"trainable_params": n_trainable, "total_params": n_total})

    peak_lr = cfg.ref_lr * cfg.batch_size
    optimizer = torch.optim.AdamW(
        trainable,
        lr=peak_lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.weight_decay,
    )
    scheduler = warmup_cosine_scheduler(optimizer, cfg.n_steps, cfg.warmup_steps)
    log.info(f"Optimizer: peak_lr={peak_lr:.2e}, warmup={cfg.warmup_steps}, betas=({cfg.adam_beta1}, {cfg.adam_beta2})")

    # Initial eval
    log.info("Initial evaluation...")
    val_loss = eval_and_log(exp, 0, avp, teacher, cfg, image_size)
    log.info(f"Init val_loss: {val_loss:.4f}")

    ema_loss = 0.0
    alpha = 2 / (cfg.log_every + 1)

    log.info("Starting training...")
    pbar = tqdm(range(1, cfg.n_steps + 1), desc="Training", unit="step")

    for step in pbar:
        # Generate batch
        images, _, _, _ = generate_multi_blob_batch(
            cfg.batch_size, image_size, cfg.n_blobs, cfg.device,
            margin=cfg.blob_margin, sigma_range=(cfg.blob_sigma_min, cfg.blob_sigma_max),
        )

        # Compute teacher target
        with torch.no_grad():
            target = teacher.forward_norm_patches(images)

        # Random viewpoints
        viewpoints = [
            random_viewpoint(cfg.batch_size, cfg.device, cfg.min_viewpoint_scale, cfg.max_viewpoint_scale)
            for _ in range(cfg.n_viewpoints_per_step)
        ]

        # Forward + loss
        loss, _, _ = avp.forward_loss(images, viewpoints, target, None, None)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        ema_loss = alpha * loss.item() + (1 - alpha) * ema_loss if step > 1 else loss.item()

        if step % cfg.log_every == 0:
            lr = scheduler.get_last_lr()[0]
            exp.log_metrics({
                "train/loss": ema_loss,
                "train/grad_norm": grad_norm.item(),
                "train/lr": lr,
            }, step=step)
            pbar.set_postfix_str(f"loss={ema_loss:.2e} grad={grad_norm:.2e} lr={lr:.2e}")

        if step % cfg.val_every == 0:
            val_loss = eval_and_log(exp, step, avp, teacher, cfg, image_size)

    # Final eval
    val_loss = eval_and_log(exp, cfg.n_steps, avp, teacher, cfg, image_size)
    log.info(f"Final: train_ema={ema_loss:.4f}, val={val_loss:.4f}")
    exp.end()


def main() -> None:
    import tyro
    torch.set_float32_matmul_precision("high")
    cfg = tyro.cli(Config)
    log.info(f"Config: {cfg}")
    train(cfg)


if __name__ == "__main__":
    main()
