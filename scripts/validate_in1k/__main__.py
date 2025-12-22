#!/usr/bin/env python3
"""ImageNet-1k validation with coarse-to-fine viewpoint policy.

Usage:
    uv run -m scripts.validate_in1k --val-dir /path/to/imagenet/val --checkpoint reference.pt
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from dinov3_probes import DINOv3LinearClassificationHead
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from avp_vit import ActiveCanViT
from avp_vit.checkpoint import load as load_ckpt, load_model
from canvit.viewpoint import Viewpoint as CoreViewpoint
from avp_vit.train.data import val_transform
from avp_vit.train.norm import PositionAwareNorm
from avp_vit.train.viewpoint import make_eval_viewpoints

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROBE_REPOS = {
    "dinov3_vits16": "yberreby/dinov3-vits16-lvd1689m-in1k-512x512-linear-clf-probe",
    "dinov3_vitb16": "yberreby/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe",
    "dinov3_vitl16": "yberreby/dinov3-vitl16-lvd1689m-in1k-512x512-linear-clf-probe",
}


@dataclass
class Config:
    val_dir: Path
    checkpoint: Path
    batch_size: int = 64
    num_workers: int = 8
    device: str = "mps"
    canvas_grid: int = 32
    glimpse_grid: int = 8


def load_cls_normalizer(ckpt_path: Path, device: torch.device) -> PositionAwareNorm:
    """Load CLS normalizer from checkpoint."""
    ckpt = load_ckpt(ckpt_path, device)
    cls_state = ckpt.get("cls_norm_state")
    assert cls_state is not None, "Checkpoint missing cls_norm_state"
    n_tokens, embed_dim = cls_state["mean"].shape
    cls_norm = PositionAwareNorm(n_tokens, embed_dim, grid_size=1)
    cls_norm.load_state_dict(cls_state)
    cls_norm.eval()
    return cls_norm.to(device)


def denormalize_cls(x: Tensor, norm: PositionAwareNorm) -> Tensor:
    """Invert PositionAwareNorm: x * sqrt(var + eps) + mean."""
    std = (norm.var + norm.eps).sqrt()
    return x * std + norm.mean


def run_trajectory(
    model: ActiveCanViT,
    images: Tensor,
    canvas_grid: int,
    glimpse_size_px: int,
) -> Tensor:
    """Run coarse-to-fine trajectory and return final CLS prediction."""
    B = images.shape[0]
    viewpoints = make_eval_viewpoints(B, images.device)
    outputs, _ = model.forward_trajectory(
        image=images,
        viewpoints=[CoreViewpoint(v.centers, v.scales) for v in viewpoints],
        canvas_grid_size=canvas_grid,
        glimpse_size_px=glimpse_size_px,
    )
    return model.compute_cls(outputs[-1].canvas)


@torch.inference_mode()
def validate(cfg: Config) -> dict[str, float]:
    device = torch.device(cfg.device)
    log.info(f"Device: {device}")

    model = load_model(cfg.checkpoint, device)
    patch_size = model.backbone.patch_size_px
    glimpse_size_px = cfg.glimpse_grid * patch_size
    img_size = cfg.canvas_grid * patch_size
    log.info(f"Grid: {cfg.canvas_grid}, glimpse: {cfg.glimpse_grid}, image: {img_size}px")

    ckpt = load_ckpt(cfg.checkpoint, device)
    backbone = ckpt["backbone"]
    assert backbone in PROBE_REPOS, f"No probe for backbone {backbone}"
    probe = DINOv3LinearClassificationHead.from_pretrained(PROBE_REPOS[backbone]).to(device)
    log.info(f"Probe: {PROBE_REPOS[backbone]}")

    cls_norm = load_cls_normalizer(cfg.checkpoint, device)

    transform = val_transform(img_size)
    dataset = ImageFolder(str(cfg.val_dir), transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    log.info(f"Dataset: {len(dataset)} images, {len(loader)} batches")

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    pbar = tqdm(loader, desc="Validating", unit="batch")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        cls_pred = run_trajectory(model, images, cfg.canvas_grid, glimpse_size_px)
        cls_raw = denormalize_cls(cls_pred, cls_norm)
        logits = probe(cls_raw)

        _, top5_pred = logits.topk(5, dim=-1)
        top1_pred = top5_pred[:, 0]

        correct_top1 += (top1_pred == labels).sum().item()
        correct_top5 += (top5_pred == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += labels.shape[0]

        acc1 = 100 * correct_top1 / total
        acc5 = 100 * correct_top5 / total
        pbar.set_postfix_str(f"top1={acc1:.2f}% top5={acc5:.2f}%")

    metrics = {
        "top1_accuracy": 100 * correct_top1 / total,
        "top5_accuracy": 100 * correct_top5 / total,
        "total_samples": total,
    }
    log.info(f"Final: top1={metrics['top1_accuracy']:.2f}%, top5={metrics['top5_accuracy']:.2f}%")
    return metrics


def main() -> None:
    cfg = tyro.cli(Config)
    validate(cfg)


if __name__ == "__main__":
    main()
