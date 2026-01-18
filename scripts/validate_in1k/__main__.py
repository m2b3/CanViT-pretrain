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
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from canvit import GlimpseOutput, RecurrentState
from canvit.backbone.dinov3 import DINOv3Backbone
from canvit.hub import create_backbone
from canvit.viewpoint import Viewpoint as CanvitViewpoint

from avp_vit import ActiveCanViT
from avp_vit.checkpoint import load as load_ckpt, load_model
from avp_vit.train.transforms import val_transform
from avp_vit.train.norm import PositionAwareNorm
from avp_vit.train.probe import load_probe
from avp_vit.train.viewpoint import make_eval_viewpoints

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


@dataclass
class Config:
    val_dir: Path
    checkpoint: Path
    batch_size: int = 64
    num_workers: int = 8
    device: str = "mps"
    canvas_grid: int = 32
    glimpse_grid: int = 8
    n_viewpoints: int = 5
    no_teacher: bool = False


def load_teacher(ckpt_path: Path, device: torch.device) -> DINOv3Backbone:
    """Load teacher backbone from checkpoint."""
    ckpt = load_ckpt(ckpt_path, "cpu")
    teacher = create_backbone(ckpt["backbone"], pretrained=True)
    assert isinstance(teacher, DINOv3Backbone)
    return teacher.to(device).eval()


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


def run_trajectory(
    model: ActiveCanViT,
    images: Tensor,
    canvas_grid: int,
    glimpse_size_px: int,
    n_viewpoints: int,
) -> list[Tensor]:
    """Run coarse-to-fine trajectory and return CLS prediction at each timestep."""
    B = images.shape[0]
    viewpoints = make_eval_viewpoints(B, images.device, n_viewpoints=n_viewpoints)

    def init_fn(_state: RecurrentState) -> list[Tensor]:
        return []

    def step_fn(acc: list[Tensor], out: GlimpseOutput, _vp: CanvitViewpoint) -> list[Tensor]:
        cls_pred = model.predict_scene_teacher_cls(out.state.recurrent_cls)
        acc.append(cls_pred)
        return acc

    cls_preds, _ = model.forward_reduce(
        image=images,
        viewpoints=viewpoints,  # pyright: ignore[reportArgumentType]
        canvas_grid_size=canvas_grid,
        glimpse_size_px=glimpse_size_px,
        init_fn=init_fn,
        step_fn=step_fn,
    )
    return cls_preds


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
    probe = load_probe(backbone, device)
    assert probe is not None, f"No probe for backbone {backbone}"
    log.info(f"Probe loaded for {backbone}")

    cls_norm = load_cls_normalizer(cfg.checkpoint, device)
    teacher: DINOv3Backbone | None = None
    if not cfg.no_teacher:
        teacher = load_teacher(cfg.checkpoint, device)
        log.info("Teacher loaded for baseline comparison")
    else:
        log.info("Teacher baseline disabled")

    transform = val_transform(img_size)
    dataset = ImageFolder(str(cfg.val_dir), transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,  # Shuffle for representative partial results in tqdm
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    log.info(f"Dataset: {len(dataset)} images, {len(loader)} batches")
    log.info(f"Viewpoints: {cfg.n_viewpoints}")

    # Accumulators on GPU - no sync until needed
    correct_top1 = torch.zeros(cfg.n_viewpoints, device=device, dtype=torch.long)
    correct_top5 = torch.zeros(cfg.n_viewpoints, device=device, dtype=torch.long)
    teacher_correct_top1 = torch.zeros(1, device=device, dtype=torch.long)
    teacher_correct_top5 = torch.zeros(1, device=device, dtype=torch.long)
    total = 0

    pbar_sync_interval = 20  # Sync for pbar every N batches
    pbar = tqdm(loader, desc="Validating", unit="batch")
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        # Model predictions at each timestep
        cls_preds = run_trajectory(model, images, cfg.canvas_grid, glimpse_size_px, cfg.n_viewpoints)
        for t, cls_pred in enumerate(cls_preds):
            cls_raw = cls_norm.denormalize(cls_pred)
            logits = probe(cls_raw)
            _, top5_pred = logits.topk(5, dim=-1)
            top1_pred = top5_pred[:, 0]
            correct_top1[t] += (top1_pred == labels).sum()
            correct_top5[t] += (top5_pred == labels.unsqueeze(1)).any(dim=1).sum()

        # Teacher baseline (raw CLS features)
        if teacher is not None:
            teacher_cls = teacher.forward_norm_features(images).cls
            teacher_logits = probe(teacher_cls)
            _, teacher_top5 = teacher_logits.topk(5, dim=-1)
            teacher_top1 = teacher_top5[:, 0]
            teacher_correct_top1 += (teacher_top1 == labels).sum()
            teacher_correct_top5 += (teacher_top5 == labels.unsqueeze(1)).any(dim=1).sum()

        total += labels.shape[0]

        # Sync only every N batches for pbar update
        if batch_idx % pbar_sync_interval == 0:
            c1 = correct_top1.tolist()
            ts = " ".join(f"t{t}={100*c1[t]/total:.1f}" for t in range(cfg.n_viewpoints))
            if teacher is not None:
                teacher_acc1 = 100 * teacher_correct_top1.item() / total
                pbar.set_postfix_str(f"{ts} teacher={teacher_acc1:.1f}")
            else:
                pbar.set_postfix_str(ts)

    # Final sync
    correct_top1_list = correct_top1.tolist()
    correct_top5_list = correct_top5.tolist()

    log.info(f"Results by timestep ({cfg.n_viewpoints} viewpoints):")
    metrics: dict[str, float] = {"total_samples": float(total)}
    for t in range(cfg.n_viewpoints):
        acc1 = 100 * correct_top1_list[t] / total
        acc5 = 100 * correct_top5_list[t] / total
        log.info(f"  t{t}: top1={acc1:.2f}%, top5={acc5:.2f}%")
        metrics[f"t{t}_top1"] = acc1
        metrics[f"t{t}_top5"] = acc5

    if teacher is not None:
        teacher_acc1 = 100 * teacher_correct_top1.item() / total
        teacher_acc5 = 100 * teacher_correct_top5.item() / total
        log.info(f"Teacher baseline: top1={teacher_acc1:.2f}%, top5={teacher_acc5:.2f}%")
        metrics["teacher_top1"] = teacher_acc1
        metrics["teacher_top5"] = teacher_acc5

    return metrics


def main() -> None:
    cfg = tyro.cli(Config)
    validate(cfg)


if __name__ == "__main__":
    main()
