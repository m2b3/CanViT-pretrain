#!/usr/bin/env python3
"""ImageNet-1k validation with coarse-to-fine viewpoint policy.

Usage:
    source slurm/env.sh
    uv run -m scripts.validate_in1k  # uses defaults from env.sh

    # Or explicit:
    uv run -m scripts.validate_in1k --checkpoint /path/to/ckpt.pt --val-dir /path/to/val
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
import tyro
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from canvit import CanViTOutput, RecurrentState
from canvit.backbone.dinov3 import DINOv3Backbone
from canvit.viewpoint import Viewpoint as CanvitViewpoint
from canvit_utils.teacher import load_teacher as _load_teacher

from avp_vit import CanViTForPretraining
from avp_vit.checkpoint import load as load_ckpt, load_model
from avp_vit.train.transforms import val_transform
from canvit import CLSStandardizer
from avp_vit.train.probe import load_probe
from avp_vit.train.viewpoint import make_eval_viewpoints
from ytch.device import get_sensible_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _default_val_dir() -> Path | None:
    if v := os.environ.get("IN1K_VAL_DIR"):
        return Path(v)
    return None


def _default_checkpoint() -> Path:
    return Path("~/projects/def-skrishna/checkpoints/CanViT-flagship.pt").expanduser()


@dataclass
class Config:
    checkpoint: Path = field(default_factory=_default_checkpoint)
    val_dir: Path | None = field(default_factory=_default_val_dir)
    batch_size: int = 64
    num_workers: int = 8
    device: torch.device = field(default_factory=get_sensible_device)
    canvas_grid: int = 32
    glimpse_grid: int = 8
    n_viewpoints: int = 10
    no_teacher: bool = False


def load_teacher(ckpt_path: Path, device: torch.device) -> DINOv3Backbone:
    """Load teacher backbone from checkpoint."""
    ckpt = load_ckpt(ckpt_path, "cpu")
    return _load_teacher(ckpt["backbone"], device)


def load_cls_normalizer(ckpt_path: Path, device: torch.device) -> CLSStandardizer:
    """Load CLS normalizer from checkpoint."""
    ckpt = load_ckpt(ckpt_path, device)
    cls_state = ckpt.get("cls_norm_state")
    assert cls_state is not None, "Checkpoint missing cls_norm_state"
    embed_dim = cls_state["mean"].shape[-1]
    cls_norm = CLSStandardizer(embed_dim)
    cls_norm.load_state_dict(cls_state)
    cls_norm.eval()
    return cls_norm.to(device)


def run_trajectory(
    model: CanViTForPretraining,
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

    def step_fn(acc: list[Tensor], out: CanViTOutput, _vp: CanvitViewpoint, _glimpse: Tensor) -> list[Tensor]:
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
    log.info("=" * 70)
    log.info("ImageNet-1k Validation")
    log.info("=" * 70)
    log.info("")
    log.info("Configuration:")
    log.info(f"  checkpoint:    {cfg.checkpoint}")
    log.info(f"  val_dir:       {cfg.val_dir}")
    log.info(f"  batch_size:    {cfg.batch_size}")
    log.info(f"  num_workers:   {cfg.num_workers}")
    log.info(f"  canvas_grid:   {cfg.canvas_grid}")
    log.info(f"  glimpse_grid:  {cfg.glimpse_grid}")
    log.info(f"  n_viewpoints:  {cfg.n_viewpoints}")
    log.info(f"  no_teacher:    {cfg.no_teacher}")
    log.info("")

    if cfg.val_dir is None:
        raise ValueError("--val-dir required (or set IN1K_VAL_DIR env var)")

    device = cfg.device
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"  GPU: {torch.cuda.get_device_name()}")
        log.info(f"  Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    log.info("")

    log.info("Loading checkpoint...")
    assert cfg.checkpoint.exists(), f"Checkpoint not found: {cfg.checkpoint}"
    ckpt = load_ckpt(cfg.checkpoint, device)
    backbone_name = ckpt["backbone"]
    step = ckpt.get("step", "unknown")
    log.info(f"  backbone: {backbone_name}")
    log.info(f"  step: {step}")
    log.info("")

    log.info("Loading model...")
    model = load_model(cfg.checkpoint, device)
    patch_size = model.backbone.patch_size_px
    glimpse_size_px = cfg.glimpse_grid * patch_size
    img_size = cfg.canvas_grid * patch_size
    log.info(f"  patch_size: {patch_size}px")
    log.info(f"  glimpse: {cfg.glimpse_grid}x{cfg.glimpse_grid} = {glimpse_size_px}px")
    log.info(f"  canvas: {cfg.canvas_grid}x{cfg.canvas_grid} = {img_size}px")
    log.info("")

    log.info("Loading probe...")
    probe = load_probe(backbone_name, device)
    assert probe is not None, f"No probe for backbone {backbone_name}"
    log.info(f"  Probe loaded for {backbone_name}")
    log.info("")

    log.info("Loading CLS normalizer...")
    cls_norm = load_cls_normalizer(cfg.checkpoint, device)
    log.info("  CLS normalizer loaded")
    log.info("")

    teacher: DINOv3Backbone | None = None
    if not cfg.no_teacher:
        log.info("Loading teacher for baseline comparison...")
        teacher = load_teacher(cfg.checkpoint, device)
        log.info(f"  Teacher: {backbone_name}")
        log.info("")
    else:
        log.info("Teacher baseline disabled")
        log.info("")

    log.info("Setting up dataset...")
    transform = val_transform(img_size)
    dataset = ImageFolder(str(cfg.val_dir), transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,  # Shuffle for representative partial results in tqdm
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    log.info(f"  val_dir: {cfg.val_dir}")
    log.info(f"  images: {len(dataset)}")
    log.info(f"  batches: {len(loader)}")
    log.info(f"  batch_size: {cfg.batch_size}")
    log.info(f"  num_workers: {cfg.num_workers}")
    log.info("")

    log.info(f"Running validation with {cfg.n_viewpoints} viewpoints...")

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
            cls_raw = cls_norm.destandardize(cls_pred)
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

    log.info("")
    log.info("=" * 70)
    log.info("RESULTS")
    log.info("=" * 70)
    log.info(f"Total samples: {total}")
    log.info("")
    log.info(f"Accuracy by timestep ({cfg.n_viewpoints} viewpoints):")
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
