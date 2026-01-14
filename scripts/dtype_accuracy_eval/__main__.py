#!/usr/bin/env python3
"""Measure IN1k val accuracy at different CLS feature storage dtypes."""

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from canvit.backbone.dinov3 import DINOv3Backbone
from canvit.hub import create_backbone

from avp_vit.train.transforms import val_transform
from avp_vit.train.probe import load_probe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Native dtypes
NATIVE_DTYPES: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp8": torch.float8_e4m3fn,
}

# Low-bit quantization (uniform per-tensor)
QUANT_BITS = [4, 2, 1]

# All formats to test (order for display)
ALL_FORMATS = ["fp32", "fp16", "bf16", "fp8", "int4", "int2", "int1"]


def quantize_uniform(x: Tensor, bits: int) -> Tensor:
    """Simulate uniform per-tensor quantization to N bits, return dequantized."""
    levels = 2**bits
    x_min, x_max = x.min(), x.max()
    if x_max == x_min:
        return x  # Constant tensor, no quantization needed
    scale = (x_max - x_min) / (levels - 1)
    x_q = torch.round((x - x_min) / scale).clamp(0, levels - 1)
    return x_q * scale + x_min


@dataclass
class Config:
    val_dir: Path
    teacher_ckpt: Path
    teacher_model: str = "dinov3_vitb16"
    image_size: int = 512
    batch_size: int = 64
    num_workers: int = 8
    device: str = "cuda"


@torch.inference_mode()
def main(cfg: Config) -> None:
    device = torch.device(cfg.device)

    # Validate inputs early
    assert cfg.val_dir.is_dir(), f"val_dir not found: {cfg.val_dir}"
    assert cfg.teacher_ckpt.exists(), f"teacher_ckpt not found: {cfg.teacher_ckpt}"

    log.info("=" * 60)
    log.info("Configuration")
    log.info("=" * 60)
    log.info(f"  val_dir: {cfg.val_dir}")
    log.info(f"  teacher_ckpt: {cfg.teacher_ckpt}")
    log.info(f"  teacher_model: {cfg.teacher_model}")
    log.info(f"  image_size: {cfg.image_size}x{cfg.image_size}")
    log.info(f"  batch_size: {cfg.batch_size}")
    log.info(f"  num_workers: {cfg.num_workers}")
    log.info(f"  device: {device}")
    log.info("=" * 60)

    # Load teacher
    log.info("Loading teacher...")
    teacher: DINOv3Backbone = create_backbone(
        cfg.teacher_model, weights=str(cfg.teacher_ckpt)
    )
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    log.info(f"  embed_dim: {teacher.embed_dim}")
    log.info(f"  patch_size: {teacher.patch_size_px}px")

    # Load probe
    log.info("Loading probe...")
    probe = load_probe(cfg.teacher_model, device)
    assert probe is not None, f"No probe available for {cfg.teacher_model}"
    log.info("  Probe loaded")

    # Dataset
    log.info("Loading dataset...")
    transform = val_transform(cfg.image_size)
    dataset = ImageFolder(str(cfg.val_dir), transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    log.info(f"  {len(dataset)} images, {len(loader)} batches")

    # Accumulators on GPU (no sync until needed)
    correct: dict[str, Tensor] = {
        name: torch.zeros(1, device=device, dtype=torch.long) for name in ALL_FORMATS
    }
    total = 0

    log.info("Running evaluation...")
    pbar = tqdm(loader, desc="Eval", unit="batch")
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Single teacher inference with AMP
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cls_fp32 = teacher.forward_norm_features(images).cls

        # Ensure fp32 baseline
        cls_fp32 = cls_fp32.float()

        # Test native dtypes
        for name, dtype in NATIVE_DTYPES.items():
            cls_test = cls_fp32 if dtype == torch.float32 else cls_fp32.to(dtype).float()
            logits = probe(cls_test)
            correct[name] += (logits.argmax(dim=-1) == labels).sum()

        # Test low-bit quantization (uniform per-tensor)
        for bits in QUANT_BITS:
            cls_test = quantize_uniform(cls_fp32, bits)
            logits = probe(cls_test)
            correct[f"int{bits}"] += (logits.argmax(dim=-1) == labels).sum()

        total += labels.shape[0]

        # Update pbar periodically (avoid sync every batch)
        if batch_idx % 10 == 0:
            accs = {n: 100.0 * correct[n].item() / total for n in ALL_FORMATS}
            pbar.set_postfix_str(" ".join(f"{n}={accs[n]:.1f}" for n in ALL_FORMATS))

    # Final results
    log.info("")
    log.info("=" * 60)
    log.info(f"Results ({total} images)")
    log.info("=" * 60)

    accs = {n: 100.0 * correct[n].item() / total for n in ALL_FORMATS}
    fp32_acc = accs["fp32"]

    for name in ALL_FORMATS:
        acc = accs[name]
        delta = acc - fp32_acc
        log.info(f"  {name:5s}: {acc:.3f}%  (Δ = {delta:+.3f}%)")

    log.info("=" * 60)


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Config))
