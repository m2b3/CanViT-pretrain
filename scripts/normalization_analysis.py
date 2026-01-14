#!/usr/bin/env python3
"""Analyze if normalization amplifies or dampens prediction errors.

Usage:
    uv run python scripts/normalization_analysis.py \
        --val-dir $IN1K_VAL_DIR \
        --teacher-ckpt $DINOV3_VITB16_CKPT \
        --shard-path $FEATURES_DIR/in21k/dinov3_vitb16/512/shards/00000.pt
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from canvit.hub import create_backbone
from avp_vit.train.transforms import val_transform
from avp_vit.train.probe import load_probe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

NOISE_LEVELS = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]


@dataclass
class Config:
    val_dir: Path
    teacher_ckpt: Path
    shard_path: Path
    teacher_model: str = "dinov3_vitb16"
    image_size: int = 512
    batch_size: int = 64
    num_workers: int = 8
    device: str = "cuda"


def compute_norm_stats(shard_path: Path, device: torch.device) -> tuple[Tensor, Tensor]:
    """Load shard and compute CLS mean/std."""
    log.info(f"Loading shard: {shard_path}")
    shard = torch.load(shard_path, map_location="cpu", weights_only=False)
    cls = shard["cls"].float()  # [N, D]
    mean = cls.mean(dim=0).to(device)  # [D]
    std = cls.std(dim=0).to(device)  # [D]
    log.info(f"  CLS shape: {cls.shape}, mean range: [{mean.min():.3f}, {mean.max():.3f}], std range: [{std.min():.3f}, {std.max():.3f}]")
    return mean, std


@torch.inference_mode()
def main(cfg: Config) -> None:
    device = torch.device(cfg.device)

    assert cfg.val_dir.is_dir(), f"val_dir not found: {cfg.val_dir}"
    assert cfg.teacher_ckpt.exists(), f"teacher_ckpt not found: {cfg.teacher_ckpt}"
    assert cfg.shard_path.exists(), f"shard not found: {cfg.shard_path}"

    # Load norm stats
    mean, std = compute_norm_stats(cfg.shard_path, device)
    eps = 1e-6

    # Load teacher
    log.info("Loading teacher...")
    teacher = create_backbone(cfg.teacher_model, weights=str(cfg.teacher_ckpt))
    teacher = teacher.to(device).eval()

    # Load probe
    log.info("Loading probe...")
    probe = load_probe(cfg.teacher_model, device)
    assert probe is not None

    # Dataset
    transform = val_transform(cfg.image_size)
    dataset = ImageFolder(str(cfg.val_dir), transform=transform)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True)
    log.info(f"Dataset: {len(dataset)} images")

    # Accumulators: [noise_level] -> correct count
    correct_raw = {σ: 0 for σ in NOISE_LEVELS}
    correct_norm = {σ: 0 for σ in NOISE_LEVELS}
    total = 0

    # For computing raw feature scale (to calibrate noise)
    raw_std_accum = torch.zeros(1, device=device)
    raw_std_count = 0
    raw_scale = 1.0  # Will be updated

    pbar = tqdm(loader, desc="Eval", unit="batch")
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        B = labels.shape[0]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            cls_raw = teacher.forward_norm_features(images).cls.float()  # [B, D]

        # Track raw feature scale
        raw_std_accum += cls_raw.std()
        raw_std_count += 1
        raw_scale = raw_std_accum.item() / raw_std_count

        # Normalize
        cls_norm = (cls_raw - mean) / (std + eps)

        for σ in NOISE_LEVELS:
            if σ == 0:
                noisy_raw = cls_raw
                noisy_norm = cls_norm
            else:
                # Noise scaled to raw feature magnitude
                noise_raw = torch.randn_like(cls_raw) * σ * raw_scale
                noisy_raw = cls_raw + noise_raw

                # Same noise magnitude in normalized space
                noise_norm = torch.randn_like(cls_norm) * σ
                noisy_norm = cls_norm + noise_norm

            # Raw path
            logits_raw = probe(noisy_raw)
            correct_raw[σ] += (logits_raw.argmax(-1) == labels).sum().item()

            # Norm path: denormalize then probe
            denormed = noisy_norm * (std + eps) + mean
            logits_norm = probe(denormed)
            correct_norm[σ] += (logits_norm.argmax(-1) == labels).sum().item()

        total += B

        if total % 5000 < cfg.batch_size:
            acc_raw_0 = 100 * correct_raw[0.0] / total
            acc_norm_0 = 100 * correct_norm[0.0] / total
            pbar.set_postfix_str(f"raw={acc_raw_0:.1f}% norm={acc_norm_0:.1f}%")

    # Results
    log.info("")
    log.info("=" * 70)
    log.info(f"Results ({total} images)")
    log.info("=" * 70)
    log.info(f"Raw feature std (avg): {raw_scale:.4f}")
    log.info("")
    log.info(f"{'Noise σ':>10} | {'Raw Acc':>10} | {'Norm→Denorm Acc':>15} | {'Δ':>10}")
    log.info("-" * 70)

    for σ in NOISE_LEVELS:
        acc_raw = 100 * correct_raw[σ] / total
        acc_norm = 100 * correct_norm[σ] / total
        delta = acc_norm - acc_raw
        log.info(f"{σ:>10.2f} | {acc_raw:>10.3f}% | {acc_norm:>15.3f}% | {delta:>+10.3f}%")

    log.info("=" * 70)
    log.info("Δ > 0 means normalized space is MORE robust to errors")
    log.info("Δ < 0 means normalized space AMPLIFIES errors")


if __name__ == "__main__":
    import tyro
    main(tyro.cli(Config))
