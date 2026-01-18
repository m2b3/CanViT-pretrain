#!/usr/bin/env python3
"""Throughput benchmark for DINOv3 ViT-B teacher at 1024px² resolution.

Usage:
    uv run python scripts/bench_throughput.py
    uv run python scripts/bench_throughput.py --compile
    uv run python scripts/bench_throughput.py --batch-size 8
"""

import time
from dataclasses import dataclass

import torch
from tqdm import tqdm

from canvit import create_backbone
from ytch.device import sync_device


@dataclass
class Config:
    backbone: str = "dinov3_vitb14"
    resolution: int = 1024
    batch_size: int = 1
    warmup: int = 20
    iters: int = 200
    compile: bool = False


def main() -> None:
    import tyro

    torch.set_float32_matmul_precision("high")
    cfg = tyro.cli(Config)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"Device: {device}")
    print(f"Backbone: {cfg.backbone}")
    print(f"Resolution: {cfg.resolution}x{cfg.resolution}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Compile: {cfg.compile}")
    print()

    # Load teacher
    teacher = create_backbone(cfg.backbone, pretrained=True).to(device).eval()
    print(f"Loaded: {cfg.backbone}")

    # Input
    x = torch.randn(cfg.batch_size, 3, cfg.resolution, cfg.resolution, device=device)
    print(f"Input shape: {tuple(x.shape)}")

    # Optionally compile
    forward_fn = teacher.forward_norm_features
    if cfg.compile:
        print("Compiling...")
        forward_fn = torch.compile(forward_fn)

    # Warmup
    print()
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for _ in tqdm(range(cfg.warmup), desc="Warmup"):
            forward_fn(x)
        sync_device(device)

    # Timed
    sync_device(device)
    t0 = time.perf_counter()
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for _ in tqdm(range(cfg.iters), desc="Bench"):
            forward_fn(x)
        sync_device(device)
    elapsed = time.perf_counter() - t0

    total_samples = cfg.iters * cfg.batch_size
    throughput = total_samples / elapsed
    latency_ms = elapsed / cfg.iters * 1000

    print()
    print(f"Total time: {elapsed:.2f}s")
    print(f"Throughput: {throughput:.1f} samples/sec")
    print(f"Latency: {latency_ms:.2f} ms/batch")


if __name__ == "__main__":
    main()
