#!/usr/bin/env python3
"""Test SLURM signal handling with real DataLoaders.

This script simulates actual training conditions:
- Uses real ImageNet data with DataLoader workers
- Workers have signal handlers that ignore SIGUSR1/SIGTERM
- Main process handles signal to trigger checkpoint
- No GPU required (CPU-only tensor operations)

Run locally:
    python slurm/test_signal_dataloader.py --data-dir /path/to/imagenet/train --index-dir /path/to/index

Run on SLURM:
    sbatch slurm/test_signal_dataloader.sbatch
"""

import argparse
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from drac_imagenet import IndexedImageFolder

# Global flag for checkpoint request
checkpoint_requested = False


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def handle_signal(signum: int, frame: object) -> None:
    """Signal handler for main process."""
    global checkpoint_requested
    checkpoint_requested = True
    sig_name = "SIGUSR1" if signum == signal.SIGUSR1 else f"signal {signum}"
    log(f">>> {sig_name} received! Will checkpoint after current step. <<<")


def worker_init_fn(worker_id: int) -> None:
    """Ignore signals in DataLoader workers - main process handles shutdown."""
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


def fake_checkpoint(step: int, ckpt_path: Path) -> None:
    """Simulate checkpoint save."""
    log(f"Saving checkpoint at step {step}...")
    state = {
        "step": step,
        "timestamp": datetime.now().isoformat(),
        "dummy_tensor": torch.randn(100, 100),
    }
    tmp = ckpt_path.with_suffix(".pt.tmp")
    torch.save(state, tmp)
    tmp.rename(ckpt_path)
    log(f"Checkpoint saved: {ckpt_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True, help="ImageNet train dir")
    parser.add_argument("--index-dir", type=Path, required=True, help="Index dir for IndexedImageFolder")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--target-steps", type=int, default=5000, help="Steps to run if not interrupted")
    parser.add_argument("--ckpt-path", type=Path, default=Path("signal_test_ckpt.pt"))
    args = parser.parse_args()

    log("=" * 60)
    log("Signal handling test with DataLoaders")
    log("=" * 60)
    log(f"data_dir: {args.data_dir}")
    log(f"index_dir: {args.index_dir}")
    log(f"num_workers: {args.num_workers}")
    log(f"batch_size: {args.batch_size}")
    log(f"target_steps: {args.target_steps}")
    log(f"ckpt_path: {args.ckpt_path}")
    log("")

    # Register signal handlers
    signal.signal(signal.SIGUSR1, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    log("Signal handlers registered (SIGUSR1, SIGTERM)")

    # Create dataset and loader
    log("Creating DataLoader...")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    dataset = IndexedImageFolder(args.data_dir, args.index_dir, transform)
    log(f"Dataset: {len(dataset):,} images")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,  # CPU only
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=worker_init_fn,
    )
    log(f"DataLoader created with {args.num_workers} workers")
    log("")

    # Training loop
    log("Starting training loop...")
    step = 0
    data_iter = iter(loader)

    while step < args.target_steps:
        # Get batch
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            images, labels = next(data_iter)

        # Fake "training" - just some CPU tensor ops
        _ = images.mean()
        _ = (images * 2).sum()

        step += 1

        if step % 100 == 0:
            log(f"Step {step}/{args.target_steps}")

        # Check for checkpoint request
        if checkpoint_requested:
            log("Checkpoint requested!")
            fake_checkpoint(step, args.ckpt_path)
            log("Exiting cleanly after checkpoint.")
            return

    log(f"Reached target {args.target_steps} steps without signal.")
    fake_checkpoint(step, args.ckpt_path)
    log("Training complete!")


if __name__ == "__main__":
    main()
