"""Benchmark: Teacher vs CanViT forward+decode overhead.

Measures 4 things:
1. Teacher: DINOv3 backbone forward_norm_features() on 128px input
2. CanViT forward only: model.forward() without decode
3. CanViT forward + decode: forward() + predict_teacher_scene()
4. Decode only: predict_teacher_scene() isolated

Goal: verify measured overhead matches theoretical FLOP overhead.

Theoretical (from scripts/flops.py for 8×8 glimpse, 32×32 canvas):
- Teacher (ViT @ glimpse): 11.98G FLOPs
- CanViT w/o heads:        15.10G FLOPs → ×1.26 expected (26%)
- CanViT w/ heads:         16.72G FLOPs → ×1.40 expected (40%)

GPU sync verification:
- Checked canvit source: NO syncs in forward path
- Checked dinov3 source: NO syncs in forward_features path
- Use --sync-debug to enable torch sync debug mode (errors on implicit syncs)
- This script syncs ONCE before timing, ONCE after each benchmark

Usage:
    uv run python scripts/bench_overhead.py --batch 2 --iters 2  # smoke test
    uv run python scripts/bench_overhead.py --device cuda --batch 64 --iters 50
    uv run python scripts/bench_overhead.py --device cuda --batch 64 --iters 50 --compile
    uv run python scripts/bench_overhead.py --device cuda --sync-debug  # verify no hidden syncs
"""

import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import tyro
from ytch.device import get_sensible_device, sync_device

from avp_vit import ACVFRP, ACVFRPConfig
from canvit import create_backbone
from canvit.viewpoint import Viewpoint

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    """Benchmark configuration."""

    # === Model config (from train/config.py - DO NOT CHANGE without checking source) ===
    glimpse_grid: int = 8
    """Glimpse grid size. Source: train/config.py:27"""
    canvas_grid: int = 32
    """Canvas grid size. Source: train/config.py:29"""
    patch_size: int = 16
    """DINOv3 ViT-B/16 patch size (constant)."""

    # === Benchmark params ===
    batch: int = 2
    """Batch size. Start small (2) for smoke test."""
    iters: int = 2
    """Timed iterations. Start small (2) for smoke test."""
    warmup: int = 5
    """Warmup iterations (includes compile graph capture if enabled)."""
    device: str = "auto"
    """Device: 'cpu', 'mps', 'cuda', or 'auto'."""

    # === Precision ===
    amp: bool = True
    """Enable AMP (bfloat16 autocast). Default ON to match training."""
    tf32: bool = True
    """Enable TF32 matmul precision (CUDA only). Default ON."""

    # === Optimization ===
    compile: bool = False
    """Enable torch.compile (adds significant warmup time)."""

    # === Debug ===
    sync_debug: bool = False
    """Enable CUDA sync debug mode - errors on implicit syncs."""
    cpu_threads: int = 8
    """CPU threads (only if device=cpu)."""


def bench(fn, iters: int, device: torch.device) -> float:
    """Benchmark a function. Returns total time in seconds.

    Syncs ONCE before, ONCE after. NO syncs inside loop.
    """
    sync_device(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync_device(device)
    return time.perf_counter() - t0


def main(cfg: Config) -> None:
    # === Device setup ===
    device = get_sensible_device() if cfg.device == "auto" else torch.device(cfg.device)
    if device.type == "cpu":
        torch.set_num_threads(cfg.cpu_threads)

    # === Precision setup ===
    if cfg.tf32 and device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    # === Sync debug mode ===
    if cfg.sync_debug and device.type == "cuda":
        # Mode 2 = error on implicit sync
        torch.cuda.set_sync_debug_mode(2)
        log.info("CUDA sync debug mode ENABLED - will error on implicit syncs")

    glimpse_px = cfg.glimpse_grid * cfg.patch_size

    # === FULL CONFIG LOGGING ===
    log.info("=" * 70)
    log.info("BENCHMARK: Teacher vs CanViT overhead")
    log.info("=" * 70)
    log.info("")
    log.info("MODEL CONFIG (from train/config.py):")
    log.info(f"  glimpse_grid:      {cfg.glimpse_grid}  (train/config.py:27)")
    log.info(f"  canvas_grid:       {cfg.canvas_grid}  (train/config.py:29)")
    log.info(f"  patch_size:        {cfg.patch_size}  (DINOv3 ViT-B/16 constant)")
    log.info(f"  → glimpse_px:      {glimpse_px}")
    log.info("")
    log.info("BENCHMARK PARAMS:")
    log.info(f"  batch:             {cfg.batch}")
    log.info(f"  iters:             {cfg.iters}")
    log.info(f"  warmup:            {cfg.warmup}")
    log.info(f"  device:            {device}")
    log.info("")
    log.info("PRECISION:")
    log.info(f"  amp:               {cfg.amp}  {'(bfloat16 autocast)' if cfg.amp else '(fp32)'}")
    log.info(f"  tf32:              {cfg.tf32}  {'(matmul precision=high)' if cfg.tf32 else ''}")
    if device.type == "cuda":
        log.info(f"  actual matmul:     {torch.get_float32_matmul_precision()}")
    log.info("")
    log.info("OPTIMIZATION:")
    log.info(f"  compile:           {cfg.compile}")
    log.info("")
    log.info("DEBUG:")
    log.info(f"  sync_debug:        {cfg.sync_debug}")
    if device.type == "cpu":
        log.info(f"  cpu_threads:       {cfg.cpu_threads}")
    log.info("")

    # === Create models (suppress dinov3 spam) ===
    logging.getLogger("dinov3").setLevel(logging.WARNING)

    log.info("Creating models...")
    backbone = create_backbone("dinov3_vitb16", pretrained=False)
    teacher = create_backbone("dinov3_vitb16", pretrained=False)

    model_cfg = ACVFRPConfig(teacher_dim=backbone.embed_dim)
    model = ACVFRP(backbone=backbone, cfg=model_cfg, policy=None)
    model.to(device).eval()
    teacher.to(device).eval()
    log.info("Models created.")
    log.info("")

    # === Log token counts ===
    n_local = 1 + 1 + 1 + backbone.n_register_tokens + cfg.glimpse_grid**2
    n_canvas = model_cfg.n_canvas_registers + cfg.canvas_grid**2
    log.info("TOKEN COUNTS:")
    log.info(f"  Local stream:      {n_local} tokens × {backbone.embed_dim}d")
    log.info(f"                     = 1 VPE + 1 recurrent_cls + 1 ephemeral_cls + {backbone.n_register_tokens} regs + {cfg.glimpse_grid**2} patches")
    log.info(f"  Canvas:            {n_canvas} tokens × {model_cfg.canvas_dim}d")
    log.info(f"                     = {model_cfg.n_canvas_registers} canvas_regs + {cfg.canvas_grid**2} spatial")
    log.info("")

    # === Theoretical FLOPs ===
    log.info("THEORETICAL (scripts/flops.py, 8×8 glimpse, 32×32 canvas):")
    log.info("  Teacher (ViT @ glimpse): 11.98G FLOPs")
    log.info("  CanViT w/o heads:        15.10G FLOPs → ×1.26 expected (+26%)")
    log.info("  CanViT w/ heads:         16.72G FLOPs → ×1.40 expected (+40%)")
    log.info("")

    # === Compile if requested ===
    if cfg.compile:
        log.info("Compiling models (this may take a while)...")
        teacher.compile()
        model.compile()
        log.info("Compilation done.")
        log.info("")

    # === Create inputs ===
    glimpse = torch.randn(cfg.batch, 3, glimpse_px, glimpse_px, device=device)
    vp = Viewpoint(
        centers=torch.zeros(cfg.batch, 2, device=device),
        scales=torch.ones(cfg.batch, device=device),
    )
    state = model.init_state(batch_size=cfg.batch, canvas_grid_size=cfg.canvas_grid)

    # Pre-allocate for decode benchmark
    canvas_for_decode = state.canvas

    # === AMP context ===
    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if cfg.amp and device.type in ("cuda", "cpu")
        else nullcontext()
    )
    amp_str = "bfloat16" if cfg.amp else "fp32"

    # === Define benchmark functions (NO syncs inside!) ===
    def teacher_fn():
        with amp_ctx:
            return teacher.forward_norm_features(glimpse)

    def canvit_fwd_fn():
        nonlocal canvas_for_decode
        with amp_ctx:
            out = model.forward(glimpse=glimpse, state=state, viewpoint=vp)
            canvas_for_decode = out.state.canvas
            return out

    def canvit_fwd_decode_fn():
        nonlocal canvas_for_decode
        with amp_ctx:
            out = model.forward(glimpse=glimpse, state=state, viewpoint=vp)
            canvas_for_decode = out.state.canvas
            return model.predict_teacher_scene(canvas_for_decode)

    def decode_only_fn():
        with amp_ctx:
            return model.predict_teacher_scene(canvas_for_decode)

    # === Warmup ===
    log.info(f"Warmup ({cfg.warmup} iters, {amp_str})...")
    with torch.no_grad():
        for _ in range(cfg.warmup):
            teacher_fn()
            canvit_fwd_decode_fn()
    sync_device(device)
    log.info("Done.")
    log.info("")

    # === Benchmark ===
    log.info(f"Benchmarking ({cfg.iters} iters each, {amp_str})...")
    log.info("  [sync once before, once after each benchmark - NO syncs in loop]")
    with torch.no_grad():
        t_teacher = bench(teacher_fn, cfg.iters, device)
        t_canvit_fwd = bench(canvit_fwd_fn, cfg.iters, device)
        t_canvit_fwd_dec = bench(canvit_fwd_decode_fn, cfg.iters, device)
        t_decode = bench(decode_only_fn, cfg.iters, device)

    # === Results ===
    def fmt(t: float) -> str:
        ms = t / cfg.iters * 1000
        gps = (cfg.iters * cfg.batch) / t
        return f"{ms:7.1f} ms/batch  ({gps:6.1f} glimpses/s)"

    ovh_fwd = t_canvit_fwd / t_teacher
    ovh_fwd_dec = t_canvit_fwd_dec / t_teacher

    log.info("")
    log.info("=" * 70)
    log.info("RESULTS")
    log.info("=" * 70)
    log.info(f"  Teacher:               {fmt(t_teacher)}")
    log.info(f"  CanViT forward only:   {fmt(t_canvit_fwd)}  ×{ovh_fwd:.2f} ({(ovh_fwd-1)*100:+.0f}%)")
    log.info(f"  CanViT forward+decode: {fmt(t_canvit_fwd_dec)}  ×{ovh_fwd_dec:.2f} ({(ovh_fwd_dec-1)*100:+.0f}%)")
    log.info(f"  Decode only:           {fmt(t_decode)}")
    log.info("")
    log.info("EXPECTED (theoretical):")
    log.info("  CanViT forward only:   ×1.26 (+26%)")
    log.info("  CanViT forward+decode: ×1.40 (+40%)")
    log.info("")

    # === Interpretation ===
    log.info("INTERPRETATION:")
    if ovh_fwd_dec < 1.1:
        log.info("  ⚠️  Overhead too LOW - likely not compute-bound. Try larger batch.")
    elif ovh_fwd_dec > 2.0:
        log.info("  ⚠️  Overhead too HIGH - possible causes:")
        log.info("      - Memory-bound (especially on CPU/MPS)")
        log.info("      - Batch too small to saturate compute")
        log.info("      - Non-FLOP overhead (RoPE, tensor creation, etc.)")
    elif 1.2 <= ovh_fwd_dec <= 1.6:
        log.info("  ✓  Overhead in expected range!")
    else:
        log.info("  ?  Overhead outside expected range but not extreme.")

    # Disable sync debug if enabled
    if cfg.sync_debug and device.type == "cuda":
        torch.cuda.set_sync_debug_mode(0)


if __name__ == "__main__":
    main(tyro.cli(Config))
