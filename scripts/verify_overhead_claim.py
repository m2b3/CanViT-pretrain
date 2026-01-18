#!/usr/bin/env python3
"""Verify the "21× capacity, <50% overhead" claim with full rigor.

This script:
1. Extracts ALL relevant config values from actual model objects (no hardcoding)
2. Computes theoretical FLOPs from first principles using canvit.flops
3. Runs timing benchmarks on GPU with proper synchronization
4. Compares theoretical vs empirical overhead

Run on Nibi:
    source slurm/env.sh
    salloc --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=0:30:00
    uv run python scripts/verify_overhead_claim.py

Or quick smoke test locally:
    uv run python scripts/verify_overhead_claim.py --batch 2 --iters 2 --device cpu
"""

import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import tyro
from ytch.device import get_sensible_device, sync_device

from avp_vit import ACVFRP, ACVFRPConfig
from avp_vit.train.config import Config as TrainConfig
from canvit import flops
from canvit import create_backbone
from canvit.viewpoint import Viewpoint

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchConfig:
    """Benchmark configuration."""
    batch: int = 64
    iters: int = 50
    warmup: int = 10
    device: str = "auto"
    amp: bool = True
    tf32: bool = True
    compile: bool = False


def fmt_flops(f: int) -> str:
    if f >= 1e12:
        return f"{f / 1e12:.2f}T"
    if f >= 1e9:
        return f"{f / 1e9:.2f}G"
    if f >= 1e6:
        return f"{f / 1e6:.1f}M"
    return str(f)


def main(cfg: BenchConfig) -> None:
    # === Device setup ===
    device = get_sensible_device() if cfg.device == "auto" else torch.device(cfg.device)
    if cfg.tf32 and device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    # Suppress model creation logs
    logging.getLogger("dinov3").setLevel(logging.WARNING)
    logging.getLogger("canvit").setLevel(logging.WARNING)

    # === Load training config defaults ===
    train_cfg = TrainConfig()
    glimpse_grid = train_cfg.glimpse_grid_size
    canvas_grid = train_cfg.grid_size

    log.info("=" * 80)
    log.info("VERIFICATION: '21× capacity, <50% overhead' claim")
    log.info("=" * 80)
    log.info("")

    # === Create models and extract ACTUAL config values ===
    backbone = create_backbone("dinov3_vitb16", pretrained=False)
    model_cfg = ACVFRPConfig(teacher_dim=backbone.embed_dim)
    model = ACVFRP(backbone=backbone, cfg=model_cfg, policy=None)
    teacher = create_backbone("dinov3_vitb16", pretrained=False)

    model.to(device).eval()
    teacher.to(device).eval()

    # === Extract all config values from ACTUAL model objects ===
    log.info("1. CONFIGURATION (extracted from model objects)")
    log.info("-" * 60)

    # Backbone params
    patch_size = backbone.patch_size_px
    local_dim = backbone.embed_dim
    n_blocks = backbone.n_blocks
    n_backbone_regs = backbone.n_register_tokens
    ffn_ratio = backbone.ffn_ratio

    log.info("  DINOv3 ViT-B/16:")
    log.info(f"    patch_size:        {patch_size}px")
    log.info(f"    embed_dim:         {local_dim}")
    log.info(f"    n_blocks:          {n_blocks}")
    log.info(f"    n_register_tokens: {n_backbone_regs}")
    log.info(f"    ffn_ratio:         {ffn_ratio}")

    # CanViT params
    canvas_dim = model.canvas_dim
    n_canvas_regs = model.cfg.n_canvas_registers
    enable_vpe = model.cfg.enable_vpe
    rw_stride = model.cfg.rw_stride
    n_adapters = len(model.read_after_blocks)

    log.info("  CanViT:")
    log.info(f"    canvas_dim:        {canvas_dim}")
    log.info(f"    n_canvas_registers:{n_canvas_regs}")
    log.info(f"    enable_vpe:        {enable_vpe}")
    log.info(f"    rw_stride:         {rw_stride}")
    log.info(f"    n_adapters:        {n_adapters}")
    log.info(f"    read_after_blocks: {model.read_after_blocks}")
    log.info(f"    write_after_blocks:{model.write_after_blocks}")

    # Training params
    log.info("  Training config:")
    log.info(f"    glimpse_grid:      {glimpse_grid}×{glimpse_grid}")
    log.info(f"    canvas_grid:       {canvas_grid}×{canvas_grid}")

    glimpse_px = glimpse_grid * patch_size
    log.info(f"    → glimpse_px:      {glimpse_px}×{glimpse_px}")
    log.info("")

    # === Token counts ===
    log.info("2. TOKEN COUNTS")
    log.info("-" * 60)

    # Teacher (vanilla ViT)
    n_teacher_tokens = 1 + n_backbone_regs + glimpse_grid ** 2  # CLS + regs + patches
    log.info("  Teacher (vanilla ViT):")
    log.info(f"    = 1 CLS + {n_backbone_regs} regs + {glimpse_grid}² patches")
    log.info(f"    = {n_teacher_tokens} tokens × {local_dim}d")

    # CanViT local stream
    n_extra = (1 if enable_vpe else 0) + 1  # VPE + recurrent_cls
    n_canvit_local = n_teacher_tokens + n_extra
    log.info("  CanViT local stream:")
    log.info(f"    = Teacher ({n_teacher_tokens}) + VPE ({1 if enable_vpe else 0}) + recurrent_cls (1)")
    log.info(f"    = {n_canvit_local} tokens × {local_dim}d")

    # Canvas
    n_canvas_tokens = n_canvas_regs + canvas_grid ** 2
    log.info("  Canvas:")
    log.info(f"    = {n_canvas_regs} regs + {canvas_grid}² spatial")
    log.info(f"    = {n_canvas_tokens} tokens × {canvas_dim}d")
    log.info("")

    # === CAPACITY CALCULATION ===
    log.info("3. CAPACITY CALCULATION")
    log.info("-" * 60)

    # Spatial only (patches vs canvas spatial)
    local_spatial_elements = glimpse_grid ** 2 * local_dim
    canvas_spatial_elements = canvas_grid ** 2 * canvas_dim
    capacity_ratio_spatial = canvas_spatial_elements / local_spatial_elements

    log.info("  Spatial tokens only:")
    log.info(f"    Local:  {glimpse_grid}² × {local_dim}d = {local_spatial_elements:,} elements")
    log.info(f"    Canvas: {canvas_grid}² × {canvas_dim}d = {canvas_spatial_elements:,} elements")
    log.info(f"    Ratio:  {canvas_spatial_elements:,} / {local_spatial_elements:,} = {capacity_ratio_spatial:.2f}×")

    # Including registers
    local_with_regs = (glimpse_grid ** 2 + n_backbone_regs) * local_dim
    canvas_with_regs = (canvas_grid ** 2 + n_canvas_regs) * canvas_dim
    capacity_ratio_with_regs = canvas_with_regs / local_with_regs

    log.info("  Including registers:")
    log.info(f"    Local:  ({glimpse_grid}² + {n_backbone_regs}) × {local_dim}d = {local_with_regs:,} elements")
    log.info(f"    Canvas: ({canvas_grid}² + {n_canvas_regs}) × {canvas_dim}d = {canvas_with_regs:,} elements")
    log.info(f"    Ratio:  {canvas_with_regs:,} / {local_with_regs:,} = {capacity_ratio_with_regs:.2f}×")

    capacity_claim = capacity_ratio_spatial >= 20
    log.info("")
    log.info(f"  ★ CLAIM: '21× higher capacity' → {capacity_ratio_spatial:.1f}× {'✓ VERIFIED' if capacity_claim else '✗ FAILED'}")
    log.info("")

    # === FLOP CALCULATION ===
    log.info("4. THEORETICAL FLOP CALCULATION (from canvit.flops)")
    log.info("-" * 60)

    # Teacher FLOPs
    n_patches = glimpse_grid ** 2
    teacher_patch_embed = flops.patch_embed(n_patches, patch_size, local_dim)
    teacher_blocks = n_blocks * flops.vit_block(n_teacher_tokens, local_dim, ffn_ratio)
    teacher_total = teacher_patch_embed + teacher_blocks

    log.info("  Teacher (ViT @ glimpse):")
    log.info(f"    patch_embed({n_patches}, {patch_size}, {local_dim}): {fmt_flops(teacher_patch_embed)}")
    log.info(f"    {n_blocks} × vit_block({n_teacher_tokens}, {local_dim}, {ffn_ratio}): {fmt_flops(teacher_blocks)}")
    log.info(f"    Total: {fmt_flops(teacher_total)}")

    # CanViT backbone FLOPs (with extra tokens)
    canvit_patch_embed = flops.patch_embed(n_patches, patch_size, local_dim)
    canvit_blocks = n_blocks * flops.vit_block(n_canvit_local, local_dim, ffn_ratio)
    canvit_backbone = canvit_patch_embed + canvit_blocks

    log.info(f"  CanViT backbone ({n_canvit_local} tokens):")
    log.info(f"    patch_embed: {fmt_flops(canvit_patch_embed)}")
    log.info(f"    {n_blocks} × vit_block({n_canvit_local}, {local_dim}, {ffn_ratio}): {fmt_flops(canvit_blocks)}")
    log.info(f"    Backbone subtotal: {fmt_flops(canvit_backbone)}")

    # Canvas adapters
    adapter_flops_each = flops.canvas_adapter(n_canvit_local, n_canvas_tokens, local_dim, canvas_dim)
    adapter_flops_total = n_adapters * adapter_flops_each

    log.info("  Canvas adapters:")
    log.info(f"    {n_adapters} × canvas_adapter({n_canvit_local}, {n_canvas_tokens}, {local_dim}, {canvas_dim})")
    log.info(f"    Each: {fmt_flops(adapter_flops_each)}")
    log.info(f"    Total: {fmt_flops(adapter_flops_total)}")

    # Scene head
    n_canvas_spatial = canvas_grid ** 2
    teacher_dim = backbone.embed_dim
    scene_head_flops = flops.scene_head(n_canvas_spatial, canvas_dim, teacher_dim)

    log.info("  Scene head:")
    log.info(f"    scene_head({n_canvas_spatial}, {canvas_dim}, {teacher_dim}): {fmt_flops(scene_head_flops)}")

    # Totals
    canvit_no_head = canvit_backbone + adapter_flops_total
    canvit_with_head = canvit_no_head + scene_head_flops

    log.info("  Totals:")
    log.info(f"    CanViT w/o head: {fmt_flops(canvit_no_head)}")
    log.info(f"    CanViT w/ head:  {fmt_flops(canvit_with_head)}")
    log.info("")

    # Overhead
    overhead_no_head = canvit_no_head / teacher_total
    overhead_with_head = canvit_with_head / teacher_total

    log.info("5. THEORETICAL OVERHEAD")
    log.info("-" * 60)
    log.info(f"  Teacher baseline:     {fmt_flops(teacher_total)}")
    log.info(f"  CanViT w/o head:      {fmt_flops(canvit_no_head)} → ×{overhead_no_head:.2f} ({(overhead_no_head-1)*100:+.0f}%)")
    log.info(f"  CanViT w/ head:       {fmt_flops(canvit_with_head)} → ×{overhead_with_head:.2f} ({(overhead_with_head-1)*100:+.0f}%)")

    overhead_claim = overhead_with_head < 1.50
    log.info("")
    log.info(f"  ★ CLAIM: '<50% overhead' → +{(overhead_with_head-1)*100:.0f}% {'✓ VERIFIED' if overhead_claim else '✗ FAILED'}")
    log.info("")

    # === EMPIRICAL BENCHMARK ===
    log.info("6. EMPIRICAL BENCHMARK")
    log.info("-" * 60)
    log.info(f"  Device: {device}")
    log.info(f"  Batch:  {cfg.batch}")
    log.info(f"  Iters:  {cfg.iters}")
    log.info(f"  AMP:    {cfg.amp}")
    log.info(f"  TF32:   {cfg.tf32}")
    log.info(f"  Compile:{cfg.compile}")
    log.info("")

    if cfg.compile:
        log.info("  Compiling models...")
        teacher.compile()
        model.compile()

    # Create inputs
    glimpse = torch.randn(cfg.batch, 3, glimpse_px, glimpse_px, device=device)
    vp = Viewpoint(
        centers=torch.zeros(cfg.batch, 2, device=device),
        scales=torch.ones(cfg.batch, device=device),
    )
    state = model.init_state(batch_size=cfg.batch, canvas_grid_size=canvas_grid)
    canvas_for_decode = state.canvas

    # AMP context
    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if cfg.amp and device.type in ("cuda", "cpu")
        else nullcontext()
    )

    def teacher_fn():
        with amp_ctx:
            return teacher.forward_norm_features(glimpse)

    def canvit_fn():
        nonlocal canvas_for_decode
        with amp_ctx:
            out = model.forward(glimpse=glimpse, state=state, viewpoint=vp)
            canvas_for_decode = out.state.canvas
            return model.predict_teacher_scene(canvas_for_decode)

    # Warmup
    log.info(f"  Warmup ({cfg.warmup} iters)...")
    with torch.no_grad():
        for _ in range(cfg.warmup):
            teacher_fn()
            canvit_fn()
    sync_device(device)

    # Benchmark
    log.info(f"  Benchmarking ({cfg.iters} iters)...")

    with torch.no_grad():
        sync_device(device)
        t0 = time.perf_counter()
        for _ in range(cfg.iters):
            teacher_fn()
        sync_device(device)
        t_teacher = time.perf_counter() - t0

        sync_device(device)
        t0 = time.perf_counter()
        for _ in range(cfg.iters):
            canvit_fn()
        sync_device(device)
        t_canvit = time.perf_counter() - t0

    empirical_overhead = t_canvit / t_teacher

    teacher_ms = t_teacher / cfg.iters * 1000
    canvit_ms = t_canvit / cfg.iters * 1000
    teacher_gps = (cfg.iters * cfg.batch) / t_teacher
    canvit_gps = (cfg.iters * cfg.batch) / t_canvit

    log.info("")
    log.info("  Results:")
    log.info(f"    Teacher:  {teacher_ms:6.1f} ms/batch  ({teacher_gps:6.1f} glimpses/s)")
    log.info(f"    CanViT:   {canvit_ms:6.1f} ms/batch  ({canvit_gps:6.1f} glimpses/s)")
    log.info(f"    Overhead: ×{empirical_overhead:.2f} ({(empirical_overhead-1)*100:+.0f}%)")
    log.info("")

    # === SUMMARY ===
    log.info("=" * 80)
    log.info("SUMMARY")
    log.info("=" * 80)
    log.info("")
    log.info(f"  Capacity ratio (spatial):  {capacity_ratio_spatial:.1f}× (claim: 21×)")
    log.info(f"  Theoretical overhead:      +{(overhead_with_head-1)*100:.0f}% (claim: <50%)")
    log.info(f"  Empirical overhead:        +{(empirical_overhead-1)*100:.0f}%")
    log.info("")

    # Final verdict
    both_claims = capacity_claim and overhead_claim
    log.info(f"  VERDICT: {'✓ BOTH CLAIMS VERIFIED' if both_claims else '✗ CLAIM(S) FAILED'}")

    if empirical_overhead > overhead_with_head * 1.3:
        log.info("")
        log.info("  ⚠ Empirical overhead significantly higher than theoretical.")
        log.info("    Possible causes: memory bandwidth, kernel launch overhead,")
        log.info("    insufficient batch size to saturate compute.")
    elif empirical_overhead < overhead_with_head * 0.8:
        log.info("")
        log.info("  ⚠ Empirical overhead significantly lower than theoretical.")
        log.info("    This is unusual - check benchmark methodology.")


if __name__ == "__main__":
    main(tyro.cli(BenchConfig))
