"""Comprehensive component-level overhead benchmark.

Measures individual operations with proper CUDA event timing and statistics.
Identifies bottlenecks by comparing isolated components to full forward pass.

Usage:
    uv run python scripts/bench_components.py
    uv run python scripts/bench_components.py --batch 32 --iters 100
"""

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import tyro

from avp_vit import ACVFRP, ACVFRPConfig
from canvit import create_backbone
from canvit.rope import RoPE, rope_apply
from canvit.viewpoint import Viewpoint

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    """Benchmark configuration."""

    # Model config (from train/config.py - DO NOT CHANGE without checking source)
    glimpse_grid: int = 8
    """Glimpse grid size (train/config.py:27)."""
    canvas_grid: int = 32
    """Canvas grid size (train/config.py:29)."""
    patch_size: int = 16
    """DINOv3 ViT-B/16 patch size (constant)."""

    # Benchmark params
    batch: int = 64
    """Batch size."""
    warmup: int = 10
    """Warmup iterations."""
    iters: int = 50
    """Timed iterations per measurement."""


def get_device() -> torch.device:
    """Get CUDA device or fail."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this benchmark")
    return torch.device("cuda")


def bench(
    name: str,
    fn: callable,
    n_warmup: int,
    n_iters: int,
    device: torch.device,
) -> dict:
    """Benchmark with CUDA events and proper statistics.

    Returns dict with min/mean/max/std in milliseconds.
    """
    # Warmup
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    # Timed iterations with per-iteration measurement
    times = []
    for _ in range(n_iters):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        fn()
        t1.record()
        torch.cuda.synchronize()
        times.append(t0.elapsed_time(t1))

    times = np.array(times)
    result = {
        "name": name,
        "min": float(times.min()),
        "mean": float(times.mean()),
        "max": float(times.max()),
        "std": float(times.std()),
    }

    log.info(
        f"{name:50s}: min={result['min']:7.3f}  mean={result['mean']:7.3f}  "
        f"max={result['max']:7.3f}  std={result['std']:6.3f} ms"
    )
    return result


def derive_token_counts(
    cfg: Config, backbone_n_register_tokens: int
) -> tuple[int, int, int]:
    """Derive token counts from config.

    Returns (n_local, n_canvas, n_spatial).
    """
    n_patches = cfg.glimpse_grid**2
    # LocalTokens layout: [vpe, recurrent_cls, ephemeral_cls, registers, patches]
    n_local = 1 + 1 + 1 + backbone_n_register_tokens + n_patches

    # Canvas layout: [registers, spatial]
    n_canvas_regs = 16  # from CanViTConfig default
    n_spatial = cfg.canvas_grid**2
    n_canvas = n_canvas_regs + n_spatial

    return n_local, n_canvas, n_spatial


def main(cfg: Config) -> None:
    device = get_device()
    torch.set_float32_matmul_precision("high")

    glimpse_px = cfg.glimpse_grid * cfg.patch_size

    log.info("=" * 70)
    log.info("COMPREHENSIVE COMPONENT BENCHMARK")
    log.info("=" * 70)
    log.info("")
    log.info("Config:")
    log.info(f"  Batch:        {cfg.batch}")
    log.info(
        f"  Glimpse:      {cfg.glimpse_grid}×{cfg.glimpse_grid} = {cfg.glimpse_grid**2} patches @ {cfg.patch_size}px = {glimpse_px}px"
    )
    log.info(
        f"  Canvas:       {cfg.canvas_grid}×{cfg.canvas_grid} = {cfg.canvas_grid**2} spatial tokens"
    )
    log.info(f"  Warmup:       {cfg.warmup} iters")
    log.info(f"  Timed:        {cfg.iters} iters")
    log.info("")

    # === CREATE MODELS ===
    log.info("Creating models...")
    # Suppress dinov3 spam
    logging.getLogger("dinov3").setLevel(logging.WARNING)

    backbone = create_backbone("dinov3_vitb16", pretrained=False)
    teacher = create_backbone("dinov3_vitb16", pretrained=False)

    model_cfg = ACVFRPConfig(teacher_dim=backbone.embed_dim)
    model = ACVFRP(backbone=backbone, cfg=model_cfg, policy=None)
    model.to(device).eval()
    teacher.to(device).eval()

    # Derive dimensions
    local_dim = backbone.embed_dim  # 768
    canvas_dim = model_cfg.canvas_dim  # 1024
    n_heads = model_cfg.canvas_num_heads  # 16
    head_dim = canvas_dim // n_heads  # 64
    n_local, n_canvas, n_spatial = derive_token_counts(cfg, backbone.n_register_tokens)

    log.info(f"  Local dim:    {local_dim}")
    log.info(f"  Canvas dim:   {canvas_dim}")
    log.info(f"  N_LOCAL:      {n_local} tokens")
    log.info(f"  N_CANVAS:     {n_canvas} tokens (16 regs + {n_spatial} spatial)")
    log.info(f"  N_HEADS:      {n_heads}, HEAD_DIM: {head_dim}")
    log.info("")

    # === PRE-ALLOCATE TENSORS ===
    log.info("Pre-allocating tensors...")

    # Full forward inputs
    glimpse = torch.randn(
        cfg.batch, 3, glimpse_px, glimpse_px, device=device, dtype=torch.float32
    )
    vp = Viewpoint(
        centers=torch.zeros(cfg.batch, 2, device=device, dtype=torch.float32),
        scales=torch.ones(cfg.batch, device=device, dtype=torch.float32),
    )
    state = model.init_state(batch_size=cfg.batch, canvas_grid_size=cfg.canvas_grid)

    # Component tensors - explicit bf16 (what autocast produces)
    q_read = torch.randn(
        cfg.batch, n_heads, n_local, head_dim, device=device, dtype=torch.bfloat16
    )
    k_read = torch.randn(
        cfg.batch, n_heads, n_canvas, head_dim, device=device, dtype=torch.bfloat16
    )
    v_read = torch.randn(
        cfg.batch, n_heads, n_canvas, head_dim, device=device, dtype=torch.bfloat16
    )

    q_write = torch.randn(
        cfg.batch, n_heads, n_canvas, head_dim, device=device, dtype=torch.bfloat16
    )
    k_write = torch.randn(
        cfg.batch, n_heads, n_local, head_dim, device=device, dtype=torch.bfloat16
    )
    v_write = torch.randn(
        cfg.batch, n_heads, n_local, head_dim, device=device, dtype=torch.bfloat16
    )

    # RoPE tensors - spatial only (registers don't get RoPE)
    x_spatial_bf16 = torch.randn(
        cfg.batch, n_heads, n_spatial, head_dim, device=device, dtype=torch.bfloat16
    )
    x_spatial_f32 = torch.randn(
        cfg.batch, n_heads, n_spatial, head_dim, device=device, dtype=torch.float32
    )

    rope_f32 = RoPE(
        sin=torch.randn(
            cfg.batch, 1, n_spatial, head_dim, device=device, dtype=torch.float32
        ),
        cos=torch.randn(
            cfg.batch, 1, n_spatial, head_dim, device=device, dtype=torch.float32
        ),
    )
    rope_bf16 = RoPE(
        sin=torch.randn(
            cfg.batch, 1, n_spatial, head_dim, device=device, dtype=torch.bfloat16
        ),
        cos=torch.randn(
            cfg.batch, 1, n_spatial, head_dim, device=device, dtype=torch.bfloat16
        ),
    )

    # Linear and LayerNorm
    proj_up = torch.nn.Linear(
        local_dim, canvas_dim, device=device, dtype=torch.bfloat16
    )
    proj_down = torch.nn.Linear(
        canvas_dim, local_dim, device=device, dtype=torch.bfloat16
    )
    ln_local = torch.nn.LayerNorm(local_dim, device=device, dtype=torch.bfloat16)
    ln_canvas = torch.nn.LayerNorm(canvas_dim, device=device, dtype=torch.bfloat16)

    local_tokens = torch.randn(
        cfg.batch, n_local, local_dim, device=device, dtype=torch.bfloat16
    )
    canvas_tokens = torch.randn(
        cfg.batch, n_canvas, canvas_dim, device=device, dtype=torch.bfloat16
    )

    log.info("Done.")
    log.info("")

    # === BENCHMARKS ===
    results = {}

    log.info("=" * 70)
    log.info("FULL MODEL FORWARD")
    log.info("=" * 70)

    with torch.no_grad():
        results["teacher"] = bench(
            "Teacher forward (baseline)",
            lambda: teacher.forward_norm_features(glimpse),
            cfg.warmup,
            cfg.iters,
            device,
        )

    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        results["canvit"] = bench(
            "CanViT forward",
            lambda: model.forward(glimpse=glimpse, state=state, viewpoint=vp),
            cfg.warmup,
            cfg.iters,
            device,
        )

        results["canvit_decode"] = bench(
            "CanViT forward + decode",
            lambda: model.predict_teacher_scene(
                model.forward(glimpse=glimpse, state=state, viewpoint=vp).state.canvas
            ),
            cfg.warmup,
            cfg.iters,
            device,
        )

    overhead = results["canvit"]["mean"] / results["teacher"]["mean"]
    log.info(f"\nOverhead: ×{overhead:.2f} ({(overhead - 1) * 100:.0f}%)")
    log.info("")

    log.info("=" * 70)
    log.info("COMPONENT BENCHMARKS")
    log.info("=" * 70)

    with torch.no_grad():
        # SDPA
        results["sdpa_read"] = bench(
            f"SDPA read ({n_local} query → {n_canvas} kv)",
            lambda: F.scaled_dot_product_attention(q_read, k_read, v_read),
            cfg.warmup,
            cfg.iters,
            device,
        )
        results["sdpa_write"] = bench(
            f"SDPA write ({n_canvas} query → {n_local} kv)",
            lambda: F.scaled_dot_product_attention(q_write, k_write, v_write),
            cfg.warmup,
            cfg.iters,
            device,
        )

        log.info("")

        # RoPE - all dtype combinations
        results["rope_bf16_bf16"] = bench(
            "rope_apply (x=bf16, rope=bf16)",
            lambda: rope_apply(x=x_spatial_bf16, rope=rope_bf16),
            cfg.warmup,
            cfg.iters,
            device,
        )
        results["rope_bf16_f32"] = bench(
            "rope_apply (x=bf16, rope=f32) ← CURRENT",
            lambda: rope_apply(x=x_spatial_bf16, rope=rope_f32),
            cfg.warmup,
            cfg.iters,
            device,
        )
        results["rope_f32_f32"] = bench(
            "rope_apply (x=f32, rope=f32)",
            lambda: rope_apply(x=x_spatial_f32, rope=rope_f32),
            cfg.warmup,
            cfg.iters,
            device,
        )
        results["rope_f32_bf16"] = bench(
            "rope_apply (x=f32, rope=bf16)",
            lambda: rope_apply(x=x_spatial_f32, rope=rope_bf16),
            cfg.warmup,
            cfg.iters,
            device,
        )

        log.info("")

        # LayerNorm
        results["ln_local"] = bench(
            f"LayerNorm local ({n_local} × {local_dim})",
            lambda: ln_local(local_tokens),
            cfg.warmup,
            cfg.iters,
            device,
        )
        results["ln_canvas"] = bench(
            f"LayerNorm canvas ({n_canvas} × {canvas_dim})",
            lambda: ln_canvas(canvas_tokens),
            cfg.warmup,
            cfg.iters,
            device,
        )

        log.info("")

        # Linear projections
        results["proj_up"] = bench(
            f"Linear {local_dim}→{canvas_dim} ({n_local} tokens)",
            lambda: proj_up(local_tokens),
            cfg.warmup,
            cfg.iters,
            device,
        )
        results["proj_down"] = bench(
            f"Linear {canvas_dim}→{local_dim} ({n_local} tokens)",
            lambda: proj_down(canvas_tokens[:, :n_local]),
            cfg.warmup,
            cfg.iters,
            device,
        )

    log.info("")
    log.info("=" * 70)
    log.info("COST BREAKDOWN PER FORWARD (estimated)")
    log.info("=" * 70)

    # Per forward: 3 read + 3 write adapters
    # Each adapter has rope_apply on Q and K
    n_rope_calls = 12  # 6 adapters × 2 (Q and K)
    n_sdpa_calls = 6
    n_ln_calls = 12  # 6 adapters × 2

    t_rope_current = results["rope_bf16_f32"]["mean"]
    t_rope_optimal = results["rope_bf16_bf16"]["mean"]
    t_sdpa_avg = (results["sdpa_read"]["mean"] + results["sdpa_write"]["mean"]) / 2
    t_ln_avg = (results["ln_local"]["mean"] + results["ln_canvas"]["mean"]) / 2
    t_proj_avg = (results["proj_up"]["mean"] + results["proj_down"]["mean"]) / 2

    log.info(
        f"rope_apply × {n_rope_calls} (current x=bf16, rope=f32): {n_rope_calls * t_rope_current:6.1f} ms"
    )
    log.info(
        f"rope_apply × {n_rope_calls} (optimal x=bf16, rope=bf16): {n_rope_calls * t_rope_optimal:6.1f} ms"
    )
    log.info(
        f"SDPA × {n_sdpa_calls}:                                   {n_sdpa_calls * t_sdpa_avg:6.1f} ms"
    )
    log.info(
        f"LayerNorm × {n_ln_calls}:                                 {n_ln_calls * t_ln_avg:6.1f} ms"
    )
    log.info(
        f"Projections × {n_ln_calls}:                               {n_ln_calls * t_proj_avg:6.1f} ms"
    )

    total_current = (
        n_rope_calls * t_rope_current
        + n_sdpa_calls * t_sdpa_avg
        + n_ln_calls * t_ln_avg
        + n_ln_calls * t_proj_avg
    )
    total_optimal = (
        n_rope_calls * t_rope_optimal
        + n_sdpa_calls * t_sdpa_avg
        + n_ln_calls * t_ln_avg
        + n_ln_calls * t_proj_avg
    )

    measured_overhead = results["canvit"]["mean"] - results["teacher"]["mean"]
    potential_savings = n_rope_calls * (t_rope_current - t_rope_optimal)

    log.info("")
    log.info(
        f"Estimated canvas attention overhead (current):   {total_current:6.1f} ms"
    )
    log.info(
        f"Estimated canvas attention overhead (bf16 rope): {total_optimal:6.1f} ms"
    )
    log.info(
        f"Measured CanViT - Teacher:                       {measured_overhead:6.1f} ms"
    )
    log.info(
        f"Potential savings from bf16 rope:                {potential_savings:6.1f} ms"
    )
    log.info("")

    # Sanity check
    if total_current > 2 * measured_overhead:
        log.warning(
            "⚠️  Estimated overhead > 2× measured. Component estimates may be wrong."
        )
    elif total_current < 0.5 * measured_overhead:
        log.warning("⚠️  Estimated overhead < 0.5× measured. Missing overhead source.")


if __name__ == "__main__":
    main(tyro.cli(Config))
