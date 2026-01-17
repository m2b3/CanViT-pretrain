"""RoPE benchmark - testing slice/cat overhead and compilation.

Tests:
1. rope_apply (core, already compiled)
2. rope_apply_with_prefix (uncompiled - slice + rope + cat)
3. rope_apply_with_prefix compiled
4. Fused version (no slice/cat, pad RoPE with identity)

Usage:
    uv run python throwaway/rope_bench.py --device cuda --iters 200
"""

import logging
import time
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor
import tyro

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    device: str = "cpu"
    batch: int = 64
    n_heads: int = 8
    n_spatial: int = 1024
    n_prefix: int = 16  # canvas registers
    head_dim: int = 128
    warmup: int = 20
    iters: int = 100


def rotate_half(x: Tensor) -> Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)


def rope_apply_core(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    """Core RoPE: bf16 -> f32 -> compute -> bf16."""
    x_dtype = x.dtype
    x = x.to(dtype=sin.dtype)
    out = x * cos + rotate_half(x) * sin
    return out.to(dtype=x_dtype)


def rope_apply_with_prefix_uncompiled(
    x: Tensor, sin: Tensor, cos: Tensor, n_prefix: int
) -> Tensor:
    """Uncompiled version with slice + cat."""
    if n_prefix == 0:
        return rope_apply_core(x, sin, cos)
    x_prefix = x[:, :, :n_prefix]
    x_spatial = rope_apply_core(x[:, :, n_prefix:], sin, cos)
    return torch.cat([x_prefix, x_spatial], dim=2)


def rope_apply_fused(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    """Fused version: RoPE applied to ALL tokens, prefix has identity rotation."""
    x_dtype = x.dtype
    x = x.to(dtype=sin.dtype)
    out = x * cos + rotate_half(x) * sin
    return out.to(dtype=x_dtype)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench(fn: Callable[[], Tensor], warmup: int, iters: int, device: torch.device) -> float:
    for _ in range(warmup):
        _ = fn()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fn()
    sync(device)
    return (time.perf_counter() - t0) / iters * 1000


def main(cfg: Config) -> None:
    device = torch.device(cfg.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        log.info(f"cuda: {torch.cuda.get_device_name()}")

    n_total = cfg.n_prefix + cfg.n_spatial
    log.info(f"shape: [{cfg.batch}, {cfg.n_heads}, {n_total}, {cfg.head_dim}]")
    log.info(f"n_prefix: {cfg.n_prefix}, n_spatial: {cfg.n_spatial}")
    log.info(f"warmup: {cfg.warmup}, iters: {cfg.iters}")
    log.info("")

    torch.manual_seed(42)

    # Full tensor (with prefix)
    x_full = torch.randn(
        cfg.batch, cfg.n_heads, n_total, cfg.head_dim,
        device=device, dtype=torch.bfloat16
    )

    # Spatial-only sin/cos (current approach)
    sin_spatial = torch.randn(
        cfg.batch, 1, cfg.n_spatial, cfg.head_dim,
        device=device, dtype=torch.float32
    )
    cos_spatial = torch.randn(
        cfg.batch, 1, cfg.n_spatial, cfg.head_dim,
        device=device, dtype=torch.float32
    )

    # Full sin/cos with identity for prefix (fused approach)
    sin_full = torch.zeros(
        cfg.batch, 1, n_total, cfg.head_dim,
        device=device, dtype=torch.float32
    )
    cos_full = torch.ones(
        cfg.batch, 1, n_total, cfg.head_dim,
        device=device, dtype=torch.float32
    )
    sin_full[:, :, cfg.n_prefix:] = sin_spatial
    cos_full[:, :, cfg.n_prefix:] = cos_spatial

    # Compile versions
    rope_core_compiled = torch.compile(rope_apply_core)
    rope_with_prefix_compiled = torch.compile(rope_apply_with_prefix_uncompiled)
    rope_fused_compiled = torch.compile(rope_apply_fused)

    # Warmup all compiled versions
    log.info("Warming up compiled versions...")
    for _ in range(cfg.warmup):
        _ = rope_core_compiled(x_full[:, :, cfg.n_prefix:], sin_spatial, cos_spatial)
        _ = rope_with_prefix_compiled(x_full, sin_spatial, cos_spatial, cfg.n_prefix)
        _ = rope_fused_compiled(x_full, sin_full, cos_full)
    sync(device)
    log.info("Done.")
    log.info("")

    # =========================================================================
    log.info("=== CORRECTNESS ===")

    out_uncompiled = rope_apply_with_prefix_uncompiled(x_full, sin_spatial, cos_spatial, cfg.n_prefix)
    out_compiled = rope_with_prefix_compiled(x_full, sin_spatial, cos_spatial, cfg.n_prefix)
    out_fused = rope_fused_compiled(x_full, sin_full, cos_full)

    diff_compiled = (out_uncompiled.float() - out_compiled.float()).abs().max().item()
    diff_fused = (out_uncompiled.float() - out_fused.float()).abs().max().item()

    log.info(f"compiled vs uncompiled: max_diff={diff_compiled:.2e}")
    log.info(f"fused vs uncompiled:    max_diff={diff_fused:.2e}")
    log.info("")

    # =========================================================================
    log.info("=== TIMING (ms per call) ===")
    log.info("")

    # 1. Core only (spatial tokens, no prefix handling)
    t_core_eager = bench(
        lambda: rope_apply_core(x_full[:, :, cfg.n_prefix:], sin_spatial, cos_spatial),
        cfg.warmup, cfg.iters, device
    )
    log.info(f"core eager (spatial only):        {t_core_eager:.3f}")

    t_core_compiled = bench(
        lambda: rope_core_compiled(x_full[:, :, cfg.n_prefix:], sin_spatial, cos_spatial),
        cfg.warmup, cfg.iters, device
    )
    log.info(f"core compiled (spatial only):     {t_core_compiled:.3f}")
    log.info("")

    # 2. With prefix (slice + cat)
    t_prefix_eager = bench(
        lambda: rope_apply_with_prefix_uncompiled(x_full, sin_spatial, cos_spatial, cfg.n_prefix),
        cfg.warmup, cfg.iters, device
    )
    log.info(f"with_prefix eager (slice+cat):    {t_prefix_eager:.3f}")

    t_prefix_compiled = bench(
        lambda: rope_with_prefix_compiled(x_full, sin_spatial, cos_spatial, cfg.n_prefix),
        cfg.warmup, cfg.iters, device
    )
    log.info(f"with_prefix compiled:             {t_prefix_compiled:.3f}")
    log.info("")

    # 3. Fused (no slice/cat, identity prefix in RoPE)
    t_fused_eager = bench(
        lambda: rope_apply_fused(x_full, sin_full, cos_full),
        cfg.warmup, cfg.iters, device
    )
    log.info(f"fused eager (no slice/cat):       {t_fused_eager:.3f}")

    t_fused_compiled = bench(
        lambda: rope_fused_compiled(x_full, sin_full, cos_full),
        cfg.warmup, cfg.iters, device
    )
    log.info(f"fused compiled:                   {t_fused_compiled:.3f}")
    log.info("")

    # =========================================================================
    log.info("=== ANALYSIS ===")
    slice_cat_overhead = t_prefix_eager - t_core_eager
    log.info(f"slice+cat overhead (eager):       {slice_cat_overhead:.3f} ms")

    compile_speedup_core = t_core_eager / t_core_compiled
    compile_speedup_prefix = t_prefix_eager / t_prefix_compiled
    compile_speedup_fused = t_fused_eager / t_fused_compiled

    log.info(f"compile speedup (core):           {compile_speedup_core:.1f}x")
    log.info(f"compile speedup (with_prefix):    {compile_speedup_prefix:.1f}x")
    log.info(f"compile speedup (fused):          {compile_speedup_fused:.1f}x")
    log.info("")

    best = min(t_core_compiled, t_prefix_compiled, t_fused_compiled)
    log.info(f"BEST: {best:.3f} ms")

    # Per-forward estimate (12 RoPE calls on canvas)
    n_rope_calls = 12
    log.info("")
    log.info(f"=== PER-FORWARD ESTIMATE ({n_rope_calls} canvas RoPE calls) ===")
    log.info(f"current (with_prefix eager):      {n_rope_calls * t_prefix_eager:.1f} ms")
    log.info(f"with_prefix compiled:             {n_rope_calls * t_prefix_compiled:.1f} ms")
    log.info(f"fused compiled:                   {n_rope_calls * t_fused_compiled:.1f} ms")


if __name__ == "__main__":
    main(tyro.cli(Config))
