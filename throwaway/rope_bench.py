"""RoPE implementation comparison and benchmarking.

Correctness: All implementations compared against f32 reference (ground truth).
Shows precision loss for bf16 variants.

Usage:
    uv run python throwaway/rope_bench.py --device cuda --iters 200
    uv run python throwaway/rope_bench.py --device cuda --dtype bfloat16 --iters 200
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
    """Benchmark configuration."""

    device: str = "cpu"
    """Device: 'cpu', 'cuda', 'mps'."""
    dtype: str = "float32"
    """Data type for benchmark: 'float32' or 'bfloat16'."""
    batch: int = 64
    """Batch size."""
    n_heads: int = 16
    """Number of attention heads."""
    n_spatial: int = 1024
    """Number of spatial tokens (32×32 = 1024)."""
    head_dim: int = 64
    """Head dimension (canvas_dim / n_heads = 1024 / 16)."""
    warmup: int = 20
    """Warmup iterations."""
    iters: int = 100
    """Timed iterations."""


# =============================================================================
# Reference implementation (from canvit/rope/impl.py)
# =============================================================================


def rotate_half(x: Tensor) -> Tensor:
    """Rotate pairs: [a, b] -> [-b, a]."""
    a, b = x.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    """Reference: x * cos + rotate_half(x) * sin."""
    return x * cos + rotate_half(x) * sin


# =============================================================================
# Utilities
# =============================================================================


def sync_device(device: torch.device) -> None:
    """Synchronize device for accurate timing."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def max_abs_diff(a: Tensor, b: Tensor) -> float:
    """Max absolute difference, comparing in f32."""
    return (a.float() - b.float()).abs().max().item()


def bench_time(
    fn: Callable[[], Tensor],
    warmup: int,
    iters: int,
    device: torch.device,
) -> float:
    """Return mean time per call in ms."""
    for _ in range(warmup):
        _ = fn()
    sync_device(device)

    sync_device(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fn()
    sync_device(device)
    return (time.perf_counter() - t0) / iters * 1000


def main(cfg: Config) -> None:
    device = torch.device(cfg.device)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[cfg.dtype]
    bytes_per_elem = 4 if dtype == torch.float32 else 2

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        log.info(f"cuda_device: {torch.cuda.get_device_name()}")

    log.info(f"device: {device}")
    log.info(f"dtype: {cfg.dtype}")
    log.info(f"shape: [{cfg.batch}, {cfg.n_heads}, {cfg.n_spatial}, {cfg.head_dim}]")
    n_elements = cfg.batch * cfg.n_heads * cfg.n_spatial * cfg.head_dim
    log.info(f"size_mb: {n_elements * bytes_per_elem / 1e6:.1f}")
    log.info(f"warmup: {cfg.warmup}, iters: {cfg.iters}")
    log.info("")

    # Create test tensors in BOTH f32 and target dtype
    torch.manual_seed(42)
    shape_x = (cfg.batch, cfg.n_heads, cfg.n_spatial, cfg.head_dim)
    shape_sc = (cfg.batch, 1, cfg.n_spatial, cfg.head_dim)

    x_f32 = torch.randn(shape_x, device=device, dtype=torch.float32)
    sin_f32 = torch.randn(shape_sc, device=device, dtype=torch.float32)
    cos_f32 = torch.randn(shape_sc, device=device, dtype=torch.float32)

    x = x_f32.to(dtype)
    sin = sin_f32.to(dtype)
    cos = cos_f32.to(dtype)

    # Ground truth: f32 eager
    ref_f32 = rope_apply(x_f32, sin_f32, cos_f32)

    # ==========================================================================
    log.info("=== CORRECTNESS (max_abs_diff vs f32 reference) ===")
    log.info("")

    # Eager same-dtype
    out_eager = rope_apply(x, sin, cos)
    diff_eager = max_abs_diff(out_eager, ref_f32)
    log.info(f"eager {cfg.dtype}: {diff_eager:.2e}")

    # Compiled same-dtype
    rope_compiled = torch.compile(rope_apply, fullgraph=True)
    for _ in range(cfg.warmup):
        _ = rope_compiled(x, sin, cos)
    sync_device(device)
    out_compiled = rope_compiled(x, sin, cos)
    diff_compiled = max_abs_diff(out_compiled, ref_f32)
    log.info(f"compiled {cfg.dtype}: {diff_compiled:.2e}")

    # Mixed: x=bf16, sin/cos=f32 (what canvit does)
    if dtype == torch.bfloat16:
        out_mixed = rope_apply(x, sin_f32, cos_f32)
        diff_mixed = max_abs_diff(out_mixed, ref_f32)
        log.info(f"mixed (x=bf16, rope=f32): {diff_mixed:.2e}")

        rope_compiled_mixed = torch.compile(rope_apply, fullgraph=True)
        for _ in range(cfg.warmup):
            _ = rope_compiled_mixed(x, sin_f32, cos_f32)
        sync_device(device)
        out_compiled_mixed = rope_compiled_mixed(x, sin_f32, cos_f32)
        diff_compiled_mixed = max_abs_diff(out_compiled_mixed, ref_f32)
        log.info(f"compiled mixed (x=bf16, rope=f32): {diff_compiled_mixed:.2e}")

    log.info("")

    # ==========================================================================
    log.info("=== TIMING (ms per call) ===")
    log.info("")

    t_eager = bench_time(lambda: rope_apply(x, sin, cos), cfg.warmup, cfg.iters, device)
    log.info(f"eager {cfg.dtype}: {t_eager:.3f}")

    t_compiled = bench_time(lambda: rope_compiled(x, sin, cos), cfg.warmup, cfg.iters, device)
    log.info(f"compiled {cfg.dtype}: {t_compiled:.3f}")

    if dtype == torch.bfloat16:
        t_mixed = bench_time(lambda: rope_apply(x, sin_f32, cos_f32), cfg.warmup, cfg.iters, device)
        log.info(f"mixed (x=bf16, rope=f32): {t_mixed:.3f}")

        t_compiled_mixed = bench_time(lambda: rope_compiled_mixed(x, sin_f32, cos_f32), cfg.warmup, cfg.iters, device)
        log.info(f"compiled mixed (x=bf16, rope=f32): {t_compiled_mixed:.3f}")

    log.info("")
    log.info(f"speedup compiled vs eager: {t_eager / t_compiled:.1f}x")


if __name__ == "__main__":
    main(tyro.cli(Config))
