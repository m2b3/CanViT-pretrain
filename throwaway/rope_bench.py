"""RoPE benchmark matching EXACT canvit implementation.

Benchmarks the actual canvit rope_apply behavior:
- x (bf16) converted to sin.dtype (f32)
- core computation in f32
- result converted back to x.dtype (bf16)

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
    n_heads: int = 16
    n_spatial: int = 1024
    head_dim: int = 64
    warmup: int = 20
    iters: int = 100


def rotate_half(x: Tensor) -> Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)


def rope_apply_canvit(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    """EXACT canvit implementation: bf16 -> f32 -> compute -> bf16."""
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
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fn()
    sync(device)
    return (time.perf_counter() - t0) / iters * 1000


def max_diff(a: Tensor, b: Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def main(cfg: Config) -> None:
    device = torch.device(cfg.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        log.info(f"cuda: {torch.cuda.get_device_name()}")

    log.info(f"shape: [{cfg.batch}, {cfg.n_heads}, {cfg.n_spatial}, {cfg.head_dim}]")
    log.info(f"warmup: {cfg.warmup}, iters: {cfg.iters}")
    log.info("")

    # Create tensors matching canvit: x=bf16, sin/cos=f32
    torch.manual_seed(42)
    shape_x = (cfg.batch, cfg.n_heads, cfg.n_spatial, cfg.head_dim)
    shape_sc = (cfg.batch, 1, cfg.n_spatial, cfg.head_dim)

    x_bf16 = torch.randn(shape_x, device=device, dtype=torch.bfloat16)
    sin_f32 = torch.randn(shape_sc, device=device, dtype=torch.float32)
    cos_f32 = torch.randn(shape_sc, device=device, dtype=torch.float32)

    # Ground truth: f32 throughout
    x_f32 = x_bf16.float()
    ref_f32 = x_f32 * cos_f32 + rotate_half(x_f32) * sin_f32

    # =========================================================================
    log.info("=== CORRECTNESS (max_abs_diff vs f32 reference) ===")

    out_eager = rope_apply_canvit(x_bf16, sin_f32, cos_f32)
    log.info(f"eager canvit (x=bf16, rope=f32): {max_diff(out_eager, ref_f32):.2e}")

    rope_compiled = torch.compile(rope_apply_canvit)
    for _ in range(cfg.warmup):
        _ = rope_compiled(x_bf16, sin_f32, cos_f32)
    sync(device)
    out_compiled = rope_compiled(x_bf16, sin_f32, cos_f32)
    log.info(f"compiled canvit (x=bf16, rope=f32): {max_diff(out_compiled, ref_f32):.2e}")

    log.info("")

    # =========================================================================
    log.info("=== TIMING (ms per call) ===")

    t_eager = bench(lambda: rope_apply_canvit(x_bf16, sin_f32, cos_f32), cfg.warmup, cfg.iters, device)
    log.info(f"eager: {t_eager:.3f}")

    t_compiled = bench(lambda: rope_compiled(x_bf16, sin_f32, cos_f32), cfg.warmup, cfg.iters, device)
    log.info(f"compiled: {t_compiled:.3f}")

    log.info("")
    log.info(f"speedup: {t_eager / t_compiled:.1f}x")


if __name__ == "__main__":
    main(tyro.cli(Config))
