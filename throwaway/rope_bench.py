"""RoPE implementation comparison and benchmarking.

Correctness testing: Each implementation is compared against the reference
(canvit's rope_apply) using torch.allclose with atol=1e-5, rtol=0.
The max absolute difference is printed for each implementation.

Usage:
    uv run python throwaway/rope_bench.py              # CPU (default)
    uv run python throwaway/rope_bench.py --device cuda  # GPU
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
    """Benchmark configuration."""

    device: str = "cpu"
    """Device: 'cpu', 'cuda', 'mps'."""
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
    atol: float = 1e-5
    """Absolute tolerance for correctness check."""


# =============================================================================
# Reference implementation (from canvit/rope/impl.py)
# =============================================================================


def rotate_half_reference(x: Tensor) -> Tensor:
    """Reference: chunk + cat."""
    a, b = x.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)


def rope_apply_reference(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    """Reference implementation from canvit."""
    return x * cos + rotate_half_reference(x) * sin


# =============================================================================
# Alternative implementations
# =============================================================================


def rope_apply_direct(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    """Direct formula without rotate_half intermediate.

    Expands: out = x * cos + rotate_half(x) * sin
    Where rotate_half([a, b]) = [-b, a]

    Result:
    - out[:half] = a * cos[:half] - b * sin[:half]
    - out[half:] = b * cos[half:] + a * sin[half:]
    """
    half = x.shape[-1] // 2
    a = x[..., :half]
    b = x[..., half:]
    out_first = a * cos[..., :half] - b * sin[..., :half]
    out_second = b * cos[..., half:] + a * sin[..., half:]
    return torch.cat([out_first, out_second], dim=-1)


def rope_apply_addcmul(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    """Use addcmul for potential fusion.

    addcmul(input, tensor1, tensor2, value) = input + value * tensor1 * tensor2
    """
    half = x.shape[-1] // 2
    a = x[..., :half]
    b = x[..., half:]

    # out_first = a * cos_first - b * sin_first
    out_first = torch.addcmul(a * cos[..., :half], b, sin[..., :half], value=-1)
    # out_second = b * cos_second + a * sin_second
    out_second = torch.addcmul(b * cos[..., half:], a, sin[..., half:], value=1)

    return torch.cat([out_first, out_second], dim=-1)


def rope_apply_inplace(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    """Pre-allocate output and write in-place."""
    half = x.shape[-1] // 2
    a = x[..., :half].contiguous()
    b = x[..., half:].contiguous()

    out = torch.empty_like(x)
    out[..., :half] = a * cos[..., :half] - b * sin[..., :half]
    out[..., half:] = b * cos[..., half:] + a * sin[..., half:]
    return out


# =============================================================================
# Utilities
# =============================================================================


def sync_device(device: torch.device) -> None:
    """Synchronize device for accurate timing."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def measure_diff(
    impl: Callable[[Tensor, Tensor, Tensor], Tensor],
    x: Tensor,
    sin: Tensor,
    cos: Tensor,
    reference: Tensor,
) -> float:
    """Return max absolute difference from reference."""
    result = impl(x, sin, cos)
    return (result - reference).abs().max().item()


def bench_time(
    impl: Callable[[Tensor, Tensor, Tensor], Tensor],
    x: Tensor,
    sin: Tensor,
    cos: Tensor,
    warmup: int,
    iters: int,
    device: torch.device,
) -> float:
    """Return mean time per call in ms."""
    # Warmup
    for _ in range(warmup):
        _ = impl(x, sin, cos)
    sync_device(device)

    # Time
    sync_device(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = impl(x, sin, cos)
    sync_device(device)
    return (time.perf_counter() - t0) / iters * 1000


def main(cfg: Config) -> None:
    device = torch.device(cfg.device)

    # Device info
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        log.info(f"cuda_device: {torch.cuda.get_device_name()}")

    # Config
    log.info(f"device: {device}")
    log.info(f"shape: [{cfg.batch}, {cfg.n_heads}, {cfg.n_spatial}, {cfg.head_dim}]")
    n_elements = cfg.batch * cfg.n_heads * cfg.n_spatial * cfg.head_dim
    log.info(f"n_elements: {n_elements}")
    log.info(f"size_mb: {n_elements * 4 / 1e6:.1f}")
    log.info(f"dtype: float32")
    log.info(f"warmup: {cfg.warmup}")
    log.info(f"iters: {cfg.iters}")
    log.info(f"atol: {cfg.atol}")
    log.info("")

    # Create test tensors
    torch.manual_seed(42)
    x = torch.randn(cfg.batch, cfg.n_heads, cfg.n_spatial, cfg.head_dim, device=device)
    sin = torch.randn(cfg.batch, 1, cfg.n_spatial, cfg.head_dim, device=device)
    cos = torch.randn(cfg.batch, 1, cfg.n_spatial, cfg.head_dim, device=device)

    # Compute reference
    reference = rope_apply_reference(x, sin, cos)

    # Implementations
    implementations = [
        ("reference", rope_apply_reference),
        ("direct", rope_apply_direct),
        ("addcmul", rope_apply_addcmul),
        ("inplace", rope_apply_inplace),
    ]

    # === CORRECTNESS ===
    log.info("=== CORRECTNESS (max_abs_diff vs reference) ===")
    for name, impl in implementations:
        diff = measure_diff(impl, x, sin, cos, reference)
        status = "PASS" if diff <= cfg.atol else "FAIL"
        log.info(f"{name}: {diff:.2e} ({status})")
    log.info("")

    # === EAGER TIMING ===
    log.info("=== EAGER TIMING (ms per call) ===")
    for name, impl in implementations:
        t = bench_time(impl, x, sin, cos, cfg.warmup, cfg.iters, device)
        log.info(f"{name}: {t:.3f}")
    log.info("")

    # === COMPILED TIMING ===
    log.info("=== COMPILED TIMING (ms per call) ===")
    compile_impls = [
        ("reference", rope_apply_reference),
        ("direct", rope_apply_direct),
        ("addcmul", rope_apply_addcmul),
        # inplace breaks dynamo with out= on non-contiguous
    ]

    for name, impl in compile_impls:
        try:
            compiled = torch.compile(impl, fullgraph=True)
            # Warmup compile
            for _ in range(cfg.warmup):
                _ = compiled(x, sin, cos)
            sync_device(device)

            # Correctness
            diff = measure_diff(compiled, x, sin, cos, reference)
            status = "PASS" if diff <= cfg.atol else "FAIL"

            # Time
            t = bench_time(compiled, x, sin, cos, cfg.warmup, cfg.iters, device)
            log.info(f"{name}: {t:.3f} (diff={diff:.2e}, {status})")
        except Exception as e:
            log.info(f"{name}: FAILED ({type(e).__name__})")

    log.info("")
    log.info("=== inplace compiled ===")
    try:
        compiled = torch.compile(rope_apply_inplace, fullgraph=True)
        for _ in range(cfg.warmup):
            _ = compiled(x, sin, cos)
        sync_device(device)
        t = bench_time(compiled, x, sin, cos, cfg.warmup, cfg.iters, device)
        diff = measure_diff(compiled, x, sin, cos, reference)
        log.info(f"inplace: {t:.3f} (diff={diff:.2e})")
    except Exception as e:
        log.info(f"inplace: FAILED ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main(tyro.cli(Config))
