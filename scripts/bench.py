"""Unified benchmark script for CanViT throughput and latency.

This script replaces 8 fragmented profiling scripts that accumulated during
performance investigation. It measures Teacher vs CanViT overhead with proper
methodology for both throughput and latency scenarios.

Usage:
    uv run python scripts/bench.py                              # Throughput (BS=64)
    uv run python scripts/bench.py --latency                    # Latency (BS=1)
    uv run python scripts/bench.py --no-compile                 # Compare eager vs compiled
    uv run python scripts/bench.py --time-budget-s 2            # Longer runs

Design Decisions:

1. CONFIG INHERITANCE (not hardcoding):
   All config values (batch_size, glimpse_grid, canvas_grid, teacher_model)
   default to None and inherit from TrainConfig() in __post_init__. This ensures
   the benchmark uses THE SAME config as training - if training config changes,
   benchmark automatically picks it up. Hardcoded values would drift silently.

2. THROUGHPUT vs LATENCY SYNC STRATEGY:
   - Throughput (default): Sync only at boundaries (before warmup, after all iters).
     Per-iter sync would kill GPU async pipelining and give misleading results.
     We only get accurate mean (total_time / n_iters), not individual iter times.

   - Latency (--latency): Sync before AND after each iteration. This is correct
     for latency measurement - in generation, you WAIT for each output before
     proceeding. Gives accurate min/max/mean/std for individual iterations.

3. TIME BUDGET (not fixed iter count):
   Run each benchmark for N seconds (default 1.0s) rather than fixed iterations.
   This keeps total wallclock predictable regardless of how fast/slow the model is.
   Useful when iterating quickly and you might Ctrl-C.

4. TQDM EVERYWHERE:
   All loops (warmup and timed) have tqdm progress bars. Your wallclock time is
   valuable - you should always see what's happening. Latency mode shows last_ms
   in the progress bar for immediate feedback.

5. OPTIONAL STATS (no lying to data structures):
   BenchStats has Optional[float] for min/max/std. In throughput mode these are
   None because we genuinely don't have per-iter times. Previous scripts would
   set min=max=mean which is misleading.

6. TRY-EXCEPT SAFETY:
   Each benchmark is wrapped in try-except. If one fails (e.g., CUDAGraph
   incompatibility during iteration), others still run. Useful when experimenting
   with compile modes that might break.

7. LOGGING FROM ACTUAL VALUES:
   All torch options are stored in variables, then applied, then logged with
   f-strings referencing those variables. This prevents the brittle pattern of:
       torch.set_foo(True)
       log.info("foo = True")  # Can go out of sync if you change the first line

8. LATENCY BS=1 BY DEFINITION:
   When --latency is passed, batch_size is FORCED to 1 in __post_init__.
   Latency with BS>1 is meaningless - you're measuring something else.
   This is not configurable to prevent user error.

Target platforms: CUDA (primary), CPU, MPS (nice-to-have, may behave oddly).
"""

import logging
import statistics
import time
from dataclasses import dataclass, field, fields
from typing import Any, Callable

import torch
import tyro
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from avp_vit import ACVFRP, ACVFRPConfig
from avp_vit.train.config import Config as TrainConfig
from canvit import create_backbone
from canvit.viewpoint import Viewpoint


logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# Placeholder teacher_dim (ViT-B). Overwritten at runtime with actual teacher.embed_dim.
_PLACEHOLDER_TEACHER_DIM = 768


def _default_model_config() -> ACVFRPConfig:
    """Model config with placeholder teacher_dim (fixed at runtime)."""
    return ACVFRPConfig(teacher_dim=_PLACEHOLDER_TEACHER_DIM)


def _is_dataclass_instance(obj: Any) -> bool:
    """Check if obj is a dataclass instance (not the class itself)."""
    return hasattr(obj, "__dataclass_fields__")


def _log_config_diff(
    cfg: Any, default_cfg: Any, prefix: str = "", indent: int = 2
) -> None:
    """Recursively log config fields, highlighting overrides. Handles nested dataclasses."""
    pad = " " * indent
    for f in fields(cfg):
        val = getattr(cfg, f.name)
        default_val = getattr(default_cfg, f.name)
        field_name = f"{prefix}{f.name}" if prefix else f.name

        if _is_dataclass_instance(val) and _is_dataclass_instance(default_val):
            # Recurse into nested dataclass
            log.info(f"{pad}{field_name}:")
            _log_config_diff(val, default_val, prefix="", indent=indent + 2)
        elif val != default_val:
            log.info(f"{pad}[OVERRIDE] {field_name} = {val} (default: {default_val})")
        else:
            log.info(f"{pad}{field_name} = {val}")


@dataclass
class BenchStats:
    """Timing statistics from a benchmark run."""

    n_iters: int
    total_s: float
    mean_ms: float
    min_ms: float | None = None
    max_ms: float | None = None
    std_ms: float | None = None

    @property
    def iters_per_sec(self) -> float:
        return self.n_iters / self.total_s if self.total_s > 0 else 0.0


@dataclass
class BenchConfig:
    """Benchmark configuration. Defaults inherited from TrainConfig()."""

    # Inherited from training config - None means use TrainConfig default
    batch_size: int | None = None
    glimpse_grid: int | None = None
    canvas_grid: int | None = None
    teacher_model: str | None = None

    # Model config - exposed for ablations
    model: ACVFRPConfig = field(default_factory=_default_model_config)

    # Bench-specific
    warmup_iters: int = 5
    time_budget_s: float = 1.0
    device: str = "cuda"
    matmul_precision: str = "medium"

    # Modes
    latency: bool = False
    no_compile: bool = False

    def __post_init__(self) -> None:
        """Inherit defaults from training config."""
        train_cfg = TrainConfig()
        if self.batch_size is None:
            self.batch_size = train_cfg.batch_size
        if self.glimpse_grid is None:
            self.glimpse_grid = train_cfg.glimpse_grid_size
        if self.canvas_grid is None:
            self.canvas_grid = train_cfg.grid_size
        if self.teacher_model is None:
            self.teacher_model = train_cfg.teacher_model

        # Latency mode: BS=1 by DEFINITION
        if self.latency:
            self.batch_size = 1


def sync(device: torch.device) -> None:
    """Sync GPU if CUDA."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def benchmark_throughput(
    fn: Callable[[], Any],
    name: str,
    warmup_iters: int,
    time_budget_s: float,
    device: torch.device,
) -> BenchStats:
    """Throughput benchmark: boundary sync only, time budget."""
    # Warmup
    for _ in tqdm(range(warmup_iters), desc=f"{name} warmup", leave=False):
        fn()
    sync(device)

    # Timed - run until budget elapsed
    n_iters = 0
    t_start = time.perf_counter()
    deadline = t_start + time_budget_s

    with tqdm(desc=name, unit="it", leave=False) as pbar:
        while time.perf_counter() < deadline:
            fn()
            n_iters += 1
            pbar.update(1)
    sync(device)
    total_s = time.perf_counter() - t_start

    return BenchStats(
        n_iters=n_iters,
        total_s=total_s,
        mean_ms=(total_s / n_iters) * 1000 if n_iters > 0 else 0.0,
    )


def benchmark_latency(
    fn: Callable[[], Any],
    name: str,
    warmup_iters: int,
    time_budget_s: float,
    device: torch.device,
) -> BenchStats:
    """Latency benchmark: per-iter sync, full stats."""
    # Warmup
    for _ in tqdm(range(warmup_iters), desc=f"{name} warmup", leave=False):
        fn()
        sync(device)

    # Timed - collect individual times
    times: list[float] = []
    t_start = time.perf_counter()
    deadline = t_start + time_budget_s

    with tqdm(desc=name, unit="it", leave=False) as pbar:
        while time.perf_counter() < deadline:
            sync(device)
            t0 = time.perf_counter()
            fn()
            sync(device)
            times.append((time.perf_counter() - t0) * 1000)
            pbar.update(1)
            pbar.set_postfix(last_ms=f"{times[-1]:.2f}")

    total_s = time.perf_counter() - t_start

    return BenchStats(
        n_iters=len(times),
        total_s=total_s,
        mean_ms=statistics.mean(times) if times else 0.0,
        min_ms=min(times) if times else None,
        max_ms=max(times) if times else None,
        std_ms=statistics.stdev(times) if len(times) > 1 else None,
    )


def run_benchmark_safe(
    fn: Callable[[], Any],
    name: str,
    bench_fn: Callable[..., BenchStats],
    warmup_iters: int,
    time_budget_s: float,
    device: torch.device,
) -> BenchStats | None:
    """Run benchmark with try-except so failures don't kill everything."""
    try:
        return bench_fn(fn, name, warmup_iters, time_budget_s, device)
    except Exception as e:
        log.error(f"FAILED: {name}: {e}")
        return None


def print_summary(
    results: dict[str, BenchStats | None], cfg: BenchConfig, console: Console
) -> None:
    """Print results as rich table."""
    console.print()
    console.rule("[bold]RESULTS[/bold]")

    mode = "Latency" if cfg.latency else "Throughput"
    table = Table(title=f"{mode} Benchmark (BS={cfg.batch_size})")
    table.add_column("Benchmark", style="bold")
    table.add_column("Iters", justify="right")
    table.add_column("Mean (ms)", justify="right")

    if cfg.latency:
        table.add_column("Min (ms)", justify="right", style="green")
        table.add_column("Max (ms)", justify="right")
        table.add_column("Std (ms)", justify="right")

    for name, stats in results.items():
        if stats is None:
            if cfg.latency:
                table.add_row(name, "FAILED", "-", "-", "-", "-")
            else:
                table.add_row(name, "FAILED", "-")
            continue

        if cfg.latency:
            table.add_row(
                name,
                str(stats.n_iters),
                f"{stats.mean_ms:.2f}",
                f"{stats.min_ms:.2f}" if stats.min_ms is not None else "-",
                f"{stats.max_ms:.2f}" if stats.max_ms is not None else "-",
                f"{stats.std_ms:.2f}" if stats.std_ms is not None else "-",
            )
        else:
            table.add_row(name, str(stats.n_iters), f"{stats.mean_ms:.2f}")

    console.print(table)

    # Overhead ratio
    teacher = results.get("Teacher.forward_norm_features")
    canvit = results.get("CanViT.forward+predict")
    if teacher and canvit:
        if cfg.latency and teacher.min_ms and canvit.min_ms:
            overhead = canvit.min_ms / teacher.min_ms
            console.print(f"\n[bold]Overhead (min):[/bold] x{overhead:.2f}")
        else:
            overhead = canvit.mean_ms / teacher.mean_ms
            console.print(f"\n[bold]Overhead (mean):[/bold] x{overhead:.2f}")


def main(cfg: BenchConfig) -> None:
    console = Console()
    device = torch.device(cfg.device)

    # Log environment
    console.rule("[bold]ENVIRONMENT[/bold]")
    log.info(f"PyTorch:      {torch.__version__}")
    if device.type == "cuda":
        log.info(f"CUDA Device:  {torch.cuda.get_device_name()}")

    # Set torch options - values defined once, then applied and logged
    log.info("")
    log.info("Torch options:")

    torch.set_float32_matmul_precision(cfg.matmul_precision)
    log.info(f"  float32_matmul_precision = {cfg.matmul_precision}")

    sdpa_flash = True
    sdpa_mem_eff = False
    sdpa_math = False
    torch.backends.cuda.enable_flash_sdp(sdpa_flash)
    torch.backends.cuda.enable_mem_efficient_sdp(sdpa_mem_eff)
    torch.backends.cuda.enable_math_sdp(sdpa_math)
    log.info(f"  SDPA: flash={sdpa_flash}, mem_efficient={sdpa_mem_eff}, math={sdpa_math}")

    # Log benchmark config
    console.print()
    console.rule("[bold]BENCHMARK CONFIGURATION[/bold]")
    log.info(f"Mode:         {'latency' if cfg.latency else 'throughput'}")
    log.info(f"Batch size:   {cfg.batch_size}")
    log.info(f"Glimpse:      {cfg.glimpse_grid}x{cfg.glimpse_grid}")
    log.info(f"Canvas:       {cfg.canvas_grid}x{cfg.canvas_grid}")
    log.info(f"Teacher:      {cfg.teacher_model}")
    log.info(f"Time budget:  {cfg.time_budget_s}s per benchmark")
    log.info(f"Warmup:       {cfg.warmup_iters} iters")
    log.info(f"Compiled:     {not cfg.no_compile}")

    # Create models
    console.print()
    console.rule("[bold]MODEL SETUP[/bold]")

    # These are guaranteed non-None after __post_init__
    assert cfg.teacher_model is not None
    assert cfg.batch_size is not None
    assert cfg.glimpse_grid is not None
    assert cfg.canvas_grid is not None

    log.info("Creating teacher...")
    teacher = create_backbone(cfg.teacher_model, pretrained=False).to(device).eval()
    log.info(f"  {teacher.n_blocks} blocks, dim={teacher.embed_dim}")

    log.info("Creating CanViT...")
    backbone = create_backbone(cfg.teacher_model, pretrained=False)
    # Update teacher_dim to match actual teacher (in case default 768 differs)
    cfg.model.teacher_dim = teacher.embed_dim
    model = ACVFRP(backbone=backbone, cfg=cfg.model, policy=None).to(device).eval()
    log.info(f"  canvas_dim={cfg.model.canvas_dim}, read_after={model.read_after_blocks}")

    # Log model config (highlight non-defaults, recurse into nested dataclasses)
    log.info("")
    log.info("Model config:")
    default_cfg = _default_model_config()
    default_cfg.teacher_dim = teacher.embed_dim  # Fair comparison
    _log_config_diff(cfg.model, default_cfg)

    # Compile
    if not cfg.no_compile:
        log.info("Compiling teacher...")
        teacher.compile()
        log.info("Compiling CanViT...")
        model.compile()
    else:
        log.info("Skipping compilation (--no-compile)")

    # Create inputs
    patch_size = teacher.patch_size_px
    glimpse_px = cfg.glimpse_grid * patch_size

    glimpse = torch.randn(cfg.batch_size, 3, glimpse_px, glimpse_px, device=device)
    viewpoint = Viewpoint(
        centers=torch.zeros(cfg.batch_size, 2, device=device),
        scales=torch.ones(cfg.batch_size, device=device),
    )
    state = model.init_state(batch_size=cfg.batch_size, canvas_grid_size=cfg.canvas_grid)

    log.info("")
    log.info("Input shapes:")
    log.info(f"  glimpse:  {list(glimpse.shape)}")
    log.info(f"  canvas:   {list(state.canvas.shape)}")

    # Select benchmark function
    bench_fn = benchmark_latency if cfg.latency else benchmark_throughput

    # Define benchmarks
    def teacher_fn() -> Any:
        return teacher.forward_norm_features(glimpse)

    def canvit_forward_fn() -> Any:
        return model.forward(glimpse=glimpse, state=state, viewpoint=viewpoint)

    def canvit_full_fn() -> Any:
        out = model.forward(glimpse=glimpse, state=state, viewpoint=viewpoint)
        return model.predict_teacher_scene(out.state.canvas)

    benchmarks = [
        ("Teacher.forward_norm_features", teacher_fn),
        ("CanViT.forward", canvit_forward_fn),
        ("CanViT.forward+predict", canvit_full_fn),
    ]

    # Run benchmarks
    console.print()
    console.rule("[bold]RUNNING BENCHMARKS[/bold]")
    results: dict[str, BenchStats | None] = {}

    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
        for name, fn in benchmarks:
            results[name] = run_benchmark_safe(
                fn, name, bench_fn, cfg.warmup_iters, cfg.time_budget_s, device
            )

    # Print results
    print_summary(results, cfg, console)


if __name__ == "__main__":
    main(tyro.cli(BenchConfig))
