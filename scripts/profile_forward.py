"""Comprehensive CanViT overhead investigation.

Tests multiple hypotheses in ONE run to minimize H100 time:
1. Eager baselines (teacher and canvit)
2. Component compile with different modes (default, max-autotune, reduce-overhead)
3. Full forward compile with different modes
4. Graph break analysis
5. Kernel profiling

Usage:
    uv run python scripts/profile_forward.py --device cuda
    uv run python scripts/profile_forward.py --device cuda --modes default max-autotune
    uv run python scripts/profile_forward.py --device cuda --skip-profile  # faster, no kernel profile
"""

import gc
import logging
import time
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch._dynamo as dynamo
import tyro

from avp_vit import ActiveCanViT, ActiveCanViTConfig
from canvit.hub import create_backbone
from canvit.viewpoint import Viewpoint

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

COMPILE_MODES = ["default", "max-autotune"]


@dataclass(frozen=True)
class Config:
    device: str = "cuda"
    batch: int = 64
    glimpse_grid: int = 8
    canvas_grid: int = 32
    patch_size: int = 16
    warmup: int = 5
    iters: int = 30
    modes: list[str] = field(default_factory=lambda: COMPILE_MODES)
    skip_profile: bool = False  # Skip kernel profiling for faster runs
    skip_graph_breaks: bool = False  # Skip graph break analysis


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench(fn: Callable, warmup: int, iters: int, device: torch.device) -> float:
    """Benchmark with proper sync. Returns ms/iter."""
    for _ in range(warmup):
        fn()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(device)
    return (time.perf_counter() - t0) / iters * 1000


def clear_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_compile_kwargs(mode: str) -> dict:
    """Get torch.compile kwargs for a given mode."""
    if mode == "default":
        return {}
    return {"mode": mode}


def create_fresh_model(device: torch.device, model_cfg: ActiveCanViTConfig):
    """Create a fresh model instance (important: avoids state pollution between tests)."""
    backbone = create_backbone("dinov3_vitb16", pretrained=False)
    model = ActiveCanViT(backbone=backbone, cfg=model_cfg, policy=None)
    model.to(device).eval()
    return model


def main(cfg: Config) -> None:
    device = torch.device(cfg.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        log.info(f"GPU: {torch.cuda.get_device_name()}")

    glimpse_px = cfg.glimpse_grid * cfg.patch_size

    log.info("=" * 70)
    log.info("COMPREHENSIVE CANVIT OVERHEAD INVESTIGATION")
    log.info("=" * 70)
    log.info(
        f"Batch: {cfg.batch}, Glimpse: {cfg.glimpse_grid}x{cfg.glimpse_grid}, Canvas: {cfg.canvas_grid}x{cfg.canvas_grid}"
    )
    log.info(f"Compile modes to test: {cfg.modes}")
    log.info(f"Warmup: {cfg.warmup}, Iters: {cfg.iters}")
    log.info("")

    # Suppress noisy logs
    logging.getLogger("dinov3").setLevel(logging.WARNING)
    logging.getLogger("canvit").setLevel(logging.WARNING)

    # Create inputs once
    glimpse = torch.randn(cfg.batch, 3, glimpse_px, glimpse_px, device=device)
    vp = Viewpoint(
        centers=torch.zeros(cfg.batch, 2, device=device),
        scales=torch.ones(cfg.batch, device=device),
    )

    # Model config (reused across tests)
    backbone_tmp = create_backbone("dinov3_vitb16", pretrained=False)
    model_cfg = ActiveCanViTConfig(teacher_dim=backbone_tmp.embed_dim)
    del backbone_tmp

    results: dict[str, float] = {}

    # =========================================================================
    # TEST 1: Teacher baselines
    # =========================================================================
    log.info("=" * 70)
    log.info("TEST 1: Teacher (DINOv3 backbone only)")
    log.info("=" * 70)

    teacher = create_backbone("dinov3_vitb16", pretrained=False).to(device).eval()

    def teacher_fn(t=teacher):
        return t.forward_norm_features(glimpse)

    # Teacher eager
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        results["teacher_eager"] = bench(teacher_fn, cfg.warmup, cfg.iters, device)
    log.info(f"  Eager:    {results['teacher_eager']:.2f} ms")

    # Teacher compiled (default mode)
    teacher.compile()
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        results["teacher_compiled"] = bench(teacher_fn, cfg.warmup, cfg.iters, device)
    log.info(f"  Compiled: {results['teacher_compiled']:.2f} ms")
    log.info("")

    del teacher, teacher_fn
    clear_cache()

    # =========================================================================
    # TEST 2: CanViT EAGER (no compilation at all)
    # =========================================================================
    log.info("=" * 70)
    log.info("TEST 2: CanViT EAGER (no compilation)")
    log.info("=" * 70)

    model_eager = create_fresh_model(device, model_cfg)
    state_eager = model_eager.init_state(
        batch_size=cfg.batch, canvas_grid_size=cfg.canvas_grid
    )

    def eager_fn(m=model_eager, s=state_eager):
        return m.forward(glimpse=glimpse, state=s, viewpoint=vp)

    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        results["canvit_eager"] = bench(eager_fn, cfg.warmup, cfg.iters, device)
    log.info(f"  Time: {results['canvit_eager']:.2f} ms")
    log.info(
        f"  vs Teacher eager: x{results['canvit_eager'] / results['teacher_eager']:.2f}"
    )
    log.info("")

    del model_eager, state_eager, eager_fn
    clear_cache()

    # =========================================================================
    # TEST 3: CanViT COMPONENT COMPILE (different modes)
    # =========================================================================
    log.info("=" * 70)
    log.info("TEST 3: CanViT COMPONENT COMPILE (backbone + attn + rope)")
    log.info("=" * 70)

    for mode in cfg.modes:
        dynamo.reset()
        clear_cache()

        model = create_fresh_model(device, model_cfg)
        compile_kwargs = get_compile_kwargs(mode)

        log.info(f"  Mode: {mode}")
        model.compile(**compile_kwargs)
        state = model.init_state(batch_size=cfg.batch, canvas_grid_size=cfg.canvas_grid)

        def comp_fn(m=model, s=state):
            return m.forward(glimpse=glimpse, state=s, viewpoint=vp)

        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            result = bench(comp_fn, cfg.warmup, cfg.iters, device)
        results[f"canvit_component_{mode}"] = result
        overhead = result / results["teacher_compiled"]
        log.info(f"    Time: {result:.2f} ms (x{overhead:.2f} vs teacher compiled)")

        del model, state, comp_fn
        clear_cache()

    log.info("")

    # =========================================================================
    # TEST 4: CanViT FULL FORWARD COMPILE (different modes)
    # =========================================================================
    log.info("=" * 70)
    log.info("TEST 4: CanViT FULL FORWARD COMPILE (wrap entire forward)")
    log.info("=" * 70)

    for mode in cfg.modes:
        dynamo.reset()
        clear_cache()

        model = create_fresh_model(device, model_cfg)
        # Don't call model.compile() - compile the entire forward instead
        state = model.init_state(batch_size=cfg.batch, canvas_grid_size=cfg.canvas_grid)

        compile_kwargs = get_compile_kwargs(mode)
        compiled_forward = torch.compile(model.forward, **compile_kwargs)

        def full_fn(cf=compiled_forward, s=state):
            return cf(glimpse=glimpse, state=s, viewpoint=vp)

        log.info(f"  Mode: {mode} (warming up...)")
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            for _ in range(cfg.warmup):
                full_fn()
        sync(device)

        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            result = bench(full_fn, cfg.warmup, cfg.iters, device)
        results[f"canvit_full_{mode}"] = result
        overhead = result / results["teacher_compiled"]
        log.info(f"    Time: {result:.2f} ms (x{overhead:.2f} vs teacher compiled)")

        del model, state, compiled_forward, full_fn
        clear_cache()

    log.info("")

    # =========================================================================
    # TEST 5: Graph break analysis
    # =========================================================================
    if not cfg.skip_graph_breaks:
        log.info("=" * 70)
        log.info("TEST 5: Graph break analysis")
        log.info("=" * 70)

        dynamo.reset()
        clear_cache()
        graph_count = [0]

        def count_backend(gm, example_inputs):
            graph_count[0] += 1
            return gm

        model_gb = create_fresh_model(device, model_cfg)
        state_gb = model_gb.init_state(
            batch_size=cfg.batch, canvas_grid_size=cfg.canvas_grid
        )

        compiled_count = torch.compile(model_gb.forward, backend=count_backend)

        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            _ = compiled_count(glimpse=glimpse, state=state_gb, viewpoint=vp)

        log.info(f"  Graph fragments when compiling model.forward: {graph_count[0]}")
        if graph_count[0] > 1:
            log.info(f"  ⚠️  {graph_count[0] - 1} graph break(s) detected!")
        else:
            log.info("  ✓ Single graph (no breaks)")

        del model_gb, state_gb
        clear_cache()
        log.info("")

    # =========================================================================
    # TEST 6: Kernel profiling (best component mode)
    # =========================================================================
    if not cfg.skip_profile:
        log.info("=" * 70)
        log.info("TEST 6: Kernel profiling")
        log.info("=" * 70)

        # Find best component mode
        best_component_mode = min(
            cfg.modes, key=lambda m: results.get(f"canvit_component_{m}", float("inf"))
        )
        log.info(f"  Profiling component compile with mode: {best_component_mode}")

        dynamo.reset()
        clear_cache()

        model_prof = create_fresh_model(device, model_cfg)
        model_prof.compile(**get_compile_kwargs(best_component_mode))
        state_prof = model_prof.init_state(
            batch_size=cfg.batch, canvas_grid_size=cfg.canvas_grid
        )

        def prof_fn(m=model_prof, s=state_prof):
            return m.forward(glimpse=glimpse, state=s, viewpoint=vp)

        # Warmup
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            for _ in range(cfg.warmup):
                prof_fn()
        sync(device)

        # Profile
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
        ) as prof:
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                for _ in range(3):
                    prof_fn()

        log.info("  Top 15 CUDA kernels by time:")
        table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=15)
        for line in table.split("\n"):
            log.info(f"  {line}")

        del model_prof, state_prof, prof_fn
        clear_cache()
        log.info("")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    log.info("=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    log.info("")

    log.info("Baselines:")
    log.info(f"  Teacher eager:     {results['teacher_eager']:7.2f} ms")
    log.info(f"  Teacher compiled:  {results['teacher_compiled']:7.2f} ms")
    log.info(
        f"  CanViT eager:      {results['canvit_eager']:7.2f} ms  (x{results['canvit_eager'] / results['teacher_eager']:.2f} vs teacher eager)"
    )
    log.info("")

    log.info("Component compile (model.compile()):")
    for mode in cfg.modes:
        key = f"canvit_component_{mode}"
        if key in results:
            overhead = results[key] / results["teacher_compiled"]
            log.info(f"  {mode:20s}: {results[key]:7.2f} ms  (x{overhead:.2f})")
    log.info("")

    log.info("Full forward compile (torch.compile(model.forward)):")
    for mode in cfg.modes:
        key = f"canvit_full_{mode}"
        if key in results:
            overhead = results[key] / results["teacher_compiled"]
            log.info(f"  {mode:20s}: {results[key]:7.2f} ms  (x{overhead:.2f})")
    log.info("")

    log.info("Expected overhead: x1.26-1.40 (from FLOP analysis)")
    log.info("")

    # Find best results
    best_component = min(
        [(m, results.get(f"canvit_component_{m}", float("inf"))) for m in cfg.modes],
        key=lambda x: x[1],
    )
    best_full = min(
        [(m, results.get(f"canvit_full_{m}", float("inf"))) for m in cfg.modes],
        key=lambda x: x[1],
    )

    log.info(
        f"BEST component: {best_component[0]} @ {best_component[1]:.2f} ms (x{best_component[1] / results['teacher_compiled']:.2f})"
    )
    log.info(
        f"BEST full:      {best_full[0]} @ {best_full[1]:.2f} ms (x{best_full[1] / results['teacher_compiled']:.2f})"
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
