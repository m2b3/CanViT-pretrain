#!/usr/bin/env python3
"""Latency benchmark for AVP-ViT model components.

Compares:
- DINOv3 teacher (standard API) at glimpse and full image sizes
- Our ActiveCanViT model forward pass
- Multi-step trajectories

All timings at batch_size=1 for real-time inference scenarios.
Min latency is the key metric (true latency without system noise).

Usage:
    uv run python scripts/bench_latency.py --ckpt reference.pt
    uv run python scripts/bench_latency.py --ckpt reference.pt --compile
"""

import logging
import statistics
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable

import torch
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from avp_vit.checkpoint import load as load_ckpt, load_model
from avp_vit.train.viewpoint import Viewpoint as NamedViewpoint
from canvit.backbone.dinov3 import DINOv3Backbone
from canvit.hub import create_backbone
from canvit.model.active import ActiveCanViT
from canvit.viewpoint import Viewpoint
from ytch.device import sync_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


@dataclass
class Config:
    ckpt: str = "reference.pt"
    device: str = "cuda"
    amp: bool = True
    compile: bool = False
    glimpse_grid: int = 8
    canvas_grid: int = 32
    warmup: int = 20
    iters: int = 100
    n_steps: int = 4


@dataclass
class BenchResult:
    min_ms: float
    mean_ms: float
    std_ms: float
    max_ms: float
    n: int

    def __str__(self) -> str:
        return f"min={self.min_ms:.3f} mean={self.mean_ms:.3f}±{self.std_ms:.3f} max={self.max_ms:.3f}"


def bench(
    name: str,
    fn: Callable[[], object],
    device: torch.device,
    warmup: int,
    iters: int,
) -> BenchResult:
    """Benchmark a function, return detailed timing stats."""
    # Warmup
    for _ in tqdm(range(warmup), desc=f"{name} (warmup)", leave=False):
        fn()
    sync_device(device)

    # Timed iterations
    times_ms: list[float] = []
    for _ in tqdm(range(iters), desc=f"{name}", leave=False):
        sync_device(device)
        t0 = time.perf_counter()
        fn()
        sync_device(device)
        times_ms.append((time.perf_counter() - t0) * 1000)

    result = BenchResult(
        min_ms=min(times_ms),
        mean_ms=statistics.mean(times_ms),
        std_ms=statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        max_ms=max(times_ms),
        n=iters,
    )
    log.info(f"{name}: {result}")
    return result


def section(title: str) -> None:
    log.info("")
    log.info("=" * 60)
    log.info(title)
    log.info("=" * 60)


def make_viewpoint(B: int, device: torch.device, scale: float) -> Viewpoint:
    return Viewpoint(
        centers=torch.zeros(B, 2, device=device, dtype=torch.float32),
        scales=torch.full((B,), scale, device=device, dtype=torch.float32),
    )


def main() -> None:
    import tyro

    torch.set_float32_matmul_precision("high")
    cfg = tyro.cli(Config)
    console = Console()

    device = torch.device(cfg.device)
    B = 1  # Fixed for latency benchmarking

    log.info("=" * 60)
    log.info("CONFIGURATION")
    log.info("=" * 60)
    log.info(f"Checkpoint: {cfg.ckpt}")
    log.info(f"Device: {device}")
    log.info(f"AMP: {'bfloat16' if cfg.amp else 'disabled'}")
    log.info(f"Compile: {cfg.compile}")
    log.info(f"Batch size: {B} (fixed for latency)")
    log.info(f"Warmup: {cfg.warmup}, Iterations: {cfg.iters}")

    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if cfg.amp
        else nullcontext()
    )

    # =========================================================================
    # Load models
    # =========================================================================
    section("LOADING MODELS")

    model = load_model(cfg.ckpt, device)
    model.eval()
    assert isinstance(model, ActiveCanViT), f"Expected ActiveCanViT, got {type(model)}"
    log.info(f"Loaded model: {type(model).__name__}")

    ckpt = load_ckpt(cfg.ckpt, "cpu")
    backbone_name = ckpt["backbone"]
    assert isinstance(backbone_name, str)

    teacher = create_backbone(backbone_name, pretrained=True).to(device).eval()
    assert isinstance(teacher, DINOv3Backbone)
    log.info(f"Loaded teacher: {backbone_name}")

    # =========================================================================
    # Derived parameters
    # =========================================================================
    patch_size = model.backbone.patch_size_px
    n_blocks = model.backbone.n_blocks
    n_reads = len(model.read_attn)
    n_writes = len(model.write_attn)
    has_vpe = model.vpe_encoder is not None

    glimpse_px = cfg.glimpse_grid * patch_size
    full_img_px = cfg.canvas_grid * patch_size
    glimpse_tokens = cfg.glimpse_grid**2
    canvas_tokens = cfg.canvas_grid**2

    log.info("")
    log.info("Model architecture:")
    log.info(f"  Backbone: {backbone_name}")
    log.info(f"  Patch size: {patch_size}px")
    log.info(f"  Blocks: {n_blocks}")
    log.info(f"  Read/write layers: {n_reads}R / {n_writes}W")
    log.info(f"  VPE: {has_vpe}")
    log.info("")
    log.info("Benchmark parameters:")
    log.info(f"  Glimpse: {cfg.glimpse_grid}x{cfg.glimpse_grid} = {glimpse_tokens} tokens ({glimpse_px}px)")
    log.info(f"  Canvas: {cfg.canvas_grid}x{cfg.canvas_grid} = {canvas_tokens} tokens")
    log.info(f"  Full image: {full_img_px}x{full_img_px}px")

    # =========================================================================
    # Optional compilation
    # =========================================================================
    if cfg.compile:
        section("COMPILATION")
        log.info("Compiling model components...")
        model.compile()
        log.info("Compiling teacher blocks...")
        teacher.compile()
        log.info("Compilation complete")

    # =========================================================================
    # Create test inputs
    # =========================================================================
    section("CREATING TEST INPUTS")

    glimpse_img = torch.randn(B, 3, glimpse_px, glimpse_px, device=device)
    full_img = torch.randn(B, 3, full_img_px, full_img_px, device=device)
    state = model.init_state(batch_size=B, canvas_grid_size=cfg.canvas_grid)
    vp = make_viewpoint(B, device, scale=cfg.glimpse_grid / cfg.canvas_grid)

    log.info(f"Glimpse image: {tuple(glimpse_img.shape)}")
    log.info(f"Full image: {tuple(full_img.shape)}")
    log.info(f"Canvas: {tuple(state.canvas.shape)}")
    log.info(f"CLS: {tuple(state.recurrent_cls.shape)}")
    log.info(f"Viewpoint scale: {vp.scales.item():.3f}")

    # Verify shapes
    assert glimpse_img.shape == (B, 3, glimpse_px, glimpse_px)
    assert full_img.shape == (B, 3, full_img_px, full_img_px)
    assert state.canvas.shape[0] == B
    assert state.canvas.shape[1] == model.n_canvas_registers + canvas_tokens

    def run(name: str, fn: Callable[[], object]) -> BenchResult:
        return bench(name, fn, device, cfg.warmup, cfg.iters)

    results: dict[str, BenchResult] = {}

    # =========================================================================
    # TEACHER BASELINES (standard DINOv3 API)
    # =========================================================================
    section("TEACHER BASELINES (standard DINOv3 API: forward_norm_features)")

    with torch.no_grad(), amp_ctx:
        results["teacher_glimpse"] = run(
            f"Teacher @ glimpse ({glimpse_tokens} tokens)",
            lambda: teacher.forward_norm_features(glimpse_img),
        )
        results["teacher_full"] = run(
            f"Teacher @ full ({canvas_tokens} tokens)",
            lambda: teacher.forward_norm_features(full_img),
        )

    # =========================================================================
    # TEACHER COMPILED (torch.compile on forward_norm_features)
    # =========================================================================
    section("TEACHER COMPILED (torch.compile(forward_norm_features))")

    compiled_teacher_forward = torch.compile(teacher.forward_norm_features)

    with torch.no_grad(), amp_ctx:
        results["teacher_glimpse_compiled"] = run(
            f"Compiled Teacher @ glimpse ({glimpse_tokens} tokens)",
            lambda: compiled_teacher_forward(glimpse_img),
        )
        results["teacher_full_compiled"] = run(
            f"Compiled Teacher @ full ({canvas_tokens} tokens)",
            lambda: compiled_teacher_forward(full_img),
        )

    # =========================================================================
    # MODEL FORWARD (single step, no sampling)
    # =========================================================================
    section("MODEL FORWARD (single step, pre-sampled glimpse)")

    with torch.no_grad(), amp_ctx:
        results["model_forward"] = run(
            f"model.forward (glimpse={glimpse_tokens}, canvas={canvas_tokens})",
            lambda: model.forward(glimpse=glimpse_img, state=state, viewpoint=vp),
        )

    # =========================================================================
    # MODEL FORWARD_STEP (single step, with sampling)
    # =========================================================================
    section("MODEL FORWARD_STEP (single step, includes grid_sample)")

    named_vp = NamedViewpoint(name="bench", centers=vp.centers, scales=vp.scales)

    with torch.no_grad(), amp_ctx:
        results["model_step"] = run(
            "model.forward_step (sample + forward)",
            lambda: model.forward_step(
                image=full_img,
                state=state,
                viewpoint=named_vp,
                glimpse_size_px=glimpse_px,
            ),
        )

    # =========================================================================
    # PREDICTION HEADS (used in real inference, not part of forward)
    # =========================================================================
    section("PREDICTION HEADS (post-forward inference overhead)")

    with torch.no_grad(), amp_ctx:
        sample_out = model.forward_step(
            image=full_img,
            state=state,
            viewpoint=named_vp,
            glimpse_size_px=glimpse_px,
        )

    log.info("These run AFTER forward_step in real inference:")

    with torch.no_grad(), amp_ctx:
        results["predict_scene"] = run(
            "predict_teacher_scene",
            lambda: model.predict_teacher_scene(sample_out.state.canvas),
        )
        results["predict_cls"] = run(
            "predict_scene_teacher_cls",
            lambda: model.predict_scene_teacher_cls(sample_out.state.recurrent_cls, sample_out.state.canvas),
        )

        if model.policy is not None and sample_out.vpe is not None:
            results["policy"] = run(
                "policy head",
                lambda: model.policy(sample_out.vpe),
            )
        else:
            results["policy"] = BenchResult(0.0, 0.0, 0.0, 0.0, 0)
            log.info("Policy head: disabled")

    # =========================================================================
    # FULL INFERENCE PIPELINE (matches inference_app.gpu_worker.step)
    # =========================================================================
    section("FULL INFERENCE PIPELINE (forward_step + predictions + policy)")

    def full_inference_step() -> None:
        out = model.forward_step(
            image=full_img,
            state=state,
            viewpoint=named_vp,
            glimpse_size_px=glimpse_px,
        )
        _ = model.predict_teacher_scene(out.state.canvas)
        _ = model.predict_scene_teacher_cls(out.state.recurrent_cls, out.state.canvas)
        if model.policy is not None and out.vpe is not None:
            _ = model.policy(out.vpe)

    with torch.no_grad(), amp_ctx:
        results["full_pipeline"] = run("full inference step", full_inference_step)

    # =========================================================================
    # MULTI-STEP TRAJECTORY (state persists across steps, init outside timing)
    # =========================================================================
    section(f"MULTI-STEP TRAJECTORY ({cfg.n_steps} steps, state persists)")

    traj_state = model.init_state(batch_size=B, canvas_grid_size=cfg.canvas_grid)

    def run_trajectory() -> None:
        nonlocal traj_state
        for _ in range(cfg.n_steps):
            out = model.forward_step(
                image=full_img,
                state=traj_state,
                viewpoint=named_vp,
                glimpse_size_px=glimpse_px,
            )
            traj_state = out.state

    with torch.no_grad(), amp_ctx:
        results["trajectory"] = run(
            f"{cfg.n_steps}-step trajectory",
            run_trajectory,
        )

    # =========================================================================
    # COMPILED FULL FORWARD (if requested)
    # =========================================================================
    if cfg.compile:
        section("FULLY COMPILED FORWARD")
        log.info("Compiling model.forward and model.forward_step...")

        compiled_forward = torch.compile(model.forward)
        compiled_step = torch.compile(model.forward_step)

        with torch.no_grad(), amp_ctx:
            results["compiled_forward"] = run(
                "torch.compile(model.forward)",
                lambda: compiled_forward(glimpse=glimpse_img, state=state, viewpoint=vp),
            )
            results["compiled_step"] = run(
                "torch.compile(model.forward_step)",
                lambda: compiled_step(
                    image=full_img,
                    state=state,
                    viewpoint=named_vp,
                    glimpse_size_px=glimpse_px,
                ),
            )

    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    section("SUMMARY")

    # Main results table
    table = Table(title=f"Latency Results (n={cfg.iters}, warmup={cfg.warmup})")
    table.add_column("Operation", style="bold")
    table.add_column("Min (ms)", justify="right", style="green")
    table.add_column("Mean±Std (ms)", justify="right")
    table.add_column("Max (ms)", justify="right", style="dim")

    def add_row(name: str, key: str) -> None:
        r = results[key]
        table.add_row(name, f"{r.min_ms:.3f}", f"{r.mean_ms:.3f}±{r.std_ms:.3f}", f"{r.max_ms:.3f}")

    add_row(f"Teacher @ glimpse ({glimpse_tokens} tok)", "teacher_glimpse")
    add_row("Teacher @ glimpse compiled", "teacher_glimpse_compiled")
    add_row(f"Teacher @ full ({canvas_tokens} tok)", "teacher_full")
    add_row("Teacher @ full compiled", "teacher_full_compiled")
    table.add_section()
    add_row("Model forward", "model_forward")
    add_row("Model forward_step", "model_step")
    add_row("Full pipeline", "full_pipeline")
    table.add_section()
    add_row("predict_teacher_scene", "predict_scene")
    add_row("predict_scene_teacher_cls", "predict_cls")
    add_row("policy head", "policy")
    table.add_section()
    add_row(f"{cfg.n_steps}-step trajectory", "trajectory")

    if cfg.compile:
        table.add_section()
        add_row("Compiled forward", "compiled_forward")
        add_row("Compiled step", "compiled_step")

    console.print()
    console.print(table)

    # Comparisons table - apples to apples
    def make_cmp_table(title: str, baseline_glimpse: float, baseline_full: float, model_key: str) -> Table:
        cmp = Table(title=title)
        cmp.add_column("Comparison", style="bold")
        cmp.add_column("Δ (ms)", justify="right")
        cmp.add_column("Ratio", justify="right")

        def add_cmp(name: str, val: float, base: float) -> None:
            delta = val - base
            ratio = val / base
            style = "green" if delta < 0 else "red" if ratio > 2 else ""
            cmp.add_row(name, f"{delta:+.3f}", f"{ratio:.2f}x", style=style)

        add_cmp("Model forward vs Teacher@glimpse", results[model_key].min_ms, baseline_glimpse)
        add_cmp("Full pipeline vs Teacher@glimpse", results["full_pipeline"].min_ms, baseline_glimpse)

        # N-step vs N × teacher@glimpse
        n_baseline = cfg.n_steps * baseline_glimpse
        add_cmp(f"{cfg.n_steps}-step vs {cfg.n_steps}×Teacher@glimpse", results["trajectory"].min_ms, n_baseline)

        # N-step vs teacher@full
        add_cmp(f"{cfg.n_steps}-step vs Teacher@full", results["trajectory"].min_ms, baseline_full)

        return cmp

    # Uncompiled comparisons
    console.print()
    console.print(make_cmp_table(
        "Comparisons: Uncompiled (min latency)",
        results["teacher_glimpse"].min_ms,
        results["teacher_full"].min_ms,
        "model_forward",
    ))

    # Compiled comparisons (if available)
    if cfg.compile:
        console.print()
        console.print(make_cmp_table(
            "Comparisons: Compiled (min latency)",
            results["teacher_glimpse_compiled"].min_ms,
            results["teacher_full_compiled"].min_ms,
            "compiled_forward",
        ))

    # Per-step amortized
    per_step = results["trajectory"].min_ms / cfg.n_steps
    console.print()
    console.print(f"[bold]Per-step amortized (min):[/bold] {per_step:.3f}ms")

    console.print()
    console.print("[dim]Note: Min latency is the key metric - represents true latency without system noise.[/dim]")


if __name__ == "__main__":
    main()
