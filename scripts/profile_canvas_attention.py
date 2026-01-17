"""Profile CanViT vs Teacher on CUDA - REAL training workload.

Run on H100:
    python scripts/profile_canvas_attention.py

Outputs:
    - Console summary of top CUDA kernels
    - Chrome trace at profile_canvit.json
"""

import torch
from torch.profiler import profile, ProfilerActivity, record_function

from canvit.hub import create_backbone
from canvit.viewpoint import Viewpoint, sample_at_viewpoint
from avp_vit import ActiveCanViT, ActiveCanViTConfig

# =============================================================================
# EXACT TRAINING CONFIG - from train/config.py
# =============================================================================
BATCH = 64
GLIMPSE_GRID = 8
CANVAS_GRID = 32
PATCH_SIZE = 16
GLIMPSE_PX = GLIMPSE_GRID * PATCH_SIZE  # 128

N_WARMUP = 10
N_PROFILE = 10


def main():
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")

    # Force FlashAttention only (match training)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)

    print("=" * 70)
    print("CONFIGURATION (matches train/config.py)")
    print("=" * 70)
    print(f"Batch:        {BATCH}")
    print(f"Glimpse:      {GLIMPSE_GRID}x{GLIMPSE_GRID} = {GLIMPSE_GRID**2} patches @ {GLIMPSE_PX}px")
    print(f"Canvas:       {CANVAS_GRID}x{CANVAS_GRID} = {CANVAS_GRID**2} spatial tokens")
    print()

    # Create teacher and student
    print("Creating models...")
    teacher = create_backbone("dinov3_vitb16", pretrained=False).to(device).eval()

    backbone = create_backbone("dinov3_vitb16", pretrained=False)
    cfg = ActiveCanViTConfig(teacher_dim=teacher.embed_dim)
    model = ActiveCanViT(backbone=backbone, cfg=cfg, policy=None).to(device).eval()

    # Print architecture info
    print(f"Teacher: DINOv3 ViT-B/16, {teacher.n_blocks} blocks, dim={teacher.embed_dim}")
    print(f"CanViT:  {model.backbone.n_blocks} blocks, canvas_dim={cfg.canvas_dim}")
    print(f"  read_after:  {model.read_after_blocks}")
    print(f"  write_after: {model.write_after_blocks}")
    print(f"  n_read_attn: {len(model.read_attn)}, n_write_attn: {len(model.write_attn)}")
    print()

    # Compile like training
    print("Compiling...")
    model.compile()
    # Teacher backbone blocks also compiled in training
    for block in teacher.vit.blocks:
        block.compile()
    print("Compilation done.")
    print()

    # Create inputs - scene size = canvas_grid * patch_size = 32 * 16 = 512
    scene = torch.randn(BATCH, 3, 512, 512, device=device)
    viewpoint = Viewpoint(
        centers=torch.zeros(BATCH, 2, device=device),
        scales=torch.ones(BATCH, device=device),
    )
    glimpse = sample_at_viewpoint(spatial=scene, viewpoint=viewpoint, out_size=GLIMPSE_PX)
    state = model.init_state(batch_size=BATCH, canvas_grid_size=CANVAS_GRID)

    print(f"Input shapes:")
    print(f"  scene:   {list(scene.shape)}")
    print(f"  glimpse: {list(glimpse.shape)}")
    print(f"  canvas:  {list(state.canvas.shape)}")
    print()

    # =========================================================================
    # WARMUP
    # =========================================================================
    print(f"Warming up ({N_WARMUP} iters)...")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(N_WARMUP):
            # Teacher forward (what we compare against)
            _ = teacher.forward_norm_features(glimpse)
            # CanViT forward + prediction head
            out = model.forward(glimpse=glimpse, state=state, viewpoint=viewpoint)
            _ = model.predict_teacher_scene(out.state.canvas)
    torch.cuda.synchronize()
    print("Warmup done.")
    print()

    # =========================================================================
    # PROFILE: Teacher only
    # =========================================================================
    print("=" * 70)
    print(f"PROFILING TEACHER ({N_PROFILE} iters)")
    print("=" * 70)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=False,
        ) as prof_teacher:
            for _ in range(N_PROFILE):
                with record_function("Teacher.forward_norm_features"):
                    _ = teacher.forward_norm_features(glimpse)
            torch.cuda.synchronize()

    teacher_total_cuda = sum(
        e.self_device_time_total for e in prof_teacher.key_averages()
        if e.self_device_time_total > 0
    )
    print(f"Teacher total CUDA time: {teacher_total_cuda/1000:.1f} ms ({N_PROFILE} iters)")
    print(f"Teacher per-iter: {teacher_total_cuda/1000/N_PROFILE:.2f} ms")
    print()

    # =========================================================================
    # PROFILE: CanViT forward + predict_teacher_scene
    # =========================================================================
    print("=" * 70)
    print(f"PROFILING CANVIT ({N_PROFILE} iters)")
    print("=" * 70)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=False,
        ) as prof_canvit:
            for _ in range(N_PROFILE):
                with record_function("CanViT.forward"):
                    out = model.forward(glimpse=glimpse, state=state, viewpoint=viewpoint)
                with record_function("CanViT.predict_teacher_scene"):
                    _ = model.predict_teacher_scene(out.state.canvas)
            torch.cuda.synchronize()

    canvit_total_cuda = sum(
        e.self_device_time_total for e in prof_canvit.key_averages()
        if e.self_device_time_total > 0
    )
    print(f"CanViT total CUDA time: {canvit_total_cuda/1000:.1f} ms ({N_PROFILE} iters)")
    print(f"CanViT per-iter: {canvit_total_cuda/1000/N_PROFILE:.2f} ms")
    print()

    # =========================================================================
    # OVERHEAD COMPARISON
    # =========================================================================
    print("=" * 70)
    print("OVERHEAD")
    print("=" * 70)
    overhead = canvit_total_cuda / teacher_total_cuda
    print(f"CanViT / Teacher = {overhead:.2f}x")
    print(f"Expected (from FLOP analysis): ~1.40x")
    print()

    # =========================================================================
    # TOP KERNELS
    # =========================================================================
    print("=" * 70)
    print("TOP 30 TEACHER CUDA KERNELS")
    print("=" * 70)
    print(prof_teacher.key_averages().table(
        sort_by="self_device_time_total",
        row_limit=30,
    ))

    print("=" * 70)
    print("TOP 30 CANVIT CUDA KERNELS")
    print("=" * 70)
    print(prof_canvit.key_averages().table(
        sort_by="self_device_time_total",
        row_limit=30,
    ))

    # Export traces
    prof_teacher.export_chrome_trace("profile_teacher.json")
    prof_canvit.export_chrome_trace("profile_canvit.json")
    print("\nChrome traces exported:")
    print("  profile_teacher.json")
    print("  profile_canvit.json")
    print("View at chrome://tracing")

    # =========================================================================
    # CATEGORIZE KERNELS
    # =========================================================================
    print("\n" + "=" * 70)
    print("CANVIT TIME BY CATEGORY")
    print("=" * 70)

    categories = {
        "LayerNorm": [],
        "SDPA/Flash": [],
        "GEMM/Linear": [],
        "Elementwise/Copy": [],
        "Other": [],
    }

    for evt in prof_canvit.key_averages():
        if evt.self_device_time_total == 0:
            continue
        name = evt.key.lower()
        if "layer_norm" in name or "layernorm" in name:
            categories["LayerNorm"].append(evt)
        elif "sdpa" in name or "flash" in name or "fmha" in name:
            categories["SDPA/Flash"].append(evt)
        elif "gemm" in name or "cutlass" in name or "mm_" in name:
            categories["GEMM/Linear"].append(evt)
        elif any(x in name for x in ["add", "mul", "copy", "cat", "elementwise"]):
            categories["Elementwise/Copy"].append(evt)
        else:
            categories["Other"].append(evt)

    for cat, evts in categories.items():
        cat_time = sum(e.self_device_time_total for e in evts)
        pct = 100 * cat_time / canvit_total_cuda if canvit_total_cuda > 0 else 0
        print(f"{cat:20s}: {cat_time/1000:8.1f} ms  ({pct:5.1f}%)")
        for e in sorted(evts, key=lambda x: x.self_device_time_total, reverse=True)[:3]:
            print(f"    {e.key[:55]:55s} {e.self_device_time_total/1000:6.1f} ms")


if __name__ == "__main__":
    main()
