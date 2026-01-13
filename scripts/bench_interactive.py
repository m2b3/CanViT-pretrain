"""Interactive benchmarking preload for ipython.

Usage:
    cd ~/scratch/avp-vit && source slurm/env.sh
    uv run ipython -i scripts/bench_interactive.py

Then:
    bench_teacher()
    bench_teacher(batch=32, amp=True, compile=True)
    bench_student(glimpse=8, canvas=32)
    bench_student(glimpse=16, canvas=64, amp=True)
"""

import os
import time
import torch
from tqdm import trange

torch.set_float32_matmul_precision('high')
DEVICE = torch.device("cuda")
N = 100
WARMUP = 5

# === Load models ===
print("Loading teacher...")
from canvit.hub import create_backbone  # noqa: E402
from canvit.viewpoint import Viewpoint  # noqa: E402
from avp_vit import ActiveCanViT, ActiveCanViTConfig  # noqa: E402

CKPT = os.path.expanduser(os.environ['DINOV3_VITB16_CKPT'])
teacher = create_backbone('dinov3_vitb16', weights=CKPT).to(DEVICE).eval()
teacher_compiled: dict[str, object] = {}
print(f"  teacher: {teacher.embed_dim}d, {teacher.n_blocks} blocks")

print("Loading student...")
backbone = create_backbone('dinov3_vitb16', pretrained=False).to(DEVICE)
student = ActiveCanViT(backbone=backbone, cfg=ActiveCanViTConfig(teacher_dim=768)).to(DEVICE).eval()
student_compiled: dict[str, object] = {}
PATCH = backbone.patch_size_px
print(f"  student: patch={PATCH}px")


def _get_teacher(compile: str | None):
    if compile is None:
        return teacher
    if compile not in teacher_compiled:
        print(f"  compiling teacher (mode={compile})...")
        teacher.compile(mode=compile)
        teacher_compiled[compile] = teacher
    return teacher_compiled[compile]


def _get_student(compile: str | None):
    if compile is None:
        return student
    if compile not in student_compiled:
        print(f"  compiling student (mode={compile})...")
        student.compile(mode=compile)
        student_compiled[compile] = student
    return student_compiled[compile]


def _bench(fn, batch, amp: bool):
    ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if amp else torch.no_grad()

    with torch.no_grad():
        with ctx if amp else torch.enable_grad():
            for _ in range(WARMUP):
                fn()
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        with ctx if amp else torch.enable_grad():
            for _ in trange(N, unit_scale=batch, unit="img"):
                fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    print(f"  → {N/elapsed:.1f} steps/s, {N*batch/elapsed:.0f} img/s")


def bench_teacher(batch=64, size=512, amp=True, compile=None):
    """compile: None, 'default', 'reduce-overhead', 'max-autotune'"""
    print(f"bench_teacher(batch={batch}, size={size}, amp={amp}, compile={compile})")
    model = _get_teacher(compile)
    x = torch.randn(batch, 3, size, size, device=DEVICE)
    _bench(lambda: model.forward_norm_features(x), batch, amp)


def bench_student(glimpse=8, canvas=32, batch=64, size=512, amp=True, compile=None):
    """compile: None, 'default', 'reduce-overhead', 'max-autotune'"""
    print(f"bench_student(glimpse={glimpse}, canvas={canvas}, batch={batch}, amp={amp}, compile={compile})")
    model = _get_student(compile)
    x = torch.randn(batch, 3, size, size, device=DEVICE)
    vp = Viewpoint.full_scene(batch_size=batch, device=DEVICE)
    glimpse_px = glimpse * PATCH
    def step():
        state = model.init_state(batch_size=batch, canvas_grid_size=canvas)
        model.forward_step(image=x, state=state, viewpoint=vp, glimpse_size_px=glimpse_px)
    _bench(step, batch, amp)


print(f"\nReady. CUDA: {torch.cuda.get_device_name()}")
print("  bench_teacher(batch, size, amp, compile)")
print("  bench_student(glimpse, canvas, batch, size, amp, compile)")
print("  compile: None, 'default', 'reduce-overhead', 'max-autotune'")
