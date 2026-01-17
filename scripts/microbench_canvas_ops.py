"""Microbenchmark individual canvas operations.

Uses ACTUAL config values from CanViTConfig and DINOv3 backbone.
"""

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity

from canvit.hub import create_backbone
from canvit.model.config import CanViTConfig

# Get ACTUAL config values
_cfg = CanViTConfig()
_bb = create_backbone("dinov3_vitb16", pretrained=False)

BATCH = 64
GLIMPSE_GRID = 8
CANVAS_GRID = 32

# From actual config
CANVAS_DIM = _cfg.canvas_dim  # 1024
CANVAS_NUM_HEADS = _cfg.canvas_num_heads  # 8
CANVAS_HEAD_DIM = _cfg.canvas_head_dim  # 128
N_CANVAS_REGS = _cfg.n_canvas_registers  # 16

LOCAL_DIM = _bb.embed_dim  # 768
BACKBONE_NUM_HEADS = _bb.num_heads  # 12
N_BACKBONE_REGS = _bb.n_register_tokens  # 4

# Token counts
LOCAL_TOKENS = 1 + 1 + 1 + N_BACKBONE_REGS + GLIMPSE_GRID**2  # VPE + rec_cls + eph_cls + regs + patches = 71
CANVAS_TOKENS = N_CANVAS_REGS + CANVAS_GRID**2  # 16 + 1024 = 1040

N_WARMUP = 50
N_BENCH = 100


def print_config():
    print("=" * 70)
    print("ACTUAL CONFIG VALUES (from CanViTConfig + DINOv3 backbone)")
    print("=" * 70)
    print(f"BATCH = {BATCH}")
    print(f"CANVAS_DIM = {CANVAS_DIM}, CANVAS_NUM_HEADS = {CANVAS_NUM_HEADS}, CANVAS_HEAD_DIM = {CANVAS_HEAD_DIM}")
    print(f"LOCAL_DIM = {LOCAL_DIM}, BACKBONE_NUM_HEADS = {BACKBONE_NUM_HEADS}")
    print(f"LOCAL_TOKENS = {LOCAL_TOKENS} (1 VPE + 1 rec_cls + 1 eph_cls + {N_BACKBONE_REGS} regs + {GLIMPSE_GRID**2} patches)")
    print(f"CANVAS_TOKENS = {CANVAS_TOKENS} ({N_CANVAS_REGS} canvas_regs + {CANVAS_GRID**2} spatial)")
    print()


def bench_layernorm_canvas():
    """LayerNorm on canvas: [B, 1040, 1024]."""
    print("=" * 70)
    print(f"LAYERNORM ON CANVAS: [{BATCH}, {CANVAS_TOKENS}, {CANVAS_DIM}]")
    print(f"  Elements: {BATCH * CANVAS_TOKENS * CANVAS_DIM:,}")
    print("=" * 70)

    device = torch.device("cuda")
    ln = nn.LayerNorm(CANVAS_DIM).to(device, dtype=torch.bfloat16)
    x = torch.randn(BATCH, CANVAS_TOKENS, CANVAS_DIM, device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = ln(x)
    torch.cuda.synchronize()

    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
            for _ in range(N_BENCH):
                _ = ln(x)
            torch.cuda.synchronize()

    total = sum(e.self_device_time_total for e in prof.key_averages() if e.self_device_time_total > 0)
    print(f"Total: {total/1000:.2f} ms ({N_BENCH} iters), Per-iter: {total/1000/N_BENCH:.3f} ms")
    print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=5))
    return total / N_BENCH


def bench_layernorm_local():
    """LayerNorm on local: [B, 71, 768]."""
    print("=" * 70)
    print(f"LAYERNORM ON LOCAL: [{BATCH}, {LOCAL_TOKENS}, {LOCAL_DIM}]")
    print(f"  Elements: {BATCH * LOCAL_TOKENS * LOCAL_DIM:,}")
    print("=" * 70)

    device = torch.device("cuda")
    ln = nn.LayerNorm(LOCAL_DIM).to(device, dtype=torch.bfloat16)
    x = torch.randn(BATCH, LOCAL_TOKENS, LOCAL_DIM, device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = ln(x)
    torch.cuda.synchronize()

    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
            for _ in range(N_BENCH):
                _ = ln(x)
            torch.cuda.synchronize()

    total = sum(e.self_device_time_total for e in prof.key_averages() if e.self_device_time_total > 0)
    print(f"Total: {total/1000:.2f} ms ({N_BENCH} iters), Per-iter: {total/1000/N_BENCH:.3f} ms")
    print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=5))
    return total / N_BENCH


def bench_sdpa_read():
    """SDPA for READ: local queries canvas.

    Q: [B, 8, 71, 128] - local tokens, canvas heads
    K/V: [B, 8, 1040, 128] - canvas tokens
    """
    print("=" * 70)
    print(f"SDPA READ (local queries canvas)")
    print(f"  Q: [{BATCH}, {CANVAS_NUM_HEADS}, {LOCAL_TOKENS}, {CANVAS_HEAD_DIM}]")
    print(f"  K/V: [{BATCH}, {CANVAS_NUM_HEADS}, {CANVAS_TOKENS}, {CANVAS_HEAD_DIM}]")
    print("=" * 70)

    device = torch.device("cuda")
    q = torch.randn(BATCH, CANVAS_NUM_HEADS, LOCAL_TOKENS, CANVAS_HEAD_DIM, device=device, dtype=torch.bfloat16)
    k = torch.randn(BATCH, CANVAS_NUM_HEADS, CANVAS_TOKENS, CANVAS_HEAD_DIM, device=device, dtype=torch.bfloat16)
    v = torch.randn(BATCH, CANVAS_NUM_HEADS, CANVAS_TOKENS, CANVAS_HEAD_DIM, device=device, dtype=torch.bfloat16)

    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)

    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()

    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
            for _ in range(N_BENCH):
                _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            torch.cuda.synchronize()

    total = sum(e.self_device_time_total for e in prof.key_averages() if e.self_device_time_total > 0)
    print(f"Total: {total/1000:.2f} ms ({N_BENCH} iters), Per-iter: {total/1000/N_BENCH:.3f} ms")
    print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=5))
    return total / N_BENCH


def bench_sdpa_write():
    """SDPA for WRITE: canvas queries local.

    Q: [B, 8, 1040, 128] - canvas tokens
    K/V: [B, 8, 71, 128] - local tokens, canvas heads
    """
    print("=" * 70)
    print(f"SDPA WRITE (canvas queries local)")
    print(f"  Q: [{BATCH}, {CANVAS_NUM_HEADS}, {CANVAS_TOKENS}, {CANVAS_HEAD_DIM}]")
    print(f"  K/V: [{BATCH}, {CANVAS_NUM_HEADS}, {LOCAL_TOKENS}, {CANVAS_HEAD_DIM}]")
    print("=" * 70)

    device = torch.device("cuda")
    q = torch.randn(BATCH, CANVAS_NUM_HEADS, CANVAS_TOKENS, CANVAS_HEAD_DIM, device=device, dtype=torch.bfloat16)
    k = torch.randn(BATCH, CANVAS_NUM_HEADS, LOCAL_TOKENS, CANVAS_HEAD_DIM, device=device, dtype=torch.bfloat16)
    v = torch.randn(BATCH, CANVAS_NUM_HEADS, LOCAL_TOKENS, CANVAS_HEAD_DIM, device=device, dtype=torch.bfloat16)

    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)

    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()

    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
            for _ in range(N_BENCH):
                _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            torch.cuda.synchronize()

    total = sum(e.self_device_time_total for e in prof.key_averages() if e.self_device_time_total > 0)
    print(f"Total: {total/1000:.2f} ms ({N_BENCH} iters), Per-iter: {total/1000/N_BENCH:.3f} ms")
    print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=5))
    return total / N_BENCH


def bench_ewa_canvas_eager():
    """EWA on canvas: split prefix/rest, scale+bias, cat."""
    print("=" * 70)
    print(f"EWA ON CANVAS (eager): [{BATCH}, {CANVAS_TOKENS}, {CANVAS_DIM}]")
    print(f"  Prefix: {N_CANVAS_REGS} tokens, Rest: {CANVAS_GRID**2} tokens")
    print("=" * 70)

    device = torch.device("cuda")
    x = torch.randn(BATCH, CANVAS_TOKENS, CANVAS_DIM, device=device, dtype=torch.bfloat16)
    prefix_scale = torch.randn(N_CANVAS_REGS, CANVAS_DIM, device=device, dtype=torch.bfloat16)
    prefix_bias = torch.randn(N_CANVAS_REGS, CANVAS_DIM, device=device, dtype=torch.bfloat16)
    rest_scale = torch.randn(CANVAS_DIM, device=device, dtype=torch.bfloat16)
    rest_bias = torch.randn(CANVAS_DIM, device=device, dtype=torch.bfloat16)

    def ewa(x):
        prefix = x[:, :N_CANVAS_REGS] * prefix_scale + prefix_bias
        rest = x[:, N_CANVAS_REGS:] * rest_scale + rest_bias
        return torch.cat([prefix, rest], dim=1)

    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = ewa(x)
    torch.cuda.synchronize()

    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
            for _ in range(N_BENCH):
                _ = ewa(x)
            torch.cuda.synchronize()

    total = sum(e.self_device_time_total for e in prof.key_averages() if e.self_device_time_total > 0)
    print(f"Total: {total/1000:.2f} ms ({N_BENCH} iters), Per-iter: {total/1000/N_BENCH:.3f} ms")
    print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=10))
    return total / N_BENCH


def bench_ewa_canvas_compiled():
    """EWA on canvas: compiled."""
    print("=" * 70)
    print(f"EWA ON CANVAS (compiled)")
    print("=" * 70)

    device = torch.device("cuda")
    x = torch.randn(BATCH, CANVAS_TOKENS, CANVAS_DIM, device=device, dtype=torch.bfloat16)
    prefix_scale = torch.randn(N_CANVAS_REGS, CANVAS_DIM, device=device, dtype=torch.bfloat16)
    prefix_bias = torch.randn(N_CANVAS_REGS, CANVAS_DIM, device=device, dtype=torch.bfloat16)
    rest_scale = torch.randn(CANVAS_DIM, device=device, dtype=torch.bfloat16)
    rest_bias = torch.randn(CANVAS_DIM, device=device, dtype=torch.bfloat16)

    def ewa(x):
        prefix = x[:, :N_CANVAS_REGS] * prefix_scale + prefix_bias
        rest = x[:, N_CANVAS_REGS:] * rest_scale + rest_bias
        return torch.cat([prefix, rest], dim=1)

    ewa_c = torch.compile(ewa)

    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = ewa_c(x)
    torch.cuda.synchronize()

    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
            for _ in range(N_BENCH):
                _ = ewa_c(x)
            torch.cuda.synchronize()

    total = sum(e.self_device_time_total for e in prof.key_averages() if e.self_device_time_total > 0)
    print(f"Total: {total/1000:.2f} ms ({N_BENCH} iters), Per-iter: {total/1000/N_BENCH:.3f} ms")
    print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=5))
    return total / N_BENCH


def main():
    assert torch.cuda.is_available()
    print(f"Device: {torch.cuda.get_device_name()}")
    print_config()

    results = {}
    results["ln_canvas"] = bench_layernorm_canvas()
    print()
    results["ln_local"] = bench_layernorm_local()
    print()
    results["sdpa_read"] = bench_sdpa_read()
    print()
    results["sdpa_write"] = bench_sdpa_write()
    print()
    results["ewa_eager"] = bench_ewa_canvas_eager()
    print()
    results["ewa_compiled"] = bench_ewa_canvas_compiled()

    print()
    print("=" * 70)
    print("SUMMARY (μs per call)")
    print("=" * 70)
    for name, us in results.items():
        print(f"  {name:20s}: {us:8.1f} μs")


if __name__ == "__main__":
    main()
