"""Benchmark EWA variants: cat vs in-place write."""

import torch
from torch.profiler import profile, ProfilerActivity

B, N, D = 64, 1040, 1024
N_PREFIX = 16
N_WARMUP, N_BENCH = 50, 100

device = torch.device("cuda")
x = torch.randn(B, N, D, device=device, dtype=torch.bfloat16)
prefix_scale = torch.randn(N_PREFIX, D, device=device, dtype=torch.bfloat16)
prefix_bias = torch.randn(N_PREFIX, D, device=device, dtype=torch.bfloat16)
rest_scale = torch.randn(D, device=device, dtype=torch.bfloat16)
rest_bias = torch.randn(D, device=device, dtype=torch.bfloat16)


def ewa_cat(x):
    """Current implementation: slice, compute, cat."""
    prefix = x[:, :N_PREFIX] * prefix_scale + prefix_bias
    rest = x[:, N_PREFIX:] * rest_scale + rest_bias
    return torch.cat([prefix, rest], dim=1)


def ewa_inplace(x):
    """Alternative: pre-allocate, write in-place."""
    out = torch.empty_like(x)
    out[:, :N_PREFIX] = x[:, :N_PREFIX] * prefix_scale + prefix_bias
    out[:, N_PREFIX:] = x[:, N_PREFIX:] * rest_scale + rest_bias
    return out


def ewa_inplace_addcmul(x):
    """Using addcmul for fused multiply-add."""
    out = torch.empty_like(x)
    torch.addcmul(prefix_bias, x[:, :N_PREFIX], prefix_scale, out=out[:, :N_PREFIX])
    torch.addcmul(rest_bias, x[:, N_PREFIX:], rest_scale, out=out[:, N_PREFIX:])
    return out


def bench(fn, name):
    # Verify correctness
    ref = ewa_cat(x)
    out = fn(x)
    if not torch.allclose(ref, out, atol=1e-2, rtol=1e-2):
        print(f"WARNING: {name} output differs from reference!")

    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = fn(x)
    torch.cuda.synchronize()

    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
            for _ in range(N_BENCH):
                _ = fn(x)
            torch.cuda.synchronize()

    total = sum(e.self_device_time_total for e in prof.key_averages() if e.self_device_time_total > 0)
    per_iter = total / 1000 / N_BENCH
    print(f"{name:25s}: {per_iter:.3f} ms/iter")
    print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=5))
    print()
    return total / N_BENCH


def main():
    print(f"Shape: [{B}, {N}, {D}], prefix={N_PREFIX}")
    print(f"Device: {torch.cuda.get_device_name()}")
    print()

    results = {}
    results["cat"] = bench(ewa_cat, "EWA cat (current)")
    results["inplace"] = bench(ewa_inplace, "EWA in-place write")
    results["addcmul"] = bench(ewa_inplace_addcmul, "EWA addcmul")
    results["cat_compiled"] = bench(torch.compile(ewa_cat), "EWA cat (compiled)")
    results["inplace_compiled"] = bench(torch.compile(ewa_inplace), "EWA in-place (compiled)")

    print("=" * 60)
    print("SUMMARY (ms/iter)")
    print("=" * 60)
    for name, us in results.items():
        print(f"  {name:25s}: {us/1000:.3f} ms")


if __name__ == "__main__":
    main()
