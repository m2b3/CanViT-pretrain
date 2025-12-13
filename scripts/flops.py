"""FLOPs calculations for AVP-ViT and teacher backbone.

Single source of truth for compute estimates. All formulas documented.
"""

from dataclasses import dataclass
from typing import NamedTuple


@dataclass(frozen=True)
class ViTConfig:
    embed_dim: int = 384
    num_heads: int = 6
    n_blocks: int = 12
    mlp_ratio: int = 4
    patch_size: int = 16
    n_prefix_tokens: int = 5  # CLS + storage


class TeacherFLOPs(NamedTuple):
    patch_embed: int
    blocks: int
    total: int
    n_tokens: int


class AVPFLOPs(NamedTuple):
    glimpse_embed: int
    read_attn: int
    backbone: int
    write_attn: int
    output_proj: int
    total: int
    n_local: int
    n_scene: int


def vit_block_flops(n_tokens: int, cfg: ViTConfig) -> int:
    """FLOPs for one ViT block.

    Self-attention:
      QKV proj: 6ND², attn: 4N²D, O proj: 2ND² → 8ND² + 4N²D

    MLP (4× expansion):
      fc1 + fc2: 16ND²

    Total: 24ND² + 4N²D

    Ignored (negligible): LayerNorm, biases, GELU, softmax, RoPE.
    """
    assert n_tokens > 0
    N, D = n_tokens, cfg.embed_dim
    return 24 * N * D * D + 4 * N * N * D


def patch_embed_flops(n_patches: int, cfg: ViTConfig) -> int:
    """FLOPs for patch embedding Conv2d(3, D, kernel_size=P, stride=P).

    = 2 × D × 3 × P² × n_patches
    """
    assert n_patches > 0
    return 2 * cfg.embed_dim * 3 * cfg.patch_size**2 * n_patches


def cross_attn_flops(
    n_q: int, n_kv: int, cfg: ViTConfig, *, q_proj: bool, kv_proj: bool, o_proj: bool
) -> int:
    """FLOPs for one cross-attention layer.

    Q proj (if on): 2 × N_q × D²
    K+V proj (if on): 4 × N_kv × D²
    Attention (Q@Kᵀ + attn@V): 4 × N_q × N_kv × D
    O proj (if on): 2 × N_q × D²

    Ignored: LayerNorm, ElementwiseAffine, RoPE.
    """
    assert n_q > 0 and n_kv > 0
    D = cfg.embed_dim
    flops = 4 * n_q * n_kv * D  # attention
    if q_proj:
        flops += 2 * n_q * D * D
    if kv_proj:
        flops += 4 * n_kv * D * D
    if o_proj:
        flops += 2 * n_q * D * D
    return flops


def teacher_flops(n_patches: int, cfg: ViTConfig) -> TeacherFLOPs:
    """Total FLOPs for teacher forward pass."""
    assert n_patches > 0
    n_tokens = n_patches + cfg.n_prefix_tokens
    patch_embed = patch_embed_flops(n_patches, cfg)
    blocks = cfg.n_blocks * vit_block_flops(n_tokens, cfg)
    return TeacherFLOPs(patch_embed, blocks, patch_embed + blocks, n_tokens)


def avp_flops(
    glimpse_patches: int, scene_patches: int, cfg: ViTConfig, *, output_proj: bool
) -> AVPFLOPs:
    """Total FLOPs for AVP forward pass.

    Per block:
      1. Read cross-attn: local queries scene (Q/O proj, K/V identity)
      2. Backbone block: self-attention + MLP on local
      3. Write cross-attn: scene queries local (Q/O identity, K/V proj)
    """
    assert glimpse_patches > 0 and scene_patches > 0
    n_local = glimpse_patches + cfg.n_prefix_tokens

    read = cfg.n_blocks * cross_attn_flops(
        n_local, scene_patches, cfg, q_proj=True, kv_proj=False, o_proj=True
    )
    backbone = cfg.n_blocks * vit_block_flops(n_local, cfg)
    write = cfg.n_blocks * cross_attn_flops(
        scene_patches, n_local, cfg, q_proj=False, kv_proj=True, o_proj=False
    )
    out_proj = 2 * scene_patches * cfg.embed_dim**2 if output_proj else 0
    glimpse_embed = patch_embed_flops(glimpse_patches, cfg)

    total = glimpse_embed + read + backbone + write + out_proj
    return AVPFLOPs(glimpse_embed, read, backbone, write, out_proj, total, n_local, scene_patches)


def fmt(flops: int) -> str:
    if flops >= 1e9:
        return f"{flops / 1e9:.2f}G"
    if flops >= 1e6:
        return f"{flops / 1e6:.2f}M"
    return f"{flops:.0f}"


def main() -> None:
    cfg = ViTConfig()
    print("=" * 60)
    print("FLOPs Calculator for AVP-ViT")
    print("=" * 60)
    print("ViT-S/16 config:")
    print(f"  embed_dim={cfg.embed_dim}, num_heads={cfg.num_heads}, n_blocks={cfg.n_blocks}")
    print(f"  patch_size={cfg.patch_size}, n_prefix_tokens={cfg.n_prefix_tokens}")
    print()

    resolutions = [256, 512, 1024]
    glimpse_grid = 7
    glimpse_patches = glimpse_grid**2

    print("=" * 60)
    print("TEACHER (full self-attention)")
    print("=" * 60)
    for px in resolutions:
        grid = px // cfg.patch_size
        t = teacher_flops(grid**2, cfg)
        print(f"{px}×{px} px ({grid}×{grid} grid, {t.n_tokens} tokens):")
        print(f"  Patch embed:  {fmt(t.patch_embed):>10}")
        print(f"  Blocks:       {fmt(t.blocks):>10}")
        print(f"  Total:        {fmt(t.total):>10}")
        print()

    print("=" * 60)
    print(f"AVP ({glimpse_grid}×{glimpse_grid} glimpse, cross-attention to scene)")
    print("=" * 60)
    for px in resolutions:
        grid = px // cfg.patch_size
        a = avp_flops(glimpse_patches, grid**2, cfg, output_proj=True)
        print(f"Scene {grid}×{grid} ({a.n_scene} tokens), local={a.n_local} tokens:")
        print(f"  Glimpse embed: {fmt(a.glimpse_embed):>10}")
        print(f"  Read attn:     {fmt(a.read_attn):>10}")
        print(f"  Backbone:      {fmt(a.backbone):>10}")
        print(f"  Write attn:    {fmt(a.write_attn):>10}")
        print(f"  Output proj:   {fmt(a.output_proj):>10}")
        print(f"  Total:         {fmt(a.total):>10}")
        print()

    print("=" * 60)
    print("COMPARISON (Teacher vs AVP)")
    print("=" * 60)
    print(f"{'Resolution':<12} {'Teacher':>12} {'AVP':>12} {'Speedup':>10}")
    print("-" * 48)
    for px in resolutions:
        grid = px // cfg.patch_size
        t = teacher_flops(grid**2, cfg)
        a = avp_flops(glimpse_patches, grid**2, cfg, output_proj=True)
        print(f"{px}×{px:<8} {fmt(t.total):>12} {fmt(a.total):>12} {t.total / a.total:>9.1f}×")


if __name__ == "__main__":
    main()
