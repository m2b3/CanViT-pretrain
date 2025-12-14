"""FLOPs calculations for AVP-ViT and teacher backbone.

Uses introspection on actual model instances - no hardcoded formulas that can drift.
"""

from pathlib import Path
from typing import NamedTuple

from torch import nn

from avp_vit import AVPConfig, AVPViT
from avp_vit.backbone.dinov3 import DINOv3Backbone

CKPT_PATH = Path("dinov3_vits16_pretrain_lvd1689m-08c60483.pth")


class TeacherFLOPs(NamedTuple):
    patch_embed: int
    blocks: int
    total: int
    n_tokens: int


class AVPStepFLOPs(NamedTuple):
    glimpse_embed: int
    read_attn: int
    backbone: int
    write_attn: int
    output_proj: int
    total: int
    n_local: int
    n_scene: int


def teacher_flops(backbone: DINOv3Backbone, n_patches: int) -> TeacherFLOPs:
    """FLOPs for teacher forward pass."""
    n_tokens = n_patches + backbone.n_prefix_tokens
    patch_embed = backbone.patch_embed_flops(n_patches)
    blocks = backbone.n_blocks * backbone.block_flops(n_tokens)
    return TeacherFLOPs(patch_embed, blocks, patch_embed + blocks, n_tokens)


def avp_step_flops(model: AVPViT, backbone: DINOv3Backbone) -> AVPStepFLOPs:
    """FLOPs for one AVP forward step. Introspects actual model structure."""
    cfg = model.cfg
    glimpse_patches = cfg.glimpse_grid_size ** 2
    scene_patches = cfg.scene_grid_size ** 2
    n_local = glimpse_patches + backbone.n_prefix_tokens

    glimpse_embed = backbone.patch_embed_flops(glimpse_patches)

    read_attn = sum(
        model.read_attn[i].flops(n_local, scene_patches)  # type: ignore[union-attr]
        for i in range(backbone.n_blocks)
    )
    blocks = backbone.n_blocks * backbone.block_flops(n_local)
    write_attn = sum(
        model.write_attn[i].flops(scene_patches, n_local)  # type: ignore[union-attr]
        for i in range(backbone.n_blocks)
    )

    if isinstance(model.output_proj, nn.Identity):
        output_proj = 0
    else:
        output_proj = 2 * scene_patches * backbone.embed_dim ** 2

    total = glimpse_embed + read_attn + blocks + write_attn + output_proj
    return AVPStepFLOPs(glimpse_embed, read_attn, blocks, write_attn, output_proj, total, n_local, scene_patches)


def fmt(f: int) -> str:
    if f >= 1e9:
        return f"{f / 1e9:.2f}G"
    if f >= 1e6:
        return f"{f / 1e6:.2f}M"
    return f"{f:.0f}"


def main() -> None:
    from dinov3.hub.backbones import dinov3_vits16

    # Load actual models
    backbone = DINOv3Backbone(dinov3_vits16(weights=str(CKPT_PATH), pretrained=True))

    print("=" * 60)
    print("FLOPs Calculator for AVP-ViT (introspection-based)")
    print("=" * 60)
    print("Backbone: DINOv3 ViT-S/16")
    print(f"  embed_dim={backbone.embed_dim}, num_heads={backbone.num_heads}")
    print(f"  n_blocks={backbone.n_blocks}, patch_size={backbone.patch_size}")
    print(f"  n_prefix_tokens={backbone.n_prefix_tokens}")
    print()

    resolutions = [256, 512, 1024]
    scene_grid = 16
    glimpse_grid = 7

    print("=" * 60)
    print("TEACHER (full self-attention)")
    print("=" * 60)
    for px in resolutions:
        grid = px // backbone.patch_size
        t = teacher_flops(backbone, grid ** 2)
        print(f"{px}×{px} ({grid}×{grid} grid, {t.n_tokens} tokens):")
        print(f"  Patch embed:  {fmt(t.patch_embed):>10}")
        print(f"  Blocks:       {fmt(t.blocks):>10}")
        print(f"  Total:        {fmt(t.total):>10}")
        print()

    print("=" * 60)
    print(f"AVP ({glimpse_grid}×{glimpse_grid} glimpse, {scene_grid}×{scene_grid} scene)")
    print("=" * 60)

    # Create AVP model for introspection
    avp_cfg = AVPConfig(scene_grid_size=scene_grid, glimpse_grid_size=glimpse_grid, use_output_proj=True)
    avp = AVPViT(backbone, avp_cfg)

    a = avp_step_flops(avp, backbone)
    print(f"Per-step FLOPs (local={a.n_local} tokens, scene={a.n_scene} tokens):")
    print(f"  Glimpse embed: {fmt(a.glimpse_embed):>10}")
    print(f"  Read attn:     {fmt(a.read_attn):>10}")
    print(f"  Backbone:      {fmt(a.backbone):>10}")
    print(f"  Write attn:    {fmt(a.write_attn):>10}")
    print(f"  Output proj:   {fmt(a.output_proj):>10}")
    print(f"  Total:         {fmt(a.total):>10}")
    print()

    print("=" * 60)
    print("COMPARISON (Teacher vs AVP step)")
    print("=" * 60)
    print(f"{'Resolution':<12} {'Teacher':>12} {'AVP step':>12} {'Speedup':>10}")
    print("-" * 48)
    for px in resolutions:
        grid = px // backbone.patch_size
        t = teacher_flops(backbone, grid ** 2)
        print(f"{px}×{px:<8} {fmt(t.total):>12} {fmt(a.total):>12} {t.total / a.total:>9.1f}×")


if __name__ == "__main__":
    main()
