"""FLOPs calculations for AVP-ViT and teacher backbone.

Uses introspection on actual model instances - no hardcoded formulas that can drift.
"""

from pathlib import Path
from typing import NamedTuple

from avp_vit import AVPConfig, AVPViT
from avp_vit.backbone.dinov3 import DINOv3Backbone

CKPT_PATH = Path("dinov3_vits16_pretrain_lvd1689m-08c60483.pth")

# Configuration
GLIMPSE_GRID = 7
N_REGISTERS = 32
SCENE_GRIDS = [16, 32, 64, 128]


class TeacherFLOPs(NamedTuple):
    patch_embed: int
    blocks: int
    total: int
    n_tokens: int


class AVPStepFLOPs(NamedTuple):
    """FLOPs for one AVP step, computed via model introspection."""

    glimpse_embed: int
    read_attn: int
    backbone: int
    write_attn: int
    scene_proj: int
    total: int
    # Token counts for reference
    n_local: int
    n_scene: int
    n_adapters: int


def teacher_flops(backbone: DINOv3Backbone, n_patches: int) -> TeacherFLOPs:
    n_tokens = n_patches + backbone.n_prefix_tokens
    patch_embed = backbone.patch_embed_flops(n_patches)
    blocks = backbone.n_blocks * backbone.block_flops(n_tokens)
    return TeacherFLOPs(patch_embed, blocks, patch_embed + blocks, n_tokens)


def avp_step_flops(model: AVPViT, backbone: DINOv3Backbone, scene_grid_size: int) -> AVPStepFLOPs:
    """FLOPs for one AVP step. Uses introspection on actual model modules."""
    cfg = model.cfg
    D = backbone.embed_dim
    n_blocks = backbone.n_blocks
    glimpse_patches = cfg.glimpse_grid_size**2
    spatial_patches = scene_grid_size**2
    n_local = glimpse_patches + backbone.n_prefix_tokens
    n_scene = model.n_registers + spatial_patches

    glimpse_embed = backbone.patch_embed_flops(glimpse_patches)

    # Use actual model introspection for attention FLOPs
    n_adapters = model.n_adapters
    read_attn = sum(model.read_attn[i].flops(n_local, n_scene) for i in range(n_adapters))  # type: ignore[union-attr]
    blocks = n_blocks * backbone.block_flops(n_local)
    write_attn = sum(model.write_attn[i].flops(n_scene, n_local) for i in range(n_adapters))  # type: ignore[union-attr]

    # scene_proj = LayerNorm + Linear(D -> teacher_dim)
    # LayerNorm: ~4*D per token, Linear: 2*D*teacher_dim per token
    # Linear dominates, so we approximate as just the Linear cost
    scene_proj = 2 * spatial_patches * D * model.teacher_dim

    total = glimpse_embed + read_attn + blocks + write_attn + scene_proj
    return AVPStepFLOPs(glimpse_embed, read_attn, blocks, write_attn, scene_proj, total, n_local, n_scene, n_adapters)


def hypothetical_full_linear_flops(
    n_local: int,
    n_scene: int,
    dim: int,
    n_adapters: int,
    convex: bool,
) -> int:
    """Hypothetical FLOPs if all cross-attention used full Linear projections (no EWA).

    Per adapter:
      Read:  SDPA + Q(Linear) + K(Linear) + V(Linear) + O(Linear)
      Write: SDPA + Q(Linear) + K(Linear) + V(Linear) + O(Linear)

    With convex gating, double the attention ops.
    """
    mult = 2 if convex else 1

    def single_attn_flops(n_q: int, n_kv: int) -> int:
        sdpa = 4 * n_q * n_kv * dim
        q_proj = 2 * n_q * dim * dim
        k_proj = 2 * n_kv * dim * dim
        v_proj = 2 * n_kv * dim * dim
        o_proj = 2 * n_q * dim * dim
        return sdpa + q_proj + k_proj + v_proj + o_proj

    read_per_adapter = mult * single_attn_flops(n_local, n_scene)
    write_per_adapter = mult * single_attn_flops(n_scene, n_local)
    return n_adapters * (read_per_adapter + write_per_adapter)


def fmt(f: int | float) -> str:
    if f >= 1e12:
        return f"{f / 1e12:.2f}T"
    if f >= 1e9:
        return f"{f / 1e9:.2f}G"
    if f >= 1e6:
        return f"{f / 1e6:.2f}M"
    return f"{f:.0f}"


def print_detailed_breakdown(
    backbone: DINOv3Backbone,
    scene_grid: int,
    glimpse_grid: int,
    n_registers: int,
    gating: str = "none",
) -> None:
    D = backbone.embed_dim
    P = backbone.patch_size
    n_blocks = backbone.n_blocks
    scene_px = scene_grid * P
    glimpse_px = glimpse_grid * P
    n_spatial = scene_grid**2

    avp = AVPViT(
        backbone,
        AVPConfig(
            glimpse_grid_size=glimpse_grid,
            n_scene_registers=n_registers,
            gating=gating,  # type: ignore[arg-type]
        ),
        teacher_dim=backbone.embed_dim,
    )
    a = avp_step_flops(avp, backbone, scene_grid)
    t_scene = teacher_flops(backbone, scene_grid**2)
    t_glimpse = teacher_flops(backbone, glimpse_grid**2)

    def pct(x: int) -> str:
        return f"({100 * x / a.total:4.1f}%)"

    print("=" * 80)
    print(f"DETAILED: {scene_grid}x{scene_grid} scene ({scene_px}x{scene_px} px), gating={gating}")
    print("=" * 80)
    print(f"Tokens: local={a.n_local}, scene={a.n_scene} (spatial={n_spatial} + registers={n_registers})")
    print(f"Adapters: {a.n_adapters} (backbone has {n_blocks} blocks)")
    print()

    print(f"  Glimpse embed: {fmt(a.glimpse_embed):>10}  {pct(a.glimpse_embed)}")
    print(f"  Read attn:     {fmt(a.read_attn):>10}  {pct(a.read_attn)}  [{a.n_adapters} adapters, local queries scene]")
    print(f"  Backbone:      {fmt(a.backbone):>10}  {pct(a.backbone)}  [{n_blocks} blocks on {a.n_local} tokens]")
    print(f"  Write attn:    {fmt(a.write_attn):>10}  {pct(a.write_attn)}  [{a.n_adapters} adapters, scene queries local]")
    print(f"  Scene proj:    {fmt(a.scene_proj):>10}  {pct(a.scene_proj)}  [LayerNorm + Linear on {n_spatial} tokens]")
    print()
    print(f"  TOTAL:         {fmt(a.total):>10}")
    print()

    # EWA savings: compare actual AVP vs hypothetical full-Linear
    convex = gating == "full"
    full_linear_attn = hypothetical_full_linear_flops(a.n_local, a.n_scene, D, a.n_adapters, convex)
    full_linear_total = a.glimpse_embed + full_linear_attn + a.backbone + a.scene_proj
    actual_attn = a.read_attn + a.write_attn
    attn_savings = full_linear_attn - actual_attn

    print("=" * 80)
    print(f"EWA SAVINGS ({scene_grid}x{scene_grid} scene)")
    print("=" * 80)
    print("Cross-attention uses EWA (ElementwiseAffine) instead of Linear for:")
    print("  Read:  K, V (on scene tokens)")
    print("  Write: Q, O (on scene tokens)")
    print()
    print(f"  Actual cross-attn:       {fmt(actual_attn):>10}")
    print(f"  Hypothetical full-Linear:{fmt(full_linear_attn):>10}")
    print(f"  Savings:                 {fmt(attn_savings):>10}  ({100 * attn_savings / full_linear_attn:.0f}% of full-Linear attn)")
    print()
    print(f"  Actual total:            {fmt(a.total):>10}")
    print(f"  Hypothetical total:      {fmt(full_linear_total):>10}")
    print(f"  Savings:                 {fmt(full_linear_total - a.total):>10}  ({100 * (full_linear_total - a.total) / full_linear_total:.0f}% of full-Linear total)")
    print()

    print("=" * 80)
    print(f"COMPARISON ({scene_grid}x{scene_grid} scene = {scene_px}x{scene_px} px)")
    print("=" * 80)
    print(f"  Teacher @ {scene_grid}x{scene_grid} ({scene_px}px):   {fmt(t_scene.total):>10}")
    print(f"  Teacher @ {glimpse_grid}x{glimpse_grid} ({glimpse_px}px):    {fmt(t_glimpse.total):>10}")
    print(f"  AVP step:                  {fmt(a.total):>10}")
    print()
    print(f"  AVP steps per Teacher({scene_grid}x{scene_grid}):    {t_scene.total / a.total:6.1f}")
    print(f"  Teacher({glimpse_grid}x{glimpse_grid}) per AVP step:       {a.total / t_glimpse.total:6.2f}")


def main() -> None:
    from dinov3.hub.backbones import dinov3_vits16

    backbone = DINOv3Backbone(dinov3_vits16(weights=str(CKPT_PATH), pretrained=True))
    D = backbone.embed_dim
    n_blocks = backbone.n_blocks
    P = backbone.patch_size

    print("=" * 80)
    print("FLOPs Calculator for AVP-ViT")
    print("=" * 80)
    print(f"Backbone: DINOv3 ViT-S/16 (D={D}, heads={backbone.num_heads}, blocks={n_blocks}, P={P})")
    print(f"Glimpse: {GLIMPSE_GRID}x{GLIMPSE_GRID} = {GLIMPSE_GRID * P}x{GLIMPSE_GRID * P} px")
    print(f"Registers: {N_REGISTERS}")
    print()

    # Table header
    print(f"{'Scene':<10} {'Teacher':<12} {'AVP':<12} {'AVP cvx':<12}")
    print(f"{'grid':<10} {'(scene)':<12} {'(default)':<12} {'(full)':<12}")
    print("-" * 58)

    for scene_grid in SCENE_GRIDS:
        t_scene = teacher_flops(backbone, scene_grid**2)

        # Default AVP (LayerScale gating)
        avp_default = AVPViT(
            backbone,
            AVPConfig(
                glimpse_grid_size=GLIMPSE_GRID,
                n_scene_registers=N_REGISTERS,
            ),
            teacher_dim=D,
        )
        a_default = avp_step_flops(avp_default, backbone, scene_grid)

        # AVP with convex gating
        avp_cvx = AVPViT(
            backbone,
            AVPConfig(
                glimpse_grid_size=GLIMPSE_GRID,
                n_scene_registers=N_REGISTERS,
                gating="full",
                layer_scale_init=1e-3,
            ),
            teacher_dim=D,
        )
        a_cvx = avp_step_flops(avp_cvx, backbone, scene_grid)

        print(
            f"{scene_grid}x{scene_grid:<7} "
            f"{fmt(t_scene.total):<12} "
            f"{fmt(a_default.total):<12} "
            f"{fmt(a_cvx.total):<12}"
        )

    print()
    print("Legend:")
    print("  Teacher (scene) = Full ViT at scene resolution")
    print("  AVP (default)   = LayerScale gating, adapter_stride=2")
    print("  AVP cvx (full)  = ConvexGating (2x attention per adapter)")
    print()

    # Detailed breakdown for largest scene
    print_detailed_breakdown(backbone, SCENE_GRIDS[-1], GLIMPSE_GRID, N_REGISTERS, gating="none")


if __name__ == "__main__":
    main()
