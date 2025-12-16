"""FLOPs calculations for AVP-ViT and teacher backbone.

Uses introspection on actual model instances - no hardcoded formulas that can drift.
"""

from pathlib import Path
from typing import NamedTuple

from torch import nn

from avp_vit import AVPConfig, AVPViT
from avp_vit.attention import AttentionConfig
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


class CrossAttnFLOPs(NamedTuple):
    sdpa: int
    q_proj: int
    k_proj: int
    v_proj: int
    o_proj: int
    total: int


class AVPStepFLOPs(NamedTuple):
    glimpse_embed: int
    read_attn: CrossAttnFLOPs
    backbone: int
    write_attn: CrossAttnFLOPs
    output_proj: int
    total: int
    n_local: int
    n_scene: int


def teacher_flops(backbone: DINOv3Backbone, n_patches: int) -> TeacherFLOPs:
    n_tokens = n_patches + backbone.n_prefix_tokens
    patch_embed = backbone.patch_embed_flops(n_patches)
    blocks = backbone.n_blocks * backbone.block_flops(n_tokens)
    return TeacherFLOPs(patch_embed, blocks, patch_embed + blocks, n_tokens)


def cross_attn_flops(
    n_q: int,
    n_kv: int,
    dim: int,
    n_blocks: int,
    q_linear: bool,
    k_linear: bool,
    v_linear: bool,
    o_linear: bool,
    convex: bool = False,
) -> CrossAttnFLOPs:
    mult = 2 if convex else 1
    sdpa = mult * n_blocks * 4 * n_q * n_kv * dim
    q_proj = mult * n_blocks * 2 * n_q * dim * dim if q_linear else 0
    k_proj = mult * n_blocks * 2 * n_kv * dim * dim if k_linear else 0
    v_proj = mult * n_blocks * 2 * n_kv * dim * dim if v_linear else 0
    o_proj = mult * n_blocks * 2 * n_q * dim * dim if o_linear else 0
    return CrossAttnFLOPs(sdpa, q_proj, k_proj, v_proj, o_proj, sdpa + q_proj + k_proj + v_proj + o_proj)


def avp_step_flops(model: AVPViT, backbone: DINOv3Backbone) -> AVPStepFLOPs:
    """FLOPs for one AVP step. Uses introspection on actual model modules."""
    cfg = model.cfg
    D = backbone.embed_dim
    n_blocks = backbone.n_blocks
    glimpse_patches = cfg.glimpse_grid_size**2
    spatial_patches = cfg.scene_grid_size**2
    n_local = glimpse_patches + backbone.n_prefix_tokens
    n_scene = model.n_ephemeral_registers + model.n_persistent_registers + spatial_patches

    glimpse_embed = backbone.patch_embed_flops(glimpse_patches)

    # Use actual model introspection for attention FLOPs
    read_attn_total = sum(
        model.read_attn[i].flops(n_local, n_scene)  # type: ignore[union-attr]
        for i in range(n_blocks)
    )
    blocks = n_blocks * backbone.block_flops(n_local)
    write_attn_total = sum(
        model.write_attn[i].flops(n_scene, n_local)  # type: ignore[union-attr]
        for i in range(n_blocks)
    )

    # Approximate breakdown (for display purposes)
    convex_mult = 2 if cfg.gating == "full" else 1
    sdpa_per_attn = convex_mult * n_blocks * 4 * n_local * n_scene * D
    read_attn = CrossAttnFLOPs(sdpa_per_attn, 0, 0, 0, 0, read_attn_total)
    write_attn = CrossAttnFLOPs(sdpa_per_attn, 0, 0, 0, 0, write_attn_total)

    if isinstance(model.output_proj, nn.Identity):
        output_proj = 0
    else:
        output_proj = 2 * spatial_patches * D**2

    total = glimpse_embed + read_attn_total + blocks + write_attn_total + output_proj
    return AVPStepFLOPs(glimpse_embed, read_attn, blocks, write_attn, output_proj, total, n_local, n_scene)


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
) -> None:
    D = backbone.embed_dim
    P = backbone.patch_size
    n_blocks = backbone.n_blocks
    scene_px = scene_grid * P
    glimpse_px = glimpse_grid * P
    n_spatial = scene_grid**2
    n_scene = n_registers + n_spatial
    n_local = glimpse_grid**2 + backbone.n_prefix_tokens

    avp = AVPViT(
        backbone,
        AVPConfig(
            scene_grid_size=scene_grid,
            glimpse_grid_size=glimpse_grid,
            n_scene_registers=n_registers,
            use_output_proj=True,
        ),
    )
    a = avp_step_flops(avp, backbone)
    t_scene = teacher_flops(backbone, scene_grid**2)
    t_glimpse = teacher_flops(backbone, glimpse_grid**2)

    def pct(x: int) -> str:
        return f"({100*x/a.total:4.1f}%)"

    print("=" * 80)
    print(f"DETAILED: {scene_grid}x{scene_grid} scene ({scene_px}x{scene_px} px)")
    print("=" * 80)
    print(f"Tokens: local={a.n_local}, scene={a.n_scene} (spatial={n_spatial} + registers={n_registers})")
    print()

    print(f"  Glimpse embed: {fmt(a.glimpse_embed):>10}  {pct(a.glimpse_embed)}")
    print()

    r = a.read_attn
    print(f"  Read attn:     {fmt(r.total):>10}  {pct(r.total)}  [local queries scene]")
    print(f"    SDPA:        {fmt(r.sdpa):>10}  {pct(r.sdpa)}  4 * {a.n_local} * {a.n_scene} * D")
    print(f"    Q proj:      {fmt(r.q_proj):>10}  {pct(r.q_proj)}  Linear on local ({a.n_local} tokens)")
    print(f"    K proj:      {fmt(r.k_proj):>10}  {pct(r.k_proj)}  EWA on scene ({a.n_scene} tokens)")
    print(f"    V proj:      {fmt(r.v_proj):>10}  {pct(r.v_proj)}  EWA on scene")
    print(f"    O proj:      {fmt(r.o_proj):>10}  {pct(r.o_proj)}  Linear on local")
    print()

    print(f"  Backbone:      {fmt(a.backbone):>10}  {pct(a.backbone)}")
    print()

    w = a.write_attn
    print(f"  Write attn:    {fmt(w.total):>10}  {pct(w.total)}  [scene queries local]")
    print(f"    SDPA:        {fmt(w.sdpa):>10}  {pct(w.sdpa)}  4 * {a.n_scene} * {a.n_local} * D")
    print(f"    Q proj:      {fmt(w.q_proj):>10}  {pct(w.q_proj)}  EWA on scene ({a.n_scene} tokens)")
    print(f"    K proj:      {fmt(w.k_proj):>10}  {pct(w.k_proj)}  Linear on local ({a.n_local} tokens)")
    print(f"    V proj:      {fmt(w.v_proj):>10}  {pct(w.v_proj)}  Linear on local")
    print(f"    O proj:      {fmt(w.o_proj):>10}  {pct(w.o_proj)}  EWA on scene")
    print()

    print(f"  Output proj:   {fmt(a.output_proj):>10}  {pct(a.output_proj)}")
    print()
    print(f"  TOTAL:         {fmt(a.total):>10}")
    print()

    # EWA savings
    read_full = cross_attn_flops(n_local, n_scene, D, n_blocks, True, True, True, True)
    write_full = cross_attn_flops(n_scene, n_local, D, n_blocks, True, True, True, True)
    full_linear_total = a.glimpse_embed + read_full.total + a.backbone + write_full.total + a.output_proj
    savings = full_linear_total - a.total

    print("=" * 80)
    print(f"EWA SAVINGS ({scene_grid}x{scene_grid} scene)")
    print("=" * 80)
    avoided_per_proj = n_blocks * 2 * n_scene * D * D
    print(f"Avoided projections (all on scene stream, {a.n_scene} tokens):")
    print(f"  Read K:  {fmt(avoided_per_proj):>10}  (2 * {n_blocks} * {a.n_scene} * D^2)")
    print(f"  Read V:  {fmt(avoided_per_proj):>10}")
    print(f"  Write Q: {fmt(avoided_per_proj):>10}")
    print(f"  Write O: {fmt(avoided_per_proj):>10}")
    print(f"  Total:   {fmt(savings):>10}  ({savings/full_linear_total*100:.0f}% of full-Linear cost)")
    print()

    print("=" * 80)
    print(f"COMPARISON ({scene_grid}x{scene_grid} scene = {scene_px}x{scene_px} px)")
    print("=" * 80)
    print(f"  Teacher @ {scene_grid}x{scene_grid} ({scene_px}px):   {fmt(t_scene.total):>10}")
    print(f"  Teacher @ {glimpse_grid}x{glimpse_grid} ({glimpse_px}px):    {fmt(t_glimpse.total):>10}")
    print(f"  AVP step (EWA):            {fmt(a.total):>10}")
    print(f"  AVP step (full-Linear):    {fmt(full_linear_total):>10}  (hypothetical)")
    print()
    print(f"  AVP steps per Teacher({scene_grid}x{scene_grid}):    {t_scene.total/a.total:6.1f}")
    print(f"  Teacher({glimpse_grid}x{glimpse_grid}) per AVP step:       {a.total/t_glimpse.total:6.1f}")
    print(f"  Full-Linear / AVP(EWA):        {full_linear_total/a.total:6.2f}x")


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
    print(f"{'Scene':<10} {'Teacher':<12} {'AVP':<12} {'AVP+V_MLP':<12} {'AVP convex':<12} {'AVP cvx+MLP':<12}")
    print(f"{'grid':<10} {'(scene)':<12} {'(base)':<12} {'(2x exp)':<12} {'(2x attn)':<12} {'(both)':<12}")
    print("-" * 82)

    for scene_grid in SCENE_GRIDS:
        t_scene = teacher_flops(backbone, scene_grid**2)

        # Base AVP (no convex, no V MLP)
        avp_base = AVPViT(backbone, AVPConfig(
            scene_grid_size=scene_grid, glimpse_grid_size=GLIMPSE_GRID,
            n_scene_registers=N_REGISTERS, use_output_proj=True,
        ))
        a_base = avp_step_flops(avp_base, backbone)

        # AVP with V MLP (2x expansion)
        avp_mlp = AVPViT(backbone, AVPConfig(
            scene_grid_size=scene_grid, glimpse_grid_size=GLIMPSE_GRID,
            n_scene_registers=N_REGISTERS, use_output_proj=True,
            attention=AttentionConfig(write_v_expansion=2),
        ))
        a_mlp = avp_step_flops(avp_mlp, backbone)

        # AVP convex (no V MLP)
        avp_cvx = AVPViT(backbone, AVPConfig(
            scene_grid_size=scene_grid, glimpse_grid_size=GLIMPSE_GRID,
            n_scene_registers=N_REGISTERS, use_output_proj=True,
            gating="full", layer_scale_init=1e-3,
        ))
        a_cvx = avp_step_flops(avp_cvx, backbone)

        # AVP convex + V MLP
        avp_cvx_mlp = AVPViT(backbone, AVPConfig(
            scene_grid_size=scene_grid, glimpse_grid_size=GLIMPSE_GRID,
            n_scene_registers=N_REGISTERS, use_output_proj=True,
            gating="full", layer_scale_init=1e-3,
            attention=AttentionConfig(write_v_expansion=2),
        ))
        a_cvx_mlp = avp_step_flops(avp_cvx_mlp, backbone)

        print(
            f"{scene_grid}x{scene_grid:<7} "
            f"{fmt(t_scene.total):<12} "
            f"{fmt(a_base.total):<12} "
            f"{fmt(a_mlp.total):<12} "
            f"{fmt(a_cvx.total):<12} "
            f"{fmt(a_cvx_mlp.total):<12}"
        )

    print()
    print("Legend:")
    print("  Teacher (scene)   = Full ViT at scene resolution")
    print("  AVP (base)        = LayerScale, V=Linear")
    print("  AVP+V_MLP         = LayerScale, V=MLP(2x expansion, SiLU)")
    print("  AVP convex        = ConvexGating (2x attention), V=Linear")
    print("  AVP cvx+MLP       = ConvexGating + V=MLP")
    print()

    # Detailed breakdown for largest scene
    print_detailed_breakdown(backbone, SCENE_GRIDS[-1], GLIMPSE_GRID, N_REGISTERS)


if __name__ == "__main__":
    main()
