"""FLOPs calculations for AVP-ViT and teacher backbone.

Uses introspection on actual model instances - no hardcoded formulas that can drift.
"""

from pathlib import Path
from typing import NamedTuple

from avp_vit import AVPConfig, AVPViT
from avp_vit.backbone.dinov3 import DINOv3Backbone

CKPT_PATH = Path("dinov3_vits16_pretrain_lvd1689m-08c60483.pth")

# Scene grid sizes to evaluate (script parameter, not model config)
SCENE_GRIDS = [16, 32, 64, 128]


class TeacherFLOPs(NamedTuple):
    patch_embed: int
    blocks: int
    total: int
    n_tokens: int


class AVPStepFLOPs(NamedTuple):
    """FLOPs for one AVP step, computed via model introspection.

    Training includes scene_proj (projects to teacher_dim for loss).
    Inference excludes scene_proj (hidden state used directly).
    """

    glimpse_embed: int
    read_attn: int
    backbone: int
    write_attn: int
    scene_proj: int  # Only needed for training (loss computation)
    total_train: int
    total_infer: int
    # Token counts for reference
    n_local: int
    n_scene: int
    n_adapters: int


def teacher_flops(backbone: DINOv3Backbone, n_patches: int) -> TeacherFLOPs:
    n_tokens = n_patches + backbone.n_prefix_tokens
    patch_embed = backbone.patch_embed_flops(n_patches)
    blocks = backbone.n_blocks * backbone.block_flops(n_tokens)
    return TeacherFLOPs(patch_embed, blocks, patch_embed + blocks, n_tokens)


def avp_step_flops(model: AVPViT, scene_grid_size: int) -> AVPStepFLOPs:
    """FLOPs for one AVP step. Uses introspection on actual model modules."""
    assert isinstance(model.backbone, DINOv3Backbone)
    backbone = model.backbone
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
    # Only needed for training (loss computation against teacher)
    scene_proj = 2 * spatial_patches * D * model.teacher_dim

    total_infer = glimpse_embed + read_attn + blocks + write_attn
    total_train = total_infer + scene_proj
    return AVPStepFLOPs(
        glimpse_embed, read_attn, blocks, write_attn, scene_proj,
        total_train, total_infer, n_local, n_scene, n_adapters,
    )


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


def print_detailed_breakdown(avp: AVPViT, scene_grid: int) -> None:
    assert isinstance(avp.backbone, DINOv3Backbone)
    backbone = avp.backbone
    cfg = avp.cfg

    D = backbone.embed_dim
    P = backbone.patch_size
    n_blocks = backbone.n_blocks
    scene_px = scene_grid * P
    glimpse_px = cfg.glimpse_grid_size * P
    n_spatial = scene_grid**2

    a = avp_step_flops(avp, scene_grid)
    t_scene = teacher_flops(backbone, scene_grid**2)
    t_glimpse = teacher_flops(backbone, cfg.glimpse_grid_size**2)

    def pct_train(x: int) -> str:
        return f"({100 * x / a.total_train:4.1f}%)"

    def pct_infer(x: int) -> str:
        return f"({100 * x / a.total_infer:4.1f}%)"

    print("=" * 80)
    print(f"DETAILED: {scene_grid}x{scene_grid} scene ({scene_px}x{scene_px} px), gating={cfg.gating}")
    print("=" * 80)
    print(f"Tokens: local={a.n_local}, scene={a.n_scene} (spatial={n_spatial} + registers={cfg.n_scene_registers})")
    print(f"Adapters: {a.n_adapters} (backbone has {n_blocks} blocks, stride={cfg.adapter_stride})")
    print()

    print("--- INFERENCE (no scene_proj) ---")
    print(f"  Glimpse embed: {fmt(a.glimpse_embed):>10}  {pct_infer(a.glimpse_embed)}")
    print(f"  Read attn:     {fmt(a.read_attn):>10}  {pct_infer(a.read_attn)}  [{a.n_adapters} adapters]")
    print(f"  Backbone:      {fmt(a.backbone):>10}  {pct_infer(a.backbone)}  [{n_blocks} blocks on {a.n_local} tokens]")
    print(f"  Write attn:    {fmt(a.write_attn):>10}  {pct_infer(a.write_attn)}  [{a.n_adapters} adapters]")
    print(f"  TOTAL:         {fmt(a.total_infer):>10}")
    print()
    print("--- TRAINING (+scene_proj) ---")
    print(f"  Scene proj:    {fmt(a.scene_proj):>10}  {pct_train(a.scene_proj)}  [Linear on {n_spatial} tokens]")
    print(f"  TOTAL:         {fmt(a.total_train):>10}")
    print()

    # EWA savings: compare actual AVP vs hypothetical full-Linear (inference mode)
    convex = cfg.gating == "full"
    full_linear_attn = hypothetical_full_linear_flops(a.n_local, a.n_scene, D, a.n_adapters, convex)
    full_linear_infer = a.glimpse_embed + full_linear_attn + a.backbone
    actual_attn = a.read_attn + a.write_attn
    attn_savings = full_linear_attn - actual_attn

    print("=" * 80)
    print(f"EWA SAVINGS ({scene_grid}x{scene_grid} scene, inference)")
    print("=" * 80)
    print("Cross-attention uses EWA (ElementwiseAffine) instead of Linear for:")
    print("  Read:  K, V (on scene tokens)")
    print("  Write: Q, O (on scene tokens)")
    print()
    print(f"  Actual cross-attn:       {fmt(actual_attn):>10}")
    print(f"  Hypothetical full-Linear:{fmt(full_linear_attn):>10}")
    print(f"  Savings:                 {fmt(attn_savings):>10}  ({100 * attn_savings / full_linear_attn:.0f}% of full-Linear attn)")
    print()
    print(f"  Actual infer total:      {fmt(a.total_infer):>10}")
    print(f"  Hypothetical infer total:{fmt(full_linear_infer):>10}")
    print(f"  Savings:                 {fmt(full_linear_infer - a.total_infer):>10}  ({100 * (full_linear_infer - a.total_infer) / full_linear_infer:.0f}% of full-Linear)")
    print()

    print("=" * 80)
    print(f"COMPARISON ({scene_grid}x{scene_grid} scene = {scene_px}x{scene_px} px)")
    print("=" * 80)
    print(f"  Teacher @ {scene_grid}x{scene_grid} ({scene_px}px):   {fmt(t_scene.total):>10}")
    print(f"  Teacher @ {cfg.glimpse_grid_size}x{cfg.glimpse_grid_size} ({glimpse_px}px):    {fmt(t_glimpse.total):>10}")
    print(f"  AVP infer:                 {fmt(a.total_infer):>10}")
    print(f"  AVP train:                 {fmt(a.total_train):>10}")
    print()
    print(f"  AVP infer steps per Teacher({scene_grid}x{scene_grid}): {t_scene.total / a.total_infer:6.1f}")
    print(f"  Teacher({cfg.glimpse_grid_size}x{cfg.glimpse_grid_size}) per AVP infer:        {a.total_infer / t_glimpse.total:6.2f}")


def main() -> None:
    from dinov3.hub.backbones import dinov3_vits16

    backbone = DINOv3Backbone(dinov3_vits16(weights=str(CKPT_PATH), pretrained=True))
    D = backbone.embed_dim

    # Create default AVP to get config values
    avp_default = AVPViT(backbone, AVPConfig(), teacher_dim=D)
    cfg = avp_default.cfg
    P = backbone.patch_size

    print("=" * 80)
    print("FLOPs Calculator for AVP-ViT")
    print("=" * 80)
    print(f"Backbone: DINOv3 ViT-S/16 (D={D}, heads={backbone.num_heads}, blocks={backbone.n_blocks}, P={P})")
    print(f"AVPConfig defaults: glimpse={cfg.glimpse_grid_size}x{cfg.glimpse_grid_size} ({cfg.glimpse_grid_size * P}px), registers={cfg.n_scene_registers}, stride={cfg.adapter_stride}")
    print()

    # Table header
    print(f"{'Scene':<8} {'Teacher':<10} {'AVP infer':<10} {'AVP train':<10} {'cvx infer':<10} {'cvx train':<10}")
    print("-" * 68)

    for scene_grid in SCENE_GRIDS:
        t_scene = teacher_flops(backbone, scene_grid**2)

        # Default AVP (LayerScale gating) - reuse if already created
        a_default = avp_step_flops(avp_default, scene_grid)

        # AVP with convex gating
        avp_cvx = AVPViT(backbone, AVPConfig(gating="full", layer_scale_init=1e-3), teacher_dim=D)
        a_cvx = avp_step_flops(avp_cvx, scene_grid)

        print(
            f"{scene_grid}x{scene_grid:<5} "
            f"{fmt(t_scene.total):<10} "
            f"{fmt(a_default.total_infer):<10} "
            f"{fmt(a_default.total_train):<10} "
            f"{fmt(a_cvx.total_infer):<10} "
            f"{fmt(a_cvx.total_train):<10}"
        )

    print()
    print("Legend:")
    print("  Teacher        = Full ViT at scene resolution")
    print(f"  AVP infer      = gating={cfg.gating}, no scene_proj")
    print(f"  AVP train      = gating={cfg.gating}, +scene_proj for loss")
    print("  cvx infer/train = ConvexGating (2x attention per adapter)")
    print()

    # Detailed breakdown for largest scene using default config
    print_detailed_breakdown(avp_default, SCENE_GRIDS[-1])


if __name__ == "__main__":
    main()
