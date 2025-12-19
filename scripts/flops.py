"""FLOPs calculations for ActiveCanViT and teacher backbone.

Uses introspection on actual model instances - no hardcoded formulas that can drift.
"""

from pathlib import Path
from typing import NamedTuple, cast

from avp_vit import ActiveCanViT, ActiveCanViTConfig
from canvit.attention import ScaledResidualAttention
from canvit.backbone.dinov3 import DINOv3Backbone

CKPT_PATH = Path("dinov3_vits16_pretrain_lvd1689m-08c60483.pth")

CANVAS_GRIDS = [16, 32, 64, 128]


class TeacherFLOPs(NamedTuple):
    patch_embed: int
    blocks: int
    total: int
    n_tokens: int


class StepFLOPs(NamedTuple):
    """FLOPs for one forward step, computed via model introspection."""

    glimpse_embed: int
    read_attn: int
    backbone: int
    write_attn: int
    scene_proj: int
    total_train: int
    total_infer: int
    n_local: int
    n_canvas: int
    n_adapters: int


def teacher_flops(backbone: DINOv3Backbone, n_patches: int) -> TeacherFLOPs:
    n_tokens = n_patches + backbone.n_prefix_tokens
    patch_embed = backbone.patch_embed_flops(n_patches)
    blocks = backbone.n_blocks * backbone.block_flops(n_tokens)
    return TeacherFLOPs(patch_embed, blocks, patch_embed + blocks, n_tokens)


def step_flops(model: ActiveCanViT, canvas_grid_size: int) -> StepFLOPs:
    """FLOPs for one step. Uses introspection on actual model modules."""
    assert isinstance(model.backbone, DINOv3Backbone)
    backbone = model.backbone
    canvit = model.canvit
    cfg = model.cfg

    D = backbone.embed_dim
    n_blocks = backbone.n_blocks
    glimpse_patches = cfg.glimpse_grid_size ** 2
    spatial_patches = canvas_grid_size ** 2
    n_local = glimpse_patches + backbone.n_prefix_tokens
    n_canvas = model.n_canvas_registers + spatial_patches

    glimpse_embed = backbone.patch_embed_flops(glimpse_patches)

    n_adapters = canvit.n_adapters
    read_attn = sum(cast(ScaledResidualAttention, canvit.read_attn[i]).flops(n_local, n_canvas) for i in range(n_adapters))
    blocks = n_blocks * backbone.block_flops(n_local)
    write_attn = sum(cast(ScaledResidualAttention, canvit.write_attn[i]).flops(n_canvas, n_local) for i in range(n_adapters))

    scene_proj = 2 * spatial_patches * D * model.teacher_dim

    total_infer = glimpse_embed + read_attn + blocks + write_attn
    total_train = total_infer + scene_proj
    return StepFLOPs(
        glimpse_embed, read_attn, blocks, write_attn, scene_proj,
        total_train, total_infer, n_local, n_canvas, n_adapters,
    )


def hypothetical_full_linear_flops(
    n_local: int,
    n_canvas: int,
    dim: int,
    n_adapters: int,
) -> int:
    """Hypothetical FLOPs if all cross-attention used full Linear projections (no EWA)."""

    def single_attn_flops(n_q: int, n_kv: int) -> int:
        sdpa = 4 * n_q * n_kv * dim
        q_proj = 2 * n_q * dim * dim
        k_proj = 2 * n_kv * dim * dim
        v_proj = 2 * n_kv * dim * dim
        o_proj = 2 * n_q * dim * dim
        return sdpa + q_proj + k_proj + v_proj + o_proj

    read_per_adapter = single_attn_flops(n_local, n_canvas)
    write_per_adapter = single_attn_flops(n_canvas, n_local)
    return n_adapters * (read_per_adapter + write_per_adapter)


def fmt(f: int | float) -> str:
    if f >= 1e12:
        return f"{f / 1e12:.2f}T"
    if f >= 1e9:
        return f"{f / 1e9:.2f}G"
    if f >= 1e6:
        return f"{f / 1e6:.2f}M"
    return f"{f:.0f}"


def print_detailed_breakdown(model: ActiveCanViT, canvas_grid: int) -> None:
    assert isinstance(model.backbone, DINOv3Backbone)
    backbone = model.backbone
    cfg = model.cfg
    canvit_cfg = cfg.canvit

    D = backbone.embed_dim
    P = backbone.patch_size_px
    n_blocks = backbone.n_blocks
    canvas_px = canvas_grid * P
    glimpse_px = cfg.glimpse_grid_size * P
    n_spatial = canvas_grid ** 2

    s = step_flops(model, canvas_grid)
    t_canvas = teacher_flops(backbone, canvas_grid ** 2)
    t_glimpse = teacher_flops(backbone, cfg.glimpse_grid_size ** 2)

    def pct_train(x: int) -> str:
        return f"({100 * x / s.total_train:4.1f}%)"

    def pct_infer(x: int) -> str:
        return f"({100 * x / s.total_infer:4.1f}%)"

    print("=" * 80)
    print(f"DETAILED: {canvas_grid}x{canvas_grid} canvas ({canvas_px}x{canvas_px} px)")
    print("=" * 80)
    print(f"Tokens: local={s.n_local}, canvas={s.n_canvas} (spatial={n_spatial} + registers={canvit_cfg.n_canvas_registers})")
    print(f"Adapters: {s.n_adapters} (backbone has {n_blocks} blocks, stride={canvit_cfg.adapter_stride})")
    print()

    print("--- INFERENCE (no scene_proj) ---")
    print(f"  Glimpse embed: {fmt(s.glimpse_embed):>10}  {pct_infer(s.glimpse_embed)}")
    print(f"  Read attn:     {fmt(s.read_attn):>10}  {pct_infer(s.read_attn)}  [{s.n_adapters} adapters]")
    print(f"  Backbone:      {fmt(s.backbone):>10}  {pct_infer(s.backbone)}  [{n_blocks} blocks on {s.n_local} tokens]")
    print(f"  Write attn:    {fmt(s.write_attn):>10}  {pct_infer(s.write_attn)}  [{s.n_adapters} adapters]")
    print(f"  TOTAL:         {fmt(s.total_infer):>10}")
    print()
    print("--- TRAINING (+scene_proj) ---")
    print(f"  Scene proj:    {fmt(s.scene_proj):>10}  {pct_train(s.scene_proj)}  [Linear on {n_spatial} tokens]")
    print(f"  TOTAL:         {fmt(s.total_train):>10}")
    print()

    full_linear_attn = hypothetical_full_linear_flops(s.n_local, s.n_canvas, D, s.n_adapters)
    full_linear_infer = s.glimpse_embed + full_linear_attn + s.backbone
    actual_attn = s.read_attn + s.write_attn
    attn_savings = full_linear_attn - actual_attn

    print("=" * 80)
    print(f"EWA SAVINGS ({canvas_grid}x{canvas_grid} canvas, inference)")
    print("=" * 80)
    print("Cross-attention uses EWA (ElementwiseAffine) instead of Linear for:")
    print("  Read:  K, V (on canvas tokens)")
    print("  Write: Q, O (on canvas tokens)")
    print()
    print(f"  Actual cross-attn:       {fmt(actual_attn):>10}")
    print(f"  Hypothetical full-Linear:{fmt(full_linear_attn):>10}")
    print(f"  Savings:                 {fmt(attn_savings):>10}  ({100 * attn_savings / full_linear_attn:.0f}% of full-Linear attn)")
    print()
    print(f"  Actual infer total:      {fmt(s.total_infer):>10}")
    print(f"  Hypothetical infer total:{fmt(full_linear_infer):>10}")
    print(f"  Savings:                 {fmt(full_linear_infer - s.total_infer):>10}  ({100 * (full_linear_infer - s.total_infer) / full_linear_infer:.0f}% of full-Linear)")
    print()

    print("=" * 80)
    print(f"COMPARISON ({canvas_grid}x{canvas_grid} canvas = {canvas_px}x{canvas_px} px)")
    print("=" * 80)
    print(f"  Teacher @ {canvas_grid}x{canvas_grid} ({canvas_px}px):   {fmt(t_canvas.total):>10}")
    print(f"  Teacher @ {cfg.glimpse_grid_size}x{cfg.glimpse_grid_size} ({glimpse_px}px):    {fmt(t_glimpse.total):>10}")
    print(f"  Model infer:               {fmt(s.total_infer):>10}")
    print(f"  Model train:               {fmt(s.total_train):>10}")
    print()
    print(f"  Model infer steps per Teacher({canvas_grid}x{canvas_grid}): {t_canvas.total / s.total_infer:6.1f}")
    print(f"  Teacher({cfg.glimpse_grid_size}x{cfg.glimpse_grid_size}) per Model infer:      {s.total_infer / t_glimpse.total:6.2f}")


def main() -> None:
    from dinov3.hub.backbones import dinov3_vits16

    backbone = DINOv3Backbone(dinov3_vits16(weights=str(CKPT_PATH), pretrained=True))
    D = backbone.embed_dim
    P = backbone.patch_size_px

    model = ActiveCanViT(backbone, ActiveCanViTConfig(), teacher_dim=D)
    cfg = model.cfg
    canvit_cfg = cfg.canvit

    print("=" * 80)
    print("FLOPs Calculator for ActiveCanViT")
    print("=" * 80)
    print(f"Backbone: DINOv3 ViT-S/16 (D={D}, heads={backbone.num_heads}, blocks={backbone.n_blocks}, P={P})")
    print(f"Config: glimpse={cfg.glimpse_grid_size}x{cfg.glimpse_grid_size} ({cfg.glimpse_grid_size * P}px), "
          f"registers={canvit_cfg.n_canvas_registers}, stride={canvit_cfg.adapter_stride}")
    print()

    print(f"{'Canvas':<10} {'Teacher':<10} {'Infer':<10} {'Train':<10}")
    print("-" * 44)

    for canvas_grid in CANVAS_GRIDS:
        t = teacher_flops(backbone, canvas_grid ** 2)
        s = step_flops(model, canvas_grid)

        print(
            f"{canvas_grid}x{canvas_grid:<7} "
            f"{fmt(t.total):<10} "
            f"{fmt(s.total_infer):<10} "
            f"{fmt(s.total_train):<10}"
        )

    print()
    print("Legend:")
    print("  Teacher = Full ViT at canvas resolution")
    print("  Infer   = One step without scene_proj")
    print("  Train   = One step with scene_proj for loss")
    print()

    print_detailed_breakdown(model, CANVAS_GRIDS[-1])


if __name__ == "__main__":
    main()
