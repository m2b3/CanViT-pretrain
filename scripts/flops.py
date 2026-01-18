"""Cross-attention FLOPs comparison: Canvas Attention vs Regular Cross-Attention."""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import tyro
from canvit import flops
from canvit.hub import create_backbone
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from canvit import CanViT
    from canvit.backbone.dinov3 import DINOv3Backbone

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    """FLOPs analysis configuration."""

    student: str = "dinov3_vitb16"
    """Student backbone (e.g. dinov3_vits16, dinov3_vitb16, dinov3_vitl16)."""
    teacher: str = "dinov3_vitb16"
    """Teacher backbone for scene head output dim."""
    glimpse_grids: tuple[int, ...] = (8,)
    """Glimpse grid sizes to analyze."""
    canvas_grids: tuple[int, ...] = (16, 32, 64, 128, 256, 512)
    """Canvas grid sizes to analyze."""


# =============================================================================
# RoPE FLOPs (MISSING from canvit.flops - critical for overhead analysis!)
# =============================================================================


def rope_flops(n_spatial: int, head_dim: int, n_heads: int) -> int:
    """RoPE application FLOPs per sample: x * cos + rotate_half(x) * sin.

    Operations per element:
    - x * cos: 1 mul
    - rotate_half: 0.5 (negate half)
    - rotate_half * sin: 1 mul
    - addition: 1 add
    Total: 3.5 FLOPs per element

    Note: RoPE is memory-bound, not compute-bound. FLOPs don't predict
    actual runtime - see rope_memory_bytes() for bandwidth analysis.
    """
    n_elements = n_heads * n_spatial * head_dim
    return int(3.5 * n_elements)


def rope_memory_bytes(
    n_spatial: int, head_dim: int, n_heads: int, dtype_bytes: int = 2
) -> int:
    """Memory bytes moved by rope_apply per sample (unfused implementation).

    Current implementation reads/writes multiple times due to unfused ops:
    - Read x, sin, cos
    - Write intermediates from rotate_half (chunk creates temp in unfused impl)
    - Write x*cos, rotate_half*sin, final sum
    - dtype conversion overhead if x.dtype != rope.dtype

    Conservative estimate: ~4× the tensor size due to unfused ops.
    Optimal fused kernel would be ~2× (read inputs, write output).
    """
    tensor_bytes = n_heads * n_spatial * head_dim * dtype_bytes
    return 4 * tensor_bytes


def rope_adapter_flops(
    n_local_spatial: int,
    n_canvas_spatial: int,
    head_dim: int,
    n_heads: int,
) -> int:
    """RoPE FLOPs per sample for one adapter (read + write attention).

    Per attention: 2 rope calls (Q and K)
    Per adapter: 1 read + 1 write = 4 rope calls

    Read: Q on local, K on canvas
    Write: Q on canvas, K on local
    """
    local_rope = rope_flops(n_local_spatial, head_dim, n_heads)
    canvas_rope = rope_flops(n_canvas_spatial, head_dim, n_heads)
    # read: Q_local + K_canvas, write: Q_canvas + K_local
    return 2 * local_rope + 2 * canvas_rope


# =============================================================================
# Regular Cross-Attention FLOPs (Linear projections on both sides, for comparison)
# =============================================================================


def _regular_read_flops(
    n_local: int, n_canvas: int, local_dim: int, canvas_dim: int
) -> int:
    """Regular cross-attention: local queries canvas (all Linear projections)."""
    q_proj = flops.linear(n_local, local_dim, canvas_dim)
    k_proj = flops.linear(n_canvas, canvas_dim, canvas_dim)
    v_proj = flops.linear(n_canvas, canvas_dim, canvas_dim)
    o_proj = flops.linear(n_local, canvas_dim, local_dim)
    sdpa = flops.sdpa(n_local, n_canvas, canvas_dim)
    return q_proj + k_proj + v_proj + o_proj + sdpa


def _regular_write_flops(
    n_local: int, n_canvas: int, local_dim: int, canvas_dim: int
) -> int:
    """Regular cross-attention: canvas queries local (all Linear projections)."""
    q_proj = flops.linear(n_canvas, canvas_dim, canvas_dim)
    k_proj = flops.linear(n_local, local_dim, canvas_dim)
    v_proj = flops.linear(n_local, local_dim, canvas_dim)
    o_proj = flops.linear(n_canvas, canvas_dim, canvas_dim)
    sdpa = flops.sdpa(n_canvas, n_local, canvas_dim)
    return q_proj + k_proj + v_proj + o_proj + sdpa


def _regular_adapter_flops(
    n_local: int, n_canvas: int, local_dim: int, canvas_dim: int
) -> int:
    """One read + one write with regular cross-attention."""
    return _regular_read_flops(
        n_local, n_canvas, local_dim, canvas_dim
    ) + _regular_write_flops(n_local, n_canvas, local_dim, canvas_dim)


# =============================================================================
# Parameter counting
# =============================================================================


def _canvas_attention_params(local_dim: int, canvas_dim: int) -> int:
    """Canvas attention params per adapter (read + write)."""
    linear_params = 4 * local_dim * canvas_dim  # 4 Linears (Q, O on read; K, V on write)
    return linear_params


def _regular_attention_params(local_dim: int, canvas_dim: int) -> int:
    """Regular cross-attention params per adapter."""
    local_linear = 4 * local_dim * canvas_dim
    canvas_linear = 4 * canvas_dim * canvas_dim
    return local_linear + canvas_linear


# =============================================================================
# Decoder / Head FLOPs
# =============================================================================


def _policy_head_flops(embed_dim: int) -> int:
    """Policy head: LN + Linear(D→D) + Linear(D→3) on 1 token."""
    return (
        flops.layer_norm(1, embed_dim)
        + flops.linear(1, embed_dim, embed_dim)
        + flops.linear(1, embed_dim, 3)
    )


# =============================================================================
# Formatting
# =============================================================================


def fmt(flops: int) -> str:
    if flops >= 1e12:
        return f"{flops / 1e12:.2f}T"
    if flops >= 1e9:
        return f"{flops / 1e9:.2f}G"
    if flops >= 1e6:
        return f"{flops / 1e6:.1f}M"
    if flops >= 1e3:
        return f"{flops / 1e3:.1f}K"
    return str(flops)


# =============================================================================
# Main
# =============================================================================


def main(cfg: Config) -> None:
    from canvit import CanViT, CanViTConfig

    backbone = create_backbone(cfg.student, pretrained=False)
    teacher_backbone = create_backbone(cfg.teacher, pretrained=False)
    teacher_dim = teacher_backbone.embed_dim

    log.info("=== FLOPs Analysis Configuration ===")
    log.info(f"  Student: {cfg.student} (dim={backbone.embed_dim})")
    log.info(f"  Teacher: {cfg.teacher} (dim={teacher_dim})")
    log.info("")

    console = Console()
    model = CanViT(backbone=backbone, cfg=CanViTConfig())

    # === VALIDATE TOKEN COUNTS WITH ACTUAL FORWARD PASS ===
    _validate_token_counts(backbone, model, cfg)

    _print_tables(console, cfg, backbone, model, teacher_dim)
    console.print()


def _validate_token_counts(
    backbone: "DINOv3Backbone", model: "CanViT", cfg: Config
) -> None:
    """Validate that our token count formulas match actual model shapes."""
    import torch
    from canvit.viewpoint import Viewpoint

    log.info("=== Validating Token Counts ===")

    g = cfg.glimpse_grids[0]
    c = cfg.canvas_grids[0]
    patch_size = backbone.patch_size_px
    glimpse_px = g * patch_size

    # Expected counts from our formulas
    n_backbone_prefix = backbone.n_prefix_tokens  # CLS + registers
    n_extra_canvit = (1 if model.cfg.enable_vpe else 0) + 1  # VPE + recurrent_cls
    expected_teacher_tokens = n_backbone_prefix + g * g
    expected_canvit_local = expected_teacher_tokens + n_extra_canvit
    expected_canvas = model.cfg.n_canvas_registers + c * c

    # Run actual forward pass
    B = 1
    glimpse = torch.randn(B, 3, glimpse_px, glimpse_px)
    vp = Viewpoint(centers=torch.zeros(B, 2), scales=torch.ones(B))
    state = model.init_state(batch_size=B, canvas_grid_size=c)

    # Teacher forward
    teacher_out = backbone.forward_norm_features(glimpse)
    actual_teacher_patches = teacher_out.patches.shape[1]  # [B, n_patches, D]
    actual_teacher_patches + 1  # +1 for CLS (patches excludes CLS)

    # CanViT forward
    with torch.no_grad():
        out = model.forward(glimpse=glimpse, state=state, viewpoint=vp)

    actual_canvas = out.state.canvas.shape[1]

    # Validate
    log.info(
        f"  Glimpse grid: {g}×{g} = {g * g} patches @ {patch_size}px = {glimpse_px}px"
    )
    log.info(f"  Canvas grid:  {c}×{c} = {c * c} spatial tokens")
    log.info("")

    # Teacher tokens: patches output excludes CLS, so we check patches + 1
    # But n_prefix_tokens = 1 (CLS) + n_registers, and forward_norm_features
    # returns only patches (no CLS, no registers in output)
    log.info(
        f"  Teacher patches (output): expected={g * g}, actual={actual_teacher_patches}"
    )
    assert actual_teacher_patches == g * g, (
        f"Teacher patch count mismatch: expected {g * g}, got {actual_teacher_patches}"
    )

    log.info(f"  Canvas tokens: expected={expected_canvas}, actual={actual_canvas}")
    assert actual_canvas == expected_canvas, (
        f"Canvas token count mismatch: expected {expected_canvas}, got {actual_canvas}"
    )

    # Validate token breakdown
    log.info("")
    log.info("  Token breakdown:")
    log.info(f"    n_backbone_prefix (CLS + regs): {n_backbone_prefix}")
    log.info(f"      = 1 CLS + {backbone.n_register_tokens} registers")
    log.info(f"    n_extra_canvit (VPE + recurrent_cls): {n_extra_canvit}")
    log.info(f"      = {1 if model.cfg.enable_vpe else 0} VPE + 1 recurrent_cls")
    log.info(
        f"    Teacher tokens: {n_backbone_prefix} + {g * g} = {expected_teacher_tokens}"
    )
    log.info(
        f"    CanViT local:   {expected_teacher_tokens} + {n_extra_canvit} = {expected_canvit_local}"
    )
    log.info(
        f"    Canvas:         {model.cfg.n_canvas_registers} regs + {c * c} spatial = {expected_canvas}"
    )
    log.info("")
    log.info("  All token counts validated!")
    log.info("")


def _print_tables(
    console: Console,
    cfg: Config,
    backbone: "DINOv3Backbone",
    model: "CanViT",
    teacher_dim: int,
) -> None:
    local_dim = model.local_dim
    canvas_dim = model.canvas_dim
    n_canvas_registers = model.cfg.n_canvas_registers
    n_heads = model.cfg.canvas_num_heads
    patch_size = backbone.patch_size_px
    n_blocks = backbone.n_blocks
    n_backbone_prefix = backbone.n_prefix_tokens
    n_adapters = len(model.read_after_blocks)
    ffn_ratio = backbone.ffn_ratio

    # CanViT adds tokens beyond backbone: VPE (if enabled) + recurrent_cls
    # LocalTokens layout: [vpe?, recurrent_cls, ephemeral_cls, registers, patches]
    n_extra_canvit_tokens = (
        1 if model.cfg.enable_vpe else 0
    ) + 1  # VPE + recurrent_cls

    console.rule(f"[bold]{cfg.student}[/bold] (local={local_dim}, canvas={canvas_dim})")

    # Table 1: Cross-Attention FLOPs per Adapter
    table = Table(title="Cross-Attention FLOPs per Adapter (read + write)")
    table.add_column("Glimpse", style="bold")
    table.add_column("Canvas", style="bold")
    table.add_column("Pixels", style="dim")
    table.add_column("Tokens", style="dim", justify="right")
    table.add_column("Attention", style="bold")
    table.add_column("Total", justify="right")

    for g in cfg.glimpse_grids:
        for c in cfg.canvas_grids:
            # CanViT local stream includes VPE + recurrent_cls beyond backbone tokens
            n_canvit_local = n_backbone_prefix + g * g + n_extra_canvit_tokens
            n_canvas = n_canvas_registers + c * c
            canvas_px = c * patch_size

            can_total = flops.canvas_adapter(
                n_canvit_local, n_canvas, local_dim, canvas_dim
            )
            reg_total = _regular_adapter_flops(
                n_canvit_local, n_canvas, local_dim, canvas_dim
            )

            table.add_row(
                f"{g}×{g}",
                f"{c}×{c}",
                f"{canvas_px}px",
                f"{n_canvas:,}",
                "Regular",
                fmt(reg_total),
            )
            table.add_row(
                "",
                "",
                "",
                "",
                "[cyan]Canvas[/cyan]",
                f"[cyan]{fmt(can_total)}[/cyan]",
            )
            table.add_row(
                "",
                "",
                "",
                "",
                "[green]Savings[/green]",
                f"[green]×{reg_total / can_total:.1f}[/green]",
                end_section=True,
            )

    console.print(table)
    console.print()

    # Params comparison
    canvas_p = _canvas_attention_params(local_dim, canvas_dim)
    regular_p = _regular_attention_params(local_dim, canvas_dim)
    console.print("[bold]Params per adapter:[/bold]")
    console.print(f"  Canvas:  {fmt(canvas_p)}")
    console.print(f"  Regular: {fmt(regular_p)} (×{regular_p / canvas_p:.1f})")
    console.print()

    # Table 2: Full Model FLOPs per Glimpse
    policy_flops = _policy_head_flops(local_dim)

    table2 = Table(
        title=f"Full Model FLOPs per Glimpse ({n_blocks} blocks, {n_adapters} adapters)"
    )
    table2.add_column("Glimpse", style="bold")
    table2.add_column("Canvas", style="bold")
    table2.add_column("Mode", style="bold")
    table2.add_column("Total", justify="right")
    table2.add_column("ViT@Glimpse", justify="right", style="dim")
    table2.add_column("Overhead", justify="right", style="yellow")
    table2.add_column("ViT@Canvas", justify="right", style="dim")
    table2.add_column("Savings", justify="right", style="green")

    for g in cfg.glimpse_grids:
        n_patches = g * g
        # Teacher baseline: CLS + registers + patches
        n_teacher_tokens = n_backbone_prefix + n_patches
        # CanViT local stream: teacher tokens + VPE + recurrent_cls
        n_canvit_local = n_teacher_tokens + n_extra_canvit_tokens

        # Teacher backbone baseline (for comparison)
        patch_emb = flops.patch_embed(n_patches, patch_size, local_dim)
        teacher_blocks = n_blocks * flops.vit_block(
            n_teacher_tokens, local_dim, ffn_ratio
        )
        teacher_backbone_flops = patch_emb + teacher_blocks

        # CanViT backbone (processes more tokens due to VPE + recurrent_cls)
        canvit_blocks = n_blocks * flops.vit_block(n_canvit_local, local_dim, ffn_ratio)
        canvit_backbone_flops = patch_emb + canvit_blocks

        for c in cfg.canvas_grids:
            n_canvas = n_canvas_registers + c * c
            n_canvas_patches = c * c

            # Canvas attention (all adapters) - uses CanViT local token count
            adapters_flops = n_adapters * flops.canvas_adapter(
                n_canvit_local, n_canvas, local_dim, canvas_dim
            )

            # RoPE FLOPs (CRITICAL: was missing from original analysis!)
            # Spatial tokens only (registers don't get RoPE)
            n_local_spatial = n_patches  # glimpse patches
            n_canvas_spatial = c * c  # canvas spatial tokens
            head_dim = canvas_dim // n_heads
            total_rope_flops = n_adapters * rope_adapter_flops(
                n_local_spatial, n_canvas_spatial, head_dim, n_heads
            )

            # Heads: policy (per glimpse) + scene (per glimpse, scales with canvas²)
            scene_flops = flops.scene_head(n_canvas_patches, canvas_dim, teacher_dim)
            heads_flops = policy_flops + scene_flops

            # CanViT totals (uses canvit_backbone_flops, not teacher_backbone_flops)
            # NOTE: RoPE FLOPs are small but runtime is dominated by memory bandwidth!
            total_no_heads = canvit_backbone_flops + adapters_flops + total_rope_flops
            total_with_heads = total_no_heads + heads_flops

            # Baseline: ViT at canvas resolution
            canvas_patch_emb = flops.patch_embed(
                n_canvas_patches, patch_size, local_dim
            )
            canvas_n_tokens = n_backbone_prefix + n_canvas_patches
            canvas_blocks = n_blocks * flops.vit_block(
                canvas_n_tokens, local_dim, ffn_ratio
            )
            vit_at_canvas = canvas_patch_emb + canvas_blocks

            # Compare CanViT total vs teacher backbone (not canvit backbone)
            table2.add_row(
                f"{g}×{g}",
                f"{c}×{c}",
                "w/o Heads",
                fmt(total_no_heads),
                fmt(teacher_backbone_flops),
                f"×{total_no_heads / teacher_backbone_flops:.2f}",
                fmt(vit_at_canvas),
                f"×{vit_at_canvas / total_no_heads:.1f}",
            )
            table2.add_row(
                "",
                "",
                "[cyan]w/ Heads[/cyan]",
                f"[cyan]{fmt(total_with_heads)}[/cyan]",
                "",
                f"[cyan]×{total_with_heads / teacher_backbone_flops:.2f}[/cyan]",
                "",
                f"[cyan]×{vit_at_canvas / total_with_heads:.1f}[/cyan]",
                end_section=True,
            )

    console.print(table2)


if __name__ == "__main__":
    main(tyro.cli(Config))
