"""Cross-attention FLOPs comparison: Canvas Attention vs Regular Cross-Attention."""

from typing import TYPE_CHECKING

from canvit import flops
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from canvit import CanViT
    from canvit.backbone.dinov3 import DINOv3Backbone


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


def _canvas_attention_params(
    local_dim: int, canvas_dim: int, use_ewa: bool = True
) -> int:
    """Canvas attention params per adapter (read + write)."""
    linear_params = 4 * local_dim * canvas_dim  # 4 Linears
    ewa_params = 4 * 2 * canvas_dim if use_ewa else 0  # 4 EWAs, each has scale + bias
    return linear_params + ewa_params


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


def main() -> None:
    from canvit import CanViT, CanViTConfig
    from canvit.backbone.dinov3 import DINOv3Backbone
    from dinov3.hub.backbones import (
        dinov3_vitb16,  # pyright: ignore[reportAttributeAccessIssue]
    )

    console = Console()

    backbones = [("ViT-B", DINOv3Backbone(dinov3_vitb16(pretrained=False)))]

    for name, backbone in backbones:
        model = CanViT(backbone=backbone, cfg=CanViTConfig())
        _print_tables(console, name, backbone, model)
        console.print()


def _print_tables(
    console: Console, name: str, backbone: "DINOv3Backbone", model: "CanViT"
) -> None:
    local_dim = model.local_dim
    canvas_dim = model.canvas_dim
    n_canvas_registers = model.cfg.n_canvas_registers
    patch_size = backbone.patch_size_px
    n_blocks = backbone.n_blocks
    n_backbone_prefix = backbone.n_prefix_tokens
    n_adapters = len(model.read_after_blocks)
    ffn_ratio = backbone.ffn_ratio

    glimpse_grids = [8]
    canvas_grids = [16, 32, 64, 128, 256, 512]

    console.rule(f"[bold]{name}[/bold] (local={local_dim}, canvas={canvas_dim})")

    # Table 1: Cross-Attention FLOPs per Adapter
    table = Table(title="Cross-Attention FLOPs per Adapter (read + write)")
    table.add_column("Glimpse", style="bold")
    table.add_column("Canvas", style="bold")
    table.add_column("Pixels", style="dim")
    table.add_column("Tokens", style="dim", justify="right")
    table.add_column("Attention", style="bold")
    table.add_column("Total", justify="right")

    for g in glimpse_grids:
        for c in canvas_grids:
            n_local = n_backbone_prefix + g * g
            n_canvas = n_canvas_registers + c * c
            canvas_px = c * patch_size

            can_total = flops.canvas_adapter(n_local, n_canvas, local_dim, canvas_dim)
            reg_total = _regular_adapter_flops(n_local, n_canvas, local_dim, canvas_dim)

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
    teacher_dim = 768  # ViT-B teacher
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

    for g in glimpse_grids:
        n_local = n_backbone_prefix + g * g
        n_patches = g * g

        # Backbone: patch embed + blocks
        patch_emb = flops.patch_embed(n_patches, patch_size, local_dim)
        blocks = n_blocks * flops.vit_block(n_local, local_dim, ffn_ratio)
        backbone_flops = patch_emb + blocks

        for c in canvas_grids:
            n_canvas = n_canvas_registers + c * c
            n_canvas_patches = c * c

            # Canvas attention (all adapters)
            adapters_flops = n_adapters * flops.canvas_adapter(
                n_local, n_canvas, local_dim, canvas_dim
            )

            # Heads: policy (per glimpse) + scene (per glimpse, scales with canvas²)
            scene_flops = flops.scene_head(n_canvas_patches, canvas_dim, teacher_dim)
            heads_flops = policy_flops + scene_flops

            total_no_heads = backbone_flops + adapters_flops
            total_with_heads = total_no_heads + heads_flops

            # Baseline: ViT at canvas resolution
            canvas_patch_emb = flops.patch_embed(n_canvas_patches, patch_size, local_dim)
            canvas_n_tokens = n_backbone_prefix + n_canvas_patches
            canvas_blocks = n_blocks * flops.vit_block(canvas_n_tokens, local_dim, ffn_ratio)
            vit_at_canvas = canvas_patch_emb + canvas_blocks

            table2.add_row(
                f"{g}×{g}",
                f"{c}×{c}",
                "w/o Heads",
                fmt(total_no_heads),
                fmt(backbone_flops),
                f"×{total_no_heads / backbone_flops:.2f}",
                fmt(vit_at_canvas),
                f"×{vit_at_canvas / total_no_heads:.1f}",
            )
            table2.add_row(
                "",
                "",
                "[cyan]w/ Heads[/cyan]",
                f"[cyan]{fmt(total_with_heads)}[/cyan]",
                "",
                f"[cyan]×{total_with_heads / backbone_flops:.2f}[/cyan]",
                "",
                f"[cyan]×{vit_at_canvas / total_with_heads:.1f}[/cyan]",
                end_section=True,
            )

    console.print(table2)


if __name__ == "__main__":
    main()
