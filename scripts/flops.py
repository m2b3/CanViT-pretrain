"""Cross-attention FLOPs comparison: Canvas Attention vs Regular Cross-Attention.

Formulas verified against model introspection.
"""

from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from canvit import CanViT
    from canvit.backbone.dinov3 import DINOv3Backbone


# =============================================================================
# Constants
# =============================================================================

# Linear(in, out) costs 2 * in * out FLOPs (matmul)
FLOPS_PER_LINEAR = 2
# Adapter = read + write = 2 directions
N_DIRECTIONS = 2


# =============================================================================
# Building block FLOPs
# =============================================================================


def sdpa_flops(n_q: int, n_kv: int, dim: int) -> int:
    """SDPA FLOPs: Q@K^T (2*n_q*n_kv*dim) + softmax@V (2*n_q*n_kv*dim)."""
    return 2 * FLOPS_PER_LINEAR * n_q * n_kv * dim


def canvas_proj_flops(n_local: int, local_dim: int, canvas_dim: int) -> int:
    """Canvas attention projection FLOPs (read + write): only local-side Linears.

    Per direction: 2 projections on local side (Q+O for read, K+V for write)
    Canvas side uses EWA (elementwise) → 0 FLOPs
    """
    projs_per_direction = 2  # Q+O or K+V on local side
    flops_per_proj = FLOPS_PER_LINEAR * n_local * local_dim * canvas_dim
    return N_DIRECTIONS * projs_per_direction * flops_per_proj


def regular_proj_flops(n_local: int, n_canvas: int, local_dim: int, canvas_dim: int) -> int:
    """Regular cross-attention projection FLOPs (read + write): local + canvas Linears.

    Per direction: 2 projections on local side + 2 projections on canvas side
    """
    projs_per_side = 2
    local_flops = N_DIRECTIONS * projs_per_side * FLOPS_PER_LINEAR * n_local * local_dim * canvas_dim
    canvas_flops = N_DIRECTIONS * projs_per_side * FLOPS_PER_LINEAR * n_canvas * canvas_dim * canvas_dim
    return local_flops + canvas_flops


# =============================================================================
# Aggregate FLOPs (per adapter = one read + one write)
# =============================================================================


def canvas_attention_flops(n_local: int, n_canvas: int, local_dim: int, canvas_dim: int) -> int:
    """Canvas attention: EWA on canvas side, Linear on local side."""
    proj = canvas_proj_flops(n_local, local_dim, canvas_dim)
    sdpa = sdpa_flops(n_local, n_canvas, canvas_dim) + sdpa_flops(n_canvas, n_local, canvas_dim)
    return proj + sdpa


def regular_attention_flops(n_local: int, n_canvas: int, local_dim: int, canvas_dim: int) -> int:
    """Regular cross-attention: all Linear projections (fair comparison with canvas_dim)."""
    proj = regular_proj_flops(n_local, n_canvas, local_dim, canvas_dim)
    sdpa = sdpa_flops(n_local, n_canvas, canvas_dim) + sdpa_flops(n_canvas, n_local, canvas_dim)
    return proj + sdpa


# =============================================================================
# Param formulas (per adapter)
# =============================================================================


def canvas_attention_params(local_dim: int, canvas_dim: int) -> int:
    """Canvas attention params: 4 Linear(local_dim, canvas_dim) + 4 EWA(canvas_dim)."""
    n_linear = 4  # Q, O for read; K, V for write
    linear = n_linear * local_dim * canvas_dim
    ewa = n_linear * 2 * canvas_dim  # scale + bias per EWA
    return linear + ewa


def regular_attention_params(local_dim: int, canvas_dim: int) -> int:
    """Regular cross-attention params: 4 Linear on local + 4 Linear on canvas."""
    n_linear_per_side = 4
    local_linear = n_linear_per_side * local_dim * canvas_dim
    canvas_linear = n_linear_per_side * canvas_dim * canvas_dim
    return local_linear + canvas_linear


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
    from dinov3.hub.backbones import dinov3_vitb16, dinov3_vitl16, dinov3_vits16  # pyright: ignore[reportAttributeAccessIssue]

    console = Console()

    backbones = [
        ("ViT-S", DINOv3Backbone(dinov3_vits16(pretrained=False))),
        ("ViT-B", DINOv3Backbone(dinov3_vitb16(pretrained=False))),
        ("ViT-L", DINOv3Backbone(dinov3_vitl16(pretrained=False))),
    ]

    for name, backbone in backbones:
        model = CanViT(backbone, CanViTConfig())
        _print_tables(console, name, backbone, model)
        console.print()


def _print_tables(console: Console, name: str, backbone: "DINOv3Backbone", model: "CanViT") -> None:
    local_dim = model.local_dim
    canvas_dim = model.canvas_dim
    n_backbone_prefix = backbone.n_prefix_tokens
    n_canvas_registers = model.n_canvas_registers
    patch_size = backbone.patch_size_px
    n_blocks = backbone.n_blocks
    n_adapters = model.n_adapters

    glimpse_grids = [2, 3]
    canvas_grids = [16, 32, 64, 128, 256, 512]

    console.rule(f"[bold]{name}[/bold] (local={local_dim}, canvas={canvas_dim})")

    # Table 1: Cross-Attention FLOPs per Adapter
    table = Table(title="Cross-Attention FLOPs per Adapter")
    table.add_column("Glimpse", style="bold")
    table.add_column("Canvas", style="bold")
    table.add_column("Pixels", style="dim")
    table.add_column("Tokens", style="dim", justify="right")
    table.add_column("Attention", style="bold")
    table.add_column("SDPA", justify="right")
    table.add_column("Proj", justify="right")
    table.add_column("Total", justify="right")

    for g in glimpse_grids:
        for c in canvas_grids:
            n_local = n_backbone_prefix + g * g  # prefix + patches
            n_canvas = 1 + n_canvas_registers + c * c  # CLS + registers + spatial
            canvas_px = c * patch_size

            # SDPA is same for both (read + write, in canvas_dim space)
            sdpa = sdpa_flops(n_local, n_canvas, canvas_dim) + sdpa_flops(n_canvas, n_local, canvas_dim)

            # Regular: local + canvas projections
            reg_proj = regular_proj_flops(n_local, n_canvas, local_dim, canvas_dim)
            reg_total = sdpa + reg_proj

            # Canvas: only local projections (canvas uses EWA)
            can_proj = canvas_proj_flops(n_local, local_dim, canvas_dim)
            can_total = sdpa + can_proj

            table.add_row(
                f"{g}×{g}",
                f"{c}×{c}",
                f"{canvas_px}px",
                f"{n_canvas:,}",
                "Regular",
                fmt(sdpa),
                fmt(reg_proj),
                fmt(reg_total),
            )
            table.add_row(
                "",
                "",
                "",
                "",
                "[cyan]Canvas[/cyan]",
                fmt(sdpa),
                f"[cyan]{fmt(can_proj)}[/cyan]",
                f"[cyan]{fmt(can_total)}[/cyan]",
            )
            table.add_row(
                "",
                "",
                "",
                "",
                "[green]×[/green]",
                "—",
                f"[green]×{reg_proj / can_proj:.1f}[/green]",
                f"[green]×{reg_total / can_total:.1f}[/green]",
                end_section=True,
            )

    console.print(table)
    console.print()

    # Params comparison
    canvas_p = canvas_attention_params(local_dim, canvas_dim)
    regular_p = regular_attention_params(local_dim, canvas_dim)
    console.print("[bold]Params per adapter:[/bold]")
    console.print(f"  Canvas:  {fmt(canvas_p)}")
    console.print(f"  Regular: {fmt(regular_p)} (×{regular_p / canvas_p:.1f})")
    console.print()

    # Table 2: Full Model FLOPs per Glimpse
    table2 = Table(title=f"Full Model FLOPs per Glimpse ({n_blocks} blocks, {n_adapters} adapters)")
    table2.add_column("Glimpse", style="bold")
    table2.add_column("Canvas", style="bold")
    table2.add_column("Backbone", justify="right")
    table2.add_column("Read (all)", justify="right")
    table2.add_column("Write (all)", justify="right")
    table2.add_column("Total", justify="right", style="bold")

    for g in glimpse_grids:
        n_local = n_backbone_prefix + g * g
        n_patches = g * g
        patch_embed = backbone.patch_embed_flops(n_patches)
        backbone_total = patch_embed + n_blocks * backbone.block_flops(n_local)

        for c in canvas_grids:
            n_canvas = 1 + n_canvas_registers + c * c

            read_flops_one: int = model.read_attn[0].flops(n_local, n_canvas)  # pyright: ignore[reportCallIssue]
            write_flops_one: int = model.write_attn[0].flops(n_canvas, n_local)  # pyright: ignore[reportCallIssue]
            read_total = n_adapters * read_flops_one
            write_total = n_adapters * write_flops_one

            total = backbone_total + read_total + write_total

            table2.add_row(
                f"{g}×{g}",
                f"{c}×{c}",
                fmt(backbone_total),
                fmt(read_total),
                fmt(write_total),
                fmt(total),
            )
        table2.add_section()

    console.print(table2)


def verify_formulas() -> None:
    """Verify formulas match actual model introspection."""
    from pathlib import Path

    from canvit import CanViT, CanViTConfig
    from canvit.attention import ScaledResidualAttention
    from canvit.backbone.dinov3 import DINOv3Backbone
    from dinov3.hub.backbones import dinov3_vits16

    ckpt = Path("dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
    if not ckpt.exists():
        print(f"Skipping verification: {ckpt} not found")
        return

    backbone = DINOv3Backbone(dinov3_vits16(weights=str(ckpt), pretrained=True))
    cfg = CanViTConfig()
    model = CanViT(backbone, cfg)

    local_dim = model.local_dim
    canvas_dim = model.canvas_dim
    glimpse_grid, canvas_grid = 2, 16
    n_local = backbone.n_prefix_tokens + glimpse_grid * glimpse_grid
    n_canvas = model.n_prefix + canvas_grid * canvas_grid

    # Get actual FLOPs from model
    read_attn: ScaledResidualAttention = model.read_attn[0]  # type: ignore[assignment]
    write_attn: ScaledResidualAttention = model.write_attn[0]  # type: ignore[assignment]
    actual_read = read_attn.flops(n_local, n_canvas)
    actual_write = write_attn.flops(n_canvas, n_local)
    actual_total = actual_read + actual_write

    # Formula prediction
    predicted = canvas_attention_flops(n_local, n_canvas, local_dim, canvas_dim)

    from rich import print as rprint
    rprint(f"\n[bold]Formula verification (2×2 → 16×16, local={local_dim}, canvas={canvas_dim}):[/bold]")
    rprint(f"  Actual (model):   {actual_total:,}")
    rprint(f"  Predicted:        {predicted:,}")
    assert actual_total == predicted, f"Mismatch: {actual_total} != {predicted}"
    rprint("  [green]✓ Match![/green]")


if __name__ == "__main__":
    main()
    verify_formulas()
