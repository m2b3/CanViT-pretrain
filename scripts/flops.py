"""Cross-attention FLOPs comparison: Canvas Attention vs Regular Cross-Attention.

Formulas derived from reading canvit source code.
"""

from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from canvit import CanViT
    from canvit.backbone.dinov3 import DINOv3Backbone


# =============================================================================
# FLOP counting primitives
# =============================================================================

# Linear(in, out): 2 * n * in * out FLOPs (matmul)
# SDPA(n_q, n_kv, dim): Q@K^T = 2*n_q*n_kv*dim, softmax@V = 2*n_q*n_kv*dim → 4*n_q*n_kv*dim


def linear_flops(n_tokens: int, in_dim: int, out_dim: int) -> int:
    return 2 * n_tokens * in_dim * out_dim


def sdpa_flops(n_q: int, n_kv: int, dim: int) -> int:
    return 4 * n_q * n_kv * dim


# =============================================================================
# Canvas Attention FLOPs (EWA on canvas side, Linear on local side)
# =============================================================================


def canvas_read_flops(
    n_local: int, n_canvas: int, local_dim: int, canvas_dim: int
) -> int:
    """CanvasReadAttention: local queries canvas.

    Q: Linear(local_dim → canvas_dim) on n_local tokens
    K: EWA (0 FLOPs)
    V: EWA (0 FLOPs)
    O: Linear(canvas_dim → local_dim) on n_local tokens
    SDPA: n_local × n_canvas in canvas_dim
    """
    q_proj = linear_flops(n_local, local_dim, canvas_dim)
    o_proj = linear_flops(n_local, canvas_dim, local_dim)
    sdpa = sdpa_flops(n_local, n_canvas, canvas_dim)
    return q_proj + o_proj + sdpa


def canvas_write_flops(
    n_local: int, n_canvas: int, local_dim: int, canvas_dim: int
) -> int:
    """CanvasWriteAttention: canvas queries local.

    Q: EWA (0 FLOPs)
    K: Linear(local_dim → canvas_dim) on n_local tokens
    V: Linear(local_dim → canvas_dim) on n_local tokens
    O: EWA (0 FLOPs)
    SDPA: n_canvas × n_local in canvas_dim
    """
    k_proj = linear_flops(n_local, local_dim, canvas_dim)
    v_proj = linear_flops(n_local, local_dim, canvas_dim)
    sdpa = sdpa_flops(n_canvas, n_local, canvas_dim)
    return k_proj + v_proj + sdpa


def canvas_adapter_flops(
    n_local: int, n_canvas: int, local_dim: int, canvas_dim: int
) -> int:
    """One read + one write = one adapter."""
    return canvas_read_flops(
        n_local, n_canvas, local_dim, canvas_dim
    ) + canvas_write_flops(n_local, n_canvas, local_dim, canvas_dim)


# =============================================================================
# Regular Cross-Attention FLOPs (Linear projections on both sides)
# =============================================================================


def regular_read_flops(
    n_local: int, n_canvas: int, local_dim: int, canvas_dim: int
) -> int:
    """Regular cross-attention: local queries canvas.

    All projections are Linear.
    Q: Linear(local_dim → canvas_dim) on n_local tokens
    K: Linear(canvas_dim → canvas_dim) on n_canvas tokens
    V: Linear(canvas_dim → canvas_dim) on n_canvas tokens
    O: Linear(canvas_dim → local_dim) on n_local tokens
    SDPA: n_local × n_canvas in canvas_dim
    """
    q_proj = linear_flops(n_local, local_dim, canvas_dim)
    k_proj = linear_flops(n_canvas, canvas_dim, canvas_dim)
    v_proj = linear_flops(n_canvas, canvas_dim, canvas_dim)
    o_proj = linear_flops(n_local, canvas_dim, local_dim)
    sdpa = sdpa_flops(n_local, n_canvas, canvas_dim)
    return q_proj + k_proj + v_proj + o_proj + sdpa


def regular_write_flops(
    n_local: int, n_canvas: int, local_dim: int, canvas_dim: int
) -> int:
    """Regular cross-attention: canvas queries local.

    Q: Linear(canvas_dim → canvas_dim) on n_canvas tokens
    K: Linear(local_dim → canvas_dim) on n_local tokens
    V: Linear(local_dim → canvas_dim) on n_local tokens
    O: Linear(canvas_dim → canvas_dim) on n_canvas tokens
    SDPA: n_canvas × n_local in canvas_dim
    """
    q_proj = linear_flops(n_canvas, canvas_dim, canvas_dim)
    k_proj = linear_flops(n_local, local_dim, canvas_dim)
    v_proj = linear_flops(n_local, local_dim, canvas_dim)
    o_proj = linear_flops(n_canvas, canvas_dim, canvas_dim)
    sdpa = sdpa_flops(n_canvas, n_local, canvas_dim)
    return q_proj + k_proj + v_proj + o_proj + sdpa


def regular_adapter_flops(
    n_local: int, n_canvas: int, local_dim: int, canvas_dim: int
) -> int:
    """One read + one write with regular cross-attention."""
    return regular_read_flops(
        n_local, n_canvas, local_dim, canvas_dim
    ) + regular_write_flops(n_local, n_canvas, local_dim, canvas_dim)


# =============================================================================
# Backbone FLOPs (DINOv3 ViT)
# =============================================================================


def backbone_block_flops(n_tokens: int, embed_dim: int, ffn_ratio: float) -> int:
    """FLOPs for one transformer block.

    Self-attention:
      - QKV projection: 2 * n * D * 3D
      - SDPA: 4 * n² * D
      - Out projection: 2 * n * D * D
      Total: 8 * n * D² + 4 * n² * D

    FFN (MLP):
      - fc1: 2 * n * D * (ffn_ratio * D)
      - fc2: 2 * n * (ffn_ratio * D) * D
      Total: 4 * ffn_ratio * n * D²
    """
    D = embed_dim
    n = n_tokens
    ffn_dim = int(ffn_ratio * D)

    # Self-attention
    qkv_proj = linear_flops(n, D, 3 * D)
    attn_sdpa = sdpa_flops(n, n, D)
    out_proj = linear_flops(n, D, D)
    self_attn = qkv_proj + attn_sdpa + out_proj

    # FFN
    fc1 = linear_flops(n, D, ffn_dim)
    fc2 = linear_flops(n, ffn_dim, D)
    ffn = fc1 + fc2

    return self_attn + ffn


def patch_embed_flops(
    n_patches: int, patch_size: int, embed_dim: int, in_chans: int = 3
) -> int:
    """Patch embedding convolution FLOPs.

    Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    FLOPs per output position: 2 * kernel_size² * in_chans * out_chans
    """
    kernel_flops = 2 * (patch_size**2) * in_chans * embed_dim
    return n_patches * kernel_flops


# =============================================================================
# Parameter counting
# =============================================================================


def canvas_attention_params(
    local_dim: int, canvas_dim: int, use_ewa: bool = True
) -> int:
    """Canvas attention params per adapter (read + write).

    Read: Q Linear + O Linear + (K EWA + V EWA if use_ewa)
    Write: K Linear + V Linear + (Q EWA + O EWA if use_ewa)
    """
    linear_params = 4 * local_dim * canvas_dim  # 4 Linears
    ewa_params = 4 * 2 * canvas_dim if use_ewa else 0  # 4 EWAs, each has scale + bias
    return linear_params + ewa_params


def regular_attention_params(local_dim: int, canvas_dim: int) -> int:
    """Regular cross-attention params per adapter.

    Read: Q Linear(local→canvas) + K,V Linear(canvas→canvas) + O Linear(canvas→local)
    Write: Q,O Linear(canvas→canvas) + K,V Linear(local→canvas)
    """
    # Read: 2 * local_dim * canvas_dim + 2 * canvas_dim * canvas_dim
    # Write: 2 * canvas_dim * canvas_dim + 2 * local_dim * canvas_dim
    local_linear = 4 * local_dim * canvas_dim
    canvas_linear = 4 * canvas_dim * canvas_dim
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
    from dinov3.hub.backbones import dinov3_vitb16, dinov3_vitl16, dinov3_vits16

    console = Console()

    backbones = [
        # ("ViT-S", DINOv3Backbone(dinov3_vits16(pretrained=False))),
        ("ViT-B", DINOv3Backbone(dinov3_vitb16(pretrained=False))),
        # ("ViT-L", DINOv3Backbone(dinov3_vitl16(pretrained=False))),
    ]

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

    glimpse_grids = [3]
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

            can_total = canvas_adapter_flops(n_local, n_canvas, local_dim, canvas_dim)
            reg_total = regular_adapter_flops(n_local, n_canvas, local_dim, canvas_dim)

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
    canvas_p = canvas_attention_params(local_dim, canvas_dim)
    regular_p = regular_attention_params(local_dim, canvas_dim)
    console.print("[bold]Params per adapter:[/bold]")
    console.print(f"  Canvas:  {fmt(canvas_p)}")
    console.print(f"  Regular: {fmt(regular_p)} (×{regular_p / canvas_p:.1f})")
    console.print()

    # Table 2: Full Model FLOPs per Glimpse
    table2 = Table(
        title=f"Full Model FLOPs per Glimpse ({n_blocks} blocks, {n_adapters} adapters)"
    )
    table2.add_column("Glimpse", style="bold")
    table2.add_column("Canvas", style="bold")
    table2.add_column("Backbone", justify="right")
    table2.add_column("Adapters", justify="right")
    table2.add_column("Total", justify="right", style="bold")
    table2.add_column("Backbone@Canvas", justify="right", style="dim")
    table2.add_column("Savings", justify="right", style="green")

    for g in glimpse_grids:
        n_local = n_backbone_prefix + g * g
        n_patches = g * g

        # Backbone: patch embed + blocks
        patch_emb = patch_embed_flops(n_patches, patch_size, local_dim)
        blocks = n_blocks * backbone_block_flops(n_local, local_dim, ffn_ratio)
        backbone_total = patch_emb + blocks

        for c in canvas_grids:
            n_canvas = n_canvas_registers + c * c
            n_canvas_patches = c * c

            # Canvas attention (all adapters)
            adapters_total = n_adapters * canvas_adapter_flops(
                n_local, n_canvas, local_dim, canvas_dim
            )

            total = backbone_total + adapters_total

            # Baseline: backbone at canvas resolution
            canvas_patch_emb = patch_embed_flops(n_canvas_patches, patch_size, local_dim)
            canvas_n_tokens = n_backbone_prefix + n_canvas_patches
            canvas_blocks = n_blocks * backbone_block_flops(
                canvas_n_tokens, local_dim, ffn_ratio
            )
            backbone_at_canvas = canvas_patch_emb + canvas_blocks

            savings = backbone_at_canvas / total

            table2.add_row(
                f"{g}×{g}",
                f"{c}×{c}",
                fmt(backbone_total),
                fmt(adapters_total),
                fmt(total),
                fmt(backbone_at_canvas),
                f"×{savings:.1f}",
            )
        table2.add_section()

    console.print(table2)


if __name__ == "__main__":
    main()
