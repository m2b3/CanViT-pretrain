"""CanViT: Canvas Cross-Attention Vision Transformer.

Resolution-decoupled ViT backbone using asymmetric cross-attention
between a small local stream and a large canvas state.
"""

import math
from dataclasses import dataclass, field
from typing import final

import torch
from torch import Tensor, nn

from canvit.attention import (
    CrossAttentionConfig,
    RoPEReadCrossAttention,
    RoPEWriteCrossAttention,
    ScaledResidualAttention,
)
from canvit.backbone import ViTBackbone


@final
@dataclass
class CanViTConfig:
    """CanViT configuration."""

    n_registers: int = 32
    adapter_stride: int = 2
    layer_scale_init: float = 1e-3
    read_attention: CrossAttentionConfig = field(default_factory=CrossAttentionConfig)
    write_attention: CrossAttentionConfig = field(default_factory=CrossAttentionConfig)


@final
class CanViT(nn.Module):
    """Canvas Cross-Attention ViT.

    Complete recurrent backbone with canvas init and normalization.
    Canvas layout: [cls | registers | spatial].

    Asymmetric projections avoid O(D²) cost on large canvas:
      - READ: Q, O are Linear on local; K, V are EWA on canvas
      - WRITE: Q, O are EWA on canvas; K, V are Linear on local
    """

    backbone: ViTBackbone
    cfg: CanViTConfig
    read_attn: nn.ModuleList
    write_attn: nn.ModuleList
    # Canvas init
    cls_init: nn.Parameter
    spatial_init: nn.Parameter
    registers: nn.Parameter
    # Canvas normalization (always enabled)
    cls_ln: nn.LayerNorm
    reg_ln: nn.LayerNorm
    spatial_ln: nn.LayerNorm

    def __init__(self, backbone: ViTBackbone, cfg: CanViTConfig) -> None:
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg

        dim = backbone.embed_dim
        heads = backbone.num_heads
        n_blocks = backbone.n_blocks
        n_adapters = (n_blocks - 1) // cfg.adapter_stride
        scale = 1.0 / math.sqrt(dim)

        # Read/write attention
        self.read_attn = nn.ModuleList([
            ScaledResidualAttention(
                RoPEReadCrossAttention(dim, heads, cfg.read_attention),
                cfg.layer_scale_init,
            )
            for _ in range(n_adapters)
        ])
        self.write_attn = nn.ModuleList([
            ScaledResidualAttention(
                RoPEWriteCrossAttention(dim, heads, cfg.write_attention),
                cfg.layer_scale_init,
            )
            for _ in range(n_adapters)
        ])

        # Canvas init params
        self.cls_init = nn.Parameter(torch.randn(1, 1, dim) * scale)
        self.spatial_init = nn.Parameter(torch.randn(1, 1, dim) * scale)
        self.registers = nn.Parameter(torch.randn(1, cfg.n_registers, dim) * scale)

        # Canvas normalization (always enabled)
        self.cls_ln = nn.LayerNorm(dim)
        self.reg_ln = nn.LayerNorm(dim)
        self.spatial_ln = nn.LayerNorm(dim)
        nn.init.constant_(self.cls_ln.weight, scale)
        nn.init.constant_(self.reg_ln.weight, scale)
        nn.init.constant_(self.spatial_ln.weight, scale)

    @property
    def n_registers(self) -> int:
        return self.cfg.n_registers

    @property
    def n_prefix(self) -> int:
        """Number of prefix tokens (cls + registers)."""
        return 1 + self.n_registers

    @property
    def n_adapters(self) -> int:
        return len(self.read_attn)

    def init_canvas(self, batch_size: int, canvas_grid_size: int) -> Tensor:
        """Create initial canvas [B, 1 + n_reg + G*G, D]."""
        B = batch_size
        n_spatial = canvas_grid_size ** 2
        cls = self.cls_init.expand(B, -1, -1)
        regs = self.registers.expand(B, -1, -1)
        spatial = self.spatial_init.expand(B, n_spatial, -1)
        out = torch.cat([cls, regs, spatial], dim=1)
        assert out.shape == (B, self.n_prefix + n_spatial, self.backbone.embed_dim)
        return out

    def forward(
        self,
        local: Tensor,
        canvas: Tensor,
        local_rope: tuple[Tensor, Tensor],
        canvas_rope: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Interleaved read-backbone-write with canvas normalization.

        Args:
            local: [B, N_local, D] - local tokens (from backbone.prepare_tokens)
            canvas: [B, N_canvas, D] - canvas state [cls | registers | spatial]
            local_rope: (sin, cos) for local positions
            canvas_rope: (sin, cos) for canvas positions

        Returns:
            (local, canvas) - updated tensors, same shapes as input
        """
        n_reg = self.n_registers

        # Normalize canvas at recurrence boundary
        cls = self.cls_ln(canvas[:, :1])
        reg = self.reg_ln(canvas[:, 1 : 1 + n_reg])
        spatial = self.spatial_ln(canvas[:, 1 + n_reg :])
        canvas = torch.cat([cls, reg, spatial], dim=1)

        # Interleaved read-backbone-write
        stride = self.cfg.adapter_stride
        for i in range(self.backbone.n_blocks):
            should_adapt = i >= stride and i % stride == 0
            if should_adapt:
                a = i // stride - 1
                local = self.read_attn[a](local, canvas, local_rope, canvas_rope)
            local = self.backbone.forward_block(i, local, local_rope)
            if should_adapt:
                canvas = self.write_attn[a](canvas, local, canvas_rope, local_rope)

        return local, canvas
