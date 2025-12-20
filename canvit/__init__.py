"""CanViT: Canvas Cross-Attention Vision Transformer.

Resolution-decoupled ViT backbone using asymmetric cross-attention
between a small local stream and a large canvas state.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import NamedTuple, final

import torch
from torch import Tensor, nn

from canvit.attention import (
    CrossAttentionConfig,
    ReadCrossAttention,
    ScaledResidualAttention,
    WriteCrossAttention,
)
from canvit.backbone import ViTBackbone
from canvit.rope import compute_rope, make_rope_periods

log = logging.getLogger(__name__)


class RoPE(NamedTuple):
    """Rotary position embedding (sin, cos)."""

    sin: Tensor
    cos: Tensor


@final
@dataclass
class CanViTConfig:
    """CanViT configuration.

    canvas_num_heads: Number of attention heads for cross-attention.
    canvas_head_dim: Dimension per attention head. Common values: 64, 96, 128.
        canvas_dim = canvas_num_heads * canvas_head_dim.
        Unusual head_dim values can cause kernel failures (e.g., cutlassF).
    """

    n_canvas_registers: int = 32
    adapter_stride: int = 4
    layer_scale_init: float = 1e-3
    canvas_num_heads: int = 4
    canvas_head_dim: int = 128  # canvas_dim = 512
    use_canvas_layer_scale: bool = True
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

    Canvas tokens live in canvas_dim = canvas_num_heads * canvas_head_dim.
    Attention happens in canvas_dim space; local tokens projected up/down.
    """

    backbone: ViTBackbone
    cfg: CanViTConfig
    read_attn: nn.ModuleList
    write_attn: nn.ModuleList
    # Canvas init
    cls_init: nn.Parameter
    spatial_init: nn.Parameter
    registers: nn.Parameter
    # Canvas gate logits (sigmoid → λ ∈ (0,1) for leaky integration)
    cls_gate_logit: nn.Parameter | None
    reg_gate_logits: nn.Parameter | None
    spatial_gate_logit: nn.Parameter | None
    # RoPE periods for cross-attention
    # Backbone self-attn uses backbone.rope_periods (backbone_head_dim = local_dim / backbone.num_heads)
    # Cross-attn uses canvas_rope_periods (canvas_head_dim from config)
    canvas_rope_periods: Tensor

    def __init__(self, backbone: ViTBackbone, cfg: CanViTConfig) -> None:
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg

        local_dim = backbone.embed_dim
        canvas_num_heads = cfg.canvas_num_heads
        canvas_head_dim = cfg.canvas_head_dim
        canvas_dim = canvas_num_heads * canvas_head_dim

        n_blocks = backbone.n_blocks
        n_adapters = (n_blocks - 1) // cfg.adapter_stride

        self.read_attn = nn.ModuleList(
            [
                ScaledResidualAttention(
                    ReadCrossAttention(local_dim, canvas_dim, canvas_num_heads, cfg.read_attention),
                    cfg.layer_scale_init,
                )
                for _ in range(n_adapters)
            ]
        )
        self.write_attn = nn.ModuleList(
            [
                ScaledResidualAttention(
                    WriteCrossAttention(
                        local_dim, canvas_dim, canvas_num_heads, cfg.write_attention
                    ),
                    cfg.layer_scale_init,
                )
                for _ in range(n_adapters)
            ]
        )

        scale = 1.0 / math.sqrt(canvas_dim)
        self.cls_init = nn.Parameter(torch.randn(1, 1, canvas_dim) * scale)
        self.spatial_init = nn.Parameter(torch.randn(1, 1, canvas_dim) * scale)
        self.registers = nn.Parameter(torch.randn(1, cfg.n_canvas_registers, canvas_dim) * scale)

        if cfg.use_canvas_layer_scale:
            # Initialize logits so sigmoid(logit) = layer_scale_init
            # logit(p) = ln(p / (1-p))
            p = cfg.layer_scale_init
            logit_init = math.log(p / (1 - p))
            self.cls_gate_logit = nn.Parameter(torch.full((canvas_dim,), logit_init))
            self.reg_gate_logits = nn.Parameter(torch.full((cfg.n_canvas_registers, canvas_dim), logit_init))
            self.spatial_gate_logit = nn.Parameter(torch.full((canvas_dim,), logit_init))
        else:
            self.cls_gate_logit = None
            self.reg_gate_logits = None
            self.spatial_gate_logit = None

        # RoPE periods for cross-attention (canvas_head_dim)
        self.register_buffer(
            "canvas_rope_periods",
            make_rope_periods(canvas_head_dim, backbone.rope_dtype),
        )

    @property
    def local_dim(self) -> int:
        return self.backbone.embed_dim

    @property
    def canvas_dim(self) -> int:
        return self.cfg.canvas_num_heads * self.cfg.canvas_head_dim

    @property
    def n_canvas_registers(self) -> int:
        return self.cfg.n_canvas_registers

    @property
    def n_prefix(self) -> int:
        """Number of prefix tokens (cls + registers)."""
        return 1 + self.n_canvas_registers

    @property
    def n_adapters(self) -> int:
        return len(self.read_attn)

    def prepare_canvas(self, previous: Tensor) -> Tensor:
        """Prepare canvas via leaky integration toward inits.

        Formula: canvas = inits + λ*(previous - inits) = (1-λ)*inits + λ*previous

        At step 0 (previous=inits from init_canvas), returns exact inits.
        At step t, blends inits with previous; older residuals decay as λ^t.
        """
        B = previous.shape[0]
        n_reg = self.n_canvas_registers
        n_spatial = previous.shape[1] - self.n_prefix

        # Expand inits to batch size
        cls_init = self.cls_init.expand(B, -1, -1)
        reg_init = self.registers.expand(B, -1, -1)
        spatial_init = self.spatial_init.expand(B, n_spatial, -1)

        if self.cls_gate_logit is not None:
            assert self.reg_gate_logits is not None and self.spatial_gate_logit is not None
            # Leaky integration: inits + λ*(prev - inits), λ = sigmoid(logit) ∈ (0,1)
            cls_gate = torch.sigmoid(self.cls_gate_logit)
            reg_gates = torch.sigmoid(self.reg_gate_logits)
            spatial_gate = torch.sigmoid(self.spatial_gate_logit)
            cls = cls_init + cls_gate * (previous[:, :1] - cls_init)
            reg = reg_init + reg_gates * (previous[:, 1 : 1 + n_reg] - reg_init)
            spatial = spatial_init + spatial_gate * (previous[:, 1 + n_reg :] - spatial_init)
        else:
            # No gating: pass through previous unchanged
            cls = previous[:, :1]
            reg = previous[:, 1 : 1 + n_reg]
            spatial = previous[:, 1 + n_reg :]

        return torch.cat([cls, reg, spatial], dim=1)

    def init_canvas(self, batch_size: int, canvas_grid_size: int) -> Tensor:
        """Create initial canvas from learned inits [B, 1 + n_reg + G*G, canvas_dim].

        Base case for recurrence: prepare_canvas(inits) = inits (exactly).
        """
        B = batch_size
        n_spatial = canvas_grid_size**2
        cls = self.cls_init.expand(B, -1, -1)
        reg = self.registers.expand(B, -1, -1)
        spatial = self.spatial_init.expand(B, n_spatial, -1)
        return torch.cat([cls, reg, spatial], dim=1)

    def compile(self, **kwargs) -> None:
        """Compile model components for faster execution."""
        self.backbone.compile(**kwargs)
        log.info(f"Compiling {self.n_adapters} read/write attention pairs")
        for attn in self.read_attn:
            attn.compile(**kwargs)
        for attn in self.write_attn:
            attn.compile(**kwargs)

    def forward(
        self,
        glimpse: Tensor,
        canvas: Tensor,
        local_positions: Tensor,
        canvas_positions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Process glimpse and update canvas via interleaved read-backbone-write.

        Two RoPE encodings used:
        - backbone_rope_periods: for backbone self-attention (head_dim = local_dim/H)
        - canvas_rope_periods: for cross-attention (head_dim = canvas_dim/H)
        Same positions, different frequency encodings.
        """
        local, H, W = self.backbone.prepare_tokens(glimpse)
        assert local_positions.shape[1] == H * W

        # RoPE for backbone self-attention
        local_rope_backbone = RoPE(*compute_rope(local_positions, self.backbone.rope_periods))
        # RoPE for cross-attention (canvas_dim head_dim)
        local_rope_xattn = RoPE(*compute_rope(local_positions, self.canvas_rope_periods))
        canvas_rope = RoPE(*compute_rope(canvas_positions, self.canvas_rope_periods))

        canvas = self.prepare_canvas(canvas)

        stride = self.cfg.adapter_stride
        for i in range(self.backbone.n_blocks):
            if i >= stride and i % stride == 0:
                a = i // stride - 1
                local = self.read_attn[a](local, canvas, local_rope_xattn, canvas_rope)
                local = self.backbone.forward_block(i, local, local_rope_backbone)
                canvas = self.write_attn[a](canvas, local, canvas_rope, local_rope_xattn)
            else:
                local = self.backbone.forward_block(i, local, local_rope_backbone)

        return local, canvas
