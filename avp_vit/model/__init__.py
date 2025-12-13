from dataclasses import dataclass
from typing import final, override

import torch
from torch import Tensor, nn

from avp_vit.attention import RoPEReadCrossAttention, RoPEWriteCrossAttention
from avp_vit.backbone import ViTBackbone
from avp_vit.rope import compute_rope, glimpse_positions, make_grid_positions


@final
@dataclass
class AVPConfig:
    scene_grid_size: int
    glimpse_grid_size: int = 7
    use_scene_registers: bool = False
    gate_init: float = 0.0
    use_output_proj: bool = False


@final
class AVPViT(nn.Module):
    """Active Visual Pondering ViT. Wraps backbone blocks with scene read/write."""

    backbone: ViTBackbone
    cfg: AVPConfig
    n_scene_registers: int
    scene_registers: nn.Parameter | None
    scene_tokens: nn.Parameter
    scene_positions: Tensor
    read_attn: nn.ModuleList
    write_attn: nn.ModuleList
    read_gate: nn.ParameterList
    write_gate: nn.ParameterList
    output_proj: nn.Module

    def __init__(self, backbone: ViTBackbone, cfg: AVPConfig) -> None:
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg

        embed_dim = backbone.embed_dim
        num_heads = backbone.num_heads
        n_blocks = backbone.n_blocks

        self.n_scene_registers = backbone.n_register_tokens if cfg.use_scene_registers else 0
        if self.n_scene_registers > 0:
            self.scene_registers = nn.Parameter(
                torch.randn(1, self.n_scene_registers, embed_dim)
            )
        else:
            self.scene_registers = None

        self.scene_tokens = nn.Parameter(
            torch.randn(1, cfg.scene_grid_size**2, embed_dim)
        )

        self.read_attn = nn.ModuleList(
            [RoPEReadCrossAttention(embed_dim, num_heads) for _ in range(n_blocks)]
        )
        self.write_attn = nn.ModuleList(
            [RoPEWriteCrossAttention(embed_dim, num_heads) for _ in range(n_blocks)]
        )

        self.read_gate = nn.ParameterList(
            [
                nn.Parameter(torch.full((embed_dim,), cfg.gate_init))
                for _ in range(n_blocks)
            ]
        )
        self.write_gate = nn.ParameterList(
            [
                nn.Parameter(torch.full((embed_dim,), cfg.gate_init))
                for _ in range(n_blocks)
            ]
        )

        pos = make_grid_positions(
            cfg.scene_grid_size, cfg.scene_grid_size, self.scene_tokens.device
        )
        self.register_buffer("scene_positions", pos)
        self.scene_positions = pos

        if cfg.use_output_proj:
            proj = nn.Linear(embed_dim, embed_dim)
            nn.init.eye_(proj.weight)
            nn.init.zeros_(proj.bias)
            self.output_proj = proj
        else:
            self.output_proj = nn.Identity()

    def _init_scene(self, B: int, scene: Tensor | None) -> Tensor:
        """Initialize scene: use provided or expand scene_tokens, prepend registers."""
        if scene is None:
            scene = self.scene_tokens.expand(B, -1, -1)
        if self.scene_registers is not None:
            scene = torch.cat([self.scene_registers.expand(B, -1, -1), scene], dim=1)
        return scene

    def forward_step(
        self,
        local: Tensor,
        centers: Tensor,
        scales: Tensor,
        scene: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Single step without output_proj. For multi-step, pass scene from previous step."""
        B = local.shape[0]
        H = W = self.cfg.glimpse_grid_size
        rope_dtype = self.backbone.rope_dtype
        periods = self.backbone.rope_periods
        n_reg = self.n_scene_registers

        scene = self._init_scene(B, scene)

        local_pos = glimpse_positions(centers, scales, H, W, dtype=rope_dtype)
        scene_pos = self.scene_positions.to(rope_dtype).unsqueeze(0).expand(B, -1, -1)

        local_rope = compute_rope(local_pos, periods)
        scene_rope = compute_rope(scene_pos, periods)

        for i in range(self.backbone.n_blocks):
            local = local + self.read_gate[i] * self.read_attn[i](
                local, scene, local_rope, scene_rope
            )
            local = self.backbone.forward_block(i, local, local_rope)
            scene = scene + self.write_gate[i] * self.write_attn[i](
                scene, local, scene_rope, local_rope
            )

        # Strip registers, return grid tokens only
        return local, scene[:, n_reg:]

    @override
    def forward(
        self,
        local: Tensor,
        centers: Tensor,
        scales: Tensor,
        scene: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Single step with output_proj. Use forward_step for multi-step loops."""
        local, scene = self.forward_step(local, centers, scales, scene)
        return local, self.output_proj(scene)
