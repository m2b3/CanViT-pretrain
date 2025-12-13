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
    gate_init: float = 0.0
    use_output_proj: bool = False


@final
class AVPViT(nn.Module):
    """Active Visual Pondering ViT. Wraps backbone blocks with scene read/write."""

    backbone: ViTBackbone
    cfg: AVPConfig
    scene_tokens: nn.Parameter
    scene_positions: Tensor
    read_attn: nn.ModuleList
    write_attn: nn.ModuleList
    read_gate: nn.ParameterList
    write_gate: nn.ParameterList
    output_proj: nn.Linear | None
    output_norm: nn.LayerNorm | None

    def __init__(self, backbone: ViTBackbone, cfg: AVPConfig) -> None:
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg

        embed_dim = backbone.embed_dim
        num_heads = backbone.num_heads
        n_blocks = backbone.n_blocks

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
            self.output_proj = nn.Linear(embed_dim, embed_dim)
            nn.init.eye_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)
            # Clone backbone's final LayerNorm for scene output normalization
            self.output_norm = nn.LayerNorm(embed_dim)
            with torch.no_grad():
                self.output_norm.weight.copy_(backbone.norm.weight)
                self.output_norm.bias.copy_(backbone.norm.bias)
        else:
            self.output_proj = None
            self.output_norm = None

    @override
    def forward(
        self,
        local: Tensor,
        centers: Tensor,
        scales: Tensor,
        return_layers: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, list[Tensor]]:
        B = local.shape[0]
        H = W = self.cfg.glimpse_grid_size
        rope_dtype = self.backbone.rope_dtype
        periods = self.backbone.rope_periods

        scene = self.scene_tokens.expand(B, -1, -1)

        local_pos = glimpse_positions(centers, scales, H, W, dtype=rope_dtype)
        scene_pos = self.scene_positions.to(rope_dtype).unsqueeze(0).expand(B, -1, -1)

        local_rope = compute_rope(local_pos, periods)
        scene_rope = compute_rope(scene_pos, periods)

        scene_layers: list[Tensor] = []
        for i in range(self.backbone.n_blocks):
            local = local + self.read_gate[i] * self.read_attn[i](
                local, scene, local_rope, scene_rope
            )
            local = self.backbone.forward_block(i, local, local_rope)
            scene = scene + self.write_gate[i] * self.write_attn[i](
                scene, local, scene_rope, local_rope
            )
            if return_layers:
                scene_layers.append(scene)

        if self.output_proj is not None:
            assert self.output_norm is not None
            scene = self.output_norm(self.output_proj(scene))

        if return_layers:
            return local, scene, scene_layers
        return local, scene
