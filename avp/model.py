from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .attention import CrossAttention
from .rope import compute_rope, glimpse_positions, make_grid_positions


@dataclass
class AVPConfig:
    scene_grid_size: int
    glimpse_grid_size: int = 7
    gate_init: float = 0.0


class AVPViT(nn.Module):
    """Active Visual Pondering ViT. Wraps backbone blocks with scene read/write."""

    def __init__(
        self,
        blocks: nn.ModuleList,
        rope_embed: nn.Module,
        embed_dim: int,
        num_heads: int,
        n_prefix_tokens: int,
        cfg: AVPConfig,
    ):
        super().__init__()
        self.blocks = blocks
        self.rope_embed = rope_embed
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.n_prefix_tokens = n_prefix_tokens
        self.cfg = cfg
        n_blocks = len(blocks)

        # Scene tokens: learnable, [1, S*S, D]
        self.scene_tokens = nn.Parameter(torch.zeros(1, cfg.scene_grid_size**2, embed_dim))

        # Per-block cross-attention
        self.read_attn = nn.ModuleList([
            CrossAttention(embed_dim, num_heads) for _ in range(n_blocks)
        ])
        self.write_attn = nn.ModuleList([
            CrossAttention(embed_dim, num_heads) for _ in range(n_blocks)
        ])

        # Per-block gating
        self.read_gate = nn.ParameterList([
            nn.Parameter(torch.full((embed_dim,), cfg.gate_init)) for _ in range(n_blocks)
        ])
        self.write_gate = nn.ParameterList([
            nn.Parameter(torch.full((embed_dim,), cfg.gate_init)) for _ in range(n_blocks)
        ])

        # Fixed scene positions
        self.register_buffer(
            "scene_positions",
            make_grid_positions(cfg.scene_grid_size, cfg.scene_grid_size, torch.device("cpu")),
        )

    def forward(
        self,
        local: Tensor,  # [B, N, D] = prefix + H*W patch tokens
        centers: Tensor,  # [B, 2] glimpse centers in [0,1]^2
        scales: Tensor,  # [B] glimpse scales
    ) -> tuple[Tensor, Tensor]:
        B = local.shape[0]
        H = W = self.cfg.glimpse_grid_size

        # Expand scene tokens
        scene = self.scene_tokens.expand(B, -1, -1)

        # Compute positions
        local_pos = glimpse_positions(centers, scales, H, W)  # [B, HW, 2]
        scene_pos = self.scene_positions.unsqueeze(0).expand(B, -1, -1)  # [B, S*S, 2]

        # Compute RoPE: [B, 1, N, head_dim]
        local_rope = compute_rope(local_pos, self.head_dim, self.rope_embed.base)
        scene_rope = compute_rope(scene_pos, self.head_dim, self.rope_embed.base)

        for i, block in enumerate(self.blocks):
            # Read: local attends to scene
            local = local + self.read_gate[i] * self.read_attn[i](
                local, scene,
                q_rope=local_rope, kv_rope=scene_rope,
                q_prefix=self.n_prefix_tokens, kv_prefix=0,
            )

            # Backbone block with scene-space positions
            local = block(local, local_rope)

            # Write: scene attends to local
            scene = scene + self.write_gate[i] * self.write_attn[i](
                scene, local,
                q_rope=scene_rope, kv_rope=local_rope,
                q_prefix=0, kv_prefix=self.n_prefix_tokens,
            )

        return local, scene

    @classmethod
    def from_dinov3(cls, backbone, cfg: AVPConfig) -> "AVPViT":
        return cls(
            blocks=backbone.blocks,
            rope_embed=backbone.rope_embed,
            embed_dim=backbone.embed_dim,
            num_heads=backbone.num_heads,
            n_prefix_tokens=1 + backbone.n_storage_tokens,
            cfg=cfg,
        )
