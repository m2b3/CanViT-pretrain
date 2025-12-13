from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..attention import CrossAttention
from ..backend import ViTBackend
from ..rope import compute_rope, glimpse_positions, make_grid_positions


@dataclass
class AVPConfig:
    scene_grid_size: int
    glimpse_grid_size: int = 7
    gate_init: float = 0.0


class AVPViT(nn.Module):
    """Active Visual Pondering ViT. Wraps backend blocks with scene read/write."""

    def __init__(self, backend: ViTBackend, cfg: AVPConfig):
        super().__init__()
        self.backend = backend
        self.cfg = cfg

        embed_dim = backend.embed_dim
        num_heads = backend.num_heads
        n_blocks = backend.n_blocks

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
        centers: Tensor,  # [B, 2] glimpse centers in [-1,1]^2
        scales: Tensor,  # [B] glimpse scales
    ) -> tuple[Tensor, Tensor]:
        B = local.shape[0]
        H = W = self.cfg.glimpse_grid_size
        rope_dtype = self.backend.rope_dtype
        periods = self.backend.rope_periods

        # Expand scene tokens
        scene = self.scene_tokens.expand(B, -1, -1)

        # Compute positions in rope dtype for numerical consistency
        local_pos = glimpse_positions(centers, scales, H, W, dtype=rope_dtype)
        scene_pos = self.scene_positions.to(rope_dtype).unsqueeze(0).expand(B, -1, -1)

        # Compute RoPE: [B, 1, N, head_dim]
        local_rope = compute_rope(local_pos, periods)
        scene_rope = compute_rope(scene_pos, periods)

        for i in range(self.backend.n_blocks):
            # Read: local attends to scene
            local = local + self.read_gate[i] * self.read_attn[i](
                local, scene, q_rope=local_rope, kv_rope=scene_rope
            )

            # Backbone block
            local = self.backend.forward_block(i, local, local_rope)

            # Write: scene attends to local
            scene = scene + self.write_gate[i] * self.write_attn[i](
                scene, local, q_rope=scene_rope, kv_rope=local_rope
            )

        return local, scene
