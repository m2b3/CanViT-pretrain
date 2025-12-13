from collections.abc import Callable
from dataclasses import dataclass
from typing import final, override

import torch
from torch import Tensor, nn

from avp_vit.attention import RoPEReadCrossAttention, RoPEWriteCrossAttention
from avp_vit.backbone import ViTBackbone
from avp_vit.glimpse import Viewpoint, extract_glimpse
from avp_vit.rope import compute_rope, glimpse_positions, make_grid_positions


@final
@dataclass
class AVPConfig:
    scene_grid_size: int
    glimpse_grid_size: int = 7
    use_scene_registers: bool = False
    gate_init: float = 0.0
    use_output_proj: bool = False
    use_policy: bool = False  # Enable learned viewpoint policy


@final
class AVPViT(nn.Module):
    """Active Visual Pondering ViT.

    Takes full images + viewpoints, handles glimpse extraction and tokenization internally.
    """

    backbone: ViTBackbone
    cfg: AVPConfig
    glimpse_size: int
    n_scene_registers: int
    scene_registers: nn.Parameter | None
    scene_tokens: nn.Parameter
    scene_positions: Tensor
    read_attn: nn.ModuleList
    write_attn: nn.ModuleList
    read_gate: nn.ParameterList
    write_gate: nn.ParameterList
    output_proj: nn.Module
    pol_token: nn.Parameter | None
    pol_norm: nn.Module | None
    pol_proj: nn.Module | None

    def __init__(self, backbone: ViTBackbone, cfg: AVPConfig) -> None:
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg
        self.glimpse_size = cfg.glimpse_grid_size * backbone.patch_size

        embed_dim = backbone.embed_dim
        num_heads = backbone.num_heads
        n_blocks = backbone.n_blocks

        # Scale scene registers proportionally to scene/glimpse token ratio
        if cfg.use_scene_registers and backbone.n_register_tokens > 0:
            ratio = (cfg.scene_grid_size / cfg.glimpse_grid_size) ** 2
            self.n_scene_registers = round(backbone.n_register_tokens * ratio)
            self.scene_registers = nn.Parameter(
                torch.randn(1, self.n_scene_registers, embed_dim)
            )
        else:
            self.n_scene_registers = 0
            self.scene_registers = None

        # Initialize scene_tokens with randn / sqrt(embed_dim)
        self.scene_tokens = nn.Parameter(
            torch.randn(1, cfg.scene_grid_size**2, embed_dim) / (embed_dim**0.5)
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
            cfg.scene_grid_size, cfg.scene_grid_size, self.scene_tokens.device, dtype=backbone.rope_dtype
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

        # Policy: learnable POL token + LayerNorm + projection to (y, x)
        if cfg.use_policy:
            self.pol_token = nn.Parameter(
                torch.randn(1, 1, embed_dim) / (embed_dim**0.5)
            )
            self.pol_norm = nn.LayerNorm(embed_dim)
            self.pol_proj = nn.Linear(embed_dim, 2)
            # Small uniform init to avoid tanh saturation at start
            nn.init.uniform_(self.pol_proj.weight, -1e-2, 1e-2)
            nn.init.zeros_(self.pol_proj.bias)
        else:
            self.pol_token = None
            self.pol_norm = None
            self.pol_proj = None

    def _init_scene(self, B: int, scene: Tensor | None) -> Tensor:
        """Initialize scene: use provided or expand scene_tokens, prepend registers."""
        if scene is None:
            scene = self.scene_tokens.expand(B, -1, -1)
        if self.scene_registers is not None:
            scene = torch.cat([self.scene_registers.expand(B, -1, -1), scene], dim=1)
        return scene

    def _process_glimpse(
        self,
        local: Tensor,
        centers: Tensor,
        scales: Tensor,
        scene: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Process one glimpse: read from scene, forward through backbone, write to scene.

        Returns (local, scene, pol_out) where pol_out is raw policy output [B, 2] or None.
        """
        B = local.shape[0]
        H = W = self.cfg.glimpse_grid_size
        rope_dtype = self.backbone.rope_dtype
        periods = self.backbone.rope_periods
        n_reg = self.n_scene_registers

        scene_t = self._init_scene(B, scene)

        # Prepend POL token to local stream if policy enabled
        has_pol = self.pol_token is not None
        if has_pol:
            pol = self.pol_token.expand(B, -1, -1)  # [B, 1, D]
            local = torch.cat([pol, local], dim=1)  # [B, 1+N, D]

        # Compute positions: POL uses viewpoint center, patches use glimpse_positions
        local_pos = glimpse_positions(centers, scales, H, W, dtype=rope_dtype)
        if has_pol:
            pol_pos = centers.unsqueeze(1).to(rope_dtype)  # [B, 1, 2]
            local_pos = torch.cat([pol_pos, local_pos], dim=1)  # [B, 1+H*W, 2]

        scene_pos = self.scene_positions.to(rope_dtype).unsqueeze(0).expand(B, -1, -1)

        local_rope = compute_rope(local_pos, periods)
        scene_rope = compute_rope(scene_pos, periods)

        for i in range(self.backbone.n_blocks):
            local = local + self.read_gate[i] * self.read_attn[i](
                local, scene_t, local_rope, scene_rope
            )
            local = self.backbone.forward_block(i, local, local_rope)
            scene_t = scene_t + self.write_gate[i] * self.write_attn[i](
                scene_t, local, scene_rope, local_rope
            )

        # Extract POL token and decode policy output
        pol_out: Tensor | None = None
        if has_pol:
            assert self.pol_norm is not None and self.pol_proj is not None
            pol_out = self.pol_proj(self.pol_norm(local[:, 0, :]))  # [B, 2]
            local = local[:, 1:, :]  # Strip POL from local

        # Strip registers, return grid tokens only
        return local, scene_t[:, n_reg:], pol_out

    def forward_step(
        self, images: Tensor, viewpoint: Viewpoint, scene: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Single step: extract glimpse, process, return (local, scene, pol_out)."""
        glimpse = extract_glimpse(images, viewpoint, self.glimpse_size)
        tokens, _, _ = self.backbone.prepare_tokens(glimpse)
        return self._process_glimpse(tokens, viewpoint.centers, viewpoint.scales, scene)

    @override
    def forward(self, images: Tensor, viewpoints: list[Viewpoint]) -> Tensor:
        """Process full images through viewpoints, return final scene representation."""
        scene: Tensor | None = None
        for vp in viewpoints:
            _, scene, _ = self.forward_step(images, vp, scene)
        assert scene is not None
        return self.output_proj(scene)

    def forward_with_loss(
        self,
        images: Tensor,
        viewpoints: list[Viewpoint],
        loss_fn: Callable[[Tensor], Tensor],
    ) -> Tensor:
        """Stream through viewpoints, compute loss at each step, return average loss.

        More memory efficient than storing all intermediate scenes - each scene
        is consumed by loss_fn immediately then discarded.
        """
        scene: Tensor | None = None
        loss_sum = torch.tensor(0.0, device=images.device)
        for vp in viewpoints:
            _, scene, _ = self.forward_step(images, vp, scene)
            scene_proj = self.output_proj(scene)
            loss_sum = loss_sum + loss_fn(scene_proj)
        return loss_sum / len(viewpoints)

    def policy_to_viewpoint(self, pol_out: Tensor, scale: float) -> Viewpoint:
        """Convert raw policy output to bounded Viewpoint.

        Args:
            pol_out: [B, 2] raw output from pol_proj
            scale: fixed scale for policy viewpoints

        Returns:
            Viewpoint with centers clamped to valid bounds
        """
        max_offset = 1.0 - scale
        centers = torch.tanh(pol_out) * max_offset  # [B, 2] in [-max_offset, max_offset]
        scales = torch.full((pol_out.shape[0],), scale, device=pol_out.device)
        return Viewpoint(name="policy", centers=centers, scales=scales)
