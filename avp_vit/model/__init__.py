from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple, cast, final, override

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from avp_vit.attention import RoPEReadCrossAttention, RoPEWriteCrossAttention
from avp_vit.backbone import ViTBackbone
from avp_vit.glimpse import Viewpoint, extract_glimpse
from avp_vit.rope import compute_rope, glimpse_positions, make_grid_positions


class StepOutput(NamedTuple):
    """Output from a single AVPViT forward step."""

    glimpse: Tensor  # [B, C, H, W] extracted glimpse image
    local: Tensor  # [B, N, D] local features (CLS + patches)
    scene: Tensor  # [B, G*G, D] updated scene representation


@final
@dataclass
class AVPConfig:
    scene_grid_size: int
    glimpse_grid_size: int = 7
    use_scene_registers: bool = False
    gate_init: float = 0.0
    use_output_proj: bool = False
    gradient_checkpointing: bool = False  # Checkpoint at timestep boundaries to save VRAM


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

    def _init_scene(self, B: int, scene: Tensor | None) -> Tensor:
        """Initialize scene: use provided or expand scene_tokens, prepend registers."""
        if scene is None:
            scene = self.scene_tokens.expand(B, -1, -1)
        if self.scene_registers is not None:
            scene = torch.cat([self.scene_registers.expand(B, -1, -1), scene], dim=1)
        return scene

    def _process_glimpse(
        self,
        glimpse: Tensor,
        local: Tensor,
        centers: Tensor,
        scales: Tensor,
        scene: Tensor | None,
    ) -> StepOutput:
        """Process one glimpse: read from scene, forward through backbone, write to scene."""
        B = local.shape[0]
        H = W = self.cfg.glimpse_grid_size
        rope_dtype = self.backbone.rope_dtype
        periods = self.backbone.rope_periods
        n_reg = self.n_scene_registers

        scene_t = self._init_scene(B, scene)
        local_pos = glimpse_positions(centers, scales, H, W, dtype=rope_dtype)
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

        # Strip registers, return grid tokens only
        return StepOutput(glimpse, local, scene_t[:, n_reg:])

    def forward_step(
        self, images: Tensor, viewpoint: Viewpoint, scene: Tensor | None = None
    ) -> StepOutput:
        """Single step: extract glimpse, process, return StepOutput."""
        glimpse = extract_glimpse(images, viewpoint, self.glimpse_size)
        tokens, _, _ = self.backbone.prepare_tokens(glimpse)
        B = tokens.shape[0]
        if scene is None:
            scene = self.scene_tokens.expand(B, -1, -1)
        if self.cfg.gradient_checkpointing and self.training:
            return cast(StepOutput, checkpoint(
                self._process_glimpse,
                glimpse,
                tokens,
                viewpoint.centers,
                viewpoint.scales,
                scene,
                use_reentrant=False,
            ))
        return self._process_glimpse(glimpse, tokens, viewpoint.centers, viewpoint.scales, scene)

    @override
    def forward(self, images: Tensor, viewpoints: list[Viewpoint]) -> Tensor:
        """Process full images through viewpoints, return final scene representation."""
        scene: Tensor | None = None
        for vp in viewpoints:
            out = self.forward_step(images, vp, scene)
            scene = out.scene
        assert scene is not None
        return self.output_proj(scene)

    def forward_with_loss(
        self,
        images: Tensor,
        viewpoints: list[Viewpoint],
        loss_fn: Callable[[Tensor], Tensor],
    ) -> Tensor:
        """Stream through viewpoints, compute loss at each step, return average."""
        scene: Tensor | None = None
        loss_sum = torch.tensor(0.0, device=images.device)
        for vp in viewpoints:
            out = self.forward_step(images, vp, scene)
            scene = out.scene
            loss_sum = loss_sum + loss_fn(self.output_proj(scene))
        return loss_sum / len(viewpoints)
