from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple, TypeVar, cast, final, override

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from avp_vit.attention import RoPEReadCrossAttention, RoPEWriteCrossAttention
from avp_vit.backbone import ViTBackbone
from avp_vit.glimpse import Viewpoint, extract_glimpse
from avp_vit.rope import compute_rope, glimpse_positions, make_grid_positions

# Type variable for forward_reduce accumulator
T = TypeVar("T")


class StepOutput(NamedTuple):
    """Output from a single AVPViT forward step.

    IMPORTANT - Two scene representations with distinct purposes:

    - hidden: Internal state passed between timesteps. Use this for CONTINUATION
              (e.g., Bernoulli survival in training). This is the raw output of
              the write attention mechanism.

    - scene:  Projected output for external use. Use this for LOSS COMPUTATION
              and VISUALIZATION. This is output_proj(hidden).

    The distinction matters because:
    1. Continuation needs the unprojected state (hidden)
    2. Loss/viz need the projected state (scene)
    3. output_proj may not be invertible, so you can't recover hidden from scene
    """

    glimpse: Tensor  # [B, C, H, W] extracted glimpse image
    local: Tensor  # [B, N, D] local features (CLS + patches)
    hidden: Tensor  # [B, G*G, D] internal state for CONTINUATION between steps
    scene: Tensor  # [B, G*G, D] projected output for LOSS and VISUALIZATION


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
    hidden_tokens: nn.Parameter  # Learned initial hidden state [1, G*G, D]
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

        # Learned initial hidden state, initialized with randn / sqrt(embed_dim)
        self.hidden_tokens = nn.Parameter(
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

        device = self.hidden_tokens.device
        assert isinstance(device, torch.device)
        pos = make_grid_positions(
            cfg.scene_grid_size, cfg.scene_grid_size, device, dtype=backbone.rope_dtype
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

    def _init_hidden(self, B: int, hidden: Tensor | None) -> Tensor:
        """Initialize hidden state: use provided or expand learned tokens, prepend registers."""
        if hidden is None:
            hidden = self.hidden_tokens.expand(B, -1, -1)
        if self.scene_registers is not None:
            hidden = torch.cat([self.scene_registers.expand(B, -1, -1), hidden], dim=1)
        return hidden

    def _process_glimpse(
        self,
        glimpse: Tensor,
        local: Tensor,
        centers: Tensor,
        scales: Tensor,
        hidden: Tensor | None,
    ) -> StepOutput:
        """Process one glimpse: read from hidden, forward through backbone, write to hidden.

        Args:
            glimpse: Extracted glimpse image [B, C, H, W]
            local: Tokenized glimpse features [B, N, D]
            centers: Viewpoint centers [B, 2]
            scales: Viewpoint scales [B]
            hidden: Previous hidden state [B, G*G, D] or None for fresh start

        Returns:
            StepOutput with both hidden (for continuation) and scene (for loss/viz)
        """
        B = local.shape[0]
        H = W = self.cfg.glimpse_grid_size
        rope_dtype = self.backbone.rope_dtype
        periods = self.backbone.rope_periods
        n_reg = self.n_scene_registers

        hidden_t = self._init_hidden(B, hidden)
        local_pos = glimpse_positions(centers, scales, H, W, dtype=rope_dtype)
        scene_pos = self.scene_positions.to(rope_dtype).unsqueeze(0).expand(B, -1, -1)
        local_rope = compute_rope(local_pos, periods)
        scene_rope = compute_rope(scene_pos, periods)

        for i in range(self.backbone.n_blocks):
            local = local + self.read_gate[i] * self.read_attn[i](
                local, hidden_t, local_rope, scene_rope
            )
            local = self.backbone.forward_block(i, local, local_rope)
            hidden_t = hidden_t + self.write_gate[i] * self.write_attn[i](
                hidden_t, local, scene_rope, local_rope
            )

        # Strip registers, get grid tokens only
        hidden_out = hidden_t[:, n_reg:]
        scene_out = self.output_proj(hidden_out)
        return StepOutput(glimpse, local, hidden_out, scene_out)

    def forward_step(
        self, images: Tensor, viewpoint: Viewpoint, hidden: Tensor | None = None
    ) -> StepOutput:
        """Process a single viewpoint.

        Args:
            images: Input images [B, C, H, W]
            viewpoint: Where to look in the images
            hidden: Previous hidden state [B, G*G, D] for CONTINUATION, or None for fresh start

        Returns:
            StepOutput containing:
            - glimpse: Extracted glimpse image
            - local: Local features from backbone
            - hidden: Updated hidden state (use this for CONTINUATION)
            - scene: Projected output (use this for LOSS/VIZ)
        """
        glimpse = extract_glimpse(images, viewpoint, self.glimpse_size)
        tokens, _, _ = self.backbone.prepare_tokens(glimpse)
        if self.cfg.gradient_checkpointing and self.training:
            return cast(StepOutput, checkpoint(
                self._process_glimpse,
                glimpse,
                tokens,
                viewpoint.centers,
                viewpoint.scales,
                hidden,
                use_reentrant=False,
            ))
        return self._process_glimpse(glimpse, tokens, viewpoint.centers, viewpoint.scales, hidden)

    # ==================== General Primitive ====================

    def forward_reduce(
        self,
        images: Tensor,
        viewpoints: list[Viewpoint],
        reducer: Callable[[T, StepOutput], T],
        init: T,
        hidden: Tensor | None = None,
    ) -> tuple[T, Tensor]:
        """Process viewpoints sequentially, reducing with a custom function.

        This is the GENERAL PRIMITIVE that all other forward methods build on.
        It implements a functional "scan" pattern over viewpoints.

        Args:
            images: Input images [B, C, H, W]
            viewpoints: Sequence of viewpoints to process
            reducer: Function (accumulator, step_output) -> new_accumulator
                     Receives full StepOutput with glimpse, local, hidden, scene.
            init: Initial accumulator value
            hidden: Initial hidden state [B, G*G, D] or None for fresh start

        Returns:
            (final_accumulator, final_hidden) where:
            - final_accumulator: Result of reducing all steps
            - final_hidden: Hidden state after last step (for CONTINUATION)
        """
        acc = init
        for vp in viewpoints:
            out = self.forward_step(images, vp, hidden)
            hidden = out.hidden
            acc = reducer(acc, out)
        assert hidden is not None
        return acc, hidden

    # ==================== Standard Invocations ====================
    # These are the common patterns, centralized here to avoid errors.

    def forward_loss(
        self,
        images: Tensor,
        viewpoints: list[Viewpoint],
        target: Tensor,
        hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Standard training: compute MSE loss against target at each step.

        Memory-efficient: does not store intermediate scenes.

        Args:
            images: Input images [B, C, H, W]
            viewpoints: Sequence of viewpoints to process
            target: Target to compare against [B, G*G, D] (e.g., teacher patches)
            hidden: Initial hidden state or None

        Returns:
            (average_loss, final_hidden) where:
            - average_loss: Mean MSE across all viewpoints (scalar)
            - final_hidden: For CONTINUATION in Bernoulli survival
        """
        def reducer(acc: Tensor, out: StepOutput) -> Tensor:
            return acc + F.mse_loss(out.scene, target)

        total, final_hidden = self.forward_reduce(
            images, viewpoints, reducer,
            init=torch.tensor(0.0, device=images.device),
            hidden=hidden,
        )
        return total / len(viewpoints), final_hidden

    def forward_trajectory(
        self,
        images: Tensor,
        viewpoints: list[Viewpoint],
        hidden: Tensor | None = None,
    ) -> tuple[list[Tensor], Tensor]:
        """Standard visualization: collect projected scenes at each step.

        Args:
            images: Input images [B, C, H, W]
            viewpoints: Sequence of viewpoints to process
            hidden: Initial hidden state or None

        Returns:
            (scenes, final_hidden) where:
            - scenes: List of projected scenes [B, G*G, D] at each timestep
            - final_hidden: For CONTINUATION if needed
        """
        def reducer(acc: list[Tensor], out: StepOutput) -> list[Tensor]:
            return [*acc, out.scene]

        return self.forward_reduce(images, viewpoints, reducer, init=[], hidden=hidden)

    def forward_trajectory_full(
        self,
        images: Tensor,
        viewpoints: list[Viewpoint],
        hidden: Tensor | None = None,
    ) -> tuple[list[StepOutput], Tensor]:
        """Full visualization: collect complete StepOutput at each step.

        Args:
            images: Input images [B, C, H, W]
            viewpoints: Sequence of viewpoints to process
            hidden: Initial hidden state or None

        Returns:
            (outputs, final_hidden) where:
            - outputs: List of StepOutput (glimpse, local, hidden, scene) at each timestep
            - final_hidden: For CONTINUATION if needed
        """
        def reducer(acc: list[StepOutput], out: StepOutput) -> list[StepOutput]:
            return [*acc, out]

        return self.forward_reduce(images, viewpoints, reducer, init=[], hidden=hidden)

    @override
    def forward(
        self,
        images: Tensor,
        viewpoints: list[Viewpoint],
        hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Standard inference: return final projected scene.

        Args:
            images: Input images [B, C, H, W]
            viewpoints: Sequence of viewpoints to process
            hidden: Initial hidden state or None

        Returns:
            (final_scene, final_hidden) where:
            - final_scene: Projected output after all viewpoints [B, G*G, D]
            - final_hidden: For CONTINUATION if needed
        """
        def reducer(acc: Tensor, out: StepOutput) -> Tensor:
            return out.scene

        # Use a dummy initial tensor (will be overwritten by first step)
        dummy = torch.empty(0, device=images.device)
        scene, final_hidden = self.forward_reduce(images, viewpoints, reducer, init=dummy, hidden=hidden)
        return scene, final_hidden
