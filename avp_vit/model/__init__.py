import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import NamedTuple, TypeVar, cast, final, override

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from ytch.nn import ElementwiseAffine
from ytch.nn.layer_scale import LayerScale

from avp_vit.attention import (
    AttentionConfig,
    RoPEReadCrossAttention,
    RoPEWriteCrossAttention,
)
from avp_vit.attention.convex import ConvexGatedAttention
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
    hidden: Tensor  # [B, n_persistent + G*G, D] internal state for CONTINUATION
    scene: Tensor  # [B, G*G, D] projected spatial for LOSS and VISUALIZATION
    context_out: (
        Tensor | None
    )  # [B, N_ctx, D] transformed context tokens (if context provided)


@final
@dataclass
class AVPConfig:
    scene_grid_size: int
    glimpse_grid_size: int = 7
    n_scene_registers: int = 32  # 0 = disabled, >0 = fixed count
    layer_scale_init: float = 1e-4  # Init for intra-step LayerScales (cross-attention)
    use_output_proj: bool = False
    use_output_proj_norm: bool = False  # LayerNorm before Linear in output_proj
    gradient_checkpointing: bool = True  # Checkpoint at timestep boundaries
    use_convex_gating: bool = False  # Dynamic per-token gating (vs static LayerScale)
    adapter_stride: int = 2  # Apply read/write adapters every N backbone blocks
    attention: AttentionConfig = field(default_factory=AttentionConfig)


@final
class AVPViT(nn.Module):
    """Active Visual Pondering ViT.

    Takes full images + viewpoints, handles glimpse extraction and tokenization internally.

    ## Naming Conventions

    - **hidden**: [persistent_registers | spatial_hidden] - internal state for CONTINUATION
    - **scene**: output_proj(spatial_hidden) - projected output for LOSS/VIZ
    - **persistent_registers**: Learnable tokens carried across timesteps
    - **ephemeral_registers**: Prepended at start of step, stripped at end (NOT in hidden)

    ## Initialization

    - spatial_hidden_init: Zero (uniform write attention pulls in glimpse info)
    - registers: Unit norm (randn / sqrt(D))
    """

    backbone: ViTBackbone
    cfg: AVPConfig
    persistent_registers: nn.Parameter | None  # [1, n_persistent, D]
    ephemeral_registers: nn.Parameter | None  # [1, n_ephemeral, D]
    spatial_hidden_init: nn.Parameter  # [1, 1, D] -> broadcast to [B, G*G, D]
    scene_positions: Tensor
    read_attn: nn.ModuleList
    write_attn: nn.ModuleList
    read_scale: nn.ModuleList | None
    write_scale: nn.ModuleList | None
    output_proj: nn.Module

    @property
    def glimpse_size(self) -> int:
        return self.cfg.glimpse_grid_size * self.backbone.patch_size

    @property
    def scene_size(self) -> int:
        return self.cfg.scene_grid_size * self.backbone.patch_size

    @property
    def n_scene_registers(self) -> int:
        """Total scene registers (persistent + ephemeral). Fixed count from config."""
        return self.cfg.n_scene_registers

    @property
    def n_persistent_registers(self) -> int:
        return self.n_scene_registers // 2

    @property
    def n_ephemeral_registers(self) -> int:
        return self.n_scene_registers - self.n_persistent_registers

    @property
    def n_local_tokens(self) -> int:
        return self.backbone.n_prefix_tokens + self.cfg.glimpse_grid_size**2

    @property
    def n_adapters(self) -> int:
        """Number of read/write adapter pairs (one per adapter_stride backbone blocks)."""
        return (
            self.backbone.n_blocks + self.cfg.adapter_stride - 1
        ) // self.cfg.adapter_stride

    def _init_state_tokens(self, embed_dim: int) -> None:
        """Initialize learnable state tokens (zero spatial, unit-norm registers)."""
        scale = 1.0 / math.sqrt(embed_dim)
        n_persistent = self.n_persistent_registers
        n_ephemeral = self.n_ephemeral_registers

        self.spatial_hidden_init = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.persistent_registers = (
            nn.Parameter(torch.randn(1, n_persistent, embed_dim) * scale)
            if n_persistent > 0 else None
        )
        self.ephemeral_registers = (
            nn.Parameter(torch.randn(1, n_ephemeral, embed_dim) * scale)
            if n_ephemeral > 0 else None
        )

    def __init__(self, backbone: ViTBackbone, cfg: AVPConfig) -> None:
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg

        embed_dim = backbone.embed_dim
        num_heads = backbone.num_heads
        n_blocks = backbone.n_blocks
        n_adapters = (n_blocks + cfg.adapter_stride - 1) // cfg.adapter_stride

        self._init_state_tokens(embed_dim)

        attn_cfg = cfg.attention
        if cfg.use_convex_gating:
            self.read_attn = nn.ModuleList(
                [
                    ConvexGatedAttention(
                        RoPEReadCrossAttention(embed_dim, num_heads, attn_cfg),
                        RoPEReadCrossAttention(embed_dim, num_heads, attn_cfg),
                        cfg.layer_scale_init,
                    )
                    for _ in range(n_adapters)
                ]
            )
            self.write_attn = nn.ModuleList(
                [
                    ConvexGatedAttention(
                        RoPEWriteCrossAttention(embed_dim, num_heads, attn_cfg),
                        RoPEWriteCrossAttention(embed_dim, num_heads, attn_cfg),
                        cfg.layer_scale_init,
                    )
                    for _ in range(n_adapters)
                ]
            )
            self.read_scale = None
            self.write_scale = None
        else:
            self.read_attn = nn.ModuleList(
                [
                    RoPEReadCrossAttention(embed_dim, num_heads, attn_cfg)
                    for _ in range(n_adapters)
                ]
            )
            self.write_attn = nn.ModuleList(
                [
                    RoPEWriteCrossAttention(embed_dim, num_heads, attn_cfg)
                    for _ in range(n_adapters)
                ]
            )
            self.read_scale = nn.ModuleList(
                [LayerScale(embed_dim, cfg.layer_scale_init) for _ in range(n_adapters)]
            )
            self.write_scale = nn.ModuleList(
                [LayerScale(embed_dim, cfg.layer_scale_init) for _ in range(n_adapters)]
            )

        device = self.spatial_hidden_init.device
        assert isinstance(device, torch.device)
        pos = make_grid_positions(
            cfg.scene_grid_size, cfg.scene_grid_size, device, dtype=backbone.rope_dtype
        )
        self.register_buffer("scene_positions", pos)
        self.scene_positions = pos

        if cfg.use_output_proj:
            layers: list[nn.Module] = []
            if cfg.use_output_proj_norm:
                layers.append(nn.LayerNorm(embed_dim))
            layers.append(nn.Linear(embed_dim, embed_dim))
            layers.append(ElementwiseAffine(embed_dim))
            self.output_proj = nn.Sequential(*layers)
        else:
            self.output_proj = nn.Identity()

    def set_scene_grid_size(self, new_size: int) -> None:
        """Update scene grid size for curriculum. Recomputes scene_positions buffer."""
        assert new_size >= self.cfg.glimpse_grid_size, (
            f"scene_grid_size {new_size} must be >= glimpse_grid_size {self.cfg.glimpse_grid_size}"
        )
        self.cfg = replace(self.cfg, scene_grid_size=new_size)
        pos = make_grid_positions(
            new_size,
            new_size,
            self.scene_positions.device,
            dtype=self.backbone.rope_dtype,
        )
        self.scene_positions = pos

    def _get_base_hidden(self, B: int) -> Tensor:
        """Get base hidden state (learnable inits expanded to batch size).

        Returns: [B, n_persistent + G*G, D] = [persistent_registers | spatial_hidden]
        """
        n_spatial = self.cfg.scene_grid_size**2
        spatial_hidden = self.spatial_hidden_init.expand(B, n_spatial, -1)
        if self.persistent_registers is not None:
            return torch.cat(
                [self.persistent_registers.expand(B, -1, -1), spatial_hidden], dim=1
            )
        return spatial_hidden

    def _init_hidden(self, B: int, hidden: Tensor | None) -> Tensor:
        """Return hidden state, or base if None."""
        if hidden is None:
            return self._get_base_hidden(B)
        return hidden

    def get_spatial(self, hidden: Tensor) -> Tensor:
        """Extract spatial_hidden from full hidden state.

        Args:
            hidden: Full hidden state [B, n_persistent + G*G, D]

        Returns:
            Spatial hidden [B, G*G, D] (excludes persistent registers)
        """
        return hidden[:, self.n_persistent_registers :]

    def get_initial_scene(self, B: int) -> Tensor:
        """Get initial scene (before any viewpoint processing).

        This is a "valid scene" - goes through same output_proj as processed scenes.
        Useful for visualization and ensuring initial state looks like network output.

        Returns:
            [B, G*G, D] initial scene
        """
        n_spatial = self.cfg.scene_grid_size**2
        spatial_hidden = self.spatial_hidden_init.expand(B, n_spatial, -1)
        return self.output_proj(spatial_hidden)

    def compute_scene(self, hidden: Tensor) -> Tensor:
        """Compute projected scene from hidden state.

        Convenience method that extracts spatial and applies output_proj.

        Args:
            hidden: Full hidden state [B, n_persistent + G*G, D]

        Returns:
            Projected scene [B, G*G, D]
        """
        return self.output_proj(self.get_spatial(hidden))

    def _process_glimpse(
        self,
        glimpse: Tensor,
        centers: Tensor,
        scales: Tensor,
        hidden: Tensor | None,
        context: Tensor | None,
    ) -> StepOutput:
        """Process one glimpse: read from hidden, forward through backbone, write to hidden.

        Args:
            glimpse: Extracted glimpse image [B, C, H, W]
            centers: Viewpoint centers [B, 2]
            scales: Viewpoint scales [B]
            hidden: Previous hidden state [B, n_persistent + G*G, D] or None for fresh start
            context: External context tokens [B, N_ctx, D] or None. Participates in attention,
                     returned transformed but NOT persisted in hidden state.

        Returns:
            StepOutput with both hidden (for continuation) and scene (for loss/viz)
        """
        D = self.backbone.embed_dim

        # Tokenize inside checkpoint so patch embedding activations aren't stored
        local_fresh, H, W = self.backbone.prepare_tokens(glimpse)
        G = self.cfg.glimpse_grid_size
        assert H == W == G, f"backbone returned {H}x{W} but config expects {G}x{G}"

        B = local_fresh.shape[0]
        rope_dtype = self.backbone.rope_dtype
        periods = self.backbone.rope_periods
        n_ephemeral = self.n_ephemeral_registers

        # Validate context shape if provided
        n_ctx = 0
        if context is not None:
            assert context.ndim == 3, (
                f"context must be [B, N_ctx, D], got {context.shape}"
            )
            assert context.shape[0] == B, (
                f"context batch {context.shape[0]} != glimpse batch {B}"
            )
            assert context.shape[2] == D, (
                f"context dim {context.shape[2]} != embed_dim {D}"
            )
            n_ctx = context.shape[1]

        local = local_fresh

        # Build hidden_t: [context, ephemeral, persistent, spatial]
        # Context is prepended first so it's at position 0:n_ctx after processing
        hidden_t = self._init_hidden(B, hidden)
        n_persistent = self.n_persistent_registers
        n_spatial = self.cfg.scene_grid_size**2
        assert hidden_t.shape == (B, n_persistent + n_spatial, D), (
            f"hidden shape {hidden_t.shape} != expected ({B}, {n_persistent + n_spatial}, {D})"
        )

        if self.ephemeral_registers is not None:
            hidden_t = torch.cat(
                [self.ephemeral_registers.expand(B, -1, -1), hidden_t], dim=1
            )

        if n_ctx > 0:
            assert context is not None
            hidden_t = torch.cat([context, hidden_t], dim=1)

        # Verify assembled shape before norm
        expected_hidden_t = n_ctx + n_ephemeral + n_persistent + n_spatial
        assert hidden_t.shape == (B, expected_hidden_t, D), (
            f"assembled hidden_t {hidden_t.shape} != expected ({B}, {expected_hidden_t}, {D})"
        )

        local_pos = glimpse_positions(centers, scales, H, W, dtype=rope_dtype)
        scene_pos = self.scene_positions.to(rope_dtype).unsqueeze(0).expand(B, -1, -1)
        local_rope = compute_rope(local_pos, periods)
        scene_rope = compute_rope(scene_pos, periods)

        stride = self.cfg.adapter_stride
        for i in range(self.backbone.n_blocks):
            if i % stride == 0:
                a = i // stride
                if self.cfg.use_convex_gating:
                    local = self.read_attn[a](local, hidden_t, local_rope, scene_rope)
                else:
                    assert self.read_scale is not None
                    local = local + self.read_scale[a](
                        self.read_attn[a](local, hidden_t, local_rope, scene_rope)
                    )
            local = self.backbone.forward_block(i, local, local_rope)
            if i % stride == 0:
                a = i // stride
                if self.cfg.use_convex_gating:
                    hidden_t = self.write_attn[a](
                        hidden_t, local, scene_rope, local_rope
                    )
                else:
                    assert self.write_scale is not None
                    hidden_t = hidden_t + self.write_scale[a](
                        self.write_attn[a](hidden_t, local, scene_rope, local_rope)
                    )

        # Verify shape unchanged after attention
        assert hidden_t.shape == (B, expected_hidden_t, D), (
            f"hidden_t after attention {hidden_t.shape} != expected ({B}, {expected_hidden_t}, {D})"
        )

        # Extract transformed context (if provided)
        context_out: Tensor | None = None
        if n_ctx > 0:
            context_out = hidden_t[:, :n_ctx]
            hidden_t = hidden_t[:, n_ctx:]
            assert isinstance(context_out, Tensor)
            assert context_out.shape == (B, n_ctx, D), (
                f"context_out {context_out.shape} != expected ({B}, {n_ctx}, {D})"
            )

        # Strip ephemeral registers -> persistent + spatial
        hidden_out = hidden_t[:, n_ephemeral:]
        assert hidden_out.shape == (B, n_persistent + n_spatial, D), (
            f"hidden_out {hidden_out.shape} != expected ({B}, {n_persistent + n_spatial}, {D})"
        )

        # Scene output is spatial-only (exclude persistent registers)
        scene_out = self.compute_scene(hidden_out)

        return StepOutput(glimpse, local, hidden_out, scene_out, context_out)

    def forward_step(
        self,
        images: Tensor,
        viewpoint: Viewpoint,
        hidden: Tensor | None = None,
        context: Tensor | None = None,
    ) -> StepOutput:
        """Process a single viewpoint.

        Args:
            images: Input images [B, C, H, W]
            viewpoint: Where to look in the images
            hidden: Previous hidden state [B, G*G, D] for CONTINUATION, or None for fresh start
            context: External context tokens [B, N_ctx, D] or None. Pre-embedded by caller.
                     Participates in attention, returned transformed in context_out.

        Returns:
            StepOutput containing:
            - glimpse: Extracted glimpse image
            - hidden: Updated hidden state (use for CONTINUATION)
            - scene: Projected output (use for LOSS/VIZ)
            - context_out: Transformed context tokens (if context provided)
        """
        glimpse = extract_glimpse(images, viewpoint, self.glimpse_size)

        if self.cfg.gradient_checkpointing and self.training:
            return cast(
                StepOutput,
                checkpoint(
                    self._process_glimpse,
                    glimpse,
                    viewpoint.centers,
                    viewpoint.scales,
                    hidden,
                    context,
                    use_reentrant=False,
                ),
            )
        return self._process_glimpse(
            glimpse, viewpoint.centers, viewpoint.scales, hidden, context
        )

    # ==================== General Primitive ====================

    def forward_reduce(
        self,
        images: Tensor,
        viewpoints: list[Viewpoint],
        reducer: Callable[[T, StepOutput], T],
        init: T,
        *,
        hidden: Tensor | None = None,
        context: Tensor | None = None,
    ) -> tuple[T, Tensor]:
        """Scan over viewpoints, reducing with custom function. Returns (accumulator, final_hidden)."""
        B = images.shape[0]
        acc = init
        for vp in viewpoints:
            out = self.forward_step(images, vp, hidden, context)
            hidden = out.hidden
            acc = reducer(acc, out)
        if hidden is None:
            hidden = self._init_hidden(B, None)
        return acc, hidden

    # ==================== Standard Invocations ====================

    def forward_loss(
        self,
        images: Tensor,
        viewpoints: list[Viewpoint],
        target: Tensor,
        *,
        hidden: Tensor | None = None,
        context: Tensor | None = None,
        loss_fn: Callable[[Tensor, Tensor], Tensor] = F.mse_loss,
    ) -> tuple[Tensor, Tensor]:
        """Compute average loss across all viewpoints. Returns (loss, final_hidden)."""
        assert len(viewpoints) > 0

        def reducer(acc: Tensor, out: StepOutput) -> Tensor:
            return acc + loss_fn(out.scene, target)

        total, final_hidden = self.forward_reduce(
            images, viewpoints, reducer,
            init=torch.tensor(0.0, device=images.device),
            hidden=hidden, context=context,
        )
        return total / len(viewpoints), final_hidden

    def forward_trajectory_full(
        self,
        images: Tensor,
        viewpoints: list[Viewpoint],
        hidden: Tensor | None = None,
        context: Tensor | None = None,
    ) -> tuple[list[StepOutput], Tensor]:
        """Collect full StepOutput at each step. Returns (outputs, final_hidden)."""
        def reducer(acc: list[StepOutput], out: StepOutput) -> list[StepOutput]:
            return [*acc, out]
        return self.forward_reduce(images, viewpoints, reducer, init=[], hidden=hidden, context=context)

    @override
    def forward(
        self,
        images: Tensor,
        viewpoints: list[Viewpoint],
        hidden: Tensor | None = None,
        context: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Process viewpoints, return final scene. Returns (scene, final_hidden)."""
        def reducer(acc: Tensor, out: StepOutput) -> Tensor:
            return out.scene
        dummy = torch.empty(0, device=images.device)
        return self.forward_reduce(images, viewpoints, reducer, init=dummy, hidden=hidden, context=context)
