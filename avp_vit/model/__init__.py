import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, NamedTuple, TypeVar, cast, final, override

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from avp_vit.attention import (
    CrossAttentionConfig,
    RoPECrossAttention,
    RoPEReadCrossAttention,
    RoPEWriteCrossAttention,
    ScaledResidualAttention,
)
from avp_vit.attention.convex import CheapConvexGatedAttention, ConvexGatedAttention
from avp_vit.backbone import ViTBackbone
from avp_vit.glimpse import Viewpoint, sample_at_viewpoint
from avp_vit.model.hidden import HiddenStreamParams
from avp_vit.rope import compute_rope, glimpse_positions, make_grid_positions

GatingMode = Literal["none", "cheap", "full"]


def _make_gated_attn(
    attn_cls: type[RoPECrossAttention],
    dim: int,
    num_heads: int,
    attn_cfg: CrossAttentionConfig,
    scale_init: float,
    gating: GatingMode,
) -> nn.Module:
    """Create attention module with appropriate gating/scaling."""
    attn = attn_cls(dim, num_heads, attn_cfg)
    if gating == "none":
        return ScaledResidualAttention(attn, scale_init)
    if gating == "cheap":
        return CheapConvexGatedAttention(attn, scale_init)
    if gating == "full":
        gate_attn = attn_cls(dim, num_heads, attn_cfg)
        return ConvexGatedAttention(attn, gate_attn, scale_init)
    raise ValueError(f"Unknown gating mode: {gating!r}")


# Type variable for forward_reduce accumulator
T = TypeVar("T")


class StepOutput(NamedTuple):
    """Output from a single AVPViT forward step."""

    glimpse: Tensor  # [B, C, H, W]
    local: Tensor  # [B, N, student_embed_dim]
    hidden: Tensor  # [B, n_registers + G*G, student_embed_dim]
    scene: Tensor  # [B, G*G, teacher_dim]
    context_out: Tensor | None  # [B, N_ctx, student_embed_dim] or None


class LossOutputs(NamedTuple):
    """Separate loss components from forward_loss. Trainer combines as needed."""

    scene: Tensor  # Scene loss (always computed)
    local: Tensor | None  # Local loss (None if use_local_loss=False)
    cls: Tensor | None  # CLS loss (None if use_cls_loss=False)


@final
@dataclass
class AVPConfig:
    glimpse_grid_size: int = 8  # 256px^2
    n_scene_registers: int = 32  # 0 = disabled, >0 = fixed count
    layer_scale_init: float = 1e-6  # Init for LayerScale (reference: 0.01)
    use_recurrence_ln: bool = False  # LN at recurrence boundary (False = Identity)
    gradient_checkpointing: bool = False  # Checkpoint at timestep boundaries
    gating: GatingMode = "cheap"  # none=LayerScale, cheap=CheapConvex, full=ConvexGated
    adapter_stride: int = 4  # Adapters every N backbone blocks (reference: 1)
    read_attention: CrossAttentionConfig = field(default_factory=CrossAttentionConfig)
    write_attention: CrossAttentionConfig = field(default_factory=CrossAttentionConfig)
    use_local_loss: bool = False  # Enable local loss (supervise glimpse predictions)
    use_cls_loss: bool = True  # Enable CLS token loss (supervise CLS predictions)


@final
class AVPViT(nn.Module):
    """Active Visual Pondering ViT.

    hidden = [cls | scene_registers | spatial_hidden], shape [B, n_cls + n_registers + G*G, D]
    scene = scene_proj(spatial_hidden), shape [B, G*G, teacher_dim]
    """

    backbone: ViTBackbone
    cfg: AVPConfig
    teacher_dim: int
    hidden_stream: HiddenStreamParams
    read_attn: nn.ModuleList
    write_attn: nn.ModuleList
    scene_proj: nn.Sequential

    @property
    def glimpse_size(self) -> int:
        return self.cfg.glimpse_grid_size * self.backbone.patch_size

    @property
    def n_cls(self) -> int:
        """Number of scene CLS tokens (always 1)."""
        return 1

    @property
    def n_registers(self) -> int:
        return self.hidden_stream.n_registers

    @property
    def n_prefix(self) -> int:
        """Number of prefix tokens in hidden state (cls + registers)."""
        return self.hidden_stream.n_prefix

    # Delegation properties for hidden stream components
    @property
    def scene_registers(self) -> nn.Parameter | None:
        return self.hidden_stream.registers

    @property
    def cls_ln(self) -> nn.Module:
        return self.hidden_stream.cls_ln

    @property
    def reg_ln(self) -> nn.Module:
        return self.hidden_stream.reg_ln

    @property
    def spatial_ln(self) -> nn.Module:
        return self.hidden_stream.spatial_ln

    @property
    def n_local_tokens(self) -> int:
        return self.backbone.n_prefix_tokens + self.cfg.glimpse_grid_size**2

    @property
    def n_adapters(self) -> int:
        """Number of read/write adapter pairs (first after adapter_stride blocks, then every stride)."""
        return (self.backbone.n_blocks - 1) // self.cfg.adapter_stride

    # === Hidden state indexing: [cls | registers | spatial] ===
    # Use these properties for ALL indexing into hidden state

    def _cls_slice(self) -> slice:
        """Slice for CLS token(s) in hidden: [0:n_cls]."""
        return slice(0, self.n_cls)

    def _reg_slice(self) -> slice:
        """Slice for registers in hidden: [n_cls:n_cls+n_reg]."""
        return slice(self.n_cls, self.n_cls + self.n_registers)

    def _spatial_slice(self) -> slice:
        """Slice for spatial tokens in hidden: [n_cls+n_reg:]."""
        return slice(self.n_prefix, None)

    def __init__(
        self,
        backbone: ViTBackbone,
        cfg: AVPConfig,
        teacher_dim: int,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg
        self.teacher_dim = teacher_dim

        embed_dim = backbone.embed_dim
        num_heads = backbone.num_heads
        n_blocks = backbone.n_blocks
        n_adapters = (n_blocks - 1) // cfg.adapter_stride

        self.hidden_stream = HiddenStreamParams(
            embed_dim, cfg.n_scene_registers, cfg.use_recurrence_ln
        )
        # Consistency assertions
        assert self.hidden_stream.n_registers == cfg.n_scene_registers
        assert self.hidden_stream.n_prefix == 1 + cfg.n_scene_registers

        self.read_attn = nn.ModuleList(
            [
                _make_gated_attn(
                    RoPEReadCrossAttention,
                    embed_dim,
                    num_heads,
                    cfg.read_attention,
                    cfg.layer_scale_init,
                    cfg.gating,
                )
                for _ in range(n_adapters)
            ]
        )
        self.write_attn = nn.ModuleList(
            [
                _make_gated_attn(
                    RoPEWriteCrossAttention,
                    embed_dim,
                    num_heads,
                    cfg.write_attention,
                    cfg.layer_scale_init,
                    cfg.gating,
                )
                for _ in range(n_adapters)
            ]
        )

        # Scene projection: LayerNorm + Linear (projects to teacher_dim for loss)
        self.scene_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, self.teacher_dim),
        )

        # Local projection (for local loss): mirrors scene_proj structure
        if cfg.use_local_loss:
            self.local_proj: nn.Sequential | None = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, self.teacher_dim),
            )
        else:
            self.local_proj = None

        # CLS projection (for CLS loss): projects CLS token to teacher_dim
        if cfg.use_cls_loss:
            self.cls_proj: nn.Sequential | None = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, self.teacher_dim),
            )
        else:
            self.cls_proj = None

    def _infer_scene_grid_size(self, hidden: Tensor) -> int:
        """Infer scene grid size from hidden state shape."""
        n_spatial = hidden.shape[1] - self.n_prefix  # subtract cls + registers
        G = int(math.sqrt(n_spatial))
        assert G * G == n_spatial, (
            f"hidden spatial dim {n_spatial} is not a perfect square"
        )
        return G

    def init_hidden(self, batch_size: int, scene_grid_size: int) -> Tensor:
        """Create initial hidden state for given batch size and scene grid size."""
        n_spatial = scene_grid_size**2
        hidden = self.hidden_stream.init_hidden(batch_size, n_spatial)
        assert hidden.shape == (
            batch_size,
            self.n_prefix + n_spatial,
            self.backbone.embed_dim,
        )
        return hidden

    def _normalize_hidden(self, hidden: Tensor) -> Tensor:
        """Normalize hidden at recurrence boundary: [cls | registers | spatial]."""
        result = self.hidden_stream.normalize(hidden)
        assert result.shape == hidden.shape
        return result

    def get_cls(self, hidden: Tensor) -> Tensor:
        """Extract CLS from hidden: [B, n_cls + n_reg + G*G, D] -> [B, D]."""
        return hidden[:, 0]  # CLS is always first

    def get_spatial(self, hidden: Tensor) -> Tensor:
        """Extract spatial from hidden: [B, n_cls + n_reg + G*G, D] -> [B, G*G, D]."""
        return hidden[:, self._spatial_slice()]

    def compute_scene(self, hidden: Tensor) -> Tensor:
        """Extract spatial tokens from hidden, project to teacher_dim."""
        return self.scene_proj(self.get_spatial(hidden))

    def compute_local(self, local: Tensor, viewpoint: Viewpoint) -> Tensor:
        """Project local patch tokens to teacher_dim."""
        assert self.local_proj is not None, (
            "local_proj not initialized (use_local_loss=False)"
        )
        n_prefix = self.backbone.n_prefix_tokens
        patch_tokens = local[:, n_prefix:]  # [B, G², D_student]
        return self.local_proj(patch_tokens)  # [B, G², D_teacher]

    def compute_cls(self, hidden: Tensor) -> Tensor:
        """Project scene CLS token to teacher_dim. Returns [B, D_teacher]."""
        assert self.cls_proj is not None, (
            "cls_proj not initialized (use_cls_loss=False)"
        )
        cls_token = self.get_cls(hidden)  # [B, D_student]
        return self.cls_proj(cls_token)  # [B, D_teacher]

    def _process_glimpse(
        self,
        glimpse: Tensor,
        centers: Tensor,
        scales: Tensor,
        hidden: Tensor,
        context: Tensor | None,
    ) -> StepOutput:
        """Process one glimpse: read from hidden, forward through backbone, write to hidden."""
        D = self.backbone.embed_dim
        B = hidden.shape[0]
        scene_grid_size = self._infer_scene_grid_size(hidden)
        n_spatial = scene_grid_size**2

        local, H, W = self.backbone.prepare_tokens(glimpse)
        assert H == W == self.cfg.glimpse_grid_size
        assert local.shape[0] == B

        # Validate context
        n_ctx = 0
        if context is not None:
            assert context.shape == (B, context.shape[1], D)
            n_ctx = context.shape[1]

        # Normalize hidden at recurrence boundary
        hidden_t = self._normalize_hidden(hidden)
        assert hidden_t.shape == (B, self.n_prefix + n_spatial, D)

        # Prepend context if provided: [context | cls | registers | spatial]
        if n_ctx > 0:
            assert context is not None
            hidden_t = torch.cat([context, hidden_t], dim=1)

        # RoPE positions
        local_pos = glimpse_positions(
            centers, scales, H, W, dtype=self.backbone.rope_dtype
        )
        scene_pos = (
            make_grid_positions(
                scene_grid_size,
                scene_grid_size,
                glimpse.device,
                dtype=self.backbone.rope_dtype,
            )
            .unsqueeze(0)
            .expand(B, -1, -1)
        )
        local_rope = compute_rope(local_pos, self.backbone.rope_periods)
        scene_rope = compute_rope(scene_pos, self.backbone.rope_periods)

        # Interleaved read/write attention (adapters start after stride blocks)
        stride = self.cfg.adapter_stride
        for i in range(self.backbone.n_blocks):
            if i >= stride and i % stride == 0:
                a = i // stride - 1
                local = self.read_attn[a](local, hidden_t, local_rope, scene_rope)
            local = self.backbone.forward_block(i, local, local_rope)
            if i >= stride and i % stride == 0:
                a = i // stride - 1
                hidden_t = self.write_attn[a](hidden_t, local, scene_rope, local_rope)

        # Extract context if provided
        context_out: Tensor | None = None
        if n_ctx > 0:
            context_out = hidden_t[:, :n_ctx]
            hidden_t = hidden_t[:, n_ctx:]

        scene_out = self.compute_scene(hidden_t)
        return StepOutput(glimpse, local, hidden_t, scene_out, context_out)

    def forward_step(
        self,
        images: Tensor,
        viewpoint: Viewpoint,
        hidden: Tensor,
        context: Tensor | None = None,
    ) -> StepOutput:
        """Process a single viewpoint."""
        glimpse = sample_at_viewpoint(images, viewpoint, self.glimpse_size)

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
        hidden: Tensor,
        reducer: Callable[[T, StepOutput, Viewpoint], T],
        init: T,
        *,
        context: Tensor | None = None,
    ) -> tuple[T, Tensor]:
        """Scan over viewpoints, reducing with custom function. Returns (accumulator, final_hidden)."""
        acc = init
        for vp in viewpoints:
            out = self.forward_step(images, vp, hidden, context)
            hidden = out.hidden
            acc = reducer(acc, out, vp)
        return acc, hidden

    # ==================== Standard Invocations ====================

    def forward_loss(
        self,
        images: Tensor,
        viewpoints: list[Viewpoint],
        target: Tensor,
        hidden: Tensor,
        *,
        cls_target: Tensor | None = None,
        context: Tensor | None = None,
        loss_fn: Callable[[Tensor, Tensor], Tensor] = F.mse_loss,
    ) -> tuple[LossOutputs, Tensor]:
        """Compute losses across all viewpoints. Returns (LossOutputs, final_hidden).

        Args:
            target: Teacher's patch tokens [B, N, D] for scene/local loss
            cls_target: Teacher's CLS token [B, D] for CLS loss (same for all viewpoints)

        LossOutputs contains scene loss (always), local loss (if enabled), and cls loss (if enabled).
        Caller is responsible for combining losses as desired.
        """
        assert len(viewpoints) > 0
        n = len(viewpoints)
        device = images.device

        scene_loss = torch.tensor(0.0, device=device)
        local_loss_acc: Tensor | None = None
        cls_loss_acc: Tensor | None = None

        # Prepare target spatial for local loss (if enabled)
        B, N_target, D = target.shape
        G_scene = int(N_target**0.5)
        target_spatial = target.view(B, G_scene, G_scene, D).permute(0, 3, 1, 2)

        for vp in viewpoints:
            out = self.forward_step(images, vp, hidden, context)
            hidden = out.hidden

            # Scene loss (always)
            scene_loss = scene_loss + loss_fn(out.scene, target)

            # Local loss (if enabled)
            if self.local_proj is not None:
                local_pred = self.compute_local(out.local, vp)
                G_glimpse = self.cfg.glimpse_grid_size
                cropped = sample_at_viewpoint(target_spatial, vp, G_glimpse)
                cropped = cropped.permute(0, 2, 3, 1).reshape(B, -1, D)
                step_local = loss_fn(local_pred, cropped)
                local_loss_acc = (
                    step_local
                    if local_loss_acc is None
                    else local_loss_acc + step_local
                )

            # CLS loss (if enabled) - scene CLS vs teacher's full-image CLS
            if self.cls_proj is not None:
                assert cls_target is not None, (
                    "cls_target required when use_cls_loss=True"
                )
                cls_pred = self.compute_cls(out.hidden)
                step_cls = loss_fn(cls_pred, cls_target)
                cls_loss_acc = (
                    step_cls if cls_loss_acc is None else cls_loss_acc + step_cls
                )

        losses = LossOutputs(
            scene=scene_loss / n,
            local=local_loss_acc / n if local_loss_acc is not None else None,
            cls=cls_loss_acc / n if cls_loss_acc is not None else None,
        )
        return losses, hidden

    def forward_trajectory_full(
        self,
        images: Tensor,
        viewpoints: list[Viewpoint],
        hidden: Tensor,
        context: Tensor | None = None,
    ) -> tuple[list[StepOutput], Tensor]:
        """Collect full StepOutput at each step. Returns (outputs, final_hidden)."""

        def reducer(
            acc: list[StepOutput], out: StepOutput, _vp: Viewpoint
        ) -> list[StepOutput]:
            return [*acc, out]

        return self.forward_reduce(
            images, viewpoints, hidden, reducer, init=[], context=context
        )

    @override
    def forward(
        self,
        images: Tensor,
        viewpoints: list[Viewpoint],
        hidden: Tensor,
        context: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Process viewpoints, return final scene. Returns (scene, final_hidden)."""

        def reducer(acc: Tensor, out: StepOutput, _vp: Viewpoint) -> Tensor:
            return out.scene

        dummy = torch.empty(0, device=images.device)
        return self.forward_reduce(
            images, viewpoints, hidden, reducer, init=dummy, context=context
        )
