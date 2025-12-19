"""Active vision wrapper for CanViT."""

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NamedTuple, TypeVar, cast, final

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from canvit import CanViT, CanViTConfig
from canvit.backbone import ViTBackbone
from canvit.rope import compute_rope, glimpse_positions, make_grid_positions

from avp_vit.glimpse import Viewpoint, sample_at_viewpoint

T = TypeVar("T")


class StepOutput(NamedTuple):
    """Output from a single ActiveCanViT forward step."""

    glimpse: Tensor  # [B, C, H, W]
    local: Tensor  # [B, N, D]
    canvas: Tensor  # [B, n_prefix + G*G, D]
    scene: Tensor  # [B, G*G, teacher_dim]


class LossOutputs(NamedTuple):
    """Loss components from forward_loss."""

    scene: Tensor | None
    cls: Tensor | None


@final
@dataclass
class ActiveCanViTConfig:
    """Config for ActiveCanViT (active vision wrapper)."""

    glimpse_grid_size: int = 8  # 8x8 patches = 128px for patch_size=16
    gradient_checkpointing: bool = False
    use_scene_loss: bool = True
    use_cls_loss: bool = True
    canvit: CanViTConfig = field(default_factory=CanViTConfig)


@final
class ActiveCanViT(nn.Module):
    """Active vision wrapper around CanViT.

    Handles viewpoint sampling, loss computation.
    CanViT owns canvas state (init, normalization).

    canvas = [cls | registers | spatial], shape [B, 1 + n_reg + G*G, D]
    scene = scene_proj(spatial), shape [B, G*G, teacher_dim]
    """

    canvit: CanViT
    cfg: ActiveCanViTConfig
    teacher_dim: int
    scene_proj: nn.Sequential
    cls_proj: nn.Sequential | None

    def __init__(
        self,
        backbone: ViTBackbone,
        cfg: ActiveCanViTConfig,
        teacher_dim: int,
    ) -> None:
        super().__init__()
        self.canvit = CanViT(backbone, cfg.canvit)
        self.cfg = cfg
        self.teacher_dim = teacher_dim

        dim = backbone.embed_dim
        self.scene_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, teacher_dim),
        )
        self.cls_proj = (
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, teacher_dim))
            if cfg.use_cls_loss
            else None
        )

        # Scale down output projection weights (no grad, simpler than scaling in CanViT)
        scale = 1.0 / math.sqrt(dim)
        with torch.no_grad():
            self.scene_proj[1].weight.mul_(scale)
            if self.cls_proj is not None:
                self.cls_proj[1].weight.mul_(scale)

    @property
    def backbone(self) -> ViTBackbone:
        return self.canvit.backbone

    @property
    def glimpse_size_px(self) -> int:
        return self.cfg.glimpse_grid_size * self.backbone.patch_size_px

    @property
    def n_registers(self) -> int:
        return self.canvit.n_registers

    @property
    def n_prefix(self) -> int:
        return self.canvit.n_prefix

    def init_canvas(self, batch_size: int, canvas_grid_size: int) -> Tensor:
        """Create initial canvas (delegates to CanViT)."""
        return self.canvit.init_canvas(batch_size, canvas_grid_size)

    def _infer_canvas_grid_size(self, canvas: Tensor) -> int:
        n_spatial = canvas.shape[1] - self.n_prefix
        G = int(math.sqrt(n_spatial))
        assert G * G == n_spatial
        return G

    def get_cls(self, canvas: Tensor) -> Tensor:
        return canvas[:, 0]

    def get_spatial(self, canvas: Tensor) -> Tensor:
        return canvas[:, self.n_prefix:]

    def compute_scene(self, canvas: Tensor) -> Tensor:
        return self.scene_proj(self.get_spatial(canvas))

    def compute_cls(self, canvas: Tensor) -> Tensor:
        assert self.cls_proj is not None
        return self.cls_proj(self.get_cls(canvas))

    def _process_glimpse(
        self,
        glimpse: Tensor,
        centers: Tensor,
        scales: Tensor,
        canvas: Tensor,
    ) -> StepOutput:
        B = canvas.shape[0]
        canvas_grid_size = self._infer_canvas_grid_size(canvas)

        local, H, W = self.backbone.prepare_tokens(glimpse)
        assert H == W == self.cfg.glimpse_grid_size

        # RoPE positions
        local_pos = glimpse_positions(centers, scales, H, W, dtype=self.backbone.rope_dtype)
        canvas_pos = (
            make_grid_positions(canvas_grid_size, canvas_grid_size, glimpse.device, self.backbone.rope_dtype)
            .unsqueeze(0)
            .expand(B, -1, -1)
        )
        local_rope = compute_rope(local_pos, self.backbone.rope_periods)
        canvas_rope = compute_rope(canvas_pos, self.backbone.rope_periods)

        local_out, canvas_out = self.canvit(local, canvas, local_rope, canvas_rope)
        scene_out = self.compute_scene(canvas_out)
        return StepOutput(glimpse, local_out, canvas_out, scene_out)

    def forward_step(
        self,
        images: Tensor,
        viewpoint: Viewpoint,
        canvas: Tensor,
    ) -> StepOutput:
        glimpse = sample_at_viewpoint(images, viewpoint, self.glimpse_size_px)

        if self.cfg.gradient_checkpointing and self.training:
            return cast(
                StepOutput,
                checkpoint(
                    self._process_glimpse,
                    glimpse,
                    viewpoint.centers,
                    viewpoint.scales,
                    canvas,
                    use_reentrant=False,
                ),
            )
        return self._process_glimpse(glimpse, viewpoint.centers, viewpoint.scales, canvas)

    def forward_reduce(
        self,
        images: Tensor,
        viewpoints: list[Viewpoint],
        canvas: Tensor,
        reducer: Callable[[T, StepOutput, Viewpoint], T],
        init: T,
    ) -> tuple[T, Tensor]:
        acc = init
        for vp in viewpoints:
            out = self.forward_step(images, vp, canvas)
            canvas = out.canvas
            acc = reducer(acc, out, vp)
        return acc, canvas

    def forward_loss(
        self,
        images: Tensor,
        viewpoints: list[Viewpoint],
        target: Tensor,
        canvas: Tensor,
        *,
        cls_target: Tensor | None = None,
        loss_fn: Callable[[Tensor, Tensor], Tensor] = F.mse_loss,
    ) -> tuple[LossOutputs, Tensor]:
        """Compute losses across all viewpoints. Includes initial canvas (n+1 terms)."""
        assert len(viewpoints) > 0
        n = len(viewpoints)

        scene_loss_acc: Tensor | None = None
        if self.cfg.use_scene_loss:
            scene_loss_acc = loss_fn(self.compute_scene(canvas), target)
        cls_loss_acc: Tensor | None = None
        if self.cls_proj is not None:
            assert cls_target is not None
            cls_loss_acc = loss_fn(self.compute_cls(canvas), cls_target)

        for vp in viewpoints:
            out = self.forward_step(images, vp, canvas)
            canvas = out.canvas
            if self.cfg.use_scene_loss:
                assert scene_loss_acc is not None
                scene_loss_acc = scene_loss_acc + loss_fn(out.scene, target)
            if self.cls_proj is not None:
                assert cls_target is not None and cls_loss_acc is not None
                cls_loss_acc = cls_loss_acc + loss_fn(self.compute_cls(canvas), cls_target)

        return LossOutputs(
            scene=scene_loss_acc / (n + 1) if scene_loss_acc is not None else None,
            cls=cls_loss_acc / (n + 1) if cls_loss_acc is not None else None,
        ), canvas

    def forward_trajectory_full(
        self,
        images: Tensor,
        viewpoints: list[Viewpoint],
        canvas: Tensor,
    ) -> tuple[list[StepOutput], Tensor]:
        def reducer(acc: list[StepOutput], out: StepOutput, _vp: Viewpoint) -> list[StepOutput]:
            return [*acc, out]
        return self.forward_reduce(images, viewpoints, canvas, reducer, init=[])

    def forward(
        self,
        images: Tensor,
        viewpoints: list[Viewpoint],
        canvas: Tensor,
    ) -> tuple[Tensor, Tensor]:
        def reducer(acc: Tensor, out: StepOutput, _vp: Viewpoint) -> Tensor:
            return out.scene
        dummy = torch.empty(0, device=images.device)
        return self.forward_reduce(images, viewpoints, canvas, reducer, init=dummy)
