"""Training step with truncated BPTT and independent branches."""

import random
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Callable, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from avp_vit import CanViTForPretraining
from canvit import CanViTOutput, RecurrentState, Viewpoint, sample_at_viewpoint

from .loss import mse_loss
from .viewpoint import Viewpoint as NamedViewpoint, ViewpointType
from .viz.sample import VizSampleData, extract_sample0_viz
from .viz.image import imagenet_denormalize


class LossOutput(NamedTuple):
    """Output from compute_loss - individual losses + combined mean."""

    scene_patches_loss: Tensor
    scene_cls_loss: Tensor
    combined: Tensor  # sum of active losses
    scene_pred: Tensor  # for cosine similarity metrics
    cls_pred: Tensor


class BranchMetrics(NamedTuple):
    """Metrics for a branch type."""

    loss: Tensor
    scene_patches_loss: Tensor
    scene_cls_loss: Tensor
    scene_cos_raw: Tensor
    scene_cos_norm: Tensor
    cls_cos_raw: Tensor
    cls_cos_norm: Tensor


@dataclass
class TrainVizData:
    """Viz data collected during one training branch (sample 0 only)."""

    image: np.ndarray  # [H, W, 3] denormalized input
    teacher_features: np.ndarray  # [G², D] normalized teacher features (target)
    viewpoints: list[NamedViewpoint]  # viewpoints used at each timestep
    viz_samples: list[VizSampleData] = field(default_factory=list)  # per-timestep
    initial_scene: np.ndarray | None = None  # [G², D] initial scene prediction
    initial_canvas_spatial: np.ndarray | None = None  # [G², C] initial canvas


class StepMetrics(NamedTuple):
    """Output from training_step."""

    total_loss: Tensor
    full_start: BranchMetrics | None  # None if n_full_start_branches=0
    random_start: BranchMetrics | None  # None if n_random_start_branches=0
    n_glimpses: int  # trajectory length this step
    viz_data: TrainVizData | None = None  # optional viz from first branch


@dataclass
class ChunkState:
    """State for TBPTT chunk processing."""

    state: RecurrentState
    vpe: Tensor | None
    chunk_combined_loss: Tensor  # with grad
    total_combined_loss: Tensor  # detached
    total_scene_patches_loss: Tensor
    total_scene_cls_loss: Tensor
    n_steps: int
    scene_pred: Tensor
    cls_pred: Tensor


class StepOutput(NamedTuple):
    """Output from forward_glimpse: model output + sampled glimpse."""

    out: CanViTOutput
    glimpse: Tensor


def training_step(
    *,
    model: CanViTForPretraining,
    images: Tensor,
    scene_target: Tensor,
    cls_target: Tensor,
    raw_scene_target: Tensor,
    raw_cls_target: Tensor,
    scene_denorm: Callable[[Tensor], Tensor],
    cls_denorm: Callable[[Tensor], Tensor],
    enable_scene_patches_loss: bool,
    enable_scene_cls_loss: bool,
    glimpse_size_px: int,
    canvas_grid_size: int,
    n_full_start_branches: int,
    n_random_start_branches: int,
    chunk_size: int,
    continue_prob: float,
    min_viewpoint_scale: float,
    amp_ctx: AbstractContextManager,
    use_checkpointing: bool,
    collect_viz: bool = False,
) -> StepMetrics:
    """Training with truncated BPTT and independent branches.

    Each branch is fully independent: own t0, own trajectory, own backward.
    No retain_graph needed. Memory is O(chunk_size), not O(n_branches).
    """
    n_branches = n_full_start_branches + n_random_start_branches
    assert n_branches >= 1
    assert chunk_size >= 2
    assert 0.0 <= continue_prob <= 1.0
    device = images.device
    B = images.shape[0]

    state_init = model.init_state(batch_size=B, canvas_grid_size=canvas_grid_size)

    # Sample trajectory length (shared across branches for this step)
    n_glimpses = chunk_size
    while random.random() < continue_prob:
        n_glimpses += chunk_size

    has_policy = model.policy is not None

    # t1_schedule[t-1][branch_idx] = viewpoint type for timestep t, branch branch_idx
    t1_schedule: list[list[ViewpointType]] = []
    for _ in range(1, n_glimpses):
        if has_policy:
            types = [ViewpointType.RANDOM] * (n_branches // 2) + [ViewpointType.POLICY] * (n_branches - n_branches // 2)
        else:
            types = [ViewpointType.RANDOM] * n_branches
        random.shuffle(types)
        t1_schedule.append(types)

    full_metrics: list[BranchMetrics] = []
    random_metrics: list[BranchMetrics] = []

    # Viz collection for first branch only (when enabled)
    viz_data: TrainVizData | None = None

    def make_named_vp(vp_type: ViewpointType, vpe: Tensor | None) -> NamedViewpoint:
        """Create a NamedViewpoint (has .name for viz, convertible to canvit Viewpoint)."""
        if vp_type == ViewpointType.RANDOM:
            return NamedViewpoint.random(batch_size=B, device=device, min_scale=min_viewpoint_scale)
        if vp_type == ViewpointType.FULL:
            return NamedViewpoint.full_scene(batch_size=B, device=device)
        assert model.policy is not None and vpe is not None
        p = model.policy(vpe)
        return NamedViewpoint(name="policy", centers=p.position, scales=p.scale)

    def to_canvit_vp(vp: NamedViewpoint) -> Viewpoint:
        return Viewpoint(centers=vp.centers, scales=vp.scales)

    def forward_glimpse(*, state: RecurrentState, vp: Viewpoint, use_ckpt: bool) -> StepOutput:
        glimpse = sample_at_viewpoint(spatial=images, viewpoint=vp, glimpse_size_px=glimpse_size_px)
        if use_ckpt:
            out = checkpoint(
                lambda g, cv, cl, ctr, sc: model.forward(
                    glimpse=g,
                    state=RecurrentState(canvas=cv, recurrent_cls=cl),
                    viewpoint=Viewpoint(centers=ctr, scales=sc),
                ),
                glimpse, state.canvas, state.recurrent_cls, vp.centers, vp.scales,
                use_reentrant=False,
            )
            assert isinstance(out, CanViTOutput)
            return StepOutput(out=out, glimpse=glimpse)
        out = model.forward(glimpse=glimpse, state=state, viewpoint=vp)
        return StepOutput(out=out, glimpse=glimpse)

    def compute_loss(out: CanViTOutput) -> LossOutput:
        scene_pred = model.predict_teacher_scene(out.state.canvas)
        cls_pred = model.predict_scene_teacher_cls(out.state.recurrent_cls)

        scene_patches_loss = torch.zeros((), device=device)
        scene_cls_loss = torch.zeros((), device=device)

        if enable_scene_patches_loss:
            scene_patches_loss = mse_loss(scene_pred, scene_target)
        if enable_scene_cls_loss:
            scene_cls_loss = mse_loss(cls_pred, cls_target)

        active: list[Tensor] = []
        if enable_scene_patches_loss:
            active.append(scene_patches_loss)
        if enable_scene_cls_loss:
            active.append(scene_cls_loss)
        assert len(active) > 0, "At least one loss must be enabled"
        combined = torch.stack(active).sum()

        return LossOutput(
            scene_patches_loss=scene_patches_loss,
            scene_cls_loss=scene_cls_loss,
            combined=combined,
            scene_pred=scene_pred,
            cls_pred=cls_pred,
        )

    def run_branch(t0_type: ViewpointType, branch_idx: int) -> BranchMetrics:
        nonlocal viz_data
        do_viz = collect_viz and branch_idx == 0

        # Capture initial state for viz (before any glimpses)
        if do_viz:
            init_scene = model.predict_teacher_scene(state_init.canvas)
            init_spatial = model.get_spatial(state_init.canvas[0:1])[0]
            viz_data = TrainVizData(
                image=imagenet_denormalize(images[0].cpu()).numpy(),
                teacher_features=scene_target[0].cpu().float().numpy(),
                viewpoints=[],
                viz_samples=[],
                initial_scene=init_scene[0].detach().cpu().float().numpy(),
                initial_canvas_spatial=init_spatial.detach().cpu().float().numpy(),
            )

        # t0 forward
        with amp_ctx:
            vp0_named = make_named_vp(t0_type, None)
            vp0 = to_canvit_vp(vp0_named)
            step_out = forward_glimpse(state=state_init, vp=vp0, use_ckpt=False)
            out, glimpse = step_out.out, step_out.glimpse
            L = compute_loss(out)

        if do_viz:
            assert viz_data is not None
            viz_data.viewpoints.append(vp0_named)
            viz_data.viz_samples.append(extract_sample0_viz(out, glimpse, L.scene_pred, model))

        chunk = ChunkState(
            state=out.state,
            vpe=out.vpe,
            chunk_combined_loss=L.combined.float(),
            total_combined_loss=L.combined.detach().float(),
            total_scene_patches_loss=L.scene_patches_loss.detach().float(),
            total_scene_cls_loss=L.scene_cls_loss.detach().float(),
            n_steps=1,
            scene_pred=L.scene_pred,
            cls_pred=L.cls_pred,
        )

        for t in range(1, n_glimpses):
            # t>=1: use pre-computed schedule (half RANDOM, half POLICY, shuffled)
            vp_type = t1_schedule[t - 1][branch_idx]
            vp_named = make_named_vp(vp_type, chunk.vpe)
            vp = to_canvit_vp(vp_named)

            with amp_ctx:
                use_ckpt = use_checkpointing and (t % 2 == 1)
                step_out = forward_glimpse(state=chunk.state, vp=vp, use_ckpt=use_ckpt)
                out, glimpse = step_out.out, step_out.glimpse
                L = compute_loss(out)

            if do_viz:
                assert viz_data is not None
                viz_data.viewpoints.append(vp_named)
                viz_data.viz_samples.append(extract_sample0_viz(out, glimpse, L.scene_pred, model))

            chunk.chunk_combined_loss = chunk.chunk_combined_loss + L.combined.float()
            chunk.total_combined_loss = chunk.total_combined_loss + L.combined.detach().float()
            chunk.total_scene_patches_loss = chunk.total_scene_patches_loss + L.scene_patches_loss.detach().float()
            chunk.total_scene_cls_loss = chunk.total_scene_cls_loss + L.scene_cls_loss.detach().float()
            chunk.scene_pred, chunk.cls_pred = L.scene_pred, L.cls_pred
            chunk.n_steps += 1

            is_chunk_end = ((t + 1) % chunk_size == 0)
            is_last = (t == n_glimpses - 1)

            if is_chunk_end:
                loss_for_backward = chunk.chunk_combined_loss / n_glimpses / n_branches
                loss_for_backward.backward()  # no retain_graph

                if not is_last:
                    chunk.state = RecurrentState(
                        canvas=out.state.canvas.detach(),
                        recurrent_cls=out.state.recurrent_cls.detach(),
                    )
                    chunk.vpe = out.vpe.detach() if out.vpe is not None else None
                    chunk.chunk_combined_loss = torch.zeros((), device=device)
                else:
                    chunk.state = out.state
                    chunk.vpe = out.vpe
            else:
                chunk.state = out.state
                chunk.vpe = out.vpe

        n = chunk.n_steps
        scene_pred_raw = scene_denorm(chunk.scene_pred)
        cls_pred_raw = cls_denorm(chunk.cls_pred.unsqueeze(1)).squeeze(1)
        return BranchMetrics(
            loss=chunk.total_combined_loss / n,
            scene_patches_loss=chunk.total_scene_patches_loss / n,
            scene_cls_loss=chunk.total_scene_cls_loss / n,
            scene_cos_raw=F.cosine_similarity(scene_pred_raw, raw_scene_target, dim=-1).mean(),
            scene_cos_norm=F.cosine_similarity(chunk.scene_pred, scene_target, dim=-1).mean(),
            cls_cos_raw=F.cosine_similarity(cls_pred_raw, raw_cls_target, dim=-1).mean(),
            cls_cos_norm=F.cosine_similarity(chunk.cls_pred, cls_target, dim=-1).mean(),
        )

    # Run all branches (full-start first, then random-start)
    branch_idx = 0
    for _ in range(n_full_start_branches):
        full_metrics.append(run_branch(ViewpointType.FULL, branch_idx))
        branch_idx += 1

    for _ in range(n_random_start_branches):
        random_metrics.append(run_branch(ViewpointType.RANDOM, branch_idx))
        branch_idx += 1

    def aggregate(metrics: list[BranchMetrics]) -> BranchMetrics | None:
        if not metrics:
            return None
        return BranchMetrics(
            loss=torch.stack([m.loss for m in metrics]).mean(),
            scene_patches_loss=torch.stack([m.scene_patches_loss for m in metrics]).mean(),
            scene_cls_loss=torch.stack([m.scene_cls_loss for m in metrics]).mean(),
            scene_cos_raw=torch.stack([m.scene_cos_raw for m in metrics]).mean(),
            scene_cos_norm=torch.stack([m.scene_cos_norm for m in metrics]).mean(),
            cls_cos_raw=torch.stack([m.cls_cos_raw for m in metrics]).mean(),
            cls_cos_norm=torch.stack([m.cls_cos_norm for m in metrics]).mean(),
        )

    full_start = aggregate(full_metrics)
    random_start = aggregate(random_metrics)

    all_losses = [m.loss for m in full_metrics] + [m.loss for m in random_metrics]
    total_loss = torch.stack(all_losses).mean()

    return StepMetrics(
        total_loss=total_loss,
        full_start=full_start,
        random_start=random_start,
        n_glimpses=n_glimpses,
        viz_data=viz_data,
    )
