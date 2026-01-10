"""Training step with truncated BPTT and independent branches."""

import random
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Callable, NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from avp_vit import ActiveCanViT, GlimpseOutput, RecurrentState
from canvit import Viewpoint

from .viewpoint import Viewpoint as NamedViewpoint, ViewpointType


class NormalizedTargets(NamedTuple):
    """Normalized teacher features (patches + CLS)."""
    patches: Tensor
    cls: Tensor


class LossOutput(NamedTuple):
    """Output from compute_loss - individual losses + combined mean."""

    scene_patches_loss: Tensor
    scene_cls_loss: Tensor
    glimpse_patches_loss: Tensor
    glimpse_cls_loss: Tensor
    combined: Tensor  # sum of active losses
    scene_pred: Tensor  # for cosine similarity metrics
    cls_pred: Tensor


class BranchMetrics(NamedTuple):
    """Metrics for a branch type."""

    loss: Tensor
    scene_patches_loss: Tensor
    scene_cls_loss: Tensor
    glimpse_patches_loss: Tensor
    glimpse_cls_loss: Tensor
    scene_cos: Tensor
    cls_cos: Tensor


class StepMetrics(NamedTuple):
    """Output from training_step."""

    total_loss: Tensor
    full_start: BranchMetrics | None  # None if n_full_start_branches=0
    random_start: BranchMetrics | None  # None if n_random_start_branches=0
    n_glimpses: int  # trajectory length this step


@dataclass
class ChunkState:
    """State for TBPTT chunk processing."""

    state: RecurrentState
    vpe: Tensor | None
    chunk_combined_loss: Tensor  # with grad
    total_combined_loss: Tensor  # detached
    total_scene_patches_loss: Tensor
    total_scene_cls_loss: Tensor
    total_glimpse_patches_loss: Tensor
    total_glimpse_cls_loss: Tensor
    n_steps: int
    scene_pred: Tensor
    cls_pred: Tensor


def training_step(
    *,
    model: ActiveCanViT,
    images: Tensor,
    scene_target: Tensor,
    cls_target: Tensor,
    compute_glimpse_targets: Callable[[Tensor], NormalizedTargets] | None,
    enable_scene_patches_loss: bool,
    enable_scene_cls_loss: bool,
    enable_glimpse_patches_loss: bool,
    enable_glimpse_cls_loss: bool,
    glimpse_size_px: int,
    canvas_grid_size: int,
    n_full_start_branches: int,
    n_random_start_branches: int,
    chunk_size: int,
    continue_prob: float,
    min_viewpoint_scale: float,
    amp_ctx: AbstractContextManager,
    use_checkpointing: bool,
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

    # Metrics accumulators
    full_metrics: list[tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]] = []
    random_metrics: list[tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]] = []

    def make_vp(vp_type: ViewpointType, vpe: Tensor | None) -> Viewpoint:
        if vp_type == ViewpointType.RANDOM:
            v = NamedViewpoint.random(batch_size=B, device=device, min_scale=min_viewpoint_scale)
            return Viewpoint(centers=v.centers, scales=v.scales)
        if vp_type == ViewpointType.FULL:
            v = NamedViewpoint.full_scene(batch_size=B, device=device)
            return Viewpoint(centers=v.centers, scales=v.scales)
        assert model.policy is not None and vpe is not None
        p = model.policy(vpe)
        return Viewpoint(centers=p.position, scales=p.scale)

    def forward_glimpse(*, state: RecurrentState, vp: Viewpoint, use_ckpt: bool) -> GlimpseOutput:
        if use_ckpt:
            out = checkpoint(
                lambda cv, cl, ctr, sc: model.forward_step(
                    image=images,
                    state=RecurrentState(canvas=cv, cls=cl),
                    viewpoint=Viewpoint(centers=ctr, scales=sc),
                    glimpse_size_px=glimpse_size_px,
                ),
                state.canvas, state.cls, vp.centers, vp.scales,
                use_reentrant=False,
            )
            assert isinstance(out, GlimpseOutput)
            return out
        return model.forward_step(
            image=images, state=state,
            viewpoint=vp, glimpse_size_px=glimpse_size_px,
        )

    def compute_loss(out: GlimpseOutput) -> LossOutput:
        scene_pred = model.predict_teacher_scene(out.state.canvas)
        cls_pred = model.predict_scene_teacher_cls(out.state.cls, out.state.canvas)

        scene_patches_loss = torch.zeros((), device=device)
        scene_cls_loss = torch.zeros((), device=device)
        glimpse_patches_loss = torch.zeros((), device=device)
        glimpse_cls_loss = torch.zeros((), device=device)

        if enable_scene_patches_loss:
            scene_patches_loss = F.mse_loss(scene_pred, scene_target)
        if enable_scene_cls_loss:
            scene_cls_loss = F.mse_loss(cls_pred, cls_target)

        if compute_glimpse_targets is not None:
            glimpse_targets = compute_glimpse_targets(out.glimpse)
            if enable_glimpse_patches_loss:
                glimpse_patches_pred = model.predict_glimpse_teacher_patches(out.local_patches)
                glimpse_patches_loss = F.mse_loss(glimpse_patches_pred, glimpse_targets.patches)
            if enable_glimpse_cls_loss:
                assert out.local_cls is not None, "local_cls required for glimpse_cls_loss"
                glimpse_cls_pred = model.predict_glimpse_teacher_cls(out.local_cls)
                glimpse_cls_loss = F.mse_loss(glimpse_cls_pred, glimpse_targets.cls)

        active: list[Tensor] = []
        if enable_scene_patches_loss:
            active.append(scene_patches_loss)
        if enable_scene_cls_loss:
            active.append(scene_cls_loss)
        if enable_glimpse_patches_loss:
            active.append(glimpse_patches_loss)
        if enable_glimpse_cls_loss:
            active.append(glimpse_cls_loss)
        assert len(active) > 0, "At least one loss must be enabled"
        combined = torch.stack(active).sum()

        return LossOutput(
            scene_patches_loss=scene_patches_loss,
            scene_cls_loss=scene_cls_loss,
            glimpse_patches_loss=glimpse_patches_loss,
            glimpse_cls_loss=glimpse_cls_loss,
            combined=combined,
            scene_pred=scene_pred,
            cls_pred=cls_pred,
        )

    def run_branch(t0_type: ViewpointType, branch_idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Run one independent branch. Returns metrics tuple."""
        # t0 forward
        with amp_ctx:
            vp0 = make_vp(t0_type, None)
            out = forward_glimpse(state=state_init, vp=vp0, use_ckpt=False)
            L = compute_loss(out)

        chunk = ChunkState(
            state=out.state,
            vpe=out.vpe,
            chunk_combined_loss=L.combined.float(),
            total_combined_loss=L.combined.detach().float(),
            total_scene_patches_loss=L.scene_patches_loss.detach().float(),
            total_scene_cls_loss=L.scene_cls_loss.detach().float(),
            total_glimpse_patches_loss=L.glimpse_patches_loss.detach().float(),
            total_glimpse_cls_loss=L.glimpse_cls_loss.detach().float(),
            n_steps=1,
            scene_pred=L.scene_pred,
            cls_pred=L.cls_pred,
        )

        for t in range(1, n_glimpses):
            # t>=1: use pre-computed schedule (half RANDOM, half POLICY, shuffled)
            vp_type = t1_schedule[t - 1][branch_idx]
            vp = make_vp(vp_type, chunk.vpe)

            with amp_ctx:
                use_ckpt = use_checkpointing and (t % 2 == 1)
                out = forward_glimpse(state=chunk.state, vp=vp, use_ckpt=use_ckpt)
                L = compute_loss(out)

            chunk.chunk_combined_loss = chunk.chunk_combined_loss + L.combined.float()
            chunk.total_combined_loss = chunk.total_combined_loss + L.combined.detach().float()
            chunk.total_scene_patches_loss = chunk.total_scene_patches_loss + L.scene_patches_loss.detach().float()
            chunk.total_scene_cls_loss = chunk.total_scene_cls_loss + L.scene_cls_loss.detach().float()
            chunk.total_glimpse_patches_loss = chunk.total_glimpse_patches_loss + L.glimpse_patches_loss.detach().float()
            chunk.total_glimpse_cls_loss = chunk.total_glimpse_cls_loss + L.glimpse_cls_loss.detach().float()
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
                        cls=out.state.cls.detach(),
                    )
                    chunk.vpe = out.vpe.detach() if out.vpe is not None else None
                    chunk.chunk_combined_loss = torch.zeros((), device=device)
                else:
                    chunk.state = out.state
                    chunk.vpe = out.vpe
            else:
                chunk.state = out.state
                chunk.vpe = out.vpe

        # Return metrics
        n = chunk.n_steps
        return (
            chunk.total_combined_loss / n,
            chunk.total_scene_patches_loss / n,
            chunk.total_scene_cls_loss / n,
            chunk.total_glimpse_patches_loss / n,
            chunk.total_glimpse_cls_loss / n,
            F.cosine_similarity(chunk.scene_pred, scene_target, dim=-1).mean(),
            F.cosine_similarity(chunk.cls_pred, cls_target, dim=-1).mean(),
        )

    # Run all branches (full-start first, then random-start)
    branch_idx = 0
    for _ in range(n_full_start_branches):
        full_metrics.append(run_branch(ViewpointType.FULL, branch_idx))
        branch_idx += 1

    for _ in range(n_random_start_branches):
        random_metrics.append(run_branch(ViewpointType.RANDOM, branch_idx))
        branch_idx += 1

    # Aggregate metrics
    def aggregate(metrics: list[tuple[Tensor, ...]]) -> BranchMetrics | None:
        if not metrics:
            return None
        stacked = [torch.stack([m[i] for m in metrics]).mean() for i in range(7)]
        return BranchMetrics(*stacked)

    full_start = aggregate(full_metrics)
    random_start = aggregate(random_metrics)

    # Total loss
    all_losses = [m[0] for m in full_metrics] + [m[0] for m in random_metrics]
    total_loss = torch.stack(all_losses).mean()

    return StepMetrics(
        total_loss=total_loss,
        full_start=full_start,
        random_start=random_start,
        n_glimpses=n_glimpses,
    )
