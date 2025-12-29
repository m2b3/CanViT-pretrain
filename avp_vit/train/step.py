"""Training step with truncated BPTT and balanced branches."""

import random
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from avp_vit import ActiveCanViT, GlimpseOutput
from canvit import Viewpoint

from .viewpoint import Viewpoint as TrainViewpoint, ViewpointType


class BranchMetrics(NamedTuple):
    """Metrics for a (t0_type, t1_type) combination."""

    loss: Tensor
    scene_loss: Tensor
    cls_loss: Tensor
    gram_loss: Tensor | None  # unused, kept for compat
    scene_cos: Tensor
    cls_cos: Tensor


class StepMetrics(NamedTuple):
    """Output from training_step."""

    total_loss: Tensor
    branches: dict[tuple[ViewpointType, ViewpointType], BranchMetrics]
    n_glimpses: int  # trajectory length this step


@dataclass
class ChunkState:
    """State for TBPTT chunk processing."""

    canvas: Tensor
    cls_tok: Tensor
    vpe: Tensor | None
    chunk_scene: Tensor  # loss accumulator (with grad)
    chunk_cls: Tensor
    total_scene: Tensor  # loss accumulator (detached, for metrics)
    total_cls: Tensor
    n_steps: int
    scene_pred: Tensor
    cls_pred: Tensor


def training_step(
    *,
    model: ActiveCanViT,
    images: Tensor,
    scene_target: Tensor,
    cls_target: Tensor,
    glimpse_size_px: int,
    canvas_grid_size: int,
    n_branches: int,
    min_glimpses: int,
    continue_prob: float,
    min_viewpoint_scale: float,
    amp_ctx: AbstractContextManager,
    use_checkpointing: bool,
) -> StepMetrics:
    """Training with truncated BPTT (1-step lookahead).

    Chunks of 2 steps: backward after every 2 glimpses, gradient flows through both.
    Trajectory length is stochastic: min_glimpses + geometric(continue_prob).
    Memory is constant due to chunking - no max needed.
    """
    assert n_branches >= 2 and n_branches % 2 == 0
    assert min_glimpses >= 2
    assert 0.0 <= continue_prob <= 1.0
    device = images.device
    B = images.shape[0]

    canvas_init = model.init_canvas(batch_size=B, canvas_grid_size=canvas_grid_size)
    cls_init = model.init_cls(batch_size=B)

    # Sample trajectory length (shared across branches)
    n_glimpses = min_glimpses
    while random.random() < continue_prob:
        n_glimpses += 1

    vp_types = _assign_viewpoint_types(
        n_glimpses=n_glimpses,
        n_branches=n_branches,
        has_policy=model.policy is not None,
    )

    full_indices = [i for i in range(n_branches) if vp_types[0][i] == ViewpointType.FULL]
    random_indices = [i for i in range(n_branches) if vp_types[0][i] == ViewpointType.RANDOM]

    # Metrics accumulators
    traj_losses = torch.zeros(n_branches, device=device)
    scene_losses = torch.zeros(n_branches, device=device)
    cls_losses = torch.zeros(n_branches, device=device)
    scene_cos = torch.zeros(n_branches, device=device)
    cls_cos = torch.zeros(n_branches, device=device)

    def make_vp(vp_type: ViewpointType, vpe: Tensor | None) -> Viewpoint:
        if vp_type == ViewpointType.RANDOM:
            v = TrainViewpoint.random(batch_size=B, device=device, min_scale=min_viewpoint_scale)
            return Viewpoint(centers=v.centers, scales=v.scales)
        if vp_type == ViewpointType.FULL:
            v = TrainViewpoint.full_scene(batch_size=B, device=device)
            return Viewpoint(centers=v.centers, scales=v.scales)
        assert model.policy is not None and vpe is not None
        p = model.policy(vpe)
        return Viewpoint(centers=p.position, scales=p.scale)

    def forward_glimpse(*, canvas: Tensor, cls: Tensor, vp: Viewpoint, use_ckpt: bool) -> GlimpseOutput:
        if use_ckpt:
            out = checkpoint(
                lambda c, cl, ctr, sc: model.forward_step(
                    image=images, canvas=c, cls=cl,
                    viewpoint=Viewpoint(centers=ctr, scales=sc),
                    glimpse_size_px=glimpse_size_px,
                ),
                canvas, cls, vp.centers, vp.scales,
                use_reentrant=False,
            )
            assert isinstance(out, GlimpseOutput)
            return out
        return model.forward_step(
            image=images, canvas=canvas, cls=cls,
            viewpoint=vp, glimpse_size_px=glimpse_size_px,
        )

    def compute_loss(out: GlimpseOutput) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        scene_pred = model.predict_teacher_scene(out.canvas)
        cls_pred = model.predict_teacher_cls(out.cls, out.canvas)
        return (
            F.mse_loss(scene_pred, scene_target),
            F.mse_loss(cls_pred, cls_target),
            scene_pred,
            cls_pred,
        )

    def run_tbptt(
        *,
        branch_idx: int,
        out_t0: GlimpseOutput,
        loss_t0: tuple[Tensor, Tensor, Tensor, Tensor],
        retain_first_chunk: bool,
    ) -> None:
        """TBPTT with 1-step lookahead: backward every 2 steps, gradient flows through both."""
        scene_t0, cls_t0, scene_pred_t0, cls_pred_t0 = loss_t0

        state = ChunkState(
            canvas=out_t0.canvas,
            cls_tok=out_t0.cls,
            vpe=out_t0.vpe,
            chunk_scene=scene_t0.float(),
            chunk_cls=cls_t0.float(),
            total_scene=scene_t0.detach().float(),
            total_cls=cls_t0.detach().float(),
            n_steps=1,
            scene_pred=scene_pred_t0,
            cls_pred=cls_pred_t0,
        )

        for t in range(1, n_glimpses):
            vp = make_vp(vp_types[t][branch_idx], state.vpe)

            with amp_ctx:
                use_ckpt = use_checkpointing and (t % 2 == 1)
                out = forward_glimpse(canvas=state.canvas, cls=state.cls_tok, vp=vp, use_ckpt=use_ckpt)
                sl, cl, state.scene_pred, state.cls_pred = compute_loss(out)

            state.chunk_scene = state.chunk_scene + sl.float()
            state.chunk_cls = state.chunk_cls + cl.float()
            state.total_scene = state.total_scene + sl.detach().float()
            state.total_cls = state.total_cls + cl.detach().float()
            state.n_steps += 1

            is_chunk_end = (t % 2 == 1)
            is_last = (t == n_glimpses - 1)

            if is_chunk_end:
                # Backward chunk (gradient flows through both steps in chunk)
                chunk_loss = (state.chunk_scene + state.chunk_cls) / n_glimpses / n_branches
                retain = retain_first_chunk if t == 1 else False
                chunk_loss.backward(retain_graph=retain)

                # Detach for next chunk
                if not is_last:
                    state.canvas = out.canvas.detach()
                    state.cls_tok = out.cls.detach()
                    state.vpe = out.vpe.detach() if out.vpe is not None else None
                    state.chunk_scene = torch.zeros((), device=device)
                    state.chunk_cls = torch.zeros((), device=device)
                else:
                    state.canvas, state.cls_tok, state.vpe = out.canvas, out.cls, out.vpe
            else:
                state.canvas, state.cls_tok, state.vpe = out.canvas, out.cls, out.vpe

        # Trailing step (if n_glimpses is odd)
        if (n_glimpses - 1) % 2 == 0:
            chunk_loss = (state.chunk_scene + state.chunk_cls) / n_glimpses / n_branches
            chunk_loss.backward()

        # Record metrics
        traj_losses[branch_idx] = (state.total_scene + state.total_cls) / state.n_steps
        scene_losses[branch_idx] = state.total_scene / state.n_steps
        cls_losses[branch_idx] = state.total_cls / state.n_steps
        scene_cos[branch_idx] = F.cosine_similarity(state.scene_pred, scene_target, dim=-1).mean()
        cls_cos[branch_idx] = F.cosine_similarity(state.cls_pred, cls_target, dim=-1).mean()

    # === FULL branches: share t=0 ===
    if full_indices:
        with amp_ctx:
            vp_full = make_vp(ViewpointType.FULL, None)
            out_full = forward_glimpse(canvas=canvas_init, cls=cls_init, vp=vp_full, use_ckpt=False)
            loss_full = compute_loss(out_full)

        for idx, i in enumerate(full_indices):
            run_tbptt(
                branch_idx=i,
                out_t0=out_full,
                loss_t0=loss_full,
                retain_first_chunk=(idx < len(full_indices) - 1),
            )

    # === RANDOM branches: unique t=0 each ===
    for i in random_indices:
        with amp_ctx:
            vp_rand = make_vp(ViewpointType.RANDOM, None)
            out = forward_glimpse(canvas=canvas_init, cls=cls_init, vp=vp_rand, use_ckpt=False)
            loss = compute_loss(out)

        run_tbptt(branch_idx=i, out_t0=out, loss_t0=loss, retain_first_chunk=False)

    # Aggregate metrics
    branches = _aggregate_branch_metrics(
        vp_types=vp_types,
        n_branches=n_branches,
        has_policy=model.policy is not None,
        traj_losses=traj_losses,
        scene_losses=scene_losses,
        cls_losses=cls_losses,
        scene_cos=scene_cos,
        cls_cos=cls_cos,
    )

    return StepMetrics(total_loss=traj_losses.mean(), branches=branches, n_glimpses=n_glimpses)


def _assign_viewpoint_types(
    *, n_glimpses: int, n_branches: int, has_policy: bool
) -> list[list[ViewpointType]]:
    """Assign viewpoint types: random permutation of half/half at each timestep."""
    vp_types: list[list[ViewpointType]] = []
    for t in range(n_glimpses):
        if t == 0:
            base = [ViewpointType.RANDOM] * (n_branches // 2) + [ViewpointType.FULL] * (n_branches // 2)
        elif has_policy:
            base = [ViewpointType.RANDOM] * (n_branches // 2) + [ViewpointType.POLICY] * (n_branches // 2)
        else:
            base = [ViewpointType.RANDOM] * n_branches
        random.shuffle(base)
        vp_types.append(base)
    return vp_types


def _aggregate_branch_metrics(
    *,
    vp_types: list[list[ViewpointType]],
    n_branches: int,
    has_policy: bool,
    traj_losses: Tensor,
    scene_losses: Tensor,
    cls_losses: Tensor,
    scene_cos: Tensor,
    cls_cos: Tensor,
) -> dict[tuple[ViewpointType, ViewpointType], BranchMetrics]:
    """Aggregate metrics by (t0_type, t1_type)."""
    branches: dict[tuple[ViewpointType, ViewpointType], BranchMetrics] = {}
    t1_options = [ViewpointType.RANDOM, ViewpointType.POLICY] if has_policy else [ViewpointType.RANDOM]

    for t0 in [ViewpointType.RANDOM, ViewpointType.FULL]:
        for t1 in t1_options:
            indices = [i for i in range(n_branches) if vp_types[0][i] == t0 and vp_types[1][i] == t1]
            if indices:
                branches[(t0, t1)] = BranchMetrics(
                    loss=traj_losses[indices].mean(),
                    scene_loss=scene_losses[indices].mean(),
                    cls_loss=cls_losses[indices].mean(),
                    gram_loss=None,
                    scene_cos=scene_cos[indices].mean(),
                    cls_cos=cls_cos[indices].mean(),
                )

    return branches
