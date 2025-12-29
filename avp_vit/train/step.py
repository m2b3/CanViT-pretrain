"""Training step with memory-efficient balanced branches."""

import random
from contextlib import AbstractContextManager
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


def training_step(
    *,
    model: ActiveCanViT,
    images: Tensor,
    scene_target: Tensor,
    cls_target: Tensor,
    glimpse_size_px: int,
    canvas_grid_size: int,
    n_branches: int,
    n_glimpses: int,
    min_viewpoint_scale: float,
    amp_ctx: AbstractContextManager,
) -> StepMetrics:
    """Memory-efficient training with balanced branches.

    n_branches (>= 2, even): parallel trajectories
    n_glimpses (>= 2): glimpses per trajectory (t=0 + at least one t>=1)

    At each timestep t:
      - Half branches use RANDOM, half use FULL (t=0) or POLICY (t>=1)
      - Which branches get which is randomly permuted each timestep

    FULL at t=0 is computed once and shared (deterministic).
    RANDOM viewpoints are independent per branch (stochastic).

    Memory: ~6.6 GB constant (FULL t=0 shared, RANDOM branches processed one at a time)
    """
    assert n_branches >= 2 and n_branches % 2 == 0
    assert n_glimpses >= 2
    device = images.device
    B = images.shape[0]

    canvas_init = model.init_canvas(batch_size=B, canvas_grid_size=canvas_grid_size)
    cls_init = model.init_cls(batch_size=B)

    # Assign viewpoint types: at each timestep, random permutation of half/half split
    vp_types: list[list[ViewpointType]] = []
    for t in range(n_glimpses):
        if t == 0:
            base = [ViewpointType.RANDOM] * (n_branches // 2) + [ViewpointType.FULL] * (n_branches // 2)
        elif model.policy is not None:
            base = [ViewpointType.RANDOM] * (n_branches // 2) + [ViewpointType.POLICY] * (n_branches // 2)
        else:
            base = [ViewpointType.RANDOM] * n_branches
        random.shuffle(base)
        vp_types.append(base)

    full_indices = [i for i in range(n_branches) if vp_types[0][i] == ViewpointType.FULL]
    random_indices = [i for i in range(n_branches) if vp_types[0][i] == ViewpointType.RANDOM]

    # Preallocate metrics
    traj_losses = torch.zeros(n_branches, device=device)
    scene_losses = torch.zeros(n_branches, device=device)
    cls_losses = torch.zeros(n_branches, device=device)
    scene_cos = torch.zeros(n_branches, device=device)
    cls_cos = torch.zeros(n_branches, device=device)

    def make_random_vp() -> Viewpoint:
        v = TrainViewpoint.random(batch_size=B, device=device, min_scale=min_viewpoint_scale)
        return Viewpoint(centers=v.centers, scales=v.scales)

    def make_policy_vp(vpe: Tensor) -> Viewpoint:
        assert model.policy is not None
        p = model.policy(vpe)
        return Viewpoint(centers=p.position, scales=p.scale)

    def fwd_step(canvas: Tensor, cls: Tensor, centers: Tensor, scales: Tensor) -> GlimpseOutput:
        return model.forward_step(
            image=images,
            canvas=canvas,
            viewpoint=Viewpoint(centers=centers, scales=scales),
            glimpse_size_px=glimpse_size_px,
            cls=cls,
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

    def run_trajectory(
        i: int,
        out_t0: GlimpseOutput,
        scene_loss_t0: Tensor,
        cls_loss_t0: Tensor,
        scene_pred_t0: Tensor,
        cls_pred_t0: Tensor,
    ) -> Tensor:
        """Run t>=1 for branch i, return trajectory loss."""
        # Accumulate in fp32 to avoid precision issues with bfloat16
        scene_loss_sum = scene_loss_t0.float()
        cls_loss_sum = cls_loss_t0.float()
        scene_pred, cls_pred = scene_pred_t0, cls_pred_t0
        canvas, cls_tok, vpe = out_t0.canvas, out_t0.cls, out_t0.vpe

        for t in range(1, n_glimpses):
            vp_type = vp_types[t][i]
            if vp_type == ViewpointType.RANDOM:
                vp = make_random_vp()
            else:
                assert vpe is not None
                vp = make_policy_vp(vpe)

            with amp_ctx:
                out_raw = checkpoint(fwd_step, canvas, cls_tok, vp.centers, vp.scales, use_reentrant=False)
                assert isinstance(out_raw, GlimpseOutput)
                out = out_raw
                scene_loss_t, cls_loss_t, scene_pred, cls_pred = compute_loss(out)
                scene_loss_sum = scene_loss_sum + scene_loss_t.float()
                cls_loss_sum = cls_loss_sum + cls_loss_t.float()
                canvas, cls_tok, vpe = out.canvas, out.cls, out.vpe

        traj_loss = (scene_loss_sum + cls_loss_sum) / n_glimpses
        with torch.no_grad():
            traj_losses[i] = traj_loss
            scene_losses[i] = scene_loss_sum / n_glimpses
            cls_losses[i] = cls_loss_sum / n_glimpses
            scene_cos[i] = F.cosine_similarity(scene_pred, scene_target, dim=-1).mean()
            cls_cos[i] = F.cosine_similarity(cls_pred, cls_target, dim=-1).mean()

        return traj_loss

    # === FULL branches: share t=0, process one-at-a-time with retain_graph ===
    if full_indices:
        with amp_ctx:
            full_vp = TrainViewpoint.full_scene(batch_size=B, device=device)
            out_full = model.forward_step(
                image=images,
                canvas=canvas_init,
                viewpoint=Viewpoint(centers=full_vp.centers, scales=full_vp.scales),
                glimpse_size_px=glimpse_size_px,
                cls=cls_init,
            )
            scene_loss_full, cls_loss_full, scene_pred_full, cls_pred_full = compute_loss(out_full)

        for idx, i in enumerate(full_indices):
            traj_loss = run_trajectory(i, out_full, scene_loss_full, cls_loss_full, scene_pred_full, cls_pred_full)
            is_last = idx == len(full_indices) - 1
            (traj_loss / n_branches).backward(retain_graph=not is_last)

    # === RANDOM branches: each has unique t=0, process one at a time ===
    for i in random_indices:
        with amp_ctx:
            rand_vp = make_random_vp()
            out = model.forward_step(
                image=images,
                canvas=canvas_init,
                viewpoint=rand_vp,
                glimpse_size_px=glimpse_size_px,
                cls=cls_init,
            )
            scene_loss, cls_loss, scene_pred, cls_pred = compute_loss(out)

        traj_loss = run_trajectory(i, out, scene_loss, cls_loss, scene_pred, cls_pred)
        (traj_loss / n_branches).backward()

    # Aggregate metrics by (t0, t1)
    branches: dict[tuple[ViewpointType, ViewpointType], BranchMetrics] = {}
    t1_options = [ViewpointType.RANDOM, ViewpointType.POLICY] if model.policy else [ViewpointType.RANDOM]
    for t0 in [ViewpointType.RANDOM, ViewpointType.FULL]:
        for t1 in t1_options:
            mask = torch.tensor([vp_types[0][i] == t0 and vp_types[1][i] == t1 for i in range(n_branches)])
            if mask.any():
                branches[(t0, t1)] = BranchMetrics(
                    loss=traj_losses[mask].mean(),
                    scene_loss=scene_losses[mask].mean(),
                    cls_loss=cls_losses[mask].mean(),
                    gram_loss=None,
                    scene_cos=scene_cos[mask].mean(),
                    cls_cos=cls_cos[mask].mean(),
                )

    return StepMetrics(total_loss=traj_losses.mean(), branches=branches)
