"""Training step with 2×3 branching for policy learning."""

from contextlib import AbstractContextManager
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor

from avp_vit import ActiveCanViT, GlimpseOutput, gram_mse
from canvit import Viewpoint
from canvit.policy import PolicyHead

from .viewpoint import Viewpoint as TrainViewpoint, ViewpointType


class StepOutput(NamedTuple):
    """Output from compute_step_loss - keeps predictions for metrics."""

    loss: Tensor  # combined loss for backward
    scene_loss: Tensor  # raw MSE
    cls_loss: Tensor  # raw MSE
    gram_loss: Tensor | None  # raw gram MSE (unweighted), None if not computed
    scene_pred: Tensor  # for metrics, no recomputation
    cls_pred: Tensor  # for metrics, no recomputation


class BranchMetrics(NamedTuple):
    """Metrics for one (t0_type, t1_type) branch."""

    loss: Tensor  # total trajectory loss
    scene_loss: Tensor
    cls_loss: Tensor
    gram_loss: Tensor | None
    scene_cos: Tensor
    cls_cos: Tensor


class StepMetrics(NamedTuple):
    """Output from training_step."""

    total_loss: Tensor
    branches: dict[tuple[ViewpointType, ViewpointType], BranchMetrics]


def compute_step_loss(
    model: ActiveCanViT,
    out: GlimpseOutput,
    scene_target: Tensor,
    cls_target: Tensor,
    compute_gram: bool,
    gram_loss_weight: float,
) -> StepOutput:
    """Compute loss for one step. Returns predictions for reuse in metrics."""
    scene_pred = model.predict_teacher_scene(out.canvas)
    cls_pred = model.predict_teacher_cls(out.cls, out.canvas)

    scene_loss = F.mse_loss(scene_pred, scene_target)
    cls_loss = F.mse_loss(cls_pred, cls_target)
    gram_loss = gram_mse(scene_pred, scene_target) if compute_gram else None

    loss = scene_loss + cls_loss
    if gram_loss is not None:
        loss = loss + gram_loss_weight * gram_loss

    return StepOutput(
        loss=loss,
        scene_loss=scene_loss,
        cls_loss=cls_loss,
        gram_loss=gram_loss,
        scene_pred=scene_pred,
        cls_pred=cls_pred,
    )


def training_step(
    *,
    model: ActiveCanViT,
    images: Tensor,
    scene_target: Tensor,
    cls_target: Tensor,
    glimpse_size_px: int,
    canvas_grid_size: int,
    t0_types: list[ViewpointType],
    t1_types: list[ViewpointType],
    min_viewpoint_scale: float,
    compute_gram: bool,
    gram_loss_weight: float,
    amp_ctx: AbstractContextManager,
) -> StepMetrics:
    """Run training step with 2×N branching.

    For each t0_type, runs forward at t=0, then branches to each t1_type at t=1.
    Uses retain_graph=True to share t=0 computation across t=1 branches.
    Accumulates gradients with proper scaling. Does NOT call optimizer.step().

    Returns metrics for logging.
    """
    device = images.device
    batch_size = images.shape[0]
    n_branches = len(t0_types) * len(t1_types)

    branches: dict[tuple[ViewpointType, ViewpointType], BranchMetrics] = {}
    total_loss_acc = torch.zeros((), device=device)

    canvas_init = model.init_canvas(batch_size=batch_size, canvas_grid_size=canvas_grid_size)
    cls_init = model.init_cls(batch_size=batch_size)

    for t0_type in t0_types:
        # Create t=0 viewpoint
        if t0_type == ViewpointType.FULL:
            vp0 = TrainViewpoint.full_scene(batch_size=batch_size, device=device)
        elif t0_type == ViewpointType.RANDOM:
            vp0 = TrainViewpoint.random(batch_size=batch_size, device=device, min_scale=min_viewpoint_scale)
        else:
            raise ValueError(f"Invalid t0_type: {t0_type}")

        # Convert to canvit Viewpoint for model
        vp0_core = Viewpoint(centers=vp0.centers, scales=vp0.scales)

        with amp_ctx:
            out_0 = model.forward_step(
                image=images,
                canvas=canvas_init,
                viewpoint=vp0_core,
                glimpse_size_px=glimpse_size_px,
                cls=cls_init,
            )
            step_0 = compute_step_loss(
                model, out_0, scene_target, cls_target, compute_gram, gram_loss_weight
            )

        for j, t1_type in enumerate(t1_types):
            # Create t=1 viewpoint
            if t1_type == ViewpointType.POLICY:
                assert out_0.vpe is not None, "VPE required for policy"
                assert isinstance(model.policy, PolicyHead), "Policy head required"
                policy_out = model.policy(out_0.vpe)
                vp1_core = Viewpoint(centers=policy_out.position, scales=policy_out.scale)
            elif t1_type == ViewpointType.FULL:
                vp1 = TrainViewpoint.full_scene(batch_size=batch_size, device=device)
                vp1_core = Viewpoint(centers=vp1.centers, scales=vp1.scales)
            elif t1_type == ViewpointType.RANDOM:
                vp1 = TrainViewpoint.random(batch_size=batch_size, device=device, min_scale=min_viewpoint_scale)
                vp1_core = Viewpoint(centers=vp1.centers, scales=vp1.scales)
            else:
                raise ValueError(f"Invalid t1_type: {t1_type}")

            with amp_ctx:
                out_1 = model.forward_step(
                    image=images,
                    canvas=out_0.canvas,
                    viewpoint=vp1_core,
                    glimpse_size_px=glimpse_size_px,
                    cls=out_0.cls,
                )
                step_1 = compute_step_loss(
                    model, out_1, scene_target, cls_target, compute_gram, gram_loss_weight
                )

                traj_loss = (step_0.loss + step_1.loss) / 2
                # No NaN check here - would cause GPU sync. NaNs will show up in logged metrics.

                is_last_for_this_t0 = j == len(t1_types) - 1
                (traj_loss / n_branches).backward(retain_graph=not is_last_for_this_t0)

            # Metrics from ALREADY COMPUTED predictions (no recomputation!)
            with torch.no_grad():
                scene_cos = F.cosine_similarity(step_1.scene_pred, scene_target, dim=-1).mean()
                cls_cos = F.cosine_similarity(step_1.cls_pred, cls_target, dim=-1).mean()

            # Average losses over t=0 and t=1
            avg_scene_loss = (step_0.scene_loss + step_1.scene_loss) / 2
            avg_cls_loss = (step_0.cls_loss + step_1.cls_loss) / 2
            avg_gram_loss: Tensor | None = None
            if step_0.gram_loss is not None and step_1.gram_loss is not None:
                avg_gram_loss = (step_0.gram_loss + step_1.gram_loss) / 2

            branches[(t0_type, t1_type)] = BranchMetrics(
                loss=traj_loss.detach(),
                scene_loss=avg_scene_loss.detach(),
                cls_loss=avg_cls_loss.detach(),
                gram_loss=avg_gram_loss.detach() if avg_gram_loss is not None else None,
                scene_cos=scene_cos,
                cls_cos=cls_cos,
            )
            total_loss_acc = total_loss_acc + traj_loss.detach()

    return StepMetrics(total_loss=total_loss_acc / n_branches, branches=branches)
