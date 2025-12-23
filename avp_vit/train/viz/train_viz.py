"""Training visualization: forward pass + PCA logging."""

from dataclasses import dataclass, field

import comet_ml
import torch
import torch.nn.functional as F
from canvit.backbone.dinov3 import DINOv3Backbone
from canvit.model.active.base import GlimpseOutput
from canvit.viewpoint import Viewpoint as CanvitViewpoint
from torch import Tensor

from avp_vit import ActiveCanViT
from ..norm import PositionAwareNorm
from ..viewpoint import Viewpoint
from .comet import log_curve, log_figure
from .image import imagenet_denormalize
from .metrics import compute_spatial_stats
from .plot import plot_multistep_pca
from .sample import VizSampleData, extract_sample0_viz


@dataclass
class _Accumulator:
    """Accumulator for training viz (uses forward_reduce, sample 0 only)."""

    scene_cos_sims: list[float] = field(default_factory=list)
    cls_cos_sims: list[float] = field(default_factory=list)
    viz_samples: list[VizSampleData] = field(default_factory=list)
    final_predicted_scene: Tensor | None = None


def viz_and_log(
    *,
    exp: comet_ml.Experiment,
    step: int,
    prefix: str,
    model: ActiveCanViT,
    teacher: DINOv3Backbone,
    normalizer: PositionAwareNorm,
    images: Tensor,
    viewpoints: list[Viewpoint],
    target: Tensor,
    canvas: Tensor,
    glimpse_size_px: int,
    cls_target: Tensor | None = None,
    log_spatial_stats: bool = True,
    log_curves: bool = True,
) -> None:
    """Run forward pass and log PCA visualization (training viz)."""
    assert isinstance(model.backbone, DINOv3Backbone)
    n_spatial = canvas.shape[1] - model.n_canvas_registers
    canvas_grid_size = int(n_spatial**0.5)
    assert canvas_grid_size**2 == n_spatial
    glimpse_grid_size = glimpse_size_px // model.backbone.patch_size_px
    has_cls = cls_target is not None

    with torch.inference_mode():
        initial_scene_np = model.predict_teacher_scene(canvas)[0].cpu().float().numpy()
        initial_canvas_spatial_np = model.get_spatial(canvas[0:1])[0].cpu().float().numpy()

        def init_fn(_canvas: Tensor, _cls: Tensor) -> _Accumulator:
            return _Accumulator()

        def step_fn(acc: _Accumulator, out: GlimpseOutput, _vp: CanvitViewpoint) -> _Accumulator:
            predicted_scene = model.predict_teacher_scene(out.canvas)

            acc.scene_cos_sims.append(
                F.cosine_similarity(predicted_scene, target, dim=-1).mean().item()
            )
            if has_cls:
                assert cls_target is not None
                predicted_cls = model.predict_teacher_cls(out.cls, out.canvas)
                acc.cls_cos_sims.append(
                    F.cosine_similarity(predicted_cls, cls_target, dim=-1).mean().item()
                )

            acc.viz_samples.append(extract_sample0_viz(out, predicted_scene, model))
            acc.final_predicted_scene = predicted_scene
            return acc

        acc, _, _ = model.forward_reduce(
            image=images,
            viewpoints=viewpoints,  # pyright: ignore[reportArgumentType]
            glimpse_size_px=glimpse_size_px,
            canvas_grid_size=canvas_grid_size,
            init_fn=init_fn,
            step_fn=step_fn,
            canvas=canvas,
        )

        if log_curves:
            log_curve(
                exp,
                f"{prefix}/scene_cos_sim_vs_timestep",
                x=list(range(len(acc.scene_cos_sims))),
                y=acc.scene_cos_sims,
                step=step,
            )
            if acc.cls_cos_sims:
                log_curve(
                    exp,
                    f"{prefix}/cls_cos_sim_vs_timestep",
                    x=list(range(len(acc.cls_cos_sims))),
                    y=acc.cls_cos_sims,
                    step=step,
                )

        if log_spatial_stats and acc.final_predicted_scene is not None:
            target_stats = compute_spatial_stats(target)
            pred_stats = compute_spatial_stats(acc.final_predicted_scene)
            exp.log_metrics(
                {
                    f"{prefix}/target_spatial_mean": target_stats["mean"],
                    f"{prefix}/target_spatial_std": target_stats["std"],
                    f"{prefix}/pred_spatial_mean": pred_stats["mean"],
                    f"{prefix}/pred_spatial_std": pred_stats["std"],
                },
                step=step,
            )

        H, W = images.shape[-2], images.shape[-1]
        boxes = [vp.to_pixel_box(0, H, W) for vp in viewpoints]
        names = [vp.name for vp in viewpoints]

        fig_pca = plot_multistep_pca(
            full_img=imagenet_denormalize(images[0].cpu()).numpy(),
            teacher=target[0].cpu().float().numpy(),
            scenes=[vs.predicted_scene for vs in acc.viz_samples],
            glimpses=[vs.glimpse for vs in acc.viz_samples],
            boxes=boxes,
            names=names,
            scene_grid_size=canvas_grid_size,
            glimpse_grid_size=glimpse_grid_size,
            initial_scene=initial_scene_np,
            hidden_spatials=[vs.canvas_spatial for vs in acc.viz_samples],
            initial_hidden_spatial=initial_canvas_spatial_np,
        )
        log_figure(exp, fig_pca, f"{prefix}/pca", step)
