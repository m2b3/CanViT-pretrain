"""Visualization and logging utilities."""

import gc
import io
import logging
from collections.abc import Callable
from typing import NamedTuple

import comet_ml
import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from torch import Tensor

from avp_vit import ActiveCanViT, StepOutput
from avp_vit.glimpse import Viewpoint, sample_at_viewpoint
from canvit.backbone.dinov3 import DINOv3Backbone, NormFeatures
from avp_vit.train import LOSS_FNS, LossType, imagenet_denormalize, plot_multistep_pca, plot_norm_stats
from avp_vit.train.norm import PositionAwareNorm
from avp_vit.train.viewpoint import make_eval_viewpoints

log = logging.getLogger(__name__)


class VizResult(NamedTuple):
    """Result from viz_and_log: per-timestep losses and model outputs."""

    losses: dict[str, list[float]]  # {loss_name: [loss_t0, loss_t1, ...]}
    outputs: list[StepOutput]  # Model outputs at each timestep


def compute_spatial_stats(x: Tensor) -> dict[str, float]:
    """Compute mean/std across spatial dimension, averaged over batch.

    Args:
        x: [B, N, D] tensor (N = spatial tokens)

    Returns:
        Dict with 'mean' and 'std' scalars:
        - mean: average of per-sample spatial means
        - std: average of per-sample spatial stds
    """
    # Per-sample spatial stats: [B, D]
    spatial_mean = x.mean(dim=1)
    spatial_std = x.std(dim=1)
    # Average across batch and dimensions to get scalars
    return {
        "mean": spatial_mean.mean().item(),
        "std": spatial_std.mean().item(),
    }


def log_figure(exp: comet_ml.Experiment, fig: Figure, name: str, step: int) -> None:
    """Log matplotlib figure to Comet. Aggressively cleans up to prevent memory leaks."""
    with io.BytesIO() as buf:
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        exp.log_image(buf, name=name, step=step)
    # Clear all axes before closing to release colorbar/patch references
    for ax in fig.axes:
        ax.clear()
    fig.clf()
    plt.close(fig)
    # Force garbage collection after complex figures
    gc.collect()


def viz_and_log(
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
    show_canvas: bool = True,
    log_spatial_stats: bool = True,
    log_curves: bool = True,
    loss_type: LossType = "mse",
    show_locals: bool = False,
) -> VizResult:
    """Run forward trajectory and log visualization."""
    assert isinstance(model.backbone, DINOv3Backbone)
    model_backbone = model.backbone
    canvas_grid_size = model._infer_canvas_grid_size(canvas)
    glimpse_grid_size = model.cfg.glimpse_grid_size

    with torch.inference_mode():
        # Compute initial scene BEFORE any forward pass
        initial_scene = model.compute_scene(canvas)

        outputs, _ = model.forward_trajectory_full(images, viewpoints, canvas)
        all_losses = {
            name: [fn(out.scene, target).item() for out in outputs]
            for name, fn in LOSS_FNS.items()
        }

        if log_curves:
            # Canvas states: initial + after each viewpoint
            canvases = [canvas] + [out.canvas for out in outputs]

            # Spatial canvas norm vs timestep
            spatial_norms = [
                model.get_spatial(c).norm(dim=-1).mean().item() for c in canvases
            ]
            exp.log_curve(
                f"{prefix}/spatial_norm_vs_timestep",
                x=list(range(len(spatial_norms))),
                y=spatial_norms,
                step=step,
            )

            # Step-to-step spatial difference norm
            diff_norms = [
                (model.get_spatial(canvases[i + 1]) - model.get_spatial(canvases[i]))
                .norm(dim=-1)
                .mean()
                .item()
                for i in range(len(canvases) - 1)
            ]
            exp.log_curve(
                f"{prefix}/spatial_diff_norm_vs_timestep",
                x=list(range(len(diff_norms))),
                y=diff_norms,
                step=step,
            )

            # Scene loss vs timestep
            losses = all_losses[loss_type]
            exp.log_curve(
                f"{prefix}/scene_loss_vs_timestep",
                x=list(range(len(losses))),
                y=losses,
                step=step,
            )

        # Log spatial stats for target and final prediction
        if log_spatial_stats:
            target_stats = compute_spatial_stats(target)
            pred_stats = compute_spatial_stats(outputs[-1].scene)
            exp.log_metrics(
                {
                    f"{prefix}/target_spatial_mean": target_stats["mean"],
                    f"{prefix}/target_spatial_std": target_stats["std"],
                    f"{prefix}/pred_spatial_mean": pred_stats["mean"],
                    f"{prefix}/pred_spatial_std": pred_stats["std"],
                },
                step=step,
            )

        # Prepare viz data for first sample
        sample_idx = 0
        n_prefix = teacher.n_prefix_tokens
        # Use actual image size for pixel coordinates (image_resolution, not scene_size_px)
        H, W = images.shape[-2], images.shape[-1]

        full_img = imagenet_denormalize(images[sample_idx].cpu()).numpy()

        # Target is already normalized; model predicts normalized features directly
        teacher_np = target[sample_idx].cpu().float().numpy()
        # Initial scene from canvas BEFORE any forward pass
        initial_np = initial_scene[sample_idx].cpu().float().numpy()

        scenes = [out.scene[sample_idx].cpu().float().numpy() for out in outputs]

        # Raw canvas spatials (before scene_proj): initial + after each viewpoint
        if show_canvas:
            initial_canvas_spatial = (
                model.get_spatial(canvas[sample_idx : sample_idx + 1])[0]
                .cpu()
                .float()
                .numpy()
            )
            canvas_spatials = [
                model.get_spatial(out.canvas[sample_idx : sample_idx + 1])[0]
                .cpu()
                .float()
                .numpy()
                for out in outputs
            ]
        else:
            canvas_spatials = None
            initial_canvas_spatial = None

        # Local features (only computed if show_locals - teacher glimpse inference is expensive)
        if show_locals:
            locals_model_raw = [
                model_backbone.output_norm(
                    out.local[sample_idx : sample_idx + 1, n_prefix:]
                ).squeeze(0)
                for out in outputs
            ]
            locals_teacher_raw = [
                teacher.forward_norm_features(
                    out.glimpse[sample_idx : sample_idx + 1]
                ).patches.squeeze(0)
                for out in outputs
            ]

            locals_model = [
                (feat).cpu().float().numpy()
                for feat, vp in zip(locals_model_raw, viewpoints, strict=True)
            ]
            locals_teacher = [
                (feat).cpu().float().numpy()
                for feat, vp in zip(locals_teacher_raw, viewpoints, strict=True)
            ]

            # Cropped teacher: sample normalized full-image targets at viewpoint positions
            # Shows "what teacher thinks at these positions with FULL image context"
            target_spatial = target.view(
                target.shape[0], canvas_grid_size, canvas_grid_size, -1
            ).permute(0, 3, 1, 2)
            locals_teacher_cropped = [
                sample_at_viewpoint(target_spatial, vp, glimpse_grid_size)[
                    sample_idx
                ]  # [D, G, G]
                .permute(1, 2, 0)
                .reshape(-1, target.shape[-1])  # [G², D]
                .cpu()
                .float()
                .numpy()
                for vp in viewpoints
            ]
        else:
            locals_model = None
            locals_teacher = None
            locals_teacher_cropped = None

        glimpses = [
            imagenet_denormalize(out.glimpse[sample_idx].cpu()).numpy()
            for out in outputs
        ]
        boxes = [vp.to_pixel_box(sample_idx, H, W) for vp in viewpoints]
        names = [vp.name for vp in viewpoints]

    fig_pca = plot_multistep_pca(
        full_img,
        teacher_np,
        scenes,
        locals_model,
        locals_teacher,
        glimpses,
        boxes,
        names,
        canvas_grid_size,
        glimpse_grid_size,
        initial_np,
        hidden_spatials=canvas_spatials,
        initial_hidden_spatial=initial_canvas_spatial,
        locals_teacher_cropped=locals_teacher_cropped,
        show_locals=show_locals,
    )
    log_figure(exp, fig_pca, f"{prefix}/pca", step)

    return VizResult(losses=all_losses, outputs=outputs)


def val_metrics_only(
    exp: comet_ml.Experiment,
    step: int,
    model: ActiveCanViT,
    compute_raw_targets: Callable[[Tensor, int], "NormFeatures"],
    scene_normalizer: PositionAwareNorm,
    cls_normalizer: PositionAwareNorm,
    images: Tensor,
    canvas_grid_size: int,
    scene_size_px: int,
    prefix: str = "val",
) -> float:
    """Fast validation without PCA. Returns final scene l1 loss (normalized)."""
    B = images.shape[0]
    viewpoints = make_eval_viewpoints(B, images.device)

    with torch.inference_mode():
        raw_feats = compute_raw_targets(images, scene_size_px)
        target = scene_normalizer(raw_feats.patches)
        canvas = model.init_canvas(B, canvas_grid_size)
        outputs, _ = model.forward_trajectory_full(images, viewpoints, canvas)
        final_scene = outputs[-1].scene

        # Scene metrics (normalized + raw)
        norms = {name: fn(final_scene, target).item() for name, fn in LOSS_FNS.items()}
        for name, val in norms.items():
            exp.log_metric(f"{prefix}/scene_{name}", val, step=step)
        for name, fn in LOSS_FNS.items():
            exp.log_metric(f"{prefix}/scene_{name}_raw", fn(final_scene, raw_feats.patches).item(), step=step)

        # CLS metrics (if enabled)
        if model.cls_proj is not None:
            cls_target = cls_normalizer(raw_feats.cls.unsqueeze(1)).squeeze(1)
            cls_pred = model.compute_cls(outputs[-1].canvas)
            for name, fn in LOSS_FNS.items():
                exp.log_metric(f"{prefix}/cls_{name}", fn(cls_pred, cls_target).item(), step=step)

    return norms["l1"]


def eval_and_log(
    exp: comet_ml.Experiment,
    step: int,
    model: ActiveCanViT,
    teacher: DINOv3Backbone,
    compute_raw_targets: Callable[[Tensor, int], "NormFeatures"],
    scene_normalizer: PositionAwareNorm,
    cls_normalizer: PositionAwareNorm,
    images: Tensor,
    canvas_grid_size: int,
    scene_size_px: int,
    prefix: str = "val",
    log_spatial_stats: bool = True,
    log_curves: bool = True,
    loss_type: LossType = "mse",
    show_locals: bool = False,
) -> float:
    """Full evaluation with PCA visualization (expensive). Returns final scene l1 loss (normalized)."""
    B = images.shape[0]
    viewpoints = make_eval_viewpoints(B, images.device)

    with torch.inference_mode():
        raw_feats = compute_raw_targets(images, scene_size_px)
        target = scene_normalizer(raw_feats.patches)
        canvas = model.init_canvas(B, canvas_grid_size)

    viz = viz_and_log(
        exp, step, prefix, model, teacher, scene_normalizer,
        images, viewpoints, target, canvas,
        log_spatial_stats=log_spatial_stats, log_curves=log_curves, loss_type=loss_type,
        show_locals=show_locals,
    )

    # Log normalized scene metrics - per timestep and final
    n_timesteps = len(viz.losses["l1"])
    for t in range(n_timesteps):
        for name, losses in viz.losses.items():
            exp.log_metric(f"{prefix}/scene_{name}_t{t}", losses[t], step=step)
    for name, losses in viz.losses.items():
        exp.log_metric(f"{prefix}/scene_{name}", losses[-1], step=step)

    # Raw scene metrics (for cross-run comparison) - reuse outputs from viz_and_log
    final_scene = viz.outputs[-1].scene
    for name, fn in LOSS_FNS.items():
        exp.log_metric(f"{prefix}/scene_{name}_raw", fn(final_scene, raw_feats.patches).item(), step=step)

    # CLS metrics (if enabled)
    if model.cls_proj is not None:
        with torch.inference_mode():
            cls_target = cls_normalizer(raw_feats.cls.unsqueeze(1)).squeeze(1)
            cls_pred = model.compute_cls(viz.outputs[-1].canvas)
            for name, fn in LOSS_FNS.items():
                exp.log_metric(f"{prefix}/cls_{name}", fn(cls_pred, cls_target).item(), step=step)

    return viz.losses["l1"][-1]


def log_norm_stats(
    exp: comet_ml.Experiment,
    scene_normalizers: dict[int, PositionAwareNorm],
    cls_normalizers: dict[int, PositionAwareNorm],
    step: int,
) -> None:
    """Log normalizer running stats to Comet: metrics and spatial heatmaps."""
    for G, scene_norm in scene_normalizers.items():
        cls_norm = cls_normalizers[G]
        exp.log_metrics(
            {
                f"norm/G{G}/scene_mean_norm": scene_norm.mean.norm().item(),
                f"norm/G{G}/scene_var_mean": scene_norm.var.mean().item(),
                f"norm/G{G}/cls_mean_norm": cls_norm.mean.norm().item(),
                f"norm/G{G}/cls_var_mean": cls_norm.var.mean().item(),
            },
            step=step,
        )
        # Spatial heatmap for scene normalizer only (CLS has no spatial structure)
        mean_np = scene_norm.mean.cpu().float().numpy()
        std_np = scene_norm.var.sqrt().cpu().float().numpy()
        fig = plot_norm_stats(mean_np, std_np, G)
        log_figure(exp, fig, f"norm/G{G}/spatial", step)
