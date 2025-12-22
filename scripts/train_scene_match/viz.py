"""Visualization and logging utilities."""

import gc
import io
import logging
from collections.abc import Callable
from typing import NamedTuple

import comet_ml
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from canvit.backbone.dinov3 import DINOv3Backbone, NormFeatures
from matplotlib.figure import Figure
from torch import Tensor

from dinov3_probes import DINOv3LinearClassificationHead

from avp_vit import ActiveCanViT, StepOutput
from avp_vit.train import imagenet_denormalize, plot_multistep_pca, plot_norm_stats
from avp_vit.train.norm import PositionAwareNorm
from avp_vit.train.probe import compute_in1k_top1
from avp_vit.train.viewpoint import Viewpoint, make_eval_viewpoints, sample_at_viewpoint

log = logging.getLogger(__name__)


class VizResult(NamedTuple):
    """Result from viz_and_log: per-timestep cosine similarities and model outputs."""

    scene_cos_sims: list[float]  # [cos_sim_t0, cos_sim_t1, ...]
    cls_cos_sims: list[float] | None  # None if no cls_target provided
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
    glimpse_size_px: int,
    cls_target: Tensor | None = None,
    show_canvas: bool = True,
    log_spatial_stats: bool = True,
    log_curves: bool = True,
    show_locals: bool = False,
) -> VizResult:
    """Run forward trajectory and log visualization."""
    assert isinstance(model.backbone, DINOv3Backbone)
    model_backbone = model.backbone
    n_spatial = canvas.shape[1] - model.n_prefix
    canvas_grid_size = int(n_spatial**0.5)
    assert canvas_grid_size**2 == n_spatial
    glimpse_grid_size = glimpse_size_px // model.backbone.patch_size_px
    assert glimpse_grid_size * model.backbone.patch_size_px == glimpse_size_px

    with torch.inference_mode():
        # Compute initial scene BEFORE any forward pass
        initial_scene = model.compute_scene(canvas)

        traj = model.forward_trajectory(
            image=images,
            viewpoints=viewpoints,
            canvas_grid_size=canvas_grid_size,
            glimpse_size_px=glimpse_size_px,
            canvas=canvas,
        )
        outputs = traj.outputs
        scene_cos_sims = [
            F.cosine_similarity(out.predicted_scene, target, dim=-1).mean().item()
            for out in outputs
        ]
        cls_cos_sims: list[float] | None = None
        if cls_target is not None:
            cls_cos_sims = [
                F.cosine_similarity(out.predicted_cls, cls_target, dim=-1).mean().item()
                for out in outputs
            ]

        if log_curves:
            exp.log_curve(
                f"{prefix}/scene_cos_sim_vs_timestep",
                x=list(range(len(scene_cos_sims))),
                y=scene_cos_sims,
                step=step,
            )
            if cls_cos_sims is not None:
                exp.log_curve(
                    f"{prefix}/cls_cos_sim_vs_timestep",
                    x=list(range(len(cls_cos_sims))),
                    y=cls_cos_sims,
                    step=step,
                )

        # Log spatial stats for target and final prediction
        if log_spatial_stats:
            target_stats = compute_spatial_stats(target)
            pred_stats = compute_spatial_stats(outputs[-1].predicted_scene)
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

        scenes = [out.predicted_scene[sample_idx].cpu().float().numpy() for out in outputs]

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
                sample_at_viewpoint(
                    spatial=target_spatial, viewpoint=vp, out_size=glimpse_grid_size
                )[sample_idx]  # [D, G, G]
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

    return VizResult(scene_cos_sims=scene_cos_sims, cls_cos_sims=cls_cos_sims, outputs=outputs)


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
    glimpse_size_px: int,
    prefix: str = "val",
    probe: DINOv3LinearClassificationHead | None = None,
    labels: Tensor | None = None,
) -> float:
    """Fast validation without PCA. Returns final scene cosine similarity.

    Args:
        probe: Optional IN1k classification probe for accuracy metrics.
        labels: Optional IN1k labels (required if probe is provided).
    """
    B = images.shape[0]
    viewpoints = make_eval_viewpoints(B, images.device)

    with torch.inference_mode():
        raw_feats = compute_raw_targets(images, scene_size_px)
        target = scene_normalizer(raw_feats.patches)
        traj = model.forward_trajectory(
            image=images,
            viewpoints=viewpoints,
            canvas_grid_size=canvas_grid_size,
            glimpse_size_px=glimpse_size_px,
        )
        outputs = traj.outputs
        final_scene = outputs[-1].predicted_scene

        cos_sim = F.cosine_similarity(final_scene, target, dim=-1).mean().item()
        scene_mse = F.mse_loss(final_scene, target).item()
        exp.log_metric(f"{prefix}/scene_cos_sim", cos_sim, step=step)
        exp.log_metric(f"{prefix}/scene_mse", scene_mse, step=step)

        if model.cls_head is not None:
            cls_target = cls_normalizer(raw_feats.cls.unsqueeze(1)).squeeze(1)
            cls_pred = model.compute_cls(outputs[-1].canvas)
            cls_cos_sim = (
                F.cosine_similarity(cls_pred, cls_target, dim=-1).mean().item()
            )
            cls_mse = F.mse_loss(cls_pred, cls_target).item()
            exp.log_metric(f"{prefix}/cls_cos_sim", cls_cos_sim, step=step)
            exp.log_metric(f"{prefix}/cls_mse", cls_mse, step=step)

            # IN1k probe metrics (optional)
            if probe is not None and labels is not None:
                in1k_accs: list[float] = []
                for t, out in enumerate(outputs):
                    cls_pred_t = model.compute_cls(out.canvas)
                    cls_raw = cls_normalizer.denormalize(cls_pred_t)
                    logits = probe(cls_raw)
                    acc = compute_in1k_top1(logits, labels)
                    in1k_accs.append(acc)
                    exp.log_metric(f"{prefix}/in1k_t{t}", acc, step=step)
                exp.log_curve(
                    f"{prefix}/in1k_vs_timestep",
                    x=list(range(len(in1k_accs))),
                    y=in1k_accs,
                    step=step,
                )

    return cos_sim


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
    glimpse_size_px: int,
    prefix: str = "val",
    log_spatial_stats: bool = True,
    log_curves: bool = True,
    show_locals: bool = False,
) -> float:
    """Full evaluation with PCA visualization (expensive). Returns final scene cosine similarity."""
    B = images.shape[0]
    viewpoints = make_eval_viewpoints(B, images.device)

    with torch.inference_mode():
        raw_feats = compute_raw_targets(images, scene_size_px)
        target = scene_normalizer(raw_feats.patches)
        cls_target = cls_normalizer(raw_feats.cls.unsqueeze(1)).squeeze(1) if model.cls_head is not None else None
        canvas = model.init_canvas(batch_size=B, canvas_grid_size=canvas_grid_size)

    viz = viz_and_log(
        exp,
        step,
        prefix,
        model,
        teacher,
        scene_normalizer,
        images,
        viewpoints,
        target,
        canvas,
        glimpse_size_px,
        cls_target=cls_target,
        log_spatial_stats=log_spatial_stats,
        log_curves=log_curves,
        show_locals=show_locals,
    )

    # Log per-timestep and final cosine similarity
    for t, cos_sim in enumerate(viz.scene_cos_sims):
        exp.log_metric(f"{prefix}/scene_cos_sim_t{t}", cos_sim, step=step)
    exp.log_metric(f"{prefix}/scene_cos_sim", viz.scene_cos_sims[-1], step=step)

    # Scene MSE (for comparison with train/scene_loss)
    final_scene = viz.outputs[-1].predicted_scene
    scene_mse = F.mse_loss(final_scene, target).item()
    exp.log_metric(f"{prefix}/scene_mse", scene_mse, step=step)

    # CLS metrics (if enabled)
    if model.cls_head is not None:
        assert cls_target is not None
        assert viz.cls_cos_sims is not None
        with torch.inference_mode():
            cls_pred = model.compute_cls(viz.outputs[-1].canvas)
            cls_cos_sim = (
                F.cosine_similarity(cls_pred, cls_target, dim=-1).mean().item()
            )
            cls_mse = F.mse_loss(cls_pred, cls_target).item()
            exp.log_metric(f"{prefix}/cls_cos_sim", cls_cos_sim, step=step)
            exp.log_metric(f"{prefix}/cls_mse", cls_mse, step=step)

    return viz.scene_cos_sims[-1]


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
