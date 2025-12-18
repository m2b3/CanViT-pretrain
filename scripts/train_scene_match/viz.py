"""Visualization and logging utilities."""

import gc
import io
import logging
import math
from collections.abc import Callable

import comet_ml
import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from torch import Tensor

from avp_vit import AVPViT
from avp_vit.backbone.dinov3 import DINOv3Backbone, NormFeatures
from avp_vit.glimpse import Viewpoint, sample_at_viewpoint
from avp_vit.train import LOSS_FNS, LossType, imagenet_denormalize, plot_multistep_pca, plot_norm_stats
from avp_vit.train.norm import PositionAwareNorm
from avp_vit.train.viewpoint import make_eval_viewpoints

log = logging.getLogger(__name__)


def _infer_scene_grid_size(target: Tensor) -> int:
    """Infer scene grid size from target shape [B, G², D]."""
    n_tokens = target.shape[1]
    G = int(math.sqrt(n_tokens))
    assert G * G == n_tokens, f"target has {n_tokens} tokens, not a perfect square"
    return G


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
    avp: AVPViT,
    teacher: DINOv3Backbone,
    normalizer: PositionAwareNorm,
    images: Tensor,
    viewpoints: list[Viewpoint],
    target: Tensor,
    hidden: Tensor,
    show_hidden: bool = True,
    log_spatial_stats: bool = True,
    log_curves: bool = True,
    loss_type: LossType = "mse",
) -> dict[str, list[float]]:
    """Run forward trajectory and log visualization. Returns {l1, mse, cos} losses per timestep."""
    assert isinstance(avp.backbone, DINOv3Backbone)
    avp_backbone = avp.backbone
    scene_grid_size = _infer_scene_grid_size(target)
    glimpse_grid_size = avp.cfg.glimpse_grid_size

    with torch.inference_mode():
        outputs, _ = avp.forward_trajectory_full(images, viewpoints, hidden)
        all_losses = {
            name: [fn(out.scene, target).item() for out in outputs]
            for name, fn in LOSS_FNS.items()
        }

        # Compute hidden states for logging and visualization
        # t=0 = initial hidden before any viewpoint, t=1,2,... = after each viewpoint
        initial_hidden = avp._normalize_hidden(hidden)

        if log_curves:
            hiddens = [initial_hidden] + [out.hidden for out in outputs]

            # Spatial hidden norm vs timestep (excludes registers)
            spatial_norms = [
                avp.get_spatial(h).norm(dim=-1).mean().item() for h in hiddens
            ]
            exp.log_curve(
                f"{prefix}/spatial_norm_vs_timestep",
                x=list(range(len(spatial_norms))),
                y=spatial_norms,
                step=step,
            )

            # Step-to-step spatial difference norm
            diff_norms = [
                (avp.get_spatial(hiddens[i + 1]) - avp.get_spatial(hiddens[i]))
                .norm(dim=-1)
                .mean()
                .item()
                for i in range(len(hiddens) - 1)
            ]
            exp.log_curve(
                f"{prefix}/spatial_diff_norm_vs_timestep",
                x=list(range(len(diff_norms))),
                y=diff_norms,
                step=step,
            )

            # Scene loss vs timestep (t=0 is initial scene before any viewpoint)
            initial_scene = avp.compute_scene(initial_hidden)
            initial_loss = LOSS_FNS[loss_type](initial_scene, target).item()
            losses = all_losses[loss_type]
            exp.log_curve(
                f"{prefix}/scene_loss_vs_timestep",
                x=list(range(len(losses) + 1)),
                y=[initial_loss] + losses,
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

        # Prepare viz data for first sample (initial_hidden already computed above)
        sample_idx = 0
        n_prefix = teacher.n_prefix_tokens
        scene_size_px = scene_grid_size * avp.backbone.patch_size
        H, W = scene_size_px, scene_size_px
        initial_scene = avp.compute_scene(initial_hidden)  # [B, N, D]

        full_img = imagenet_denormalize(images[sample_idx].cpu()).numpy()

        # Target is already normalized; model predicts normalized features directly
        teacher_np = target[sample_idx].cpu().float().numpy()
        initial_np = initial_scene[sample_idx].cpu().float().numpy()

        scenes = [out.scene[sample_idx].cpu().float().numpy() for out in outputs]

        # Raw hidden spatials (before output_proj)
        if show_hidden:
            initial_hidden_spatial = (
                avp.get_spatial(initial_hidden[sample_idx : sample_idx + 1])[0]
                .cpu()
                .float()
                .numpy()
            )
            hidden_spatials = [
                avp.get_spatial(out.hidden[sample_idx : sample_idx + 1])[0]
                .cpu()
                .float()
                .numpy()
                for out in outputs
            ]
        else:
            initial_hidden_spatial = None
            hidden_spatials = None

        # Local features from AVP backbone
        locals_avp_raw = [
            avp_backbone.output_norm(
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

        locals_avp = [
            (feat).cpu().float().numpy()
            for feat, vp in zip(locals_avp_raw, viewpoints, strict=True)
        ]
        locals_teacher = [
            (feat).cpu().float().numpy()
            for feat, vp in zip(locals_teacher_raw, viewpoints, strict=True)
        ]

        # Cropped teacher: sample normalized full-image targets at viewpoint positions
        # Shows "what teacher thinks at these positions with FULL image context"
        # Target is already normalized, so cropped features are too
        target_spatial = target.view(
            target.shape[0], scene_grid_size, scene_grid_size, -1
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
        locals_avp,
        locals_teacher,
        glimpses,
        boxes,
        names,
        scene_grid_size,
        glimpse_grid_size,
        initial_np,
        hidden_spatials=hidden_spatials,
        initial_hidden_spatial=initial_hidden_spatial,
        locals_teacher_cropped=locals_teacher_cropped,
        use_local_loss=False,
    )
    log_figure(exp, fig_pca, f"{prefix}/pca", step)

    return all_losses


def val_metrics_only(
    exp: comet_ml.Experiment,
    step: int,
    avp: AVPViT,
    compute_raw_targets: Callable[[Tensor], "NormFeatures"],
    scene_normalizer: PositionAwareNorm,
    cls_normalizer: PositionAwareNorm,
    images: Tensor,
    scene_grid_size: int,
    prefix: str = "val",
) -> float:
    """Fast validation without PCA. Returns final scene l1 loss (normalized)."""
    B = images.shape[0]
    viewpoints = make_eval_viewpoints(B, images.device)

    with torch.inference_mode():
        raw_feats = compute_raw_targets(images)
        target = scene_normalizer(raw_feats.patches)
        hidden = avp.init_hidden(B, scene_grid_size)
        outputs, _ = avp.forward_trajectory_full(images, viewpoints, hidden)
        final_scene = outputs[-1].scene

        # Scene metrics (normalized + raw)
        norms = {name: fn(final_scene, target).item() for name, fn in LOSS_FNS.items()}
        for name, val in norms.items():
            exp.log_metric(f"{prefix}/scene_{name}", val, step=step)
        for name, fn in LOSS_FNS.items():
            exp.log_metric(f"{prefix}/scene_{name}_raw", fn(final_scene, raw_feats.patches).item(), step=step)

        # CLS metrics (if enabled)
        if avp.cls_proj is not None:
            cls_target = cls_normalizer(raw_feats.cls.unsqueeze(1)).squeeze(1)
            cls_pred = avp.compute_cls(outputs[-1].hidden)
            for name, fn in LOSS_FNS.items():
                exp.log_metric(f"{prefix}/cls_{name}", fn(cls_pred, cls_target).item(), step=step)

    return norms["l1"]


def eval_and_log(
    exp: comet_ml.Experiment,
    step: int,
    avp: AVPViT,
    teacher: DINOv3Backbone,
    compute_raw_targets: Callable[[Tensor], "NormFeatures"],
    scene_normalizer: PositionAwareNorm,
    cls_normalizer: PositionAwareNorm,
    images: Tensor,
    scene_grid_size: int,
    prefix: str = "val",
    log_spatial_stats: bool = True,
    log_curves: bool = True,
    loss_type: LossType = "mse",
) -> float:
    """Full evaluation with PCA visualization (expensive). Returns final scene l1 loss (normalized)."""
    B = images.shape[0]
    viewpoints = make_eval_viewpoints(B, images.device)

    with torch.inference_mode():
        raw_feats = compute_raw_targets(images)
        target = scene_normalizer(raw_feats.patches)
        hidden = avp.init_hidden(B, scene_grid_size)

    all_losses = viz_and_log(
        exp, step, prefix, avp, teacher, scene_normalizer,
        images, viewpoints, target, hidden,
        log_spatial_stats=log_spatial_stats, log_curves=log_curves, loss_type=loss_type,
    )

    # Log normalized scene metrics - per timestep and final
    n_timesteps = len(all_losses["l1"])
    for t in range(n_timesteps):
        for name, losses in all_losses.items():
            exp.log_metric(f"{prefix}/scene_{name}_t{t}", losses[t], step=step)
    for name, losses in all_losses.items():
        exp.log_metric(f"{prefix}/scene_{name}", losses[-1], step=step)

    # Raw scene metrics (for cross-run comparison)
    with torch.inference_mode():
        outputs, _ = avp.forward_trajectory_full(images, viewpoints, hidden)
        final_scene = outputs[-1].scene
        for name, fn in LOSS_FNS.items():
            exp.log_metric(f"{prefix}/scene_{name}_raw", fn(final_scene, raw_feats.patches).item(), step=step)

        # CLS metrics (if enabled)
        if avp.cls_proj is not None:
            cls_target = cls_normalizer(raw_feats.cls.unsqueeze(1)).squeeze(1)
            cls_pred = avp.compute_cls(outputs[-1].hidden)
            for name, fn in LOSS_FNS.items():
                exp.log_metric(f"{prefix}/cls_{name}", fn(cls_pred, cls_target).item(), step=step)

    return all_losses["l1"][-1]


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
