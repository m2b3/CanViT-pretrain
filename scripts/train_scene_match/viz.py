"""Visualization and logging utilities."""

import io
import logging
from collections.abc import Callable
from pathlib import Path

import comet_ml
import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from torch import Tensor

from avp_vit import AVPViT
from avp_vit.backbone.dinov3 import DINOv3Backbone
from avp_vit.glimpse import Viewpoint
from avp_vit.train import imagenet_denormalize, plot_multistep_pca, plot_trajectory
from avp_vit.train.norm import PositionAwareNorm
from avp_vit.train.viewpoint import make_curriculum_eval_viewpoints

log = logging.getLogger(__name__)


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
    """Log matplotlib figure to Comet."""
    with io.BytesIO() as buf:
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        exp.log_image(buf, name=name, step=step)
    plt.close(fig)


def viz_and_log(
    exp: comet_ml.Experiment,
    step: int,
    prefix: str,
    avp: AVPViT,
    teacher: DINOv3Backbone,
    images: Tensor,
    viewpoints: list[Viewpoint],
    target: Tensor,
    hidden: Tensor | None,
    target_norm: PositionAwareNorm | None = None,
    show_hidden: bool = True,
    log_spatial_stats: bool = True,
) -> tuple[list[float], list[float]]:
    """Run forward trajectory and log visualization.

    Args:
        show_hidden: If True, show raw hidden spatial column in PCA plot.

    Returns (l1_losses, mse_losses) per timestep.
    """
    from torch.nn.functional import l1_loss, mse_loss

    assert isinstance(avp.backbone, DINOv3Backbone)
    avp_backbone = avp.backbone
    G = avp.cfg.glimpse_grid_size

    with torch.inference_mode():
        outputs, _, _ = avp.forward_trajectory_full(images, viewpoints, hidden)
        l1_losses = [l1_loss(out.scene, target).item() for out in outputs]
        mse_losses = [mse_loss(out.scene, target).item() for out in outputs]

        # Log per-timestep hidden norm (should be flat if recurrence is stable)
        # Include initial hidden at t=0, then outputs at t=1,2,...
        initial_hidden = hidden if hidden is not None else avp._get_base_hidden(images.shape[0])
        hidden_norms = [initial_hidden.norm(dim=-1).mean().item()]
        hidden_norms.extend(out.hidden.norm(dim=-1).mean().item() for out in outputs)
        exp.log_curve(
            f"{prefix}/hidden_norm_vs_timestep",
            x=list(range(len(hidden_norms))),
            y=hidden_norms,
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

        # Initial scene from hidden (or base hidden if None)
        if hidden is not None:
            initial_hidden_full = hidden[0:1]
        else:
            initial_hidden_full = avp._get_base_hidden(1)
        initial_scene = avp.compute_scene(initial_hidden_full)[0]

        # Prepare viz data for first sample
        sample_idx = 0
        n_prefix = teacher.n_prefix_tokens
        H, W = avp.scene_size, avp.scene_size

        full_img = imagenet_denormalize(images[sample_idx].cpu()).numpy()
        teacher_np = target[sample_idx].cpu().float().numpy()
        initial_np = initial_scene.cpu().float().numpy()

        scenes = [out.scene[sample_idx].cpu().float().numpy() for out in outputs]

        # Raw hidden spatials (before output_proj)
        if show_hidden:
            initial_hidden_spatial = avp.get_spatial(initial_hidden_full)[0].cpu().float().numpy()
            hidden_spatials = [
                avp.get_spatial(out.hidden[sample_idx : sample_idx + 1])[0].cpu().float().numpy()
                for out in outputs
            ]
        else:
            initial_hidden_spatial = None
            hidden_spatials = None

        # Local features - normalize with interpolated stats if available
        locals_avp_raw = [
            avp_backbone.output_norm(
                out.local[sample_idx : sample_idx + 1, n_prefix:]
            ).squeeze(0)
            for out in outputs
        ]
        locals_teacher_raw = [
            teacher.forward_norm_patches(
                out.glimpse[sample_idx : sample_idx + 1]
            ).squeeze(0)
            for out in outputs
        ]

        if target_norm is not None and target_norm.initialized:
            locals_avp = [
                target_norm.normalize_at_viewpoint(feat, vp, G).cpu().float().numpy()
                for feat, vp in zip(locals_avp_raw, viewpoints, strict=True)
            ]
            locals_teacher = [
                target_norm.normalize_at_viewpoint(feat, vp, G).cpu().float().numpy()
                for feat, vp in zip(locals_teacher_raw, viewpoints, strict=True)
            ]
        else:
            locals_avp = [feat.cpu().float().numpy() for feat in locals_avp_raw]
            locals_teacher = [feat.cpu().float().numpy() for feat in locals_teacher_raw]

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
        avp.cfg.scene_grid_size,
        avp.cfg.glimpse_grid_size,
        initial_np,
        hidden_spatials=hidden_spatials,
        initial_hidden_spatial=initial_hidden_spatial,
    )
    log_figure(exp, fig_pca, f"{prefix}/pca", step)

    fig_traj = plot_trajectory(full_img, boxes, names)
    log_figure(exp, fig_traj, f"{prefix}/trajectory", step)

    return l1_losses, mse_losses


def eval_and_log(
    exp: comet_ml.Experiment,
    step: int,
    avp: AVPViT,
    teacher: DINOv3Backbone,
    compute_targets: Callable[[Tensor], Tensor],
    images: Tensor,
    target_norm: PositionAwareNorm | None = None,
    prefix: str = "val",
    log_spatial_stats: bool = True,
) -> float:
    """Evaluate on one batch with curriculum viewpoints. Returns final L1 loss.

    Args:
        compute_targets: Function mapping images → normalized targets (what AVP outputs).
    """
    B = images.shape[0]
    G = avp.cfg.scene_grid_size
    g = avp.cfg.glimpse_grid_size
    viewpoints = make_curriculum_eval_viewpoints(B, G, g, images.device)

    with torch.inference_mode():
        target = compute_targets(images)

    l1_losses, mse_losses = viz_and_log(
        exp, step, prefix, avp, teacher, images, viewpoints, target, None, target_norm,
        log_spatial_stats=log_spatial_stats,
    )

    for t, (l1, mse) in enumerate(zip(l1_losses, mse_losses, strict=True)):
        exp.log_metric(f"{prefix}/l1_t{t}", l1, step=step)
        exp.log_metric(f"{prefix}/mse_t{t}", mse, step=step)

    exp.log_metric(f"{prefix}/l1", l1_losses[-1], step=step)
    exp.log_metric(f"{prefix}/mse", mse_losses[-1], step=step)
    return l1_losses[-1]


def save_checkpoint(
    avp: AVPViT,
    norms: dict[int, PositionAwareNorm],
    path: Path,
    exp: comet_ml.Experiment,
    step: int,
    train_loss: float,
    current_grid_size: int,
) -> None:
    """Save checkpoint with model and all norm states."""
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "avp": avp.state_dict(),
        "step": step,
        "current_grid_size": current_grid_size,
        "norm_states": {G: norm.state_dict() for G, norm in norms.items()},
    }
    torch.save(checkpoint, path)
    size_mb = path.stat().st_size / (1024 * 1024)
    log.info(f"Saved checkpoint: {path} ({size_mb:.1f} MB), train_loss={train_loss:.4f}")
    exp.log_metric("ckpt/train_loss", train_loss, step=step)
