"""Main training loop with random grid size sampling."""

import logging
import random
from collections.abc import Callable

import comet_ml
import optuna
import torch
from torch import Tensor
from torch.nn.functional import l1_loss, mse_loss
from tqdm import tqdm
from ymc.lr import get_linear_scaled_lr
from ytch.model import count_parameters

from avp_vit import AVPViT
from avp_vit.backbone.dinov3 import NormFeatures
from avp_vit.train import InfiniteLoader, SurvivalBatch
from avp_vit.train.norm import PositionAwareNorm
from avp_vit.train.viewpoint import random_viewpoint

from .config import Config
from .data import ResolutionStage, create_loaders, create_resolution_stages
from .model import compile_avp, compile_teacher, create_avp, load_student_backbone, load_teacher
from .scheduler import create_scheduler
from .viz import (
    eval_and_log,
    load_avp_checkpoint,
    log_norm_stats,
    save_checkpoint,
    val_metrics_only,
    viz_and_log,
)

log = logging.getLogger(__name__)


def init_survival_batch(
    avp: AVPViT,
    train_loader: InfiniteLoader,
    compute_targets: Callable[[Tensor], NormFeatures],
    scene_normalizer: PositionAwareNorm,
    cls_normalizer: PositionAwareNorm,
    batch_size: int,
    fresh_count: int,
    scene_grid_size: int,
    device: torch.device,
) -> SurvivalBatch:
    """Initialize survival batch by loading fresh_count images at a time.

    Targets are normalized using the provided normalizers (expected to be in eval mode).
    """
    n_init_batches = (batch_size + fresh_count - 1) // fresh_count
    log.info(f"Initializing survival batch: batch_size={batch_size}, fresh_count={fresh_count}")

    init_imgs_list: list[Tensor] = []
    init_patches_list: list[Tensor] = []
    init_cls_list: list[Tensor] = []
    with torch.no_grad():
        for _ in range(n_init_batches):
            batch = train_loader.next_batch().to(device)
            feats = compute_targets(batch)
            norm_patches = scene_normalizer(feats.patches)
            norm_cls = cls_normalizer(feats.cls.unsqueeze(1)).squeeze(1)  # [B,D]->[B,1,D]->[B,D]
            init_imgs_list.append(batch)
            init_patches_list.append(norm_patches)
            init_cls_list.append(norm_cls)

    init_imgs = torch.cat(init_imgs_list, dim=0)[:batch_size]
    init_targets = torch.cat(init_patches_list, dim=0)[:batch_size]
    init_cls_targets = torch.cat(init_cls_list, dim=0)[:batch_size]
    hidden_init = avp.init_hidden(batch_size, scene_grid_size)

    return SurvivalBatch.init(init_imgs, init_targets, init_cls_targets, hidden_init)


def _log_stage(stage: ResolutionStage) -> None:
    log.info(
        f"  G={stage.scene_grid_size}: batch={stage.batch_size}, fresh={stage.fresh_count}, "
        f"ratio={stage.fresh_ratio_actual:.3f} (desired={stage.fresh_ratio_desired:.3f}), "
        f"E[glimpses]={stage.e_glimpses_actual:.1f} (desired={stage.e_glimpses_desired:.1f}), "
        f"min_scale={stage.min_viewpoint_scale:.2f}"
    )


def warmup_normalizers(
    scene_normalizers: dict[int, PositionAwareNorm],
    cls_normalizer: PositionAwareNorm,
    train_loaders: dict[int, InfiniteLoader],
    compute_raw_targets: Callable[[Tensor], NormFeatures],
    warmup_images: int,
    device: torch.device,
) -> None:
    """Warm up normalizer running stats before training begins.

    CLS normalizer is warmed up alongside the first scene normalizer (grid-size independent).
    Normalizers stay in train mode - stats continue updating during training.
    """
    cls_warmed_up = False
    for G, normalizer in scene_normalizers.items():
        normalizer.train()
        if not cls_warmed_up:
            cls_normalizer.train()
        loader = train_loaders[G]
        images_seen = 0
        pbar = tqdm(total=warmup_images, desc=f"Warmup G={G}", unit="img", leave=False)
        while images_seen < warmup_images:
            batch = loader.next_batch().to(device)
            with torch.no_grad():
                feats = compute_raw_targets(batch)
                normalizer(feats.patches)
                if not cls_warmed_up:
                    cls_normalizer(feats.cls.unsqueeze(1))  # [B, D] -> [B, 1, D]
            images_seen += batch.shape[0]
            pbar.update(batch.shape[0])
        pbar.close()
        log.info(
            f"Warmup G={G}: {images_seen} images, "
            f"mean_norm={normalizer.mean.norm().item():.4f}, "
            f"var_mean={normalizer.var.mean().item():.4f}"
        )
        if not cls_warmed_up:
            log.info(
                f"Warmup CLS: {images_seen} images, "
                f"mean_norm={cls_normalizer.mean.norm().item():.4f}, "
                f"var_mean={cls_normalizer.var.mean().item():.4f}"
            )
            cls_warmed_up = True


def train(cfg: Config, trial: optuna.Trial) -> float:
    """Train AVP model with random grid size sampling. Returns best val_loss."""
    log.info(f"Starting trial {trial.number}")
    log.info(f"Device: {cfg.device}")

    exp = comet_ml.Experiment(project_name="avp-vit-scene-match", auto_metric_logging=False)
    from dataclasses import asdict

    def flatten_dict(d: dict, prefix: str = "") -> dict[str, object]:
        flat: dict[str, object] = {}
        for k, v in d.items():
            key = f"{prefix}{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(flatten_dict(v, f"{key}."))
            else:
                flat[key] = str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v
        return flat

    exp.log_parameters(flatten_dict(asdict(cfg)))
    exp.log_parameters({"trial_number": trial.number})

    teacher = load_teacher(cfg)
    log.info(f"Teacher params: {count_parameters(teacher):,}")

    student_backbone = load_student_backbone(cfg)
    log.info(f"Student backbone params: {count_parameters(student_backbone):,}")

    avp = create_avp(student_backbone, teacher.embed_dim, cfg)

    if cfg.compile and cfg.avp.gradient_checkpointing:
        raise ValueError(
            "compile=True and gradient_checkpointing=True may be incompatible. "
            "This combination has caused CheckpointError (tensor metadata mismatch) in some configurations. "
            "Disable one of them."
        )

    if cfg.compile:
        log.info("Compiling teacher and AVP")
        compile_teacher(teacher)
        compile_avp(avp)

    loss_fn = {"l1": l1_loss, "mse": mse_loss}[cfg.loss]
    log.info(f"Loss function: {cfg.loss}")

    # Create resolution stages and resources
    patch_size = teacher.patch_size
    stages = create_resolution_stages(cfg, patch_size)
    log.info("Resolution stages:")
    for stage in stages.values():
        _log_stage(stage)

    train_loaders, val_loaders = create_loaders(cfg, stages)

    trainable = [p for p in avp.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = count_parameters(avp)
    log.info(f"AVP total: {n_total:,}, trainable: {n_trainable:,} ({100 * n_trainable / n_total:.1f}%)")
    exp.log_parameters({"trainable_params": n_trainable, "total_params": n_total})

    peak_lr = get_linear_scaled_lr(cfg.ref_lr, cfg.batch_size)
    optimizer = torch.optim.AdamW(trainable, lr=peak_lr, weight_decay=cfg.weight_decay)
    scheduler = create_scheduler(optimizer, cfg)
    log.info(f"Optimizer: AdamW, peak_lr={peak_lr:.2e}, weight_decay={cfg.weight_decay:.2e}")

    # Load AVP weights from checkpoint if specified
    if cfg.resume_ckpt is not None:
        load_avp_checkpoint(cfg.resume_ckpt, avp)

    ckpt_path = cfg.ckpt_dir / f"{exp.get_key()}.pt"

    def compute_raw_targets(images: Tensor) -> NormFeatures:
        """Compute teacher features (not normalized)."""
        with torch.autocast(device_type=cfg.device.type, dtype=torch.bfloat16):
            feats = teacher.forward_norm_features(images)
            return NormFeatures(patches=feats.patches.float(), cls=feats.cls.float())

    # Create normalizers for each grid size (different n_tokens)
    normalizers: dict[int, PositionAwareNorm] = {}
    for G in cfg.grid_sizes:
        normalizers[G] = PositionAwareNorm(
            n_tokens=G * G,
            embed_dim=teacher.embed_dim,
            grid_size=G,
            momentum=cfg.norm_momentum,
        ).to(cfg.device)

    # CLS normalizer (grid-size independent, single token)
    cls_normalizer = PositionAwareNorm(
        n_tokens=1,
        embed_dim=teacher.embed_dim,
        grid_size=1,
        momentum=cfg.norm_momentum,
    ).to(cfg.device)

    # Warm up normalizer stats before training
    log.info(f"Warming up normalizers ({cfg.norm_warmup_images} images per grid size)...")
    warmup_normalizers(normalizers, cls_normalizer, train_loaders, compute_raw_targets, cfg.norm_warmup_images, cfg.device)

    # Initialize all survival batches upfront (with normalized targets)
    log.info("Initializing survival batches...")
    states: dict[int, SurvivalBatch] = {}
    for G, stage in stages.items():
        states[G] = init_survival_batch(
            avp, train_loaders[G], compute_raw_targets, normalizers[G], cls_normalizer,
            stage.batch_size, stage.fresh_count, G, cfg.device,
        )

    # EMA tracking for losses (scene, local, cls, total)
    ema_scene_t = torch.tensor(0.0, device=cfg.device)
    ema_local_t = torch.tensor(0.0, device=cfg.device)
    ema_cls_t = torch.tensor(0.0, device=cfg.device)
    ema_loss_t = torch.tensor(0.0, device=cfg.device)
    alpha = 2 / (cfg.log_every + 1)

    log.info("Starting training loop...")
    pbar = tqdm(range(cfg.n_steps), desc="Training", unit="step")

    for step in pbar:
        G = random.choice(cfg.grid_sizes)
        stage = stages[G]
        state = states[G]
        train_loader = train_loaders[G]
        val_loader = val_loaders[G]

        # Load fresh images and normalize targets (both scene patches and CLS)
        normalizer = normalizers[G]
        fresh_imgs = train_loader.next_batch().to(cfg.device)
        with torch.no_grad():
            fresh_feats = compute_raw_targets(fresh_imgs)
            fresh_patches_norm = normalizer(fresh_feats.patches)
            fresh_cls_norm = cls_normalizer(fresh_feats.cls.unsqueeze(1)).squeeze(1)  # [B,D]->[B,1,D]->[B,D]

        # Viewpoint scale bounds for current grid size
        min_scale = stage.min_viewpoint_scale
        max_scale = 1.0

        # Inner loop: multiple viewpoints per optimizer step
        viewpoints = [
            random_viewpoint(stage.batch_size, cfg.device, min_scale, max_scale)
            for _ in range(cfg.n_viewpoints_per_step)
        ]
        losses, final_hidden = avp.forward_loss(
            state.images,
            viewpoints,
            state.targets,
            state.hidden,
            cls_target=state.cls_targets,
            loss_fn=loss_fn,
        )

        # Combine losses (trainer's responsibility)
        total_loss = losses.scene
        if losses.local is not None:
            total_loss = total_loss + losses.local
        if losses.cls is not None:
            total_loss = total_loss + losses.cls

        if not torch.isfinite(total_loss):
            log.warning(f"NaN/Inf loss at step {step}, pruning trial")
            exp.end()
            raise optuna.TrialPruned()

        optimizer.zero_grad()
        total_loss.backward()
        grad_norm_t = torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        # Update EMAs
        scene_t = losses.scene.detach()
        local_t = losses.local.detach() if losses.local is not None else torch.tensor(0.0, device=cfg.device)
        cls_t = losses.cls.detach() if losses.cls is not None else torch.tensor(0.0, device=cfg.device)
        total_t = total_loss.detach()
        if step > 0:
            ema_scene_t = alpha * scene_t + (1 - alpha) * ema_scene_t
            ema_local_t = alpha * local_t + (1 - alpha) * ema_local_t
            ema_cls_t = alpha * cls_t + (1 - alpha) * ema_cls_t
            ema_loss_t = alpha * total_t + (1 - alpha) * ema_loss_t
        else:
            ema_scene_t, ema_local_t, ema_cls_t, ema_loss_t = scene_t, local_t, cls_t, total_t

        if step % cfg.log_every == 0:
            ema_loss = ema_loss_t.item()
            ema_scene = ema_scene_t.item()
            ema_local = ema_local_t.item()
            ema_cls = ema_cls_t.item()
            grad_norm = grad_norm_t.item()
            lr = scheduler.get_last_lr()[0]
            metrics = {
                f"grid{G}/train/loss": ema_loss,
                f"grid{G}/train/scene_loss": ema_scene,
                "train/loss": ema_loss,
                "train/scene_loss": ema_scene,
                "train/grad_norm": grad_norm,
                "train/lr": lr,
                "train/grid_size": G,
                "train/spatial_hidden_init_norm": avp.spatial_hidden_init.norm().item(),
                "train/cls_hidden_init_norm": avp.cls_hidden_init.norm().item(),
            }
            if avp.cls_proj is not None:
                cls_linear = avp.cls_proj[1]
                assert isinstance(cls_linear, torch.nn.Linear)
                metrics["train/cls_proj_weight_norm"] = cls_linear.weight.norm().item()
            if losses.local is not None:
                metrics[f"grid{G}/train/local_loss"] = ema_local
                metrics["train/local_loss"] = ema_local
            if losses.cls is not None:
                metrics[f"grid{G}/train/cls_loss"] = ema_cls
                metrics["train/cls_loss"] = ema_cls
            exp.log_metrics(metrics, step=step)
            pbar.set_postfix_str(f"G={G} loss={ema_loss:.2e} grad={grad_norm:.2e} lr={lr:.2e}")

        # Validation (metrics only, fast)
        if step % cfg.val_every == 0:
            val_images = val_loader.next_batch().to(cfg.device)
            val_loss = val_metrics_only(
                exp, step, avp, compute_raw_targets, normalizer, cls_normalizer,
                val_images, G, f"grid{G}/val",
            )
            exp.log_metric("val/loss", val_loss, step=step)

            if step > 0:
                trial.report(ema_loss_t.item(), step)
                if trial.should_prune():
                    exp.end()
                    raise optuna.TrialPruned()

        # Full viz with PCA (expensive, less frequent)
        if step % cfg.viz_every == 0:
            train_l1, train_mse = viz_and_log(
                exp, step, f"grid{G}/train", avp, teacher, normalizer,
                state.images, viewpoints, state.targets, state.hidden,
                log_spatial_stats=cfg.log_spatial_stats, log_curves=False, loss_type=cfg.loss,
            )
            exp.log_metric(f"grid{G}/train/viz_l1", train_l1[-1], step=step)
            exp.log_metric(f"grid{G}/train/viz_mse", train_mse[-1], step=step)

            val_images = val_loader.next_batch().to(cfg.device)
            eval_and_log(
                exp, step, avp, teacher, compute_raw_targets, normalizer, cls_normalizer,
                val_images, G, f"grid{G}/val",
                log_spatial_stats=cfg.log_spatial_stats,
                log_curves=(step % cfg.curve_every == 0),
                loss_type=cfg.loss,
            )

        # Checkpointing
        if step % cfg.ckpt_every == 0:
            save_checkpoint(
                avp=avp, path=ckpt_path, exp=exp,
                step=step, train_loss=ema_loss_t.item(), current_grid_size=G,
            )
            log_norm_stats(exp, normalizers, cls_normalizer, step)

        # Fresh ratio survival: permute batch, replace first K with fresh (normalized targets)
        hidden_init = avp.init_hidden(stage.fresh_count, G)
        states[G] = state.step(
            fresh_images=fresh_imgs,
            fresh_targets=fresh_patches_norm,
            fresh_cls_targets=fresh_cls_norm,
            next_hidden=final_hidden,
            hidden_init=hidden_init,
        )

    # Final checkpoint
    save_checkpoint(
        avp=avp, path=ckpt_path, exp=exp,
        step=cfg.n_steps, train_loss=ema_loss_t.item(), current_grid_size=cfg.max_grid_size,
    )

    log.info(f"Final: train_ema={ema_loss_t.item():.4f}")
    exp.end()
    return ema_loss_t.item()
