"""Main training loop with random grid size sampling."""

import logging
import random
from collections.abc import Callable

import comet_ml
import optuna
import torch
from torch import Tensor, nn
from tqdm import tqdm
from ymc.lr import get_linear_scaled_lr
from ytch.model import count_parameters

from avp_vit import ActiveCanViT, ActiveCanViTConfig
from canvit.backbone.dinov3 import NormFeatures
from avp_vit.checkpoint import load as load_checkpoint
from avp_vit.checkpoint import save as save_checkpoint
from avp_vit.train import InfiniteLoader, SurvivalBatch, get_loss_fn, warmup_cosine_scheduler
from avp_vit.train.norm import PositionAwareNorm
from avp_vit.train.viewpoint import random_viewpoint

from .config import Config
from .data import ResolutionStage, create_loaders, create_resolution_stages
from .model import compile_model, compile_teacher, create_model, load_student_backbone, load_teacher
from .viz import eval_and_log, log_norm_stats, val_metrics_only, viz_and_log

log = logging.getLogger(__name__)


def grad_norms_by_module(model: nn.Module, depth: int = 1) -> dict[str, float]:
    """Gradient norms grouped by module path prefix."""
    groups: dict[str, list[Tensor]] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        parts = name.split(".")
        prefix = ".".join(parts[:depth])
        groups.setdefault(prefix, []).append(param.grad)
    return {
        prefix: torch.cat([g.flatten() for g in grads]).norm().item()
        for prefix, grads in groups.items()
    }


def init_survival_batch(
    model: ActiveCanViT,
    train_loader: InfiniteLoader,
    compute_targets: Callable[[Tensor], NormFeatures],
    scene_normalizer: PositionAwareNorm,
    cls_normalizer: PositionAwareNorm,
    batch_size: int,
    fresh_count: int,
    canvas_grid_size: int,
    device: torch.device,
) -> SurvivalBatch:
    """Initialize survival batch. Normalizers expected to be in eval mode."""
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
            norm_cls = cls_normalizer(feats.cls.unsqueeze(1)).squeeze(1)
            init_imgs_list.append(batch)
            init_patches_list.append(norm_patches)
            init_cls_list.append(norm_cls)

    init_imgs = torch.cat(init_imgs_list, dim=0)[:batch_size]
    init_targets = torch.cat(init_patches_list, dim=0)[:batch_size]
    init_cls_targets = torch.cat(init_cls_list, dim=0)[:batch_size]
    canvas_init = model.init_canvas(batch_size, canvas_grid_size)

    return SurvivalBatch.init(init_imgs, init_targets, init_cls_targets, canvas_init)


def _log_stage(stage: ResolutionStage) -> None:
    log.info(
        f"  G={stage.scene_grid_size}: batch={stage.batch_size}, fresh={stage.fresh_count}, "
        f"fresh_ratio={stage.fresh_ratio:.2f}"
    )


def warmup_normalizers(
    scene_normalizers: dict[int, PositionAwareNorm],
    cls_normalizers: dict[int, PositionAwareNorm],
    train_loaders: dict[int, InfiniteLoader],
    compute_raw_targets: Callable[[Tensor], NormFeatures],
    warmup_images: int,
    device: torch.device,
) -> None:
    """Warm up normalizer running stats before training begins.

    Each CLS normalizer is warmed up alongside its corresponding scene normalizer
    (CLS statistics depend on input resolution, which varies per grid size).
    """
    for G, scene_norm in scene_normalizers.items():
        cls_norm = cls_normalizers[G]
        scene_norm.train()
        cls_norm.train()
        loader = train_loaders[G]
        images_seen = 0
        pbar = tqdm(total=warmup_images, desc=f"Warmup G={G}", unit="img", leave=False)
        while images_seen < warmup_images:
            batch = loader.next_batch().to(device)
            with torch.no_grad():
                feats = compute_raw_targets(batch)
                scene_norm(feats.patches)
                cls_norm(feats.cls.unsqueeze(1))
            images_seen += batch.shape[0]
            pbar.update(batch.shape[0])
        pbar.close()
        log.info(
            f"Warmup G={G}: {images_seen} images, "
            f"scene mean_norm={scene_norm.mean.norm().item():.4f}, "
            f"cls mean_norm={cls_norm.mean.norm().item():.4f}"
        )


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

    model = create_model(student_backbone, teacher.embed_dim, cfg)

    if cfg.compile and cfg.model.gradient_checkpointing:
        raise ValueError(
            "compile=True and gradient_checkpointing=True may be incompatible. "
            "This combination has caused CheckpointError (tensor metadata mismatch) in some configurations. "
            "Disable one of them."
        )

    if cfg.compile:
        log.info("Compiling teacher and model")
        compile_teacher(teacher)
        compile_model(model)

    loss_fn = get_loss_fn(cfg.loss)
    log.info(f"Loss function: {cfg.loss}")

    # Create resolution stages and resources
    patch_size = teacher.patch_size_px
    stages = create_resolution_stages(cfg, patch_size)
    log.info("Resolution stages:")
    for stage in stages.values():
        _log_stage(stage)

    train_loaders, val_loaders = create_loaders(cfg, stages)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = count_parameters(model)
    log.info(f"Model total: {n_total:,}, trainable: {n_trainable:,} ({100 * n_trainable / n_total:.1f}%)")
    exp.log_parameters({"trainable_params": n_trainable, "total_params": n_total})

    peak_lr = get_linear_scaled_lr(cfg.ref_lr, cfg.batch_size)
    optimizer = torch.optim.AdamW(trainable, lr=peak_lr, weight_decay=cfg.weight_decay)
    scheduler = warmup_cosine_scheduler(optimizer, cfg.n_steps, cfg.warmup_steps)
    log.info(f"Optimizer: AdamW, peak_lr={peak_lr:.2e}, weight_decay={cfg.weight_decay:.2e}")

    # Load model weights from checkpoint if specified
    if cfg.resume_ckpt is not None:
        ckpt_data = load_checkpoint(cfg.resume_ckpt, cfg.device)
        ckpt_cfg = ActiveCanViTConfig(**ckpt_data["model_config"])
        if ckpt_cfg != cfg.model:
            log.warning("Checkpoint config differs from current config!")
            log.warning(f"  Checkpoint: {ckpt_cfg}")
            log.warning(f"  Current: {cfg.model}")
        model.load_state_dict(ckpt_data["state_dict"])

    ckpt_path = cfg.ckpt_dir / f"{exp.get_key()}.pt"

    def compute_raw_targets(images: Tensor) -> NormFeatures:
        """Compute teacher features (not normalized)."""
        with torch.autocast(device_type=cfg.device.type, dtype=torch.bfloat16):
            feats = teacher.forward_norm_features(images)
            return NormFeatures(patches=feats.patches.float(), cls=feats.cls.float())

    # Create normalizers for each grid size
    # Both scene and CLS normalizers are per-grid-size (CLS stats depend on input resolution)
    scene_normalizers: dict[int, PositionAwareNorm] = {}
    cls_normalizers: dict[int, PositionAwareNorm] = {}
    for G in cfg.grid_sizes:
        scene_normalizers[G] = PositionAwareNorm(
            n_tokens=G * G, embed_dim=teacher.embed_dim, grid_size=G, momentum=cfg.norm_momentum,
        ).to(cfg.device)
        cls_normalizers[G] = PositionAwareNorm(
            n_tokens=1, embed_dim=teacher.embed_dim, grid_size=1, momentum=cfg.norm_momentum,
        ).to(cfg.device)

    log.info(f"Warming up normalizers ({cfg.norm_warmup_images} images per grid size)...")
    warmup_normalizers(scene_normalizers, cls_normalizers, train_loaders, compute_raw_targets, cfg.norm_warmup_images, cfg.device)

    log.info("Initializing survival batches...")
    states: dict[int, SurvivalBatch] = {}
    for G, stage in stages.items():
        states[G] = init_survival_batch(
            model, train_loaders[G], compute_raw_targets, scene_normalizers[G], cls_normalizers[G],
            stage.batch_size, stage.fresh_count, G, cfg.device,
        )

    # EMA tracking for losses (scene, cls, total)
    ema_scene_t = torch.tensor(0.0, device=cfg.device)
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
        scene_norm = scene_normalizers[G]
        cls_norm = cls_normalizers[G]
        fresh_imgs = train_loader.next_batch().to(cfg.device)
        with torch.no_grad():
            fresh_feats = compute_raw_targets(fresh_imgs)
            fresh_patches_norm = scene_norm(fresh_feats.patches)
            fresh_cls_norm = cls_norm(fresh_feats.cls.unsqueeze(1)).squeeze(1)

        # Inner loop: multiple viewpoints per optimizer step
        viewpoints = [
            random_viewpoint(stage.batch_size, cfg.device)
            for _ in range(cfg.n_viewpoints_per_step)
        ]
        losses, final_canvas = model.forward_loss(
            state.images,
            viewpoints,
            state.targets,
            state.canvas,
            cls_target=state.cls_targets,
            loss_fn=loss_fn,
        )

        # Combine losses (trainer's responsibility)
        total_loss = torch.tensor(0.0, device=cfg.device)
        if losses.scene is not None:
            total_loss = total_loss + losses.scene
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
        scene_t = losses.scene.detach() if losses.scene is not None else torch.tensor(0.0, device=cfg.device)
        cls_t = losses.cls.detach() if losses.cls is not None else torch.tensor(0.0, device=cfg.device)
        total_t = total_loss.detach()
        if step > 0:
            ema_scene_t = alpha * scene_t + (1 - alpha) * ema_scene_t
            ema_cls_t = alpha * cls_t + (1 - alpha) * ema_cls_t
            ema_loss_t = alpha * total_t + (1 - alpha) * ema_loss_t
        else:
            ema_scene_t, ema_cls_t, ema_loss_t = scene_t, cls_t, total_t

        if step % cfg.log_every == 0:
            # Raw (instantaneous) values
            loss_raw = total_t.item()
            scene_raw = scene_t.item()
            cls_raw = cls_t.item()
            # EMA-smoothed values
            loss_ema = ema_loss_t.item()
            scene_ema = ema_scene_t.item()
            cls_ema = ema_cls_t.item()

            grad_norm = grad_norm_t.item()
            lr = scheduler.get_last_lr()[0]
            metrics = {
                # Raw losses
                f"grid{G}/train/loss": loss_raw,
                "train/loss": loss_raw,
                # EMA losses
                f"grid{G}/train/loss_ema": loss_ema,
                "train/loss_ema": loss_ema,
                # Other
                "train/grad_norm": grad_norm,
                "train/lr": lr,
                "train/grid_size": G,
                "train/spatial_canvas_init_norm": model.canvit.spatial_init.norm().item(),
                "train/cls_canvas_init_norm": model.canvit.cls_init.norm().item(),
            }
            if model.cls_proj is not None:
                cls_linear = model.cls_proj[1]
                assert isinstance(cls_linear, torch.nn.Linear)
                metrics["train/cls_proj_weight_norm"] = cls_linear.weight.norm().item()
            if losses.scene is not None:
                metrics[f"grid{G}/train/scene_loss"] = scene_raw
                metrics["train/scene_loss"] = scene_raw
                metrics[f"grid{G}/train/scene_loss_ema"] = scene_ema
                metrics["train/scene_loss_ema"] = scene_ema
            if losses.cls is not None:
                metrics[f"grid{G}/train/cls_loss"] = cls_raw
                metrics["train/cls_loss"] = cls_raw
                metrics[f"grid{G}/train/cls_loss_ema"] = cls_ema
                metrics["train/cls_loss_ema"] = cls_ema
            exp.log_metrics(metrics, step=step)
            pbar.set_postfix_str(f"G={G} loss={loss_ema:.2e} grad={grad_norm:.2e} lr={lr:.2e}")

        # Validation - always log gradient norms and report to Optuna at val_every
        if step % cfg.val_every == 0:
            for name, norm in grad_norms_by_module(model, depth=1).items():
                exp.log_metric(f"grad_norm/{name}", norm, step=step)

            # Fast validation (skip at viz_every - eval_and_log covers the same metrics)
            if step % cfg.viz_every != 0:
                val_images = val_loader.next_batch().to(cfg.device)
                val_scene_l1 = val_metrics_only(
                    exp, step, model, compute_raw_targets, scene_norm, cls_norm,
                    val_images, G, f"grid{G}/val",
                )
                exp.log_metric("val/scene_l1", val_scene_l1, step=step)

            if step > 0:
                trial.report(ema_loss_t.item(), step)
                if trial.should_prune():
                    exp.end()
                    raise optuna.TrialPruned()

        # Full viz with PCA (expensive, less frequent)
        if step % cfg.viz_every == 0:
            train_viz = viz_and_log(
                exp, step, f"grid{G}/train", model, teacher, scene_norm,
                state.images, viewpoints, state.targets, state.canvas,
                log_spatial_stats=cfg.log_spatial_stats, log_curves=False, loss_type=cfg.loss,
            )
            for name, losses in train_viz.losses.items():
                exp.log_metric(f"grid{G}/train/viz_{name}", losses[-1], step=step)

            val_images = val_loader.next_batch().to(cfg.device)
            val_scene_l1 = eval_and_log(
                exp, step, model, teacher, compute_raw_targets, scene_norm, cls_norm,
                val_images, G, f"grid{G}/val",
                log_spatial_stats=cfg.log_spatial_stats,
                log_curves=(step % cfg.curve_every == 0),
                loss_type=cfg.loss,
            )
            exp.log_metric("val/scene_l1", val_scene_l1, step=step)

        # Checkpointing
        if step % cfg.ckpt_every == 0:
            save_checkpoint(
                ckpt_path, model, cfg.student_model,
                step=step, train_loss=ema_loss_t.item(), comet_id=exp.get_key(),
            )
            log_norm_stats(exp, scene_normalizers, cls_normalizers, step)

        # Fresh ratio survival: permute batch, replace first K with fresh (normalized targets)
        canvas_init = model.init_canvas(stage.fresh_count, G)
        states[G] = state.step(
            fresh_images=fresh_imgs,
            fresh_targets=fresh_patches_norm,
            fresh_cls_targets=fresh_cls_norm,
            next_canvas=final_canvas,
            canvas_init=canvas_init,
        )

    # Final checkpoint
    save_checkpoint(
        ckpt_path, model, cfg.student_model,
        step=cfg.n_steps, train_loss=ema_loss_t.item(), comet_id=exp.get_key(),
    )

    log.info(f"Final: train_ema={ema_loss_t.item():.4f}")
    exp.end()
    return ema_loss_t.item()
