"""Main training loop."""

import logging
import traceback
from collections.abc import Callable
from contextlib import nullcontext
from typing import NamedTuple

import comet_ml
import dacite
import numpy as np
import optuna
import torch
from torch import Tensor, nn
from tqdm import tqdm


class TrainBatch(NamedTuple):
    """A training batch with precomputed targets (viewpoints sampled separately)."""
    images: Tensor
    labels: Tensor  # ImageNet class labels (for probe-based accuracy, if probe available)
    scene_target: Tensor  # Normalized teacher scene features
    cls_target: Tensor  # Normalized teacher CLS features

# Force FlashAttention for SDPA - fail loud if unavailable
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(False)
from ytch.model import count_parameters  # noqa: E402

from avp_vit import ActiveCanViTConfig  # noqa: E402
from avp_vit.checkpoint import CheckpointData, load as load_checkpoint  # noqa: E402
from avp_vit.checkpoint import save as save_checkpoint  # noqa: E402
from canvit.backbone.dinov3 import NormFeatures  # noqa: E402

from .config import Config  # noqa: E402
from .data import InfiniteLoader, create_loaders, scene_size_px  # noqa: E402
from .ema import EMATracker  # noqa: E402
from .model import compile_model, compile_teacher, create_model, load_student_backbone, load_teacher  # noqa: E402
from .norm import PositionAwareNorm  # noqa: E402
from .probe import load_probe  # noqa: E402
from .scheduler import warmup_cosine_scheduler  # noqa: E402
from .step import training_step  # noqa: E402
from .viewpoint import ViewpointType  # noqa: E402
from .viz import validate  # noqa: E402

log = logging.getLogger(__name__)


def log_spaced_steps(n: int, max_step: int, K: float | None = None) -> frozenset[int]:
    """Generate n steps from 0 to max_step with geometrically increasing gaps."""
    assert n > 1
    n_gaps = n - 1
    K = K if K is not None else float(n)
    r = K ** (1 / (n_gaps - 1))
    g1 = max_step * (r - 1) / (r**n_gaps - 1)
    gaps = g1 * (r ** np.arange(n_gaps))
    steps = np.concatenate([[0], np.cumsum(gaps)])
    return frozenset(int(round(s)) for s in steps)


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


def warmup_normalizer(
    scene_norm: PositionAwareNorm,
    cls_norm: PositionAwareNorm,
    train_loader: InfiniteLoader,
    compute_raw_targets: Callable[[Tensor, int], NormFeatures],
    warmup_images: int,
    scene_size: int,
    device: torch.device,
) -> None:
    """Warm up normalizer running stats."""
    scene_norm.train()
    cls_norm.train()
    images_seen = 0
    pbar = tqdm(total=warmup_images, desc="Warmup normalizers", unit="img", leave=False)
    while images_seen < warmup_images:
        batch = train_loader.next_batch().to(device)
        with torch.no_grad():
            feats = compute_raw_targets(batch, scene_size)
            scene_norm(feats.patches)
            cls_norm(feats.cls.unsqueeze(1))
        images_seen += batch.shape[0]
        pbar.update(batch.shape[0])
    pbar.close()
    log.info(f"Warmup done: {images_seen} images")


def train(cfg: Config, trial: optuna.Trial) -> float:
    """Train with stochastic reset. Returns best val_loss."""
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

    probe = load_probe(cfg.teacher_model, cfg.device)
    if probe is not None:
        log.info(f"Loaded IN1k probe for {cfg.teacher_model}")

    student_backbone = load_student_backbone(cfg)
    log.info(f"Student backbone params: {count_parameters(student_backbone):,}")

    bundle = create_model(student_backbone, teacher.embed_dim, cfg)
    model, glimpse_size_px = bundle.model, bundle.glimpse_size_px

    if cfg.compile and cfg.model.gradient_checkpointing:
        raise ValueError("compile=True and gradient_checkpointing=True may be incompatible.")

    if cfg.enable_policy and not cfg.model.enable_vpe:
        raise ValueError("enable_policy=True requires cfg.model.enable_vpe=True (VPE provides policy input)")

    if cfg.enable_policy:
        log.info("Policy training enabled - VPE token will be used for policy input")

    if cfg.compile:
        log.info("Compiling teacher and model")
        compile_teacher(teacher)
        compile_model(model)

    G = cfg.grid_size
    patch_size = teacher.patch_size_px
    scene_size = scene_size_px(G, patch_size)
    log.info(f"Grid size: {G}, scene size: {scene_size}px")

    train_loader, val_loader = create_loaders(cfg)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = count_parameters(model)
    log.info(f"Model total: {n_total:,}, trainable: {n_trainable:,} ({100 * n_trainable / n_total:.1f}%)")
    exp.log_parameters({"trainable_params": n_trainable, "total_params": n_total})

    optimizer = torch.optim.AdamW(trainable, lr=cfg.peak_lr, weight_decay=cfg.weight_decay)
    scheduler = warmup_cosine_scheduler(
        optimizer, cfg.n_steps, cfg.warmup_steps, cfg.peak_lr,
        start_lr=cfg.start_lr, end_lr=cfg.end_lr,
    )
    start_lr = cfg.start_lr if cfg.start_lr is not None else cfg.peak_lr / cfg.warmup_steps
    end_lr = cfg.end_lr if cfg.end_lr is not None else 0.0
    log.info(f"Optimizer: AdamW, lr={start_lr:.2e}→{cfg.peak_lr:.2e}→{end_lr:.2e}, wd={cfg.weight_decay:.2e}")

    # Generate viz/curve steps as multiples of val_every (so they actually trigger!)
    n_val_steps = cfg.n_steps // cfg.val_every
    viz_indices = log_spaced_steps(cfg.total_viz, n_val_steps)
    curve_indices = log_spaced_steps(cfg.total_curves, n_val_steps)
    viz_steps = frozenset(i * cfg.val_every for i in viz_indices)
    curve_steps = frozenset(i * cfg.val_every for i in curve_indices)
    log.info(f"Viz steps: {len(viz_steps)} points, curves: {len(curve_steps)} points")

    amp_ctx = (
        torch.autocast(device_type=cfg.device.type, dtype=torch.bfloat16)
        if cfg.amp else nullcontext()
    )
    log.info(f"AMP: {'bfloat16' if cfg.amp else 'disabled'}")

    ckpt_data: CheckpointData | None = None
    if cfg.resume_ckpt is not None:
        ckpt_data = load_checkpoint(cfg.resume_ckpt, cfg.device)
        ckpt_cfg = dacite.from_dict(ActiveCanViTConfig, ckpt_data["model_config"])
        if ckpt_cfg != cfg.model:
            log.warning("Checkpoint config differs from current config!")
            log.warning(f"  Checkpoint: {ckpt_cfg}")
            log.warning(f"  Current:    {cfg.model}")
        state_dict = ckpt_data["state_dict"]
        if cfg.reset_policy:
            n_before = len(state_dict)
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("policy.")}
            n_removed = n_before - len(state_dict)
            log.info(f"Reset policy: removed {n_removed} policy keys, will use fresh init")
        incompat = model.load_state_dict(state_dict, strict=False)
        if incompat.missing_keys:
            log.warning(f"Checkpoint missing keys (freshly initialized): {incompat.missing_keys}")
        if incompat.unexpected_keys:
            log.warning(f"Checkpoint has unexpected keys (ignored): {incompat.unexpected_keys}")

    ckpt_path = cfg.ckpt_dir / f"{exp.get_key()}.pt"

    def compute_raw_targets(images: Tensor, sz: int) -> NormFeatures:
        with amp_ctx:
            if images.shape[-1] != sz:
                images = torch.nn.functional.interpolate(
                    images, size=(sz, sz), mode="bilinear", align_corners=False
                )
            feats = teacher.forward_norm_features(images)
            return NormFeatures(patches=feats.patches.float(), cls=feats.cls.float())

    scene_norm = PositionAwareNorm(
        n_tokens=G * G, embed_dim=teacher.embed_dim, grid_size=G, momentum=cfg.norm_momentum,
    ).to(cfg.device)
    cls_norm = PositionAwareNorm(
        n_tokens=1, embed_dim=teacher.embed_dim, grid_size=1, momentum=cfg.norm_momentum,
    ).to(cfg.device)

    if ckpt_data is not None and ckpt_data.get("scene_norm_state") is not None:
        scene_norm_state = ckpt_data["scene_norm_state"]
        cls_norm_state = ckpt_data["cls_norm_state"]
        assert scene_norm_state is not None and cls_norm_state is not None
        scene_norm.load_state_dict(scene_norm_state)
        cls_norm.load_state_dict(cls_norm_state)
        log.info("Loaded normalizer states from checkpoint")
    else:
        log.info(f"Warming up normalizers ({cfg.norm_warmup_images} images)...")
        warmup_normalizer(scene_norm, cls_norm, train_loader, compute_raw_targets, cfg.norm_warmup_images, scene_size, cfg.device)

    # Build viewpoint type lists for branching
    t0_types = [ViewpointType.RANDOM, ViewpointType.FULL]
    t1_types = [ViewpointType.RANDOM, ViewpointType.FULL]
    if cfg.enable_policy:
        t1_types.append(ViewpointType.POLICY)
    log.info(f"Training branches: t0={[t.name for t in t0_types]} × t1={[t.name for t in t1_types]}")

    # EMA tracking for all metrics
    ema = EMATracker(alpha=cfg.ema_alpha)

    def load_train_batch() -> TrainBatch:
        """Load training batch and compute normalized teacher targets."""
        images, labels = train_loader.next_batch_with_labels()
        images = images.to(cfg.device)
        labels = labels.to(cfg.device)
        with torch.no_grad():
            feats = compute_raw_targets(images, scene_size)
            scene_target = scene_norm(feats.patches)
            cls_target = cls_norm(feats.cls.unsqueeze(1)).squeeze(1)
        return TrainBatch(
            images=images,
            labels=labels,
            scene_target=scene_target,
            cls_target=cls_target,
        )

    # Step semantics: step S = model state after S gradient updates
    # step=0: before any gradient (initial model)
    # step=n_steps: after all n_steps gradient updates (final model)
    log.info("Starting training loop...")
    pbar = tqdm(range(cfg.n_steps + 1), desc="Training", unit="step")

    for step in pbar:
        batch: TrainBatch | None = None

        # === VALIDATION/VIZ PHASE (state after `step` gradient updates) ===
        if step % cfg.val_every == 0:
            do_curves = step in curve_steps
            do_pca = step in viz_steps

            # Validation on val batch (always at val_every)
            val_images, val_labels = val_loader.next_batch_with_labels()
            val_images = val_images.to(cfg.device)
            val_labels = val_labels.to(cfg.device) if probe is not None else None
            try:
                with amp_ctx:
                    validate(
                        exp=exp,
                        step=step,
                        model=model,
                        compute_raw_targets=compute_raw_targets,
                        scene_normalizer=scene_norm,
                        cls_normalizer=cls_norm,
                        images=val_images,
                        canvas_grid_size=G,
                        scene_size_px=scene_size,
                        glimpse_size_px=glimpse_size_px,
                        n_eval_viewpoints=cfg.n_eval_viewpoints,
                        min_viewpoint_scale=cfg.min_viewpoint_scale,
                        prefix="val",
                        probe=probe,
                        labels=val_labels,
                        log_curves=do_curves,
                        log_pca=do_pca,
                        teacher=teacher,
                        log_spatial_stats=cfg.log_spatial_stats,
                        backbone=cfg.teacher_model,
                    )
            except Exception:
                log.error(f"!!! VALIDATION FAILED at step {step} !!!\n{traceback.format_exc()}")

        # === CHECKPOINT PHASE ===
        if step % cfg.ckpt_every == 0:
            ema_loss = ema.get("total_loss")
            save_checkpoint(
                ckpt_path, model, cfg.student_model,
                step=step, train_loss=ema_loss.item() if ema_loss is not None else None,
                comet_id=exp.get_key(),
                scene_norm_state=scene_norm.state_dict(),
                cls_norm_state=cls_norm.state_dict(),
            )
            exp.log_metric("norm/scene_mean_norm", scene_norm.mean.norm().item(), step=step)
            exp.log_metric("norm/cls_mean_norm", cls_norm.mean.norm().item(), step=step)

        # === TRAINING PHASE (only for step < n_steps) ===
        if step < cfg.n_steps:
            if batch is None:
                batch = load_train_batch()

            optimizer.zero_grad()

            step_metrics = training_step(
                model=model,
                images=batch.images,
                scene_target=batch.scene_target,
                cls_target=batch.cls_target,
                glimpse_size_px=glimpse_size_px,
                canvas_grid_size=G,
                t0_types=t0_types,
                t1_types=t1_types,
                min_viewpoint_scale=cfg.min_viewpoint_scale,
                compute_gram=cfg.gram_loss_weight > 0,
                gram_loss_weight=cfg.gram_loss_weight,
                amp_ctx=amp_ctx,
            )

            grad_norm_t = torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            # Update EMA for all metrics
            ema.update("total_loss", step_metrics.total_loss)
            for (t0, t1), m in step_metrics.branches.items():
                prefix = f"{t0.name.lower()}_{t1.name.lower()}"
                ema.update(f"{prefix}/loss", m.loss)
                ema.update(f"{prefix}/scene_loss", m.scene_loss)
                ema.update(f"{prefix}/cls_loss", m.cls_loss)
                if m.gram_loss is not None:
                    ema.update(f"{prefix}/gram_loss", m.gram_loss)
                ema.update(f"{prefix}/scene_cos", m.scene_cos)
                ema.update(f"{prefix}/cls_cos", m.cls_cos)

            if step % cfg.log_every == 0:
                grad_norm = grad_norm_t.item()
                lr = scheduler.get_last_lr()[0]

                # Log all EMA metrics
                metrics = {f"train/{k}": v.item() for k, v in ema.items()}
                metrics["train/lr"] = lr
                metrics["train/grad_norm"] = grad_norm
                exp.log_metrics(metrics, step=step)

                ema_loss = ema.get("total_loss")
                assert ema_loss is not None
                pbar.set_postfix_str(f"loss={ema_loss.item():.2e} grad={grad_norm:.2e} lr={lr:.2e}")

            # Per-module grad norms (at val intervals, after training)
            if step % cfg.val_every == 0:
                for name, norm in grad_norms_by_module(model, depth=1).items():
                    exp.log_metric(f"grad_norm/{name}", norm, step=step)

            # Optuna pruning (skip step 0 - EMA not meaningful yet)
            if step > 0 and step % cfg.val_every == 0:
                ema_loss = ema.get("total_loss")
                assert ema_loss is not None
                trial.report(ema_loss.item(), step)
                if trial.should_prune():
                    exp.end()
                    raise optuna.TrialPruned()

    # Final checkpoint (if not already saved at step=n_steps)
    if cfg.n_steps % cfg.ckpt_every != 0:
        ema_loss = ema.get("total_loss")
        save_checkpoint(
            ckpt_path, model, cfg.student_model,
            step=cfg.n_steps, train_loss=ema_loss.item() if ema_loss is not None else None,
            comet_id=exp.get_key(),
            scene_norm_state=scene_norm.state_dict(),
            cls_norm_state=cls_norm.state_dict(),
        )

    ema_loss = ema.get("total_loss")
    final_loss = ema_loss.item() if ema_loss is not None else float("inf")
    log.info(f"Final: train_ema={final_loss:.4f}")
    exp.end()
    return final_loss
