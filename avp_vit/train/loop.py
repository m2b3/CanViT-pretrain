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
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm import tqdm


class TrainBatch(NamedTuple):
    """A training batch with precomputed targets."""
    images: Tensor
    labels: Tensor  # ImageNet class labels (for probe-based accuracy, if probe available)
    scene_target: Tensor  # Normalized teacher scene features
    cls_target: Tensor  # Normalized teacher CLS features
    canvas: Tensor  # Initial canvas state
    viewpoints: list  # Sampled viewpoints for this step

# Force FlashAttention for SDPA - fail loud if unavailable
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(False)
from ytch.model import count_parameters  # noqa: E402

from avp_vit import ActiveCanViTConfig  # noqa: E402
from avp_vit.checkpoint import load as load_checkpoint  # noqa: E402
from avp_vit.checkpoint import save as save_checkpoint  # noqa: E402
from canvit.backbone.dinov3 import NormFeatures  # noqa: E402

from .config import Config  # noqa: E402
from .data import InfiniteLoader, create_loaders, scene_size_px  # noqa: E402
from .model import compile_model, compile_teacher, create_model, load_student_backbone, load_teacher  # noqa: E402
from .norm import PositionAwareNorm  # noqa: E402
from .probe import compute_in1k_top1, labels_are_in1k, load_probe  # noqa: E402
from .scheduler import warmup_cosine_scheduler  # noqa: E402
from .viewpoint import random_viewpoint  # noqa: E402
from .viz import validate, viz_and_log  # noqa: E402

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
    scheduler = warmup_cosine_scheduler(optimizer, cfg.n_steps, cfg.warmup_steps)
    log.info(f"Optimizer: AdamW, peak_lr={cfg.peak_lr:.2e}, weight_decay={cfg.weight_decay:.2e}")

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

    if cfg.resume_ckpt is not None:
        ckpt_data = load_checkpoint(cfg.resume_ckpt, cfg.device)
        ckpt_cfg = dacite.from_dict(ActiveCanViTConfig, ckpt_data["model_config"])
        if ckpt_cfg != cfg.model:
            log.warning("Checkpoint config differs from current config!")
        model.load_state_dict(ckpt_data["state_dict"])

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

    log.info(f"Warming up normalizers ({cfg.norm_warmup_images} images)...")
    warmup_normalizer(scene_norm, cls_norm, train_loader, compute_raw_targets, cfg.norm_warmup_images, scene_size, cfg.device)

    # EMA tracking
    alpha = 2 / (cfg.log_every + 1)
    ema_scene: Tensor | None = None
    ema_cls: Tensor | None = None
    ema_gram: Tensor | None = None
    ema_loss: Tensor | None = None

    def ema_update(ema: Tensor | None, val: Tensor) -> Tensor:
        v = val.detach()
        return v if ema is None else alpha * v + (1 - alpha) * ema

    def load_train_batch() -> TrainBatch:
        """Load training batch and compute normalized teacher targets."""
        images, labels = train_loader.next_batch_with_labels()
        images = images.to(cfg.device)
        labels = labels.to(cfg.device)
        with torch.no_grad():
            feats = compute_raw_targets(images, scene_size)
            scene_target = scene_norm(feats.patches)
            cls_target = cls_norm(feats.cls.unsqueeze(1)).squeeze(1)
        canvas = model.init_canvas(batch_size=cfg.batch_size, canvas_grid_size=G)
        viewpoints = [random_viewpoint(cfg.batch_size, cfg.device, min_scale=cfg.min_viewpoint_scale) for _ in range(cfg.n_viewpoints_per_step)]
        return TrainBatch(
            images=images,
            labels=labels,
            scene_target=scene_target,
            cls_target=cls_target,
            canvas=canvas,
            viewpoints=viewpoints,
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

            # Train PCA viz (only at viz_steps, uses train batch)
            if do_pca:
                batch = load_train_batch()
                try:
                    with amp_ctx:
                        viz_and_log(
                            exp=exp,
                            step=step,
                            prefix="train",
                            model=model,
                            teacher=teacher,
                            normalizer=scene_norm,
                            images=batch.images,
                            viewpoints=batch.viewpoints,
                            target=batch.scene_target,
                            canvas=batch.canvas,
                            glimpse_size_px=glimpse_size_px,
                            log_spatial_stats=cfg.log_spatial_stats,
                            log_curves=False,
                        )
                except Exception:
                    log.error(f"!!! VIZ FAILED at step {step} !!!\n{traceback.format_exc()}")

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

            with amp_ctx:
                result = model.forward_loss(
                    image=batch.images,
                    viewpoints=batch.viewpoints,  # pyright: ignore[reportArgumentType] - Viewpoint subclass
                    spatial_target=batch.scene_target,
                    cls_target=batch.cls_target,
                    glimpse_size_px=glimpse_size_px,
                    canvas_grid_size=G,
                    canvas=batch.canvas,
                    compute_gram=cfg.gram_loss_weight > 0,
                    include_init=cfg.include_init,
                )
                losses = result.losses
                final_canvas = result.canvas

                total_loss = losses.scene + losses.cls
                if losses.gram is not None:
                    total_loss = total_loss + cfg.gram_loss_weight * losses.gram

                if not torch.isfinite(total_loss):
                    log.warning(f"NaN/Inf loss at step {step}, pruning trial")
                    exp.end()
                    raise optuna.TrialPruned()

                # Compute cos_sim only when logging
                scene_cos_sim: float | None = None
                cls_cos_sim: float | None = None
                if step % cfg.log_every == 0:
                    with torch.no_grad():
                        scene_pred = model.predict_teacher_scene(final_canvas)
                        scene_cos_sim = F.cosine_similarity(scene_pred, batch.scene_target, dim=-1).mean().item()
                        if model.cls_proj is not None:
                            cls_pred = model.predict_teacher_cls(result.cls, result.canvas)
                            cls_cos_sim = F.cosine_similarity(cls_pred, batch.cls_target, dim=-1).mean().item()

                optimizer.zero_grad()
                total_loss.backward()

            grad_norm_t = torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            # Update EMAs
            ema_scene = ema_update(ema_scene, losses.scene)
            ema_cls = ema_update(ema_cls, losses.cls)
            if losses.gram is not None:
                ema_gram = ema_update(ema_gram, losses.gram)
            ema_loss = ema_update(ema_loss, total_loss)

            # Train metrics logging
            if step % cfg.log_every == 0:
                assert ema_loss is not None
                grad_norm = grad_norm_t.item()
                lr = scheduler.get_last_lr()[0]
                metrics = {
                    "train/loss": total_loss.item(),
                    "train/loss_ema": ema_loss.item(),
                    "train/grad_norm": grad_norm,
                    "train/lr": lr,
                }
                if losses.gram is not None:
                    assert ema_gram is not None
                    metrics["train/gram_mse"] = losses.gram.item()
                    metrics["train/gram_mse_ema"] = ema_gram.item()
                assert ema_scene is not None and ema_cls is not None
                metrics["train/scene_loss"] = losses.scene.item()
                metrics["train/scene_loss_ema"] = ema_scene.item()
                metrics["train/cls_loss"] = losses.cls.item()
                metrics["train/cls_loss_ema"] = ema_cls.item()
                assert scene_cos_sim is not None
                metrics["train/scene_cos_sim"] = scene_cos_sim
                if cls_cos_sim is not None:
                    metrics["train/cls_cos_sim"] = cls_cos_sim
                # Train IN1k accuracy (TTS = Teacher-to-Student probe)
                # Skip for IN21k training (labels >= 1000)
                if probe is not None and model.cls_proj is not None and labels_are_in1k(batch.labels):
                    with torch.no_grad():
                        cls_pred = model.predict_teacher_cls(result.cls, result.canvas)
                        cls_raw = cls_norm.denormalize(cls_pred)
                        logits = probe(cls_raw)
                        train_in1k = compute_in1k_top1(logits, batch.labels)
                    metrics["train/in1k_tts_top1"] = train_in1k
                exp.log_metrics(metrics, step=step)
                pbar.set_postfix_str(f"loss={ema_loss.item():.2e} grad={grad_norm:.2e} lr={lr:.2e}")

            # Per-module grad norms (at val intervals, after training)
            if step % cfg.val_every == 0:
                for name, norm in grad_norms_by_module(model, depth=1).items():
                    exp.log_metric(f"grad_norm/{name}", norm, step=step)

            # Optuna pruning (skip step 0 - EMA not meaningful yet)
            if step > 0 and step % cfg.val_every == 0:
                assert ema_loss is not None
                trial.report(ema_loss.item(), step)
                if trial.should_prune():
                    exp.end()
                    raise optuna.TrialPruned()

    # Final checkpoint (if not already saved at step=n_steps)
    if cfg.n_steps % cfg.ckpt_every != 0:
        assert ema_loss is not None
        save_checkpoint(
            ckpt_path, model, cfg.student_model,
            step=cfg.n_steps, train_loss=ema_loss.item(), comet_id=exp.get_key(),
            scene_norm_state=scene_norm.state_dict(),
            cls_norm_state=cls_norm.state_dict(),
        )

    assert ema_loss is not None
    log.info(f"Final: train_ema={ema_loss.item():.4f}")
    exp.end()
    return ema_loss.item()
