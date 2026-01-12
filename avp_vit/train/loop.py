"""Main training loop."""

import logging
import os
import signal
import traceback
from datetime import datetime, timezone
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
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
    raw_scene_target: Tensor  # Raw teacher scene features (for metrics)
    raw_cls_target: Tensor  # Raw teacher CLS features (for metrics)

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
from .data import create_loaders, scene_size_px  # noqa: E402
from .ema import EMATracker  # noqa: E402
from .model import compile_model, compile_teacher, create_model, load_student_backbone, load_teacher  # noqa: E402
from .norm import PositionAwareNorm  # noqa: E402
from .probe import load_probe  # noqa: E402
from .scheduler import warmup_cosine_scheduler  # noqa: E402
from .step import TeacherTargets, training_step  # noqa: E402
from .viz import log_figure, plot_multistep_pca, validate  # noqa: E402

log = logging.getLogger(__name__)

# Signal-triggered checkpoint
_checkpoint_requested = False


def _handle_sigusr1(signum: int, frame: object) -> None:
    global _checkpoint_requested
    _checkpoint_requested = True
    log.info("SIGUSR1 received - will save checkpoint after current step")


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


def init_normalizer_stats_from_shard(
    shard_path: Path,
    scene_norm: PositionAwareNorm,
    cls_norm: PositionAwareNorm,
    device: torch.device,
) -> None:
    """Initialize normalizer stats from one precomputed shard."""
    log.info(f"Computing normalizer stats from shard: {shard_path.name}")
    shard = torch.load(shard_path, map_location=device, weights_only=False)
    patches = shard["patches"].float()  # [N, n_tokens, D]
    cls = shard["cls"].float()  # [N, D]
    scene_norm.set_stats(patches)
    cls_norm.set_stats(cls.unsqueeze(1))  # [N, 1, D] for n_tokens=1
    log.info(f"  Scene/CLS stats from {patches.shape[0]} samples")
    del shard
    torch.cuda.empty_cache()


def train(cfg: Config, trial: optuna.Trial) -> float:
    """Train with stochastic reset. Returns best val_loss."""
    signal.signal(signal.SIGUSR1, _handle_sigusr1)
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
    if slurm_job_id := os.environ.get("SLURM_JOB_ID"):
        exp.log_parameters({"slurm_job_id": slurm_job_id})

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
    has_features = cfg.feature_base_dir is not None

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
    log.info(f"Non-blocking transfers: {'enabled' if cfg.non_blocking_transfer else 'DISABLED (sync)'}")

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

        # Filter out shape-incompatible weights (e.g., when changing canvas_num_heads)
        model_state = model.state_dict()
        mismatched_params = 0
        for k, v in list(state_dict.items()):
            if k in model_state and v.shape != model_state[k].shape:
                mismatched_params += model_state[k].numel()
                del state_dict[k]
        if mismatched_params > 0:
            total_params = sum(p.numel() for p in model.parameters())
            pct = 100 * mismatched_params / total_params
            log.warning(f"Shape mismatch: {mismatched_params:,} params ({pct:.1f}%) freshly initialized")

        incompat = model.load_state_dict(state_dict, strict=False)
        if incompat.missing_keys:
            log.warning(f"Checkpoint missing keys (freshly initialized): {incompat.missing_keys}")
        if incompat.unexpected_keys:
            log.warning(f"Checkpoint has unexpected keys (ignored): {incompat.unexpected_keys}")

        # Load optimizer + scheduler state (tied together)
        if cfg.reset_opt_and_sched:
            log.info("Reset optimizer+scheduler: using fresh state")
        else:
            opt_state = ckpt_data.get("optimizer_state")
            sched_state = ckpt_data.get("scheduler_state")
            if opt_state is not None and sched_state is not None:
                optimizer.load_state_dict(opt_state)
                scheduler.load_state_dict(sched_state)
                log.info(f"Loaded optimizer+scheduler from checkpoint (step={sched_state.get('last_epoch', '?')})")
            elif opt_state is not None or sched_state is not None:
                log.warning("Checkpoint has only one of optimizer/scheduler - using fresh init for both")
            else:
                log.warning("Checkpoint has no optimizer/scheduler state, using fresh init")

    # Build training config history (tracks config across resumes)
    training_config_history: dict[str, dict] = {}
    if ckpt_data is not None:
        training_config_history = ckpt_data.get("training_config_history") or {}
    training_config_history[datetime.now(timezone.utc).isoformat()] = flatten_dict(asdict(cfg))

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
        n_tokens=G * G, embed_dim=teacher.embed_dim, grid_size=G,
    ).to(cfg.device)
    cls_norm = PositionAwareNorm(
        n_tokens=1, embed_dim=teacher.embed_dim, grid_size=1,
    ).to(cfg.device)

    any_glimpse_loss = cfg.enable_glimpse_patches_loss or cfg.enable_glimpse_cls_loss

    norm_loaded = False
    if ckpt_data is not None and not cfg.reset_normalizer:
        scene_norm_state = ckpt_data.get("scene_norm_state")
        cls_norm_state = ckpt_data.get("cls_norm_state")
        if scene_norm_state is not None and cls_norm_state is not None:
            scene_norm.load_state_dict(scene_norm_state)
            cls_norm.load_state_dict(cls_norm_state)
            norm_loaded = True
            log.info("Loaded scene/cls normalizer states from checkpoint")
        else:
            log.warning("Checkpoint has no normalizer states")
    elif cfg.reset_normalizer:
        log.info("Reset normalizer: will re-init stats")

    if not norm_loaded:
        assert cfg.feature_base_dir is not None, "feature_base_dir required for normalizer init"
        shards_dir = cfg.feature_base_dir / cfg.teacher_model / str(cfg.image_resolution) / "shards"
        shard_files = sorted(shards_dir.glob("*.pt"))
        assert shard_files, f"No shards in {shards_dir}"
        init_normalizer_stats_from_shard(shard_files[0], scene_norm, cls_norm, cfg.device)

    log.info(f"Training: {cfg.n_full_start_branches} full + {cfg.n_random_start_branches} random branches, chunk_size={cfg.chunk_size}, continue_prob={cfg.continue_prob}")
    log.info(f"Scene loss type: {cfg.scene_loss_type.value}")

    # EMA tracking for all metrics
    ema = EMATracker(alpha=cfg.ema_alpha)

    nb = cfg.non_blocking_transfer  # Ablation flag for async transfers

    def load_train_batch() -> TrainBatch:
        """Load training batch and compute/load normalized scene targets."""
        # non_blocking=True: CPU returns immediately, GPU ops serialize on same stream
        # Safe because we don't mutate source tensors after transfer
        if has_features:
            images, raw_patches, raw_cls, labels = train_loader.next()
            images = images.to(cfg.device, non_blocking=nb)
            labels = labels.to(cfg.device, non_blocking=nb)
            # .float() for consistency - stored features may be fp16
            raw_patches = raw_patches.to(device=cfg.device, dtype=torch.float32, non_blocking=nb)
            raw_cls = raw_cls.to(device=cfg.device, dtype=torch.float32, non_blocking=nb)
        else:
            images, labels = train_loader.next_batch_with_labels()
            images = images.to(cfg.device, non_blocking=nb)
            labels = labels.to(cfg.device, non_blocking=nb)
            raw = compute_raw_targets(images, scene_size)
            raw_patches, raw_cls = raw.patches, raw.cls
        norm_patches = scene_norm(raw_patches)
        norm_cls = cls_norm(raw_cls.unsqueeze(1)).squeeze(1)
        return TrainBatch(images, labels, norm_patches, norm_cls, raw_patches, raw_cls)

    # Glimpse targets: raw features (cosine similarity loss, no normalization)
    if any_glimpse_loss:
        def _compute_glimpse_targets(glimpse: Tensor) -> TeacherTargets:
            with torch.no_grad():
                feats = compute_raw_targets(glimpse, glimpse_size_px)
            return TeacherTargets(patches=feats.patches, cls=feats.cls)
        compute_glimpse_targets_fn: Callable[[Tensor], TeacherTargets] | None = _compute_glimpse_targets
    else:
        compute_glimpse_targets_fn = None

    # Step semantics: step S = model state after S gradient updates
    # step=0: before any gradient (initial model)
    # step=n_steps: after all n_steps gradient updates (final model)
    log.info("Starting training loop...")
    model.train()  # Explicit: validate() restores, but be clear about initial state
    pbar = tqdm(range(cfg.n_steps + 1), desc="Training", unit="step")

    for step in pbar:
        batch: TrainBatch | None = None
        do_pca = step in viz_steps  # Used by both validation and training viz

        # === VALIDATION/VIZ PHASE (state after `step` gradient updates) ===
        if step % cfg.val_every == 0:
            do_curves = step in curve_steps

            # Validation on val batch (always at val_every)
            val_images, val_labels = val_loader.next_batch_with_labels()
            val_images = val_images.to(cfg.device, non_blocking=nb)
            val_labels = val_labels.to(cfg.device, non_blocking=nb) if probe is not None else None
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
        global _checkpoint_requested
        if step % cfg.ckpt_every == 0 or _checkpoint_requested:
            if _checkpoint_requested:
                log.info(f"Saving signal-triggered checkpoint at step {step}")
                _checkpoint_requested = False
            ema_loss = ema.get("total_loss")
            save_checkpoint(
                ckpt_path, model, cfg.student_model,
                step=step, train_loss=ema_loss.item() if ema_loss is not None else None,
                comet_id=exp.get_key(),
                scene_norm_state=scene_norm.state_dict(),
                cls_norm_state=cls_norm.state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                training_config_history=training_config_history,
            )
            exp.log_metric("norm/scene_mean_norm", scene_norm.mean.norm().item(), step=step)
            exp.log_metric("norm/cls_mean_norm", cls_norm.mean.norm().item(), step=step)

        # === TRAINING PHASE (only for step < n_steps) ===
        if step < cfg.n_steps:
            if batch is None:
                batch = load_train_batch()

            optimizer.zero_grad()

            # Warmup continue_prob: 0 → peak over warmup steps
            if cfg.continue_prob_warmup_steps > 0:
                continue_prob = cfg.continue_prob * min(step / cfg.continue_prob_warmup_steps, 1.0)
            else:
                continue_prob = cfg.continue_prob

            step_metrics = training_step(
                model=model,
                images=batch.images,
                scene_target=batch.scene_target,
                cls_target=batch.cls_target,
                raw_scene_target=batch.raw_scene_target,
                raw_cls_target=batch.raw_cls_target,
                scene_denorm=scene_norm.denormalize,
                cls_denorm=cls_norm.denormalize,
                compute_glimpse_targets=compute_glimpse_targets_fn,
                enable_scene_patches_loss=cfg.enable_scene_patches_loss,
                enable_scene_cls_loss=cfg.enable_scene_cls_loss,
                enable_glimpse_patches_loss=cfg.enable_glimpse_patches_loss,
                enable_glimpse_cls_loss=cfg.enable_glimpse_cls_loss,
                scene_loss_type=cfg.scene_loss_type,
                glimpse_size_px=glimpse_size_px,
                canvas_grid_size=G,
                n_full_start_branches=cfg.n_full_start_branches,
                n_random_start_branches=cfg.n_random_start_branches,
                chunk_size=cfg.chunk_size,
                continue_prob=continue_prob,
                min_viewpoint_scale=cfg.min_viewpoint_scale,
                amp_ctx=amp_ctx,
                use_checkpointing=cfg.use_checkpointing,
                collect_viz=do_pca,
            )

            # Clip policy grads first (if present), then whole model
            if model.policy is not None:
                torch.nn.utils.clip_grad_norm_(model.policy.parameters(), cfg.policy_grad_clip)
            grad_norm_t = torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            # Update EMA for all metrics
            ema.update("total_loss", step_metrics.total_loss)
            ema.update("n_glimpses", torch.tensor(step_metrics.n_glimpses, dtype=torch.float32))
            for prefix, m in [("full", step_metrics.full_start), ("random", step_metrics.random_start)]:
                if m is None:
                    continue
                ema.update(f"{prefix}/loss", m.loss)
                if cfg.enable_scene_patches_loss:
                    ema.update(f"{prefix}/scene_patches_loss", m.scene_patches_loss)
                if cfg.enable_scene_cls_loss:
                    ema.update(f"{prefix}/scene_cls_loss", m.scene_cls_loss)
                if cfg.enable_glimpse_patches_loss:
                    ema.update(f"{prefix}/glimpse_patches_loss", m.glimpse_patches_loss)
                if cfg.enable_glimpse_cls_loss:
                    ema.update(f"{prefix}/glimpse_cls_loss", m.glimpse_cls_loss)
                ema.update(f"{prefix}/scene_cos_raw", m.scene_cos_raw)
                ema.update(f"{prefix}/scene_cos_norm", m.scene_cos_norm)
                ema.update(f"{prefix}/cls_cos_raw", m.cls_cos_raw)
                ema.update(f"{prefix}/cls_cos_norm", m.cls_cos_norm)

            if step % cfg.log_every == 0:
                grad_norm = grad_norm_t.item()
                lr = scheduler.get_last_lr()[0]

                # Log all EMA metrics
                metrics = {f"train/{k}": v.item() for k, v in ema.items()}
                metrics["train/lr"] = lr
                metrics["train/grad_norm"] = grad_norm
                metrics["train/continue_prob"] = continue_prob
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

            # Training batch PCA visualization (same data as training, no recomputation)
            if step_metrics.viz_data is not None:
                vd = step_metrics.viz_data
                assert vd.image.ndim == 3 and vd.image.shape[2] == 3, f"Expected [H,W,3], got {vd.image.shape}"
                H, W = vd.image.shape[:2]
                boxes = [vp.to_pixel_box(0, H, W) for vp in vd.viewpoints]
                names = [vp.name for vp in vd.viewpoints]
                scenes = [vs.predicted_scene for vs in vd.viz_samples]
                glimpses = [vs.glimpse for vs in vd.viz_samples]
                canvas_spatials = [vs.canvas_spatial for vs in vd.viz_samples]
                assert vd.initial_scene is not None
                fig = plot_multistep_pca(
                    full_img=vd.image,
                    teacher=vd.teacher_features,
                    scenes=scenes,
                    glimpses=glimpses,
                    boxes=boxes,
                    names=names,
                    scene_grid_size=G,
                    glimpse_grid_size=cfg.glimpse_grid_size,
                    initial_scene=vd.initial_scene,
                    hidden_spatials=canvas_spatials if canvas_spatials[0] is not None else None,
                    initial_hidden_spatial=vd.initial_canvas_spatial,
                )
                log_figure(exp, fig, "train/pca", step)

    # Final checkpoint (if not already saved at step=n_steps)
    if cfg.n_steps % cfg.ckpt_every != 0:
        ema_loss = ema.get("total_loss")
        save_checkpoint(
            ckpt_path, model, cfg.student_model,
            step=cfg.n_steps, train_loss=ema_loss.item() if ema_loss is not None else None,
            comet_id=exp.get_key(),
            scene_norm_state=scene_norm.state_dict(),
            cls_norm_state=cls_norm.state_dict(),
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
            training_config_history=training_config_history,
        )

    ema_loss = ema.get("total_loss")
    final_loss = ema_loss.item() if ema_loss is not None else float("inf")
    log.info(f"Final: train_ema={final_loss:.4f}")
    exp.end()
    return final_loss
