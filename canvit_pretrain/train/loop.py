"""Main training loop."""

# backward() runs outside autocast (as PyTorch recommends). torch.compile's default
# "same_as_forward" assumption silently corrupts gradients when that's the case.
import torch._functorch.config
torch._functorch.config.backward_pass_autocast = "off"  # type: ignore[attr-defined]

import logging
import os
import signal
import subprocess
import time
import traceback
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import comet_ml
import dacite
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
from canvit import CLSStandardizer, PatchStandardizer  # noqa: E402
from canvit.backbone.vit import NormFeatures  # noqa: E402
from ytch.model import count_parameters  # noqa: E402

from canvit_pretrain import CanViTForPretrainingConfig  # noqa: E402
from canvit_pretrain.checkpoint import CheckpointData, current_provenance, find_latest, update_symlink  # noqa: E402
from canvit_pretrain.checkpoint import load as load_checkpoint  # noqa: E402
from canvit_pretrain.checkpoint import save as save_checkpoint  # noqa: E402

from .config import Config  # noqa: E402
from .data import ShardedFeatureLoader, create_loaders, scene_size_px  # noqa: E402
from .ema import EMATracker  # noqa: E402
from .model import compile_model, compile_teacher, create_model, load_student_backbone, load_teacher  # noqa: E402
from .probe import load_probe  # noqa: E402
from .scheduler import warmup_constant_scheduler  # noqa: E402
from .step import training_step  # noqa: E402
from .viz import log_figure, plot_multistep_pca, validate  # noqa: E402

log = logging.getLogger(__name__)

# Signal-triggered checkpoint
_checkpoint_requested = False


def _handle_sigusr1(signum: int, frame: object) -> None:
    global _checkpoint_requested
    _checkpoint_requested = True
    log.info("SIGUSR1 received - will save checkpoint after current step")


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
    scene_norm: PatchStandardizer,
    cls_norm: CLSStandardizer,
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

    # === RUN NAME AND CHECKPOINT RESOLUTION ===
    run_name = cfg.run_name
    if run_name is None:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_dir = cfg.ckpt_dir / run_name
    log.info(f"Run: {run_name} (dir: {run_dir})")

    # Check for failure marker (prevents infinite crash loops in job arrays)
    failed_marker = run_dir / "FAILED"
    if failed_marker.exists():
        log.error(f"FAILED marker exists: {failed_marker}")
        log.error(f"Previous job crashed. Delete marker to retry: rm {failed_marker}")
        cancel_slurm_array()
        raise RuntimeError(f"Refusing to start: {failed_marker} exists")

    # Create run_dir early so we can write FAILED marker on crash
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        return training_loop(cfg=cfg, trial=trial, run_name=run_name, run_dir=run_dir)
    except Exception:
        log.exception("Training crashed - writing FAILED marker")
        failed_marker.write_text(f"Crashed at {datetime.now(UTC).isoformat()}\n")
        cancel_slurm_array()
        raise


def cancel_slurm_array() -> None:
    """Cancel remaining SLURM array tasks. No-op if not in a SLURM job."""
    job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    if job_id is None:
        return
    log.info(f"Cancelling SLURM array job {job_id}")
    try:
        subprocess.run(["scancel", job_id], check=True)
    except Exception:
        log.exception(f"Failed to cancel SLURM job {job_id}")


def training_loop(*, cfg: Config, trial: optuna.Trial, run_name: str, run_dir: Path) -> float:
    """Main training loop. Called by train() with crash handling wrapper."""
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

    # Determine checkpoint source and load mode
    # Priority: run_dir/latest.pt (RESUME) > seed_ckpt (SEED) > fresh start
    ckpt_path_to_load: Path | None = None
    is_seeding = False  # True = seed mode (weights only), False = resume mode (full state)
    latest = find_latest(run_dir)
    if latest is not None:
        ckpt_path_to_load = latest
        is_seeding = False
        log.info(f"RESUME mode: continuing from {ckpt_path_to_load}")
    elif cfg.seed_ckpt is not None:
        ckpt_path_to_load = cfg.seed_ckpt
        is_seeding = True
        log.info(f"SEED mode: loading weights from {ckpt_path_to_load} (fresh opt/sched/step)")
    else:
        log.info("FRESH mode: no checkpoint, starting from scratch")

    # Load checkpoint BEFORE creating Comet experiment
    ckpt_data: CheckpointData | None = None
    prev_comet_id: str | None = None
    if ckpt_path_to_load is not None:
        ckpt_data = load_checkpoint(ckpt_path_to_load, cfg.device)
        prev_comet_id = ckpt_data["comet_id"]

    # === COMET EXPERIMENT ===
    # RESUME mode: continue existing experiment. SEED/FRESH mode: new experiment.
    comet_cfg = comet_ml.ExperimentConfig(auto_metric_logging=False)
    if prev_comet_id is not None and not is_seeding:
        log.info(f"Continuing Comet experiment: {prev_comet_id}")
        exp = comet_ml.start(
            experiment_key=prev_comet_id,
            project_name=cfg.comet_project,
            workspace=cfg.comet_workspace,
            experiment_config=comet_cfg,
        )
    else:
        if is_seeding and prev_comet_id:
            log.info(f"SEED mode: creating new experiment (seed source had {prev_comet_id})")
        else:
            log.info("Creating NEW Comet experiment")
        exp = comet_ml.start(
            project_name=cfg.comet_project,
            workspace=cfg.comet_workspace,
            experiment_config=comet_cfg,
        )

    exp.log_parameters(flatten_dict(asdict(cfg)))
    exp.log_parameters({"trial_number": trial.number, "run_name": run_name})
    if slurm_job_id := os.environ.get("SLURM_JOB_ID"):
        exp.log_parameters({"slurm_job_id": slurm_job_id})

    teacher = load_teacher(cfg)
    log.info(f"Teacher params: {count_parameters(teacher):,}")

    probe = load_probe(cfg.teacher_name, cfg.device)
    if probe is not None:
        log.info(f"Loaded IN1k probe for {cfg.teacher_name}")

    student_backbone = load_student_backbone(cfg)
    log.info(f"Student backbone params: {count_parameters(student_backbone):,}")

    bundle = create_model(student_backbone, teacher.embed_dim, cfg)
    model, glimpse_size_px = bundle.model, bundle.glimpse_size_px

    if cfg.compile:
        if cfg.combo_kernels:
            torch._inductor.config.combo_kernels = True  # type: ignore[attr-defined]
            log.info("Compiling teacher and model (combo_kernels=True, backward_pass_autocast=off)")
        else:
            log.info("Compiling teacher and model (backward_pass_autocast=off)")
        compile_teacher(teacher)
        compile_model(model)

    G = cfg.canvas_patch_grid_size
    patch_size = teacher.model.config.patch_size
    scene_size = scene_size_px(G, patch_size)
    log.info(f"Grid size: {G}, scene size: {scene_size}px")

    # Extract start_step from checkpoint scheduler state (BEFORE creating loaders)
    # NOTE: PyTorch LRScheduler uses "last_epoch" but we call scheduler.step() once per
    # training step, so last_epoch == number of gradient updates == our "step"
    # SEED mode always starts at step=0 (fresh training run)
    if ckpt_data is not None and not is_seeding:
        sched_state = ckpt_data["scheduler_state"]
        assert sched_state is not None, "Checkpoint has no scheduler_state — cannot determine start_step"
        start_step = sched_state["last_epoch"]  # PyTorch API: last_epoch = num scheduler.step() calls
        log.info("=" * 60)
        log.info(f"RESUME: start_step={start_step}")
        log.info("=" * 60)
    else:
        start_step = 0
        log.info("=" * 60)
        log.info(f"{'SEED' if is_seeding else 'FRESH'}: start_step=0")
        log.info("=" * 60)

    train_loader, val_loader = create_loaders(cfg, start_step=start_step)

    # Feature-based training is the only supported path
    assert cfg.feature_base_dir is not None, "feature_base_dir required (raw image training removed)"
    assert isinstance(train_loader, ShardedFeatureLoader)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = count_parameters(model)
    log.info(f"Model total: {n_total:,}, trainable: {n_trainable:,} ({100 * n_trainable / n_total:.1f}%)")
    exp.log_parameters({"trainable_params": n_trainable, "total_params": n_total})

    optimizer = torch.optim.AdamW(trainable, lr=cfg.peak_lr, weight_decay=cfg.weight_decay)
    start_lr = cfg.start_lr if cfg.start_lr is not None else cfg.peak_lr / cfg.warmup_steps
    scheduler = warmup_constant_scheduler(
        optimizer, cfg.warmup_steps, cfg.peak_lr,
        start_lr=cfg.start_lr,
    )
    log.info(f"Optimizer: AdamW, lr={start_lr:.2e}→{cfg.peak_lr:.2e} (constant), wd={cfg.weight_decay:.2e}")

    amp_ctx = (
        torch.autocast(device_type=cfg.device.type, dtype=torch.bfloat16)
        if cfg.amp else nullcontext()
    )
    log.info(f"AMP: {'bfloat16' if cfg.amp else 'disabled'}")
    log.info(f"Non-blocking transfers: {'enabled' if cfg.non_blocking_transfer else 'DISABLED (sync)'}")

    # === RESTORE MODEL/OPTIMIZER STATE FROM CHECKPOINT ===
    if ckpt_data is not None:
        ckpt_cfg = dacite.from_dict(CanViTForPretrainingConfig, ckpt_data["model_config"])
        if ckpt_cfg != cfg.model:
            log.warning("Checkpoint config differs from current config!")
            log.warning(f"  Checkpoint: {ckpt_cfg}")
            log.warning(f"  Current:    {cfg.model}")
        model.load_state_dict(ckpt_data["state_dict"], strict=True)
        log.info("Model state loaded (strict=True)")

        # Load optimizer + scheduler state (RESUME mode only)
        if is_seeding:
            log.info("SEED mode: fresh optimizer+scheduler (step=0)")
        else:
            opt_state = ckpt_data["optimizer_state"]
            sched_state = ckpt_data["scheduler_state"]
            assert opt_state is not None, "Checkpoint missing optimizer_state — cannot resume"
            assert sched_state is not None, "Checkpoint missing scheduler_state — cannot resume"
            optimizer.load_state_dict(opt_state)
            scheduler.load_state_dict(sched_state)
            log.info(f"RESUME mode: restored optimizer+scheduler (step={sched_state['last_epoch']})")

    # Build training config history (tracks config across resumes)
    training_config_history: dict[str, dict] = {}
    if ckpt_data is not None:
        training_config_history = ckpt_data["training_config_history"] or {}
    now = datetime.now(UTC).isoformat()
    training_config_history[now] = flatten_dict(asdict(cfg))

    # Build provenance history (tracks git/host/slurm across resumes)
    provenance_history: dict[str, dict] = {}
    if ckpt_data is not None:
        provenance_history = ckpt_data.get("provenance_history") or {}
    provenance_history[now] = current_provenance()

    def make_ckpt_path(step: int) -> Path:
        """Generate versioned checkpoint path: {run_dir}/step-{step}.pt"""
        return run_dir / f"step-{step}.pt"

    def compute_raw_targets(images: Tensor, sz: int) -> NormFeatures:
        with amp_ctx:
            if images.shape[-1] != sz:
                images = torch.nn.functional.interpolate(
                    images, size=(sz, sz), mode="bilinear", align_corners=False
                )
            feats = teacher.forward_norm_features(images)
            return NormFeatures(patches=feats.patches.float(), cls=feats.cls.float())

    scene_norm = PatchStandardizer(grid_size=G, embed_dim=teacher.embed_dim).to(cfg.device)
    cls_norm = CLSStandardizer(embed_dim=teacher.embed_dim).to(cfg.device)


    norm_loaded = False
    if ckpt_data is not None and not cfg.reset_normalizer:
        scene_norm_state = ckpt_data["scene_norm_state"]
        cls_norm_state = ckpt_data["cls_norm_state"]
        assert scene_norm_state is not None, "Checkpoint missing scene_norm_state"
        assert cls_norm_state is not None, "Checkpoint missing cls_norm_state"
        scene_norm.load_state_dict(scene_norm_state)
        cls_norm.load_state_dict(cls_norm_state)
        norm_loaded = True
        log.info("Loaded scene/cls normalizer states from checkpoint")
    elif cfg.reset_normalizer:
        log.info("Reset normalizer: will re-init stats")

    if not norm_loaded:
        assert cfg.feature_base_dir is not None, "feature_base_dir required for normalizer init"
        shards_dir = cfg.feature_base_dir / cfg.teacher_name / str(cfg.scene_resolution) / "shards"
        shard_files = sorted(shards_dir.glob("*.pt"))
        assert shard_files, f"No shards in {shards_dir}"
        init_normalizer_stats_from_shard(shard_files[0], scene_norm, cls_norm, cfg.device)

    log.info(
        f"Training: {cfg.n_full_start_branches} full + {cfg.n_random_start_branches} random branches,"
        f" chunk_size={cfg.chunk_size}, continue_prob={cfg.continue_prob}"
    )

    # EMA tracking for all metrics
    ema = EMATracker(alpha=cfg.ema_alpha)

    nb = cfg.non_blocking_transfer  # Ablation flag for async transfers

    def load_train_batch() -> TrainBatch:
        """Load training batch from precomputed features."""
        # non_blocking=True: CPU returns immediately, GPU ops serialize on same stream
        # Safe because we don't mutate source tensors after transfer
        images, raw_patches, raw_cls, labels = train_loader.next()
        images = images.to(cfg.device, non_blocking=nb)
        labels = labels.to(cfg.device, non_blocking=nb)
        # .float() for consistency - stored features may be fp16
        raw_patches = raw_patches.to(device=cfg.device, dtype=torch.float32, non_blocking=nb)
        raw_cls = raw_cls.to(device=cfg.device, dtype=torch.float32, non_blocking=nb)
        norm_patches = scene_norm(raw_patches)
        norm_cls = cls_norm(raw_cls.unsqueeze(1)).squeeze(1)
        return TrainBatch(images, labels, norm_patches, norm_cls, raw_patches, raw_cls)

    # Step semantics: step S = model state after S gradient updates
    # step=0: before any gradient (initial model)
    # Scheduler last_epoch tracks gradient updates done so far
    start_step = scheduler.last_epoch
    end_step = start_step + cfg.steps_per_job
    if ckpt_data is not None and start_step == 0:
        log.error("!!! CHECKPOINT LOADED BUT start_step=0 - optimizer/scheduler state was not restored !!!")
    log.info(f"Starting training loop: steps {start_step} → {end_step}")
    model.train()  # Explicit: validate() restores, but be clear about initial state
    # NOTE: tqdm shows job-local progress (e.g. "199/5001"), but `step` is global
    pbar = tqdm(range(start_step, end_step + 1), desc="Training", unit="step")

    for step in pbar:
        batch: TrainBatch | None = None
        # Determine viz/curve based on validation count
        val_count = step // cfg.val_every
        do_pca = step % cfg.val_every == 0 and val_count % cfg.viz_every_n_vals == 0
        do_curves = step % cfg.val_every == 0 and val_count % cfg.curve_every_n_vals == 0

        # === VALIDATION/VIZ PHASE (state after `step` gradient updates) ===
        if step % cfg.val_every == 0:

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
                        backbone=cfg.teacher_name,
                    )
            except Exception:
                log.error(f"!!! VALIDATION FAILED at step {step} !!!\n{traceback.format_exc()}")

        # === SIGUSR1 CHECKPOINT ===
        global _checkpoint_requested
        if _checkpoint_requested:
            log.info(f"Saving signal-triggered checkpoint at step {step}")
            _checkpoint_requested = False
            ema_loss = ema.get("total_loss")
            ckpt_path = make_ckpt_path(step)
            save_checkpoint(
                ckpt_path, model, cfg.backbone_name,
                teacher_repo_id=cfg.teacher_repo_id,
                teacher_name=cfg.teacher_name,
                dataset=cfg.dataset,
                glimpse_grid_size=cfg.glimpse_grid_size,
                scene_resolution=cfg.scene_resolution,
                step=step, train_loss=ema_loss.item() if ema_loss is not None else None,
                comet_id=exp.get_key(),
                scene_norm_state=scene_norm.state_dict(),
                cls_norm_state=cls_norm.state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                training_config_history=training_config_history,
                provenance_history=provenance_history,
            )
            update_symlink(run_dir / "latest.pt", ckpt_path)

        # === TRAINING PHASE (only for step < end_step) ===
        if step < end_step:
            if batch is None:
                batch = load_train_batch()

            optimizer.zero_grad()
            t_step_start = time.perf_counter()

            step_metrics = training_step(
                model=model,
                images=batch.images,
                scene_target=batch.scene_target,
                cls_target=batch.cls_target,
                raw_scene_target=batch.raw_scene_target,
                raw_cls_target=batch.raw_cls_target,
                scene_denorm=scene_norm.destandardize,
                cls_denorm=cls_norm.destandardize,
                enable_scene_patches_loss=cfg.enable_scene_patches_loss,
                enable_scene_cls_loss=cfg.enable_scene_cls_loss,
                glimpse_size_px=glimpse_size_px,
                canvas_grid_size=G,
                n_full_start_branches=cfg.n_full_start_branches,
                n_random_start_branches=cfg.n_random_start_branches,
                chunk_size=cfg.chunk_size,
                continue_prob=cfg.continue_prob,
                min_viewpoint_scale=cfg.min_viewpoint_scale,
                amp_ctx=amp_ctx,

                collect_viz=do_pca,
            )

            if step == start_step:
                log.info(f"First training_step took {time.perf_counter() - t_step_start:.1f}s (includes compile)")

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
                metrics["train/continue_prob"] = cfg.continue_prob
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

    # End-of-job checkpoint (always saved)
    ema_loss = ema.get("total_loss")
    ckpt_path = make_ckpt_path(end_step)
    save_checkpoint(
        ckpt_path, model, cfg.backbone_name,
        teacher_repo_id=cfg.teacher_repo_id,
        teacher_name=cfg.teacher_name,
        dataset=cfg.dataset,
        glimpse_grid_size=cfg.glimpse_grid_size,
        scene_resolution=cfg.scene_resolution,
        step=end_step, train_loss=ema_loss.item() if ema_loss is not None else None,
        comet_id=exp.get_key(),
        scene_norm_state=scene_norm.state_dict(),
        cls_norm_state=cls_norm.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict(),
        training_config_history=training_config_history,
        provenance_history=provenance_history,
    )
    update_symlink(run_dir / "latest.pt", ckpt_path)

    ema_loss = ema.get("total_loss")
    final_loss = ema_loss.item() if ema_loss is not None else float("inf")
    log.info(f"Final: train_ema={final_loss:.4f}")
    exp.end()
    return final_loss
