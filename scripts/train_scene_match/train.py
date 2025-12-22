"""Main training loop."""

import logging
from contextlib import nullcontext

import comet_ml
import dacite
import numpy as np
import optuna
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm import tqdm

# Force FlashAttention for SDPA - fail loud if unavailable
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(False)
from ytch.model import count_parameters  # noqa: E402

from avp_vit import ActiveCanViTConfig  # noqa: E402
from canvit.backbone.dinov3 import NormFeatures  # noqa: E402
from avp_vit.checkpoint import load as load_checkpoint  # noqa: E402
from avp_vit.checkpoint import save as save_checkpoint  # noqa: E402
from avp_vit.train import InfiniteLoader, warmup_cosine_scheduler  # noqa: E402
from avp_vit.train.norm import PositionAwareNorm  # noqa: E402
from avp_vit.train.probe import load_probe  # noqa: E402
from avp_vit.train.viewpoint import random_viewpoint  # noqa: E402

from .config import Config  # noqa: E402
from .data import create_loaders, scene_size_px  # noqa: E402
from .model import compile_model, compile_teacher, create_model, load_student_backbone, load_teacher  # noqa: E402
from .viz import eval_and_log, val_metrics_only, viz_and_log  # noqa: E402

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
    compute_raw_targets: callable,
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

    viz_steps = log_spaced_steps(cfg.total_viz, cfg.n_steps)
    curve_steps = log_spaced_steps(cfg.total_curves, cfg.n_steps)
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

    log.info("Starting training loop...")
    pbar = tqdm(range(cfg.n_steps), desc="Training", unit="step")

    for step in pbar:
        # Fresh batch and canvas each step
        images = train_loader.next_batch().to(cfg.device)
        with torch.no_grad():
            feats = compute_raw_targets(images, scene_size)
            targets = scene_norm(feats.patches)
            cls_targets = cls_norm(feats.cls.unsqueeze(1)).squeeze(1)
        canvas = model.init_canvas(batch_size=cfg.batch_size, canvas_grid_size=G)

        viewpoints = [random_viewpoint(cfg.batch_size, cfg.device, min_scale=cfg.min_viewpoint_scale) for _ in range(cfg.n_viewpoints_per_step)]

        with amp_ctx:
            losses, final_canvas = model.forward_loss(
                image=images,
                viewpoints=viewpoints,
                spatial_target=targets,
                cls_target=cls_targets,
                glimpse_size_px=glimpse_size_px,
                canvas_grid_size=G,
                canvas=canvas,
                compute_gram=cfg.gram_loss_weight > 0,
                include_init=cfg.include_init,
            )

            total_loss = losses.scene + losses.cls
            if losses.gram is not None:
                total_loss = total_loss + cfg.gram_loss_weight * losses.gram

            if not torch.isfinite(total_loss):
                log.warning(f"NaN/Inf loss at step {step}, pruning trial")
                exp.end()
                raise optuna.TrialPruned()

            # Compute cos_sim only when logging (cheap: projection heads + dot product)
            scene_cos_sim: float | None = None
            cls_cos_sim: float | None = None
            if step % cfg.log_every == 0:
                with torch.no_grad():
                    scene_pred = model.compute_scene(final_canvas)
                    scene_cos_sim = F.cosine_similarity(scene_pred, targets, dim=-1).mean().item()
                    if model.cls_head is not None:
                        cls_pred = model.compute_cls(final_canvas)
                        cls_cos_sim = F.cosine_similarity(cls_pred, cls_targets, dim=-1).mean().item()

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
            exp.log_metrics(metrics, step=step)
            pbar.set_postfix_str(f"loss={ema_loss.item():.2e} grad={grad_norm:.2e} lr={lr:.2e}")

        if step % cfg.val_every == 0:
            for name, norm in grad_norms_by_module(model, depth=1).items():
                exp.log_metric(f"grad_norm/{name}", norm, step=step)

            if step not in viz_steps:
                val_images, val_labels = val_loader.next_batch_with_labels()
                val_images = val_images.to(cfg.device)
                val_labels = val_labels.to(cfg.device) if probe is not None else None
                with amp_ctx:
                    val_metrics_only(
                        exp, step, model, compute_raw_targets, scene_norm, cls_norm,
                        val_images, G, scene_size, glimpse_size_px, "val",
                        probe=probe, labels=val_labels,
                    )

            if step > 0:
                trial.report(ema_loss.item(), step)
                if trial.should_prune():
                    exp.end()
                    raise optuna.TrialPruned()

        if step in viz_steps:
            with amp_ctx:
                viz_and_log(
                    exp, step, "train", model, teacher, scene_norm,
                    images, viewpoints, targets, canvas, glimpse_size_px,
                    log_spatial_stats=cfg.log_spatial_stats, log_curves=False,
                )

            val_images = val_loader.next_batch().to(cfg.device)
            with amp_ctx:
                eval_and_log(
                    exp, step, model, teacher, compute_raw_targets, scene_norm, cls_norm,
                    val_images, G, scene_size, glimpse_size_px, "val",
                    log_spatial_stats=cfg.log_spatial_stats,
                    log_curves=(step in curve_steps),
                )

        if step % cfg.ckpt_every == 0:
            save_checkpoint(
                ckpt_path, model, cfg.student_model,
                step=step, train_loss=ema_loss.item(), comet_id=exp.get_key(),
                scene_norm_state=scene_norm.state_dict(),
                cls_norm_state=cls_norm.state_dict(),
            )
            exp.log_metric("norm/scene_mean_norm", scene_norm.mean.norm().item(), step=step)
            exp.log_metric("norm/cls_mean_norm", cls_norm.mean.norm().item(), step=step)

    assert ema_loss is not None
    save_checkpoint(
        ckpt_path, model, cfg.student_model,
        step=cfg.n_steps, train_loss=ema_loss.item(), comet_id=exp.get_key(),
        scene_norm_state=scene_norm.state_dict(),
        cls_norm_state=cls_norm.state_dict(),
    )

    log.info(f"Final: train_ema={ema_loss.item():.4f}")
    exp.end()
    return ema_loss.item()
