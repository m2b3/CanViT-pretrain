"""ADE20K probe training for CanViT.

Trains segmentation probes on frozen CanViT features:
- ONE probe per feature type, shared weights across timesteps (anytime decoding)
- Training: loss averaged across timesteps, single backward pass
- Eval: mIoU computed per timestep, logged as curves

Training protocol aligned with DINOv3's linear probing (Appendix D.1).
Note: We use whole-image inference, not sliding window (intentional simplification).
"""

import logging
import math
import time
from dataclasses import asdict
from pathlib import Path

import comet_ml
import torch
import torch.nn as nn
import tyro
from canvit import CanViTForPretrainingHFHub
from canvit_utils.teacher import load_teacher
from dinov3.eval.segmentation.schedulers import WarmupOneCycleLR
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassJaccardIndex
from tqdm import tqdm

from canvit_eval.ade20k.dataset import IGNORE_LABEL, NUM_CLASSES, ADE20kDataset
from canvit_eval.ade20k.probe import ProbeHead
from canvit_eval.ade20k.viz import log_viz
from canvit_eval.utils import make_viewpoints

from .config import STATIC_FEATURES, Config, FeatureType
from .features import ExtractedFeatures, extract_features
from .loss import ce_loss, focal_loss, upsample_preds
from .state import ProbeState

log = logging.getLogger(__name__)


def _make_probe(name: str, dim: int, cfg: Config, device: torch.device, *, use_ln: bool) -> ProbeState:
    """Create probe with DINOv3's WarmupOneCycleLR scheduler."""
    head = ProbeHead(dim, dropout=cfg.dropout, use_ln=use_ln).to(device)
    opt = AdamW(head.parameters(), lr=cfg.peak_lr, weight_decay=cfg.weight_decay)
    scheduler = WarmupOneCycleLR(
        opt,
        max_lr=cfg.peak_lr,
        total_steps=cfg.max_steps,
        warmup_iters=cfg.warmup_steps,
        warmup_ratio=cfg.warmup_lr_ratio,
        pct_start=0,  # no rising phase after warmup
        anneal_strategy="cos",
        final_div_factor=float("inf"),  # decay to 0
        use_beta1=False,
        update_momentum=False,
    )
    return ProbeState(name, head, opt, scheduler)


def _save_checkpoint(path: Path, probes: dict[FeatureType, ProbeState], step: int) -> None:
    data = {
        "step": step,
        "probe_state_dicts": {name: p.head.state_dict() for name, p in probes.items()},
        "best_mean_mious": {name: p.best_mean_miou for name, p in probes.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    log.info(f"Saved checkpoint: {path} ({path.stat().st_size / 1e6:.1f} MB)")


def train(cfg: Config) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_float32_matmul_precision("high")
    device = torch.device(cfg.device)

    log.info("=" * 60)
    log.info("ADE20K Probe Training")
    log.info("=" * 60)
    log.info(f"Model: {cfg.model_repo}")
    log.info(f"Features: {cfg.features}")
    log.info(f"Timesteps: {cfg.n_timesteps}")
    log.info(f"Viewpoint: scale=[{cfg.min_vp_scale}, {cfg.max_vp_scale}], train_start_full={cfg.train_start_full}")
    log.info(f"Training: BS={cfg.batch_size}, steps={cfg.max_steps}, LR={cfg.peak_lr}")

    # Model
    log.info("Loading model...")
    model = CanViTForPretrainingHFHub.from_pretrained(cfg.model_repo).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    log.info(f"  params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    teacher = load_teacher(model.backbone_name, device)
    log.info(f"  teacher: {model.backbone_name}, dim={teacher.embed_dim}")

    patch_size = model.backbone.patch_size_px
    canvas_grid = cfg.image_size // patch_size
    log.info(f"  canvas: {canvas_grid}x{canvas_grid}, glimpse: {cfg.glimpse_px}px")

    # Probes
    dims: dict[FeatureType, int] = {
        "hidden": model.canvas_dim,
        "predicted_norm": teacher.embed_dim,
        "teacher_glimpse": teacher.embed_dim,
        "teacher_full": teacher.embed_dim,
    }
    # Only raw canvas features need LN; others are already normalized
    needs_ln: dict[FeatureType, bool] = {
        "hidden": True, "predicted_norm": False, "teacher_glimpse": False, "teacher_full": False,
    }

    compute_teacher_full = "teacher_full" in cfg.features
    need_canvit = any(f not in STATIC_FEATURES for f in cfg.features)
    probes: dict[FeatureType, ProbeState] = {
        feat: _make_probe(feat, dims[feat], cfg, device, use_ln=needs_ln[feat]) for feat in cfg.features
    }
    for feat, probe in probes.items():
        log.info(f"  probe[{feat}]: dim={dims[feat]}, params={sum(p.numel() for p in probe.head.parameters()):,}")

    # IoU metrics
    val_iou = {
        feat: [MulticlassJaccardIndex(NUM_CLASSES, ignore_index=IGNORE_LABEL, average="macro").to(device)
               for _ in range(cfg.n_timesteps)]
        for feat in cfg.features
    }
    train_iou = {
        feat: [MulticlassJaccardIndex(NUM_CLASSES, ignore_index=IGNORE_LABEL, average="macro").to(device)
               for _ in range(cfg.n_timesteps)]
        for feat in cfg.features
    }

    # Data
    train_ds = ADE20kDataset(
        cfg.ade20k_root, "training", cfg.image_size,
        augment=True, aug_scale_range=cfg.aug_scale_range, aug_flip_prob=cfg.aug_flip_prob,
    )
    val_ds = ADE20kDataset(
        cfg.ade20k_root, "validation", cfg.image_size,
        augment=False, aug_scale_range=cfg.aug_scale_range, aug_flip_prob=cfg.aug_flip_prob,
    )
    train_loader = DataLoader(
        train_ds, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(val_ds, cfg.eval_batch_size, num_workers=cfg.num_workers, pin_memory=True)

    # Comet
    exp = comet_ml.Experiment(project_name=cfg.comet_project, workspace=cfg.comet_workspace)
    exp.log_parameters(asdict(cfg))
    log.info(f"Comet: {cfg.comet_workspace}/{cfg.comet_project}")

    amp_dtype = torch.bfloat16 if cfg.amp else torch.float32
    amp_ctx = torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=cfg.amp)

    log.info("=" * 60)
    log.info("Starting training...")

    step = 0
    train_iter = iter(train_loader)
    pbar = tqdm(total=cfg.max_steps, desc="Training")
    val_viz_batch: tuple[Tensor, Tensor, ExtractedFeatures] | None = None

    while step < cfg.max_steps:
        try:
            images, masks = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, masks = next(train_iter)
        images, masks = images.to(device), masks.to(device)

        # === Validation ===
        if step % cfg.val_every == 0:
            val_start = time.perf_counter()
            for p in probes.values():
                p.head.eval()
            for feat in cfg.features:
                for m in val_iou[feat]:
                    m.reset()

            first_val_batch = True
            with torch.no_grad():
                for vi, vm in val_loader:
                    vi, vm = vi.to(device), vm.to(device)
                    B = vi.shape[0]
                    viewpoints = make_viewpoints(
                        "random", B, device, cfg.n_timesteps,
                        min_scale=cfg.min_vp_scale, max_scale=cfg.max_vp_scale,
                        start_with_full_scene=True,  # Val ALWAYS starts full
                    ) if need_canvit else None
                    with amp_ctx:
                        val_feats = extract_features(
                            model=model, teacher=teacher, images=vi,
                            canvas_grid=canvas_grid, glimpse_px=cfg.glimpse_px,
                            viewpoints=viewpoints, compute_teacher_full=compute_teacher_full,
                        )
                    if first_val_batch:
                        val_viz_batch = (vi, vm, val_feats)
                        first_val_batch = False
                    for feat_type in cfg.features:
                        t_range = [0] if feat_type in STATIC_FEATURES else range(cfg.n_timesteps)
                        for t in t_range:
                            logits = probes[feat_type].head(val_feats.get(feat_type, t).float())
                            preds_up = upsample_preds(logits.argmax(1), vm.shape[1], vm.shape[2])
                            val_iou[feat_type][t].update(preds_up, vm)

            any_improved = False
            for feat_type in cfg.features:
                if feat_type in STATIC_FEATURES:
                    miou = val_iou[feat_type][0].compute().item()
                    exp.log_metric(f"{feat_type}/val_miou", miou, step=step)
                    mean_miou = miou
                else:
                    mious = [val_iou[feat_type][t].compute().item() for t in range(cfg.n_timesteps)]
                    mean_miou = sum(mious) / len(mious)
                    for t, miou in enumerate(mious):
                        exp.log_metric(f"{feat_type}/val_miou_t{t}", miou, step=step)
                    exp.log_metric(f"{feat_type}/val_miou_mean", mean_miou, step=step)
                    exp.log_curve(f"{feat_type}/val_miou_curve", x=list(range(cfg.n_timesteps)), y=mious, step=step)

                if mean_miou > probes[feat_type].best_mean_miou:
                    probes[feat_type].best_mean_miou = mean_miou
                    any_improved = True

            if any_improved and cfg.probe_ckpt_dir:
                _save_checkpoint(cfg.probe_ckpt_dir / "best.pt", probes, step)

            val_time = time.perf_counter() - val_start
            log.info(f"Step {step}: validation took {val_time:.1f}s")
            exp.log_metric("timing/val_seconds", val_time, step=step)

        # === Training step ===
        for p in probes.values():
            p.head.train()

        B = images.shape[0]
        viewpoints = make_viewpoints(
            "random", B, device, cfg.n_timesteps,
            min_scale=cfg.min_vp_scale, max_scale=cfg.max_vp_scale,
            start_with_full_scene=cfg.train_start_full,
        ) if need_canvit else None
        with amp_ctx:
            feats = extract_features(
                model=model, teacher=teacher, images=images,
                canvas_grid=canvas_grid, glimpse_px=cfg.glimpse_px,
                viewpoints=viewpoints, compute_teacher_full=compute_teacher_full,
            )

        for feat_type in cfg.features:
            probe = probes[feat_type]
            probe.optimizer.zero_grad()
            t_range = [0] if feat_type in STATIC_FEATURES else range(cfg.n_timesteps)
            logits_list = [probe.head(feats.get(feat_type, t).float()) for t in t_range]
            if cfg.loss_type == "ce":
                losses = [ce_loss(logits, masks) for logits in logits_list]
            else:
                losses = [focal_loss(logits, masks, cfg.focal_gamma) for logits in logits_list]
            loss = torch.stack(losses).mean()
            loss.backward()
            if math.isfinite(cfg.grad_clip):
                grad_norm = nn.utils.clip_grad_norm_(probe.head.parameters(), cfg.grad_clip)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(probe.head.parameters(), float("inf"))
            probe.optimizer.step()
            probe.scheduler.step()
            probe.accumulate(loss, grad_norm)

            # Train IoU (NO GPU sync here)
            with torch.no_grad():
                for t, logits in zip(t_range, logits_list, strict=True):
                    preds = logits.detach().argmax(1)
                    preds_up = upsample_preds(preds, masks.shape[1], masks.shape[2])
                    train_iou[feat_type][t].update(preds_up, masks)

        # === Visualization (before increment, like val) ===
        if step % cfg.viz_every == 0:
            viz_start = time.perf_counter()
            for p in probes.values():
                p.head.eval()
            with torch.no_grad():
                log_viz(exp, step, probes, feats, images, masks, cfg.viz_samples, cfg.n_timesteps, split="train")
                if val_viz_batch is not None:
                    v_img, v_mask, v_feats = val_viz_batch
                    log_viz(exp, step, probes, v_feats, v_img, v_mask, cfg.viz_samples, cfg.n_timesteps, split="val")
            viz_time = time.perf_counter() - viz_start
            log.info(f"Step {step}: viz took {viz_time:.1f}s")
            exp.log_metric("timing/viz_seconds", viz_time, step=step)

        step += 1
        pbar.update(1)

        # === Logging (GPU sync here) ===
        if step % cfg.log_every == 0:
            lr = list(probes.values())[0].scheduler.get_last_lr()[0]
            log_dict: dict[str, float] = {"lr": float(lr)}
            for name, p in probes.items():
                avg_loss, avg_grad = p.get_and_reset()
                log_dict[f"{name}/loss"] = avg_loss
                log_dict[f"{name}/grad_norm"] = avg_grad

            # Log train mIoU mean (scalar) at log_every
            # Log train mIoU curves only at val_every to save Comet curve budget
            log_curves = (step % cfg.val_every == 0)
            for feat_type in cfg.features:
                if feat_type in STATIC_FEATURES:
                    miou = train_iou[feat_type][0].compute().item()
                    log_dict[f"{feat_type}/train_miou"] = miou
                    train_iou[feat_type][0].reset()
                else:
                    mious = [train_iou[feat_type][t].compute().item() for t in range(cfg.n_timesteps)]
                    log_dict[f"{feat_type}/train_miou_mean"] = sum(mious) / len(mious)
                    if log_curves:
                        xs = list(range(cfg.n_timesteps))
                        exp.log_curve(f"{feat_type}/train_miou_curve", x=xs, y=mious, step=step)
                    for m in train_iou[feat_type]:
                        m.reset()

            exp.log_metrics(log_dict, step=step)

    pbar.close()

    if cfg.probe_ckpt_dir:
        _save_checkpoint(cfg.probe_ckpt_dir / "final.pt", probes, step)

    log.info("=" * 60)
    log.info("Training complete. Best mean mIoU:")
    for name, p in probes.items():
        log.info(f"  {name}: {p.best_mean_miou:.4f}")
        exp.log_metric(f"best/{name}", p.best_mean_miou)
    log.info("=" * 60)


def main() -> None:
    cfg = tyro.cli(Config)
    train(cfg)
