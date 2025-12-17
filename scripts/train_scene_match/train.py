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
from avp_vit.train import InfiniteLoader, SurvivalBatch
from avp_vit.train.viewpoint import random_viewpoint

from .config import Config
from .data import ResolutionStage, create_loaders, create_resolution_stages
from .model import compile_avp, compile_teacher, create_avp, load_student_backbone, load_teacher
from .scheduler import create_scheduler
from .viz import eval_and_log, save_checkpoint, viz_and_log

log = logging.getLogger(__name__)


def init_survival_batch(
    avp: AVPViT,
    train_loader: InfiniteLoader,
    compute_targets: Callable[[Tensor], Tensor],
    batch_size: int,
    fresh_count: int,
    scene_grid_size: int,
    device: torch.device,
) -> SurvivalBatch:
    """Initialize survival batch by loading fresh_count images at a time."""
    n_init_batches = (batch_size + fresh_count - 1) // fresh_count
    log.info(f"Initializing survival batch: batch_size={batch_size}, fresh_count={fresh_count}")

    init_imgs_list, init_targets_list = [], []
    with torch.no_grad():
        for _ in range(n_init_batches):
            batch = train_loader.next_batch().to(device)
            init_imgs_list.append(batch)
            init_targets_list.append(compute_targets(batch))

    init_imgs = torch.cat(init_imgs_list, dim=0)[:batch_size]
    init_targets = torch.cat(init_targets_list, dim=0)[:batch_size]
    hidden_init = avp.init_hidden(batch_size, scene_grid_size)

    return SurvivalBatch.init(init_imgs, init_targets, hidden_init)


def _log_stage(stage: ResolutionStage) -> None:
    log.info(
        f"  G={stage.scene_grid_size}: batch={stage.batch_size}, fresh={stage.fresh_count}, "
        f"ratio={stage.fresh_ratio_actual:.3f} (desired={stage.fresh_ratio_desired:.3f}), "
        f"E[glimpses]={stage.e_glimpses_actual:.1f} (desired={stage.e_glimpses_desired:.1f}), "
        f"min_scale={stage.min_viewpoint_scale:.2f}"
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

    ckpt_path = cfg.ckpt_dir / f"{exp.get_key()}.pt"

    def compute_targets(images: Tensor) -> Tensor:
        with torch.autocast(device_type=cfg.device.type, dtype=torch.bfloat16):
            return teacher.forward_norm_patches(images).float()

    # Initialize all survival batches upfront
    log.info("Initializing survival batches...")
    states: dict[int, SurvivalBatch] = {}
    for G, stage in stages.items():
        states[G] = init_survival_batch(
            avp, train_loaders[G], compute_targets,
            stage.batch_size, stage.fresh_count, G, cfg.device,
        )

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

        # Load fresh images
        fresh_imgs = train_loader.next_batch().to(cfg.device)
        with torch.no_grad():
            fresh_targets = compute_targets(fresh_imgs)

        # Viewpoint scale bounds for current grid size
        min_scale = stage.min_viewpoint_scale
        max_scale = 1.0

        # Inner loop: multiple viewpoints per optimizer step
        viewpoints = [
            random_viewpoint(stage.batch_size, cfg.device, min_scale, max_scale)
            for _ in range(cfg.n_viewpoints_per_step)
        ]
        loss, final_hidden = avp.forward_loss(
            state.images, viewpoints, state.targets, state.hidden, loss_fn=loss_fn,
        )

        if not torch.isfinite(loss):
            log.warning(f"NaN/Inf loss at step {step}, pruning trial")
            exp.end()
            raise optuna.TrialPruned()

        optimizer.zero_grad()
        loss.backward()
        grad_norm_t = torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        ema_loss_t = alpha * loss.detach() + (1 - alpha) * ema_loss_t if step > 0 else loss.detach()

        if step % cfg.log_every == 0:
            ema_loss = ema_loss_t.item()
            grad_norm = grad_norm_t.item()
            lr = scheduler.get_last_lr()[0]
            exp.log_metrics({
                f"grid{G}/train/loss": ema_loss,
                "train/loss": ema_loss,
                "train/grad_norm": grad_norm,
                "train/lr": lr,
                "train/grid_size": G,
                "train/spatial_hidden_init_norm": avp.spatial_hidden_init.norm().item(),
            }, step=step)
            pbar.set_postfix_str(f"G={G} loss={ema_loss:.2e} grad={grad_norm:.2e} lr={lr:.2e}")

        if step % cfg.val_every == 0:
            # Training viz (no curves)
            train_l1, train_mse = viz_and_log(
                exp, step, f"grid{G}/train", avp, teacher,
                state.images, viewpoints, state.targets, state.hidden,
                log_spatial_stats=cfg.log_spatial_stats, log_curves=False, loss_type=cfg.loss,
            )
            exp.log_metric(f"grid{G}/train/viz_l1", train_l1[-1], step=step)
            exp.log_metric(f"grid{G}/train/viz_mse", train_mse[-1], step=step)

            # Validation
            val_images = val_loader.next_batch().to(cfg.device)
            val_loss = eval_and_log(
                exp, step, avp, teacher, compute_targets, val_images, G, f"grid{G}/val",
                log_spatial_stats=cfg.log_spatial_stats,
                log_curves=(step % cfg.curve_every == 0),
                loss_type=cfg.loss,
            )
            exp.log_metric("val/loss", val_loss, step=step)

            if step % cfg.ckpt_every == 0:
                save_checkpoint(
                    avp=avp, path=ckpt_path, exp=exp,
                    step=step, train_loss=ema_loss_t.item(), current_grid_size=G,
                )

            if step > 0:
                trial.report(ema_loss_t.item(), step)
                if trial.should_prune():
                    exp.end()
                    raise optuna.TrialPruned()

        # Fresh ratio survival: permute batch, replace first K with fresh
        hidden_init = avp.init_hidden(stage.fresh_count, G)
        states[G] = state.step(
            fresh_images=fresh_imgs,
            fresh_targets=fresh_targets,
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
