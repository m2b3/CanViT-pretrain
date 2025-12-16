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
from avp_vit.train.curriculum import CurriculumStage, log_curriculum_stage
from avp_vit.train.norm import PositionAwareNorm
from avp_vit.train.viewpoint import random_viewpoint

from .config import Config
from .data import create_curriculum_stages, create_loaders_for_curriculum
from .model import compile_avp, compile_teacher, create_avp, load_teacher
from .scheduler import create_scheduler
from .viz import eval_and_log, save_checkpoint, viz_and_log

log = logging.getLogger(__name__)


def create_norms(
    stages: dict[int, CurriculumStage], embed_dim: int, device: torch.device
) -> dict[int, PositionAwareNorm]:
    """Create position-aware norms for each grid size."""
    norms: dict[int, PositionAwareNorm] = {}
    for G, stage in stages.items():
        norms[G] = PositionAwareNorm(
            n_tokens=stage.n_scene_tokens,
            embed_dim=embed_dim,
            grid_size=G,
        ).to(device)
    return norms


def init_survival_batch(
    avp: AVPViT,
    train_loader: InfiniteLoader,
    compute_targets: Callable[[Tensor], Tensor],
    batch_size: int,
    fresh_count: int,
    device: torch.device,
) -> SurvivalBatch:
    """Initialize survival batch by loading fresh_count images at a time."""
    n_init_batches = (batch_size + fresh_count - 1) // fresh_count
    log.info(
        f"Initializing survival batch: batch_size={batch_size}, fresh_count={fresh_count}, "
        f"loading {n_init_batches} mini-batches"
    )
    init_imgs_list, init_targets_list = [], []

    with torch.no_grad():
        for _ in range(n_init_batches):
            batch = train_loader.next_batch().to(device)
            init_imgs_list.append(batch)
            init_targets_list.append(compute_targets(batch))

    init_imgs = torch.cat(init_imgs_list, dim=0)[:batch_size]
    init_targets = torch.cat(init_targets_list, dim=0)[:batch_size]
    hidden_init = avp._init_hidden(batch_size, None)
    local_init = (
        avp.local_init.expand(batch_size, -1, -1) if avp.local_init is not None else None
    )

    log.info(
        f"Survival batch initialized: images={init_imgs.shape}, targets={init_targets.shape}"
    )
    return SurvivalBatch.init(init_imgs, init_targets, hidden_init, local_init)


def train(cfg: Config, trial: optuna.Trial) -> float:
    """Train AVP model with random grid size sampling. Returns best val_loss."""
    log.info(f"Starting trial {trial.number}")
    log.info(f"Device: {cfg.device}")

    exp = comet_ml.Experiment(
        project_name="avp-vit-scene-match", auto_metric_logging=False
    )
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

    log.info("Loading teacher...")
    teacher = load_teacher(cfg)
    log.info(f"Teacher params: {count_parameters(teacher):,}")

    log.info("Creating AVP model...")
    avp = create_avp(teacher, cfg)

    if cfg.compile:
        log.info("Compilation enabled - compiling teacher and AVP with dynamic=True")
        compile_teacher(teacher)
        compile_avp(avp)
    else:
        log.info("Compilation disabled")

    loss_fn = {"l1": l1_loss, "mse": mse_loss}[cfg.loss]
    log.info(f"Loss function: {cfg.loss}")

    # Create curriculum stages and resources
    patch_size = teacher.patch_size
    stages = create_curriculum_stages(cfg, patch_size)
    for stage in stages.values():
        log_curriculum_stage(stage, log)

    train_loaders, val_loaders = create_loaders_for_curriculum(cfg, stages)
    norms = create_norms(stages, teacher.embed_dim, cfg.device)

    trainable = [p for p in avp.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = count_parameters(avp)
    log.info(
        f"AVP total: {n_total:,}, trainable: {n_trainable:,} "
        f"({100 * n_trainable / n_total:.1f}%)"
    )
    exp.log_parameters({"trainable_params": n_trainable, "total_params": n_total})

    peak_lr = get_linear_scaled_lr(cfg.ref_lr, cfg.batch_size)
    optimizer = torch.optim.AdamW(trainable, lr=peak_lr, weight_decay=cfg.weight_decay)
    scheduler = create_scheduler(optimizer, cfg)
    log.info(f"Optimizer: AdamW, peak_lr={peak_lr:.2e}, weight_decay={cfg.weight_decay:.2e}")

    ckpt_path = cfg.ckpt_dir / f"{exp.get_key()}.pt"

    def make_target_fn(norm: PositionAwareNorm) -> Callable[[Tensor], Tensor]:
        def compute_targets(images: Tensor) -> Tensor:
            with torch.autocast(device_type=cfg.device.type, dtype=torch.bfloat16):
                teacher_patches = teacher.forward_norm_patches(images)
            return norm(teacher_patches.float())
        return compute_targets

    # Initialize all survival batches upfront
    log.info("Initializing survival batches for all grid sizes...")
    states: dict[int, SurvivalBatch] = {}
    for G, stage in stages.items():
        avp.set_scene_grid_size(G)
        states[G] = init_survival_batch(
            avp,
            train_loaders[G],
            make_target_fn(norms[G]),
            stage.batch_size,
            stage.fresh_count,
            cfg.device,
        )

    ema_loss_t = torch.tensor(0.0, device=cfg.device)
    alpha = 2 / (cfg.log_every + 1)

    log.info("Starting training loop...")
    pbar = tqdm(range(cfg.n_steps), desc="Training", unit="step")

    for step in pbar:
        G = random.choice(cfg.grid_sizes)
        stage = stages[G]
        state = states[G]
        norm = norms[G]
        train_loader = train_loaders[G]
        val_loader = val_loaders[G]
        avp.set_scene_grid_size(G)
        compute_targets = make_target_fn(norm)

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
        loss, final_hidden, final_local = avp.forward_loss(
            state.images,
            viewpoints,
            state.targets,
            state.hidden,
            state.local_prev,
            loss_fn=loss_fn,
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

        ema_loss_t = (
            alpha * loss.detach() + (1 - alpha) * ema_loss_t
            if step > 0
            else loss.detach()
        )

        if step % cfg.log_every == 0:
            ema_loss = ema_loss_t.item()
            grad_norm = grad_norm_t.item()
            lr = scheduler.get_last_lr()[0]
            # Log actual gate values (sigmoid of logits), not raw logits
            scene_gate_mean = torch.sigmoid(avp.scene_temporal_gate).mean().item()
            metrics = {
                f"grid{G}/train/loss": ema_loss,
                "train/loss": ema_loss,
                "train/grad_norm": grad_norm,
                "train/lr": lr,
                "train/grid_size": G,
                "train/scene_temporal_gate_mean": scene_gate_mean,
                "train/spatial_hidden_init_norm": avp.spatial_hidden_init.norm().item(),
            }
            if avp.local_temporal_gate is not None:
                metrics["train/local_temporal_gate_mean"] = torch.sigmoid(avp.local_temporal_gate).mean().item()
            exp.log_metrics(metrics, step=step)
            pbar.set_postfix_str(
                f"G={G} loss={ema_loss:.2e} grad={grad_norm:.2e} lr={lr:.2e}"
            )

        if step % cfg.val_every == 0:
            # Training viz (no curves)
            train_l1, train_mse = viz_and_log(
                exp,
                step,
                f"grid{G}/train",
                avp,
                teacher,
                state.images,
                viewpoints,
                state.targets,
                state.hidden,
                norm,
                log_spatial_stats=cfg.log_spatial_stats,
                log_curves=False,
                loss_type=cfg.loss,
            )
            exp.log_metric(f"grid{G}/train/viz_l1", train_l1[-1], step=step)
            exp.log_metric(f"grid{G}/train/viz_mse", train_mse[-1], step=step)

            # Validation (curves at curve_every intervals)
            val_images = val_loader.next_batch().to(cfg.device)
            norm.eval()
            val_loss = eval_and_log(
                exp, step, avp, teacher, compute_targets, val_images, norm, f"grid{G}/val",
                log_spatial_stats=cfg.log_spatial_stats,
                log_curves=(step % cfg.curve_every == 0),
                loss_type=cfg.loss,
            )
            norm.train()
            exp.log_metric("val/loss", val_loss, step=step)

            if step % cfg.ckpt_every == 0:
                save_checkpoint(avp, norms, ckpt_path, exp, step, ema_loss_t.item(), G)

            if step > 0:
                trial.report(ema_loss_t.item(), step)
                if trial.should_prune():
                    exp.end()
                    raise optuna.TrialPruned()

        # Fresh ratio survival: permute batch, replace first K with fresh
        hidden_init = avp._init_hidden(stage.fresh_count, None)
        local_init = (
            avp.local_init.expand(stage.fresh_count, -1, -1)
            if avp.local_init is not None
            else None
        )
        states[G] = state.step(
            fresh_imgs,
            fresh_targets,
            final_hidden,
            final_local,
            hidden_init,
            local_init,
        )

    # Final checkpoint
    save_checkpoint(avp, norms, ckpt_path, exp, cfg.n_steps, ema_loss_t.item(), cfg.max_grid_size)

    log.info(f"Final: train_ema={ema_loss_t.item():.4f}")
    exp.end()
    return ema_loss_t.item()
