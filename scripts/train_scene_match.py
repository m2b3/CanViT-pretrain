"""Train AVP scene representation to match frozen teacher backbone patches."""

import copy
import io
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

import comet_ml
import matplotlib.pyplot as plt
import optuna
import torch
import torch.nn as nn
from dinov3.hub.backbones import dinov3_vits16
from matplotlib.figure import Figure
from torch import Tensor
from tqdm import tqdm
from ymc.lr import get_linear_scaled_lr
from ytch.device import get_sensible_device
from ytch.model import count_parameters

from avp_vit import AVPConfig, AVPViT
from avp_vit.backbone.dinov3 import DINOv3Backbone
from avp_vit.train import (
    InfiniteLoader,
    TrainState,
    fit_pca,
    imagenet_denormalize,
    make_eval_viewpoints,
    make_loader,
    plot_pca_grid,
    plot_trajectory,
    random_viewpoint,
    train_transform,
    val_transform,
    warmup_cosine_scheduler,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


@dataclass
class Config:
    # Paths
    teacher_ckpt: Path = Path("dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
    train_dir: Path = Path("/datasets/ILSVRC/Data/CLS-LOC/train")
    val_dir: Path = Path("/datasets/ILSVRC/Data/CLS-LOC/val")
    ckpt_dir: Path = Path("checkpoints")
    # Model
    scene_grid_size: int = 16
    glimpse_grid_size: int = 7
    gate_init: float = 1e-5
    use_output_proj: bool = True
    use_scene_registers: bool = True
    freeze_inner_backbone: bool = False
    gradient_checkpointing: bool = True
    # Training
    survival_prob: float = 0.5
    n_steps: int = 20000
    batch_size: int = 64
    num_workers: int = 8
    ref_lr: float = 1e-5
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.5
    grad_clip: float = 1.0
    crop_scale_min: float = 0.4
    # Logging
    log_every: int = 20
    val_every: int = 200
    # Optuna
    n_trials: int = 100
    # Runtime
    device: torch.device = field(default_factory=get_sensible_device)

    @property
    def min_viewpoint_scale(self) -> float:
        return self.glimpse_grid_size / self.scene_grid_size

    @property
    def max_viewpoint_scale(self) -> float:
        return 1.0

    @property
    def scene_size(self) -> int:
        return 16 * self.scene_grid_size  # patch_size=16 for DINOv3


def load_teacher(cfg: Config) -> DINOv3Backbone:
    model = dinov3_vits16(weights=str(cfg.teacher_ckpt), pretrained=True)
    backbone = DINOv3Backbone(model.eval().to(cfg.device))
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone


def create_avp(teacher: DINOv3Backbone, cfg: Config) -> AVPViT:
    backbone_copy = copy.deepcopy(teacher)
    for p in backbone_copy.parameters():
        p.requires_grad = not cfg.freeze_inner_backbone
    avp_cfg = AVPConfig(
        scene_grid_size=cfg.scene_grid_size,
        glimpse_grid_size=cfg.glimpse_grid_size,
        gate_init=cfg.gate_init,
        use_output_proj=cfg.use_output_proj,
        use_scene_registers=cfg.use_scene_registers,
        gradient_checkpointing=cfg.gradient_checkpointing,
    )
    return AVPViT(backbone_copy, avp_cfg).to(cfg.device)


def log_figure(exp: comet_ml.Experiment, fig: Figure, name: str, step: int) -> None:
    with io.BytesIO() as buf:
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        exp.log_image(buf, name=name, step=step)
    plt.close(fig)


def eval_and_log(
    exp: comet_ml.Experiment,
    step: int,
    avp: AVPViT,
    teacher: DINOv3Backbone,
    images: Tensor,
    cfg: Config,
) -> float:
    """Evaluate on one batch and log to Comet. Returns final MSE."""
    B = images.shape[0]
    viewpoints = make_eval_viewpoints(B, cfg.device)

    with torch.inference_mode():
        target = teacher.forward_norm_patches(images)
        # Use forward_trajectory: returns (list[scene], final_hidden)
        # scene = projected output for loss/viz (no need to call output_proj)
        scenes, _ = avp.forward_trajectory(images, viewpoints)
        mses = [nn.functional.mse_loss(s, target).item() for s in scenes]
        scenes_np = [s[0].cpu().float().numpy() for s in scenes]

    for t, mse in enumerate(mses):
        exp.log_metric(f"val/mse_t{t}", mse, step=step)

    # PCA visualization
    teacher_np = target[0].cpu().float().numpy()
    pca = fit_pca(teacher_np)
    titles = [f"t={i} ({vp.name})" for i, vp in enumerate(viewpoints)]
    fig_pca = plot_pca_grid(pca, teacher_np, scenes_np, cfg.scene_grid_size, titles)
    log_figure(exp, fig_pca, "val/pca", step)

    # Trajectory visualization
    img_np = imagenet_denormalize(images[0].cpu()).numpy()
    H, W = img_np.shape[:2]
    boxes = [vp.to_pixel_box(0, H, W) for vp in viewpoints]
    names = [vp.name for vp in viewpoints]
    fig_traj = plot_trajectory(img_np, boxes, names)
    log_figure(exp, fig_traj, "val/trajectory", step)

    val_loss = mses[-1]
    exp.log_metric("val/loss", val_loss, step=step)
    return val_loss


def save_checkpoint(avp: AVPViT, path: Path, exp: comet_ml.Experiment, step: int, val_loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(avp.state_dict(), path)
    size_mb = path.stat().st_size / (1024 * 1024)
    log.info(f"Saved checkpoint: {path} ({size_mb:.1f} MB), val_loss={val_loss:.4f}")
    exp.log_metric("ckpt/val_loss", val_loss, step=step)


def train(cfg: Config, trial: optuna.Trial) -> float:
    """Train AVP model and return best val_loss for HP optimization."""
    exp = comet_ml.Experiment(project_name="avp-vit-scene-match", auto_metric_logging=False)
    exp.log_parameters({
        k: str(v) if isinstance(v, (torch.device, Path)) else v
        for k, v in cfg.__dict__.items()
    })
    exp.log_parameters({"trial_number": trial.number})

    teacher = load_teacher(cfg)
    avp = create_avp(teacher, cfg)

    train_loader = InfiniteLoader(make_loader(
        cfg.train_dir,
        train_transform(cfg.scene_size, (cfg.crop_scale_min, 1.0)),
        cfg.batch_size, cfg.num_workers, shuffle=True,
    ))
    val_loader = InfiniteLoader(make_loader(
        cfg.val_dir,
        val_transform(cfg.scene_size),
        cfg.batch_size, cfg.num_workers, shuffle=True,
    ))

    trainable = [p for p in avp.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    log.info(f"Trainable: {n_trainable:,}, Teacher: {count_parameters(teacher):,}")
    exp.log_parameters({"trainable_params": n_trainable})

    peak_lr = get_linear_scaled_lr(cfg.ref_lr, cfg.batch_size)
    optimizer = torch.optim.AdamW(trainable, lr=peak_lr, weight_decay=cfg.weight_decay)
    warmup_steps = int(cfg.n_steps * cfg.warmup_ratio)
    scheduler = warmup_cosine_scheduler(optimizer, cfg.n_steps, warmup_steps)

    ckpt_path = cfg.ckpt_dir / f"{exp.get_key()}_best.pt"
    best_val_loss = float("inf")

    # Initial eval
    val_images = val_loader.next_batch().to(cfg.device)
    val_loss = eval_and_log(exp, 0, avp, teacher, val_images, cfg)
    save_checkpoint(avp, ckpt_path, exp, 0, val_loss)
    best_val_loss = val_loss

    # Initialize training state
    fresh_imgs = train_loader.next_batch().to(cfg.device)
    with torch.no_grad():
        fresh_targets = teacher.forward_norm_patches(fresh_imgs)
    state = TrainState.init(fresh_imgs, fresh_targets)

    ema_loss_t = torch.tensor(0.0, device=cfg.device)
    alpha = 2 / (cfg.log_every + 1)
    pbar = tqdm(range(cfg.n_steps), desc="Training", unit="step")

    for step in pbar:
        fresh_imgs = train_loader.next_batch().to(cfg.device)
        with torch.no_grad():
            fresh_targets = teacher.forward_norm_patches(fresh_imgs)

        vp = random_viewpoint(cfg.batch_size, cfg.device, cfg.min_viewpoint_scale, cfg.max_viewpoint_scale)
        out = avp.forward_step(state.images, vp, state.hidden)
        # out.scene = projected output for loss (no need to call output_proj)
        # out.hidden = internal state for continuation
        loss = nn.functional.mse_loss(out.scene, state.targets)

        if not torch.isfinite(loss):
            log.warning(f"NaN/Inf loss at step {step}, pruning trial")
            exp.end()
            raise optuna.TrialPruned()

        optimizer.zero_grad()
        loss.backward()
        grad_norm_t = torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        # Use out.hidden for continuation, avp.hidden_tokens for reset
        state = state.step(fresh_imgs, fresh_targets, out.hidden, cfg.survival_prob, avp.hidden_tokens)

        ema_loss_t = alpha * loss.detach() + (1 - alpha) * ema_loss_t if step > 0 else loss.detach()

        if step % cfg.log_every == 0:
            ema_loss = ema_loss_t.item()
            grad_norm = grad_norm_t.item()
            lr = scheduler.get_last_lr()[0]
            exp.log_metrics({"train/loss": ema_loss, "train/grad_norm": grad_norm, "train/lr": lr}, step=step)
            pbar.set_postfix_str(f"loss={ema_loss:.2e} grad={grad_norm:.2e} lr={lr:.2e}")

        if step > 0 and step % cfg.val_every == 0:
            val_images = val_loader.next_batch().to(cfg.device)
            val_loss = eval_and_log(exp, step, avp, teacher, val_images, cfg)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(avp, ckpt_path, exp, step, val_loss)
            trial.report(val_loss, step)
            if trial.should_prune():
                exp.end()
                raise optuna.TrialPruned()

    val_images = val_loader.next_batch().to(cfg.device)
    val_loss = eval_and_log(exp, cfg.n_steps, avp, teacher, val_images, cfg)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(avp, ckpt_path, exp, cfg.n_steps, val_loss)
    log.info(f"Final: train_ema={ema_loss_t.item():.4f}, val={val_loss:.4f}, best={best_val_loss:.4f}")
    exp.end()
    return best_val_loss


def main() -> None:
    import tyro

    torch.set_float32_matmul_precision("high")
    cfg = tyro.cli(Config)
    log.info(f"Config: {cfg}")

    def objective(trial: optuna.Trial) -> float:
        ref_lr = trial.suggest_float("ref_lr", 1e-6, 1e-2, log=True)
        train_cfg = replace(cfg, ref_lr=ref_lr)
        return train(train_cfg, trial)

    study = optuna.create_study(direction="minimize")
    study.enqueue_trial({"ref_lr": cfg.ref_lr})
    study.optimize(objective, n_trials=cfg.n_trials)

    log.info(f"Best trial: {study.best_trial.params}")
    log.info(f"Best val_loss: {study.best_value:.4f}")


if __name__ == "__main__":
    main()
