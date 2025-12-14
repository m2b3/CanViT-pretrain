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

from avp_vit import AVPConfig, AVPViT, StepOutput
from avp_vit.backbone.dinov3 import DINOv3Backbone
from avp_vit.glimpse import Viewpoint
from avp_vit.train import (
    InfiniteLoader,
    TrainState,
    imagenet_denormalize,
    make_eval_viewpoints,
    make_loader,
    plot_multistep_pca,
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
    avp: AVPConfig = field(default_factory=lambda: AVPConfig(
        scene_grid_size=16,
        glimpse_grid_size=7,
        gate_init=1e-5,
        use_output_proj=True,
        use_scene_registers=True,
        gradient_checkpointing=True,
    ))
    freeze_inner_backbone: bool = False
    # Training
    survival_prob: float = 0.5
    n_viewpoints_per_step: int = 2  # Inner loop: viewpoints per optimizer step (>=2 for length generalization)
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
        return self.avp.glimpse_grid_size / self.avp.scene_grid_size

    @property
    def max_viewpoint_scale(self) -> float:
        return 1.0


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
    return AVPViT(backbone_copy, cfg.avp).to(cfg.device)


def log_figure(exp: comet_ml.Experiment, fig: Figure, name: str, step: int) -> None:
    with io.BytesIO() as buf:
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        exp.log_image(buf, name=name, step=step)
    plt.close(fig)


def log_multistep_viz(
    exp: comet_ml.Experiment,
    step: int,
    prefix: str,
    image: Tensor,
    teacher_patches: Tensor,
    outputs: list[StepOutput],
    viewpoints: list[Viewpoint],
    initial_scene: Tensor,
    avp: AVPViT,
    teacher: DINOv3Backbone,
) -> None:
    """Log full multi-row PCA visualization and trajectory to Comet."""
    assert isinstance(avp.backbone, DINOv3Backbone)
    avp_backbone = avp.backbone

    sample_idx = 0
    n_prefix = teacher.n_prefix_tokens
    H, W = avp.scene_size, avp.scene_size

    full_img = imagenet_denormalize(image.cpu()).numpy()
    teacher_np = teacher_patches.cpu().float().numpy()
    initial_np = initial_scene.cpu().float().numpy()

    scenes_np = [out.scene[sample_idx].cpu().float().numpy() for out in outputs]
    # CRITICAL: locals_avp uses AVP's TRAINABLE backbone, locals_teacher uses FROZEN teacher
    # These will diverge as training progresses - comparing them shows representation drift
    locals_avp_np = [
        avp_backbone.output_norm(out.local[sample_idx : sample_idx + 1, n_prefix:])
        .squeeze(0).cpu().float().numpy()
        for out in outputs
    ]
    locals_teacher_np = [
        teacher.output_norm(out.local[sample_idx : sample_idx + 1, n_prefix:])
        .squeeze(0).cpu().float().numpy()
        for out in outputs
    ]
    glimpses_np = [
        imagenet_denormalize(out.glimpse[sample_idx].cpu()).numpy()
        for out in outputs
    ]
    boxes = [vp.to_pixel_box(sample_idx, H, W) for vp in viewpoints]
    names = [vp.name for vp in viewpoints]

    fig_pca = plot_multistep_pca(
        full_img, teacher_np, scenes_np, locals_avp_np, locals_teacher_np, glimpses_np,
        boxes, names, avp.cfg.scene_grid_size, avp.cfg.glimpse_grid_size, initial_np,
    )
    log_figure(exp, fig_pca, f"{prefix}/pca", step)

    fig_traj = plot_trajectory(full_img, boxes, names)
    log_figure(exp, fig_traj, f"{prefix}/trajectory", step)


def eval_and_log(
    exp: comet_ml.Experiment,
    step: int,
    avp: AVPViT,
    teacher: DINOv3Backbone,
    images: Tensor,
) -> float:
    """Evaluate on one batch and log to Comet. Returns final MSE."""
    B = images.shape[0]
    viewpoints = make_eval_viewpoints(B, images.device)

    with torch.inference_mode():
        target = teacher.forward_norm_patches(images)
        outputs, _ = avp.forward_trajectory_full(images, viewpoints)
        mses = [nn.functional.mse_loss(out.scene, target).item() for out in outputs]
        initial_scene = avp.output_proj(avp.hidden_tokens.expand(1, -1, -1))[0]

    for t, mse in enumerate(mses):
        exp.log_metric(f"val/mse_t{t}", mse, step=step)

    log_multistep_viz(
        exp, step, "val", images[0], target[0], outputs, viewpoints,
        initial_scene, avp, teacher,
    )

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
        train_transform(avp.scene_size, (cfg.crop_scale_min, 1.0)),
        cfg.batch_size, cfg.num_workers, shuffle=True,
    ))
    val_loader = InfiniteLoader(make_loader(
        cfg.val_dir,
        val_transform(avp.scene_size),
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
    val_loss = eval_and_log(exp, 0, avp, teacher, val_images)
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

        # Inner loop: multiple viewpoints per optimizer step (for length generalization)
        viewpoints = [
            random_viewpoint(cfg.batch_size, cfg.device, cfg.min_viewpoint_scale, cfg.max_viewpoint_scale)
            for _ in range(cfg.n_viewpoints_per_step)
        ]
        loss, final_hidden = avp.forward_loss(state.images, viewpoints, state.targets, state.hidden)

        if not torch.isfinite(loss):
            log.warning(f"NaN/Inf loss at step {step}, pruning trial")
            exp.end()
            raise optuna.TrialPruned()

        optimizer.zero_grad()
        loss.backward()
        grad_norm_t = torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        # Bernoulli survival at optimizer step boundary
        state = state.step(fresh_imgs, fresh_targets, final_hidden, cfg.survival_prob, avp.hidden_tokens)

        ema_loss_t = alpha * loss.detach() + (1 - alpha) * ema_loss_t if step > 0 else loss.detach()

        if step % cfg.log_every == 0:
            ema_loss = ema_loss_t.item()
            grad_norm = grad_norm_t.item()
            lr = scheduler.get_last_lr()[0]
            exp.log_metrics({"train/loss": ema_loss, "train/grad_norm": grad_norm, "train/lr": lr}, step=step)
            pbar.set_postfix_str(f"loss={ema_loss:.2e} grad={grad_norm:.2e} lr={lr:.2e}")

        if step > 0 and step % cfg.val_every == 0:
            val_images = val_loader.next_batch().to(cfg.device)
            val_loss = eval_and_log(exp, step, avp, teacher, val_images)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(avp, ckpt_path, exp, step, val_loss)
            trial.report(val_loss, step)
            if trial.should_prune():
                exp.end()
                raise optuna.TrialPruned()

    val_images = val_loader.next_batch().to(cfg.device)
    val_loss = eval_and_log(exp, cfg.n_steps, avp, teacher, val_images)
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
