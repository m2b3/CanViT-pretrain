"""Train AVP scene representation to match frozen teacher backbone patches."""

import copy
import io
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

import comet_ml
import matplotlib
import matplotlib.pyplot as plt
import optuna
import torch
import torch.nn as nn
from dinov3.hub.backbones import dinov3_vits16
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from ymc.lr import get_linear_scaled_lr
from ytch.device import get_sensible_device
from ytch.model import count_parameters

from avp_vit import AVPConfig, AVPViT
from avp_vit.backbone.dinov3 import DINOv3Backbone
from avp_vit.glimpse import Viewpoint
from avp_vit.train import TrainState, fit_pca, imagenet_denormalize, make_eval_viewpoints, pca_rgb, random_viewpoint

matplotlib.use("Agg")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

CKPT_PATH = Path("dinov3_vits16_pretrain_lvd1689m-08c60483.pth")


@dataclass
class Config:
    # Data
    train_dir: Path = Path("/datasets/ILSVRC/Data/CLS-LOC/train")
    val_dir: Path = Path("/datasets/ILSVRC/Data/CLS-LOC/val")
    # Model
    scene_grid_size: int = 16
    glimpse_grid_size: int = 7
    gate_init: float = 1e-5
    use_output_proj: bool = True
    use_scene_registers: bool = True
    freeze_inner_backbone: bool = False
    gradient_checkpointing: bool = True
    # Training
    survival_prob: float = 0.5  # Bernoulli survival: prob of carrying over batch item
    n_steps: int = 20000
    batch_size: int = 64
    num_workers: int = 8
    ref_lr: float = 1e-5
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    # Logging
    log_every: int = 20
    viz_every: int = 50
    val_every: int = 200
    ckpt_every: int = 1000
    ckpt_dir: Path = Path("checkpoints")
    # Optuna
    n_trials: int = 100
    # Runtime
    device: torch.device = field(default_factory=get_sensible_device)

    @property
    def min_viewpoint_scale(self) -> float:
        """Finest meaningful scale: glimpse covers exactly one scene grid cell."""
        return self.glimpse_grid_size / self.scene_grid_size

    @property
    def max_viewpoint_scale(self) -> float:
        return 1.0


def load_teacher(device: torch.device) -> DINOv3Backbone:
    model = dinov3_vits16(weights=str(CKPT_PATH), pretrained=True)
    backbone = DINOv3Backbone(model.eval().to(device))
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


def make_train_loader(cfg: Config, scene_size: int) -> DataLoader[tuple[Tensor, Tensor]]:
    transform = transforms.Compose([
        transforms.RandomResizedCrop(scene_size, scale=(0.4, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    dataset = ImageFolder(str(cfg.train_dir), transform=transform)
    return DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )


def make_val_loader(cfg: Config, scene_size: int) -> DataLoader[tuple[Tensor, Tensor]]:
    transform = transforms.Compose([
        transforms.Resize(scene_size),
        transforms.CenterCrop(scene_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    dataset = ImageFolder(str(cfg.val_dir), transform=transform)
    return DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )


class InfiniteLoader:
    """Infinite iterator over a DataLoader."""

    def __init__(self, loader: DataLoader[tuple[Tensor, Tensor]]) -> None:
        self._loader = loader
        self._iter = iter(loader)

    def next_batch(self) -> Tensor:
        try:
            imgs, _ = next(self._iter)
        except StopIteration:
            self._iter = iter(self._loader)
            imgs, _ = next(self._iter)
        return imgs


# --- Visualization ---


def _timestep_colors(n: int) -> list[tuple[float, float, float, float]]:
    cmap = plt.get_cmap("viridis")
    return [cmap(i / max(1, n - 1)) for i in range(n)]


def plot_viewpoint_trajectory(image: Tensor, viewpoints: list[Viewpoint], batch_idx: int = 0) -> Figure:
    """Plot image with viewpoint boxes for a single sample."""
    img = imagenet_denormalize(image.detach().cpu()).numpy()
    H, W = img.shape[:2]
    colors = _timestep_colors(len(viewpoints))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)

    for i, vp in enumerate(viewpoints):
        cy, cx = vp.centers[batch_idx].cpu().numpy()
        s = vp.scales[batch_idx].item()
        px_cx, px_cy = (cx + 1) / 2 * W, (cy + 1) / 2 * H
        hw, hh = s * W / 2, s * H / 2

        rect = Rectangle(
            (px_cx - hw, px_cy - hh), 2 * hw, 2 * hh,
            linewidth=2, edgecolor=colors[i], facecolor="none",
            label=f"t={i} ({vp.name}, s={s:.2f})",
        )
        ax.add_patch(rect)
        ax.plot(px_cx, px_cy, "o", color=colors[i], markersize=6)

    ax.set_title(f"Trajectory (sample {batch_idx})")
    ax.legend(loc="upper right", fontsize=8)
    ax.axis("off")
    plt.tight_layout()
    return fig


def run_eval_inference(
    avp: AVPViT, teacher: DINOv3Backbone, images: Tensor, device: torch.device
) -> tuple[list[float], list[Viewpoint]]:
    """Run evaluation scheme and return MSEs at each step."""
    B = images.shape[0]
    viewpoints = make_eval_viewpoints(B, device)

    with torch.inference_mode():
        target = teacher.forward_norm_patches(images)
        scene = None
        mses = []
        for vp in viewpoints:
            out = avp.forward_step(images, vp, scene)
            scene = out.scene
            mse = nn.functional.mse_loss(avp.output_proj(scene), target).item()
            mses.append(mse)
    return mses, viewpoints


def plot_eval_pca(
    avp: AVPViT, teacher: DINOv3Backbone, images: Tensor,
    scene_grid_size: int, sample_idx: int = 0,
) -> Figure:
    """Create PCA visualization for evaluation."""
    B = images.shape[0]
    device = images.device
    viewpoints = make_eval_viewpoints(B, device)
    S = scene_grid_size

    with torch.inference_mode():
        target = teacher.forward_norm_patches(images)
        scene = None
        scenes = []
        for vp in viewpoints:
            out = avp.forward_step(images, vp, scene)
            scene = out.scene
            scenes.append(avp.output_proj(scene))

    # Move to CPU for visualization
    teacher_np = target[sample_idx].cpu().float().numpy()
    pca = fit_pca(teacher_np)
    teacher_pca = pca_rgb(pca, teacher_np, S, S)

    n_views = len(viewpoints)
    fig, axes = plt.subplots(1, n_views + 1, figsize=(4 * (n_views + 1), 4))

    axes[0].imshow(teacher_pca)
    axes[0].set_title("Teacher")
    axes[0].axis("off")

    for i, scene_t in enumerate(scenes):
        scene_np = scene_t[sample_idx].cpu().float().numpy()
        scene_pca = pca_rgb(pca, scene_np, S, S)
        mse = ((scene_np - teacher_np) ** 2).mean()
        axes[i + 1].imshow(scene_pca)
        axes[i + 1].set_title(f"t={i} ({viewpoints[i].name})\nMSE={mse:.4f}")
        axes[i + 1].axis("off")

    plt.tight_layout()
    return fig


def log_figure(exp: comet_ml.Experiment, fig: Figure, name: str, step: int) -> None:
    with io.BytesIO() as buf:
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        exp.log_image(buf, name=name, step=step)
    plt.close(fig)


# --- Training ---


def eval_and_log(
    exp: comet_ml.Experiment, step: int,
    avp: AVPViT, teacher: DINOv3Backbone,
    val_loader: InfiniteLoader, cfg: Config,
) -> float:
    """Evaluate on one batch and log to Comet."""
    images = val_loader.next_batch().to(cfg.device)
    mses, viewpoints = run_eval_inference(avp, teacher, images, cfg.device)

    for t, mse in enumerate(mses):
        exp.log_metric(f"val/mse_t{t}", mse, step=step)

    fig = plot_eval_pca(avp, teacher, images, cfg.scene_grid_size)
    log_figure(exp, fig, "val/pca", step)

    fig_traj = plot_viewpoint_trajectory(images[0], viewpoints)
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

    teacher = load_teacher(cfg.device)
    avp = create_avp(teacher, cfg)
    scene_size = teacher.patch_size * cfg.scene_grid_size

    train_loader = InfiniteLoader(make_train_loader(cfg, scene_size))
    val_loader = InfiniteLoader(make_val_loader(cfg, scene_size))

    trainable = [p for p in avp.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    log.info(f"Trainable: {n_trainable:,}, Teacher: {count_parameters(teacher):,}")
    exp.log_parameters({"trainable_params": n_trainable})

    peak_lr = get_linear_scaled_lr(cfg.ref_lr, cfg.batch_size)
    optimizer = torch.optim.AdamW(trainable, lr=peak_lr, weight_decay=cfg.weight_decay)
    warmup_steps = int(cfg.n_steps * cfg.warmup_ratio)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.n_steps - warmup_steps, eta_min=0.0)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    ckpt_path = cfg.ckpt_dir / f"{exp.get_key()}_best.pt"
    best_val_loss = float("inf")

    # Initial eval
    val_loss = eval_and_log(exp, 0, avp, teacher, val_loader, cfg)
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
        # Get fresh batch for potential replacement
        fresh_imgs = train_loader.next_batch().to(cfg.device)
        with torch.no_grad():
            fresh_targets = teacher.forward_norm_patches(fresh_imgs)

        # Forward step with current state
        vp = random_viewpoint(cfg.batch_size, cfg.device, cfg.min_viewpoint_scale, cfg.max_viewpoint_scale)
        out = avp.forward_step(state.images, vp, state.scene)
        loss = nn.functional.mse_loss(avp.output_proj(out.scene), state.targets)

        if not torch.isfinite(loss):
            log.warning(f"NaN/Inf loss at step {step}, pruning trial")
            exp.end()
            raise optuna.TrialPruned()

        # Backward
        optimizer.zero_grad()
        loss.backward()
        grad_norm_t = torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        # Update state with Bernoulli survival
        state = state.step(fresh_imgs, fresh_targets, out.scene, cfg.survival_prob, avp.scene_tokens)

        # EMA loss tracking
        ema_loss_t = alpha * loss.detach() + (1 - alpha) * ema_loss_t if step > 0 else loss.detach()

        # Logging
        if step % cfg.log_every == 0:
            ema_loss = ema_loss_t.item()
            grad_norm = grad_norm_t.item()
            lr = scheduler.get_last_lr()[0]
            exp.log_metrics({"train/loss": ema_loss, "train/grad_norm": grad_norm, "train/lr": lr}, step=step)
            pbar.set_postfix_str(f"loss={ema_loss:.2e} grad={grad_norm:.2e} lr={lr:.2e}")

        # Validation
        if step > 0 and step % cfg.val_every == 0:
            val_loss = eval_and_log(exp, step, avp, teacher, val_loader, cfg)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if step % cfg.ckpt_every == 0:
                    save_checkpoint(avp, ckpt_path, exp, step, val_loss)
            trial.report(val_loss, step)
            if trial.should_prune():
                exp.end()
                raise optuna.TrialPruned()

    # Final eval
    val_loss = eval_and_log(exp, cfg.n_steps, avp, teacher, val_loader, cfg)
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
