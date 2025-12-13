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
from sklearn.decomposition import PCA
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
from avp_vit.glimpse import Viewpoint, extract_glimpse

matplotlib.use("Agg")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

CKPT_PATH = Path("dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PATCH_SIZE = 16


@dataclass
class Config:
    # Data
    train_dir: Path = Path("/datasets/ILSVRC/Data/CLS-LOC/train")
    val_dir: Path = Path("/datasets/ILSVRC/Data/CLS-LOC/val")
    # Model
    scene_grid_size: int = 14
    glimpse_grid_size: int = 7
    gate_init: float = 1e-5
    use_output_proj: bool = True
    use_scene_registers: bool = True
    freeze_inner_backbone: bool = False
    # Viewpoints
    n_random_viewpoints: int = 1
    min_viewpoint_scale: float = 0.3
    max_viewpoint_scale: float = 1.0
    # Training
    n_steps: int = 50000
    batch_size: int = 32
    num_workers: int = 8
    ref_lr: float = 1e-5
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    # Logging
    log_every: int = 20
    viz_every: int = 200
    val_every: int = 200
    ckpt_every: int = 1000
    ckpt_dir: Path = Path("checkpoints")
    # Optuna
    n_trials: int = 1
    # Runtime
    device: torch.device = field(default_factory=get_sensible_device)

    @property
    def scene_size(self) -> int:
        return self.scene_grid_size * PATCH_SIZE

    @property
    def glimpse_size(self) -> int:
        return self.glimpse_grid_size * PATCH_SIZE


def random_viewpoint(
    B: int, device: torch.device, min_scale: float, max_scale: float
) -> Viewpoint:
    """Random viewpoint with scale in [min_scale, max_scale], center constrained to stay in bounds."""
    scales = torch.rand(B, device=device) * (max_scale - min_scale) + min_scale
    # Center must satisfy |c| <= 1 - s to keep glimpse in [-1, 1]
    max_offset = (1 - scales).unsqueeze(1)  # [B, 1]
    centers = (torch.rand(B, 2, device=device) * 2 - 1) * max_offset  # [B, 2] in [-max_offset, max_offset]
    return Viewpoint(name="random", centers=centers, scales=scales)


def make_viewpoints(B: int, cfg: Config) -> list[Viewpoint]:
    """Full scene + n_random random viewpoints."""
    vps = [Viewpoint.full_scene(B, cfg.device)]
    for _ in range(cfg.n_random_viewpoints):
        vps.append(random_viewpoint(B, cfg.device, cfg.min_viewpoint_scale, cfg.max_viewpoint_scale))
    return vps


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
    )
    return AVPViT(backbone_copy, avp_cfg).to(cfg.device)


def make_train_loader(cfg: Config) -> DataLoader[tuple[Tensor, Tensor]]:
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(cfg.scene_size, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    dataset = ImageFolder(str(cfg.train_dir), transform=transform)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )


def make_val_loader(cfg: Config) -> DataLoader[tuple[Tensor, Tensor]]:
    transform = transforms.Compose(
        [
            transforms.Resize(cfg.scene_size),
            transforms.CenterCrop(cfg.scene_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    dataset = ImageFolder(str(cfg.val_dir), transform=transform)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )


class ValIter:
    """Iterator over val loader that auto-restarts on exhaustion."""

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


def train_step(
    avp: AVPViT,
    teacher: DINOv3Backbone,
    images: Tensor,
    viewpoints: list[Viewpoint],
) -> Tensor:
    """Compute average MSE loss across viewpoints."""
    with torch.inference_mode():
        target = teacher.forward_norm_patches(images)
    target = target.clone()
    return avp.forward_with_loss(
        images, viewpoints, lambda scene: nn.functional.mse_loss(scene, target)
    )


def fit_pca(features: Tensor) -> PCA:
    pca = PCA(n_components=3, whiten=True)
    pca.fit(features.float().cpu().numpy())
    return pca


def pca_rgb(pca: PCA, features: Tensor, H: int, W: int) -> Tensor:
    proj = pca.transform(features.float().cpu().numpy())
    return torch.sigmoid(torch.from_numpy(proj).view(H, W, 3) * 2.0)


def tensor_to_img(t: Tensor) -> Tensor:
    """Convert [3, H, W] tensor to displayable [H, W, 3] in [0, 1]."""
    t = t.detach().cpu().float()
    t = (t - t.min()) / (t.max() - t.min() + 1e-8)
    return t.permute(1, 2, 0)


@dataclass
class MultistepResult:
    """Results from multi-step AVP inference."""

    teacher_patches: Tensor
    scenes: list[Tensor]
    locals: list[Tensor]
    glimpse_imgs: list[Tensor]
    mses: list[float]
    viewpoints: list[Viewpoint]


def run_multistep_inference(
    avp: AVPViT,
    teacher: DINOv3Backbone,
    images: Tensor,
    viewpoints: list[Viewpoint],
) -> MultistepResult:
    """Run multi-step inference, collecting intermediate scenes, locals, and MSEs."""
    n_prefix = teacher.n_prefix_tokens
    with torch.inference_mode():
        teacher_patches = teacher.forward_norm_patches(images)
        scene: Tensor | None = None
        scenes: list[Tensor] = []
        locals_list: list[Tensor] = []
        glimpse_imgs: list[Tensor] = []
        mses: list[float] = []

        for vp in viewpoints:
            glimpse_imgs.append(extract_glimpse(images, vp, avp.glimpse_size))
            local, scene = avp.forward_step(images, vp, scene)
            # Strip prefix tokens and apply teacher's output norm for fair comparison
            local_patches = teacher.output_norm(local[:, n_prefix:])
            locals_list.append(local_patches)
            scene_proj = avp.output_proj(scene)
            scenes.append(scene_proj)
            mse = nn.functional.mse_loss(scene_proj, teacher_patches).item()
            mses.append(mse)

    return MultistepResult(
        teacher_patches, scenes, locals_list, glimpse_imgs, mses, viewpoints
    )


def plot_multistep_pca(
    result: MultistepResult,
    teacher_img: Tensor,
    scene_grid_size: int,
    glimpse_grid_size: int,
) -> Figure:
    """Create multi-row PCA visualization figure.

    Columns: Glimpse | Teacher | Scene | Local | Delta | Error
    - Local: 7x7 processed glimpse features using same PCA as scene
    - Delta: spatial change in scene from previous timestep (blank for t=0)
    """
    S = scene_grid_size
    G = glimpse_grid_size
    B = teacher_img.shape[0]
    n_views = len(result.viewpoints)

    idx = int(torch.randint(B, (1,)).item())
    teacher_i = result.teacher_patches[idx]
    pca = fit_pca(teacher_i)
    teacher_pca = pca_rgb(pca, teacher_i, S, S)

    fig, axes = plt.subplots(n_views, 6, figsize=(24, 4 * n_views))

    for row, vp in enumerate(result.viewpoints):
        glimpse_i = result.glimpse_imgs[row][idx]
        scene_i = result.scenes[row][idx]
        local_i = result.locals[row][idx]
        scene_pca = pca_rgb(pca, scene_i, S, S)
        local_pca = pca_rgb(pca, local_i, G, G)
        error_map = (scene_i - teacher_i).pow(2).mean(dim=-1).view(S, S).cpu()

        # Delta: change from previous scene (L2 norm of difference per patch)
        if row > 0:
            prev_scene_i = result.scenes[row - 1][idx]
            delta_map = (scene_i - prev_scene_i).pow(2).mean(dim=-1).view(S, S).cpu()
        else:
            delta_map = torch.zeros(S, S)

        axes[row, 0].imshow(tensor_to_img(glimpse_i).numpy())
        axes[row, 0].set_title(f"Glimpse ({vp.name})")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(teacher_pca.numpy())
        axes[row, 1].set_title("Teacher (HR)")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(scene_pca.numpy())
        axes[row, 2].set_title(f"Scene t={row}")
        axes[row, 2].axis("off")

        axes[row, 3].imshow(local_pca.numpy())
        axes[row, 3].set_title(f"Local t={row}")
        axes[row, 3].axis("off")

        im_delta = axes[row, 4].imshow(delta_map.numpy(), cmap="viridis")
        axes[row, 4].set_title("Δ Scene" if row > 0 else "Δ (n/a)")
        axes[row, 4].axis("off")
        fig.colorbar(im_delta, ax=axes[row, 4], fraction=0.046, pad=0.04)

        im_err = axes[row, 5].imshow(error_map.numpy(), cmap="hot")
        axes[row, 5].set_title(f"MSE={result.mses[row]:.2f}")
        axes[row, 5].axis("off")
        fig.colorbar(im_err, ax=axes[row, 5], fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


def log_train_pca(
    exp: comet_ml.Experiment,
    step: int,
    avp: AVPViT,
    teacher: DINOv3Backbone,
    images: Tensor,
    viewpoints: list[Viewpoint],
    cfg: Config,
) -> None:
    """Log multi-row train PCA viz to Comet."""
    result = run_multistep_inference(avp, teacher, images, viewpoints)
    fig = plot_multistep_pca(
        result, images, cfg.scene_grid_size, cfg.glimpse_grid_size
    )
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    exp.log_image(buf, name="train/pca", step=step)
    plt.close(fig)


def eval_and_log(
    exp: comet_ml.Experiment,
    step: int,
    avp: AVPViT,
    teacher: DINOv3Backbone,
    val_iter: "ValIter",
    cfg: Config,
) -> float:
    """Eval on one random val batch, log multi-row PCA plots to Comet."""
    imgs = val_iter.next_batch()
    images = imgs.to(cfg.device)
    viewpoints = make_viewpoints(images.shape[0], cfg)

    result = run_multistep_inference(avp, teacher, images, viewpoints)

    fig = plot_multistep_pca(
        result, images, cfg.scene_grid_size, cfg.glimpse_grid_size
    )
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    exp.log_image(buf, name="val/pca", step=step)
    plt.close(fig)

    # Log MSE evolution
    for t, mse in enumerate(result.mses):
        exp.log_metric(f"val/mse_t{t}", mse, step=step)

    val_loss = result.mses[-1]
    exp.log_metric("val/loss", val_loss, step=step)
    return val_loss


def save_checkpoint(
    avp: AVPViT, ckpt_path: Path, exp: comet_ml.Experiment, step: int, val_loss: float
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(avp.state_dict(), ckpt_path)
    size_mb = ckpt_path.stat().st_size / (1024 * 1024)
    log.info(
        f"Saved checkpoint: {ckpt_path} ({size_mb:.1f} MB), val_loss={val_loss:.4f}"
    )
    exp.log_metric("ckpt/val_loss", val_loss, step=step)


def train(cfg: Config, trial: optuna.Trial) -> float:
    """Train AVP model and return best val_loss for HP optimization."""
    exp = comet_ml.Experiment(
        project_name="avp-vit-scene-match", auto_metric_logging=False
    )
    exp.log_parameters(
        {
            k: str(v) if isinstance(v, (torch.device, Path)) else v
            for k, v in cfg.__dict__.items()
        }
    )
    exp.log_parameters({"trial_number": trial.number})

    teacher = load_teacher(cfg.device)
    avp = create_avp(teacher, cfg)

    train_loader = make_train_loader(cfg)
    val_iter = ValIter(make_val_loader(cfg))
    train_iter = iter(train_loader)

    trainable = [p for p in avp.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_teacher = count_parameters(teacher)
    n_train = len(train_loader.dataset)  # type: ignore[arg-type]
    n_val = len(val_iter._loader.dataset)  # type: ignore[arg-type]
    log.info(f"Trainable: {n_trainable:,}, Teacher: {n_teacher:,}")
    log.info(f"Train: {n_train:,}, Val: {n_val:,}")
    exp.log_parameters(
        {
            "trainable_params": n_trainable,
            "teacher_params": n_teacher,
            "train_size": n_train,
            "val_size": n_val,
        }
    )

    peak_lr = get_linear_scaled_lr(cfg.ref_lr, cfg.batch_size)
    optimizer = torch.optim.AdamW(trainable, lr=peak_lr, weight_decay=cfg.weight_decay)
    warmup_steps = int(cfg.n_steps * cfg.warmup_ratio)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.n_steps - warmup_steps, eta_min=0.0
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
    )

    ckpt_path = cfg.ckpt_dir / f"{exp.get_key()}_best.pt"
    best_val_loss = float("inf")

    val_loss = eval_and_log(exp, 0, avp, teacher, val_iter, cfg)
    save_checkpoint(avp, ckpt_path, exp, 0, val_loss)
    best_val_loss = val_loss

    ema_loss_t = torch.tensor(0.0, device=cfg.device)
    alpha = 2 / (cfg.log_every + 1)
    pbar = tqdm(range(cfg.n_steps), desc="Training", unit="step")
    for step in pbar:
        try:
            imgs, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            imgs, _ = next(train_iter)

        teacher_img = imgs.to(cfg.device)
        viewpoints = make_viewpoints(teacher_img.shape[0], cfg)

        optimizer.zero_grad()
        loss = train_step(avp, teacher, teacher_img, viewpoints)
        if not torch.isfinite(loss):
            log.warning(f"NaN/Inf loss at step {step}, pruning trial")
            exp.end()
            raise optuna.TrialPruned()
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
            sps = (
                pbar.format_dict["rate"] * cfg.batch_size
                if pbar.format_dict["rate"]
                else 0
            )
            exp.log_metrics(
                {"train/loss": ema_loss, "train/grad_norm": grad_norm, "train/lr": lr},
                step=step,
            )
            pbar.set_postfix_str(
                f"loss={ema_loss:.2e} grad={grad_norm:.2e} lr={lr:.2e} sps={sps:.0f}"
            )

        if step > 0 and step % cfg.viz_every == 0:
            log_train_pca(exp, step, avp, teacher, teacher_img, viewpoints, cfg)

        if step > 0 and step % cfg.val_every == 0:
            val_loss = eval_and_log(exp, step, avp, teacher, val_iter, cfg)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if step % cfg.ckpt_every == 0:
                    save_checkpoint(avp, ckpt_path, exp, step, val_loss)
            # Report intermediate for pruning
            trial.report(val_loss, step)
            if trial.should_prune():
                exp.end()
                raise optuna.TrialPruned()

    val_loss = eval_and_log(exp, cfg.n_steps, avp, teacher, val_iter, cfg)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(avp, ckpt_path, exp, cfg.n_steps, val_loss)
    log.info(
        f"Final: train_ema={ema_loss_t.item():.4f}, val={val_loss:.4f}, best={best_val_loss:.4f}"
    )
    exp.end()
    return best_val_loss


def main() -> None:
    import tyro

    torch.set_float32_matmul_precision("high")
    cfg = tyro.cli(Config)
    log.info(f"Config: {cfg}")

    def objective(trial: optuna.Trial) -> float:
        # Suggest HPs - when enqueued, returns the enqueued value
        ref_lr = trial.suggest_float("ref_lr", 1e-6, 1e-2, log=True)
        train_cfg = replace(cfg, ref_lr=ref_lr)
        return train(train_cfg, trial)

    study = optuna.create_study(direction="minimize")
    # Enqueue defaults so first trial (n_trials=1) uses CLI config
    study.enqueue_trial({"ref_lr": cfg.ref_lr})
    study.optimize(objective, n_trials=cfg.n_trials)

    log.info(f"Best trial: {study.best_trial.params}")
    log.info(f"Best val_loss: {study.best_value:.4f}")


if __name__ == "__main__":
    main()
