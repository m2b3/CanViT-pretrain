"""Train AVP scene representation to match frozen teacher backbone patches."""

import copy
import io
import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import comet_ml
import matplotlib.pyplot as plt
import optuna
import torch
from dinov3.hub.backbones import dinov3_vits16
from matplotlib.figure import Figure
from torch import Tensor
from torch.nn.functional import (  # noqa: F401 (l1_loss for easy switching)
    l1_loss,
    mse_loss,
)
from tqdm import tqdm
from ymc.lr import get_linear_scaled_lr
from ytch.device import get_sensible_device
from ytch.model import count_parameters

from avp_vit import AVPConfig, AVPViT
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


class TargetNorm(torch.nn.Module):
    """Position-aware running normalization for [B, N, D] targets.

    Always updates AND uses running stats - no train/eval mode distinction.
    First batch initializes stats directly (no warmup period with wrong stats).
    All ops are in-place on GPU buffers (no sync).

    Stats shape: [N, D] - one mean/std per token position per dimension.
    """

    mean: Tensor
    var: Tensor

    def __init__(
        self, n_tokens: int, embed_dim: int, momentum: float = 0.1, eps: float = 1e-5
    ):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("mean", torch.zeros(n_tokens, embed_dim))
        self.register_buffer("var", torch.ones(n_tokens, embed_dim))
        self.initialized = False  # Python bool, not tensor - no GPU sync on check

    def forward(self, x: Tensor) -> Tensor:
        """Update running stats and normalize. x: [B, N, D] -> [B, N, D]."""
        with torch.no_grad():
            batch_mean = x.mean(dim=0)  # [N, D]
            batch_var = x.var(
                dim=0, unbiased=True
            )  # [N, D], unbiased for population estimate

            if not self.initialized:
                self.mean.copy_(batch_mean)
                self.var.copy_(batch_var)
                self.initialized = True
            else:
                self.mean.lerp_(batch_mean, self.momentum)
                self.var.lerp_(batch_var, self.momentum)

        return (x - self.mean) / (self.var + self.eps).sqrt()


LOSS_FN = mse_loss

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
    avp: AVPConfig = field(
        default_factory=lambda: AVPConfig(
            scene_grid_size=32,
            glimpse_grid_size=7,
            layer_scale_init=1e-3,
            use_output_proj=True,
            use_scene_registers=True,
            gradient_checkpointing=True,
            use_convex_gating=True,
            use_local_temporal=True,
        )
    )
    freeze_inner_backbone: bool = False
    # Training
    fresh_ratio: float = 0.25  # Fraction of batch replaced each step
    n_viewpoints_per_step: int = (
        2  # Inner loop: viewpoints per optimizer step (>=2 for length generalization)
    )
    n_steps: int = 200000
    batch_size: int = 64
    num_workers: int = 8
    ref_lr: float = 2.5e-5
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.01
    grad_clip: float = 1.0
    crop_scale_min: float = 0.4
    # Logging
    log_every: int = 20
    val_every: int = 50
    ckpt_every: int = 500
    # Compilation
    compile: bool = True
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


def compile_avp(avp: AVPViT) -> None:
    """Wrap DINOv3 blocks and cross-attention modules with torch.compile."""
    n_blocks = avp.backbone.n_blocks
    log.info(
        f"Wrapping {n_blocks} DINOv3 blocks + {n_blocks} read/write attention pairs for compilation"
    )

    assert isinstance(avp.backbone, DINOv3Backbone)
    blocks = avp.backbone._backbone.blocks
    for i in range(n_blocks):
        blocks[i] = torch.compile(blocks[i])  # type: ignore[assignment]

    for i in range(n_blocks):
        avp.read_attn[i] = torch.compile(avp.read_attn[i])  # type: ignore[assignment]
        avp.write_attn[i] = torch.compile(avp.write_attn[i])  # type: ignore[assignment]


def log_figure(exp: comet_ml.Experiment, fig: Figure, name: str, step: int) -> None:
    with io.BytesIO() as buf:
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        exp.log_image(buf, name=name, step=step)
    plt.close(fig)


def viz_and_log(
    exp: comet_ml.Experiment,
    step: int,
    prefix: str,
    avp: AVPViT,
    teacher: DINOv3Backbone,
    images: Tensor,
    viewpoints: list[Viewpoint],
    target: Tensor,
    hidden: Tensor | None,
) -> tuple[list[float], list[float]]:
    """Run forward trajectory and log visualization.

    This is the core visualization function used by both validation and training viz.
    Returns (l1_losses, mse_losses) per timestep.
    """
    assert isinstance(avp.backbone, DINOv3Backbone)
    avp_backbone = avp.backbone

    with torch.inference_mode():
        outputs, _, _ = avp.forward_trajectory_full(images, viewpoints, hidden)
        l1_losses = [l1_loss(out.scene, target).item() for out in outputs]
        mse_losses = [mse_loss(out.scene, target).item() for out in outputs]

        # Initial scene from hidden (or spatial_init if None)
        if hidden is not None:
            initial_scene = avp.compute_scene(hidden[0:1])[0]
        else:
            initial_scene = avp.output_proj(avp.spatial_init)[0]

        # Prepare viz data for first sample
        sample_idx = 0
        n_prefix = teacher.n_prefix_tokens
        H, W = avp.scene_size, avp.scene_size

        full_img = imagenet_denormalize(images[sample_idx].cpu()).numpy()
        teacher_np = target[sample_idx].cpu().float().numpy()
        initial_np = initial_scene.cpu().float().numpy()

        scenes = [out.scene[sample_idx].cpu().float().numpy() for out in outputs]
        locals_avp = [
            avp_backbone.output_norm(out.local[sample_idx : sample_idx + 1, n_prefix:])
            .squeeze(0)
            .cpu()
            .float()
            .numpy()
            for out in outputs
        ]
        locals_teacher = [
            teacher.forward_norm_patches(out.glimpse[sample_idx : sample_idx + 1])
            .squeeze(0)
            .cpu()
            .float()
            .numpy()
            for out in outputs
        ]
        glimpses = [
            imagenet_denormalize(out.glimpse[sample_idx].cpu()).numpy()
            for out in outputs
        ]
        boxes = [vp.to_pixel_box(sample_idx, H, W) for vp in viewpoints]
        names = [vp.name for vp in viewpoints]

    fig_pca = plot_multistep_pca(
        full_img,
        teacher_np,
        scenes,
        locals_avp,
        locals_teacher,
        glimpses,
        boxes,
        names,
        avp.cfg.scene_grid_size,
        avp.cfg.glimpse_grid_size,
        initial_np,
    )
    log_figure(exp, fig_pca, f"{prefix}/pca", step)

    fig_traj = plot_trajectory(full_img, boxes, names)
    log_figure(exp, fig_traj, f"{prefix}/trajectory", step)

    return l1_losses, mse_losses


def eval_and_log(
    exp: comet_ml.Experiment,
    step: int,
    avp: AVPViT,
    teacher: DINOv3Backbone,
    get_targets: Callable[[Tensor], Tensor],
    images: Tensor,
) -> float:
    """Evaluate on one batch with fixed viewpoints. Returns final L1 loss."""
    B = images.shape[0]
    viewpoints = make_eval_viewpoints(B, images.device)

    with torch.inference_mode():
        target = get_targets(images)

    l1_losses, mse_losses = viz_and_log(
        exp, step, "val", avp, teacher, images, viewpoints, target, None
    )

    for t, (l1, mse) in enumerate(zip(l1_losses, mse_losses, strict=True)):
        exp.log_metric(f"val/l1_t{t}", l1, step=step)
        exp.log_metric(f"val/mse_t{t}", mse, step=step)

    exp.log_metric("val/l1", l1_losses[-1], step=step)
    exp.log_metric("val/mse", mse_losses[-1], step=step)
    return l1_losses[-1]


def save_checkpoint(
    avp: AVPViT, path: Path, exp: comet_ml.Experiment, step: int, val_loss: float
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(avp.state_dict(), path)
    size_mb = path.stat().st_size / (1024 * 1024)
    log.info(f"Saved checkpoint: {path} ({size_mb:.1f} MB), val_loss={val_loss:.4f}")
    exp.log_metric("ckpt/val_loss", val_loss, step=step)


def train(cfg: Config, trial: optuna.Trial) -> float:
    """Train AVP model and return best val_loss for HP optimization."""
    log.info(f"Starting trial {trial.number}")
    log.info(f"Device: {cfg.device}")

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

    log.info("Loading teacher...")
    teacher = load_teacher(cfg)
    log.info(f"Teacher params: {count_parameters(teacher):,}")

    log.info("Creating AVP model...")
    avp = create_avp(teacher, cfg)
    if cfg.compile:
        compile_avp(avp)

    # Target normalization: position-aware running stats
    n_tokens = cfg.avp.scene_grid_size**2
    target_norm = TargetNorm(n_tokens, teacher.embed_dim).to(cfg.device)

    def get_targets(images: Tensor) -> Tensor:
        """Get normalized teacher targets for images."""
        return target_norm(teacher.forward_norm_patches(images))

    # Fresh ratio: only load/compute teacher for this many images per step
    fresh_count = max(1, int(cfg.fresh_ratio * cfg.batch_size))
    log.info(
        f"Fresh ratio: {cfg.fresh_ratio} -> {fresh_count}/{cfg.batch_size} fresh per step"
    )

    log.info("Setting up data loaders...")
    train_loader = InfiniteLoader(
        make_loader(
            cfg.train_dir,
            train_transform(avp.scene_size, (cfg.crop_scale_min, 1.0)),
            fresh_count,
            cfg.num_workers,
            shuffle=True,
        )
    )
    val_loader = InfiniteLoader(
        make_loader(
            cfg.val_dir,
            val_transform(avp.scene_size),
            fresh_count,
            cfg.num_workers,
            shuffle=True,
        )
    )

    trainable = [p for p in avp.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = count_parameters(avp)
    log.info(
        f"AVP total: {n_total:,}, trainable: {n_trainable:,} ({100 * n_trainable / n_total:.1f}%)"
    )
    exp.log_parameters({"trainable_params": n_trainable, "total_params": n_total})

    peak_lr = get_linear_scaled_lr(cfg.ref_lr, cfg.batch_size)
    optimizer = torch.optim.AdamW(trainable, lr=peak_lr, weight_decay=cfg.weight_decay)
    warmup_steps = int(cfg.n_steps * cfg.warmup_ratio)
    scheduler = warmup_cosine_scheduler(optimizer, cfg.n_steps, warmup_steps)
    log.info(
        f"Optimizer: AdamW, peak_lr={peak_lr:.2e}, weight_decay={cfg.weight_decay:.2e}"
    )
    log.info(f"Schedule: {warmup_steps:,} warmup steps, {cfg.n_steps:,} total steps")

    # Fresh ratio survival: geometric distribution of lifetimes via random permutation
    # E[optimizer steps per image] = B / fresh_count = 1 / fresh_ratio
    # E[glimpses per image] = n_viewpoints_per_step / fresh_ratio
    expected_steps = cfg.batch_size / fresh_count
    expected_glimpses = cfg.n_viewpoints_per_step * expected_steps
    log.info(
        f"Fresh ratio survival: {fresh_count}/{cfg.batch_size} fresh/step, expected {expected_steps:.1f} steps, {expected_glimpses:.1f} glimpses per image"
    )

    ckpt_path = cfg.ckpt_dir / f"{exp.get_key()}_best.pt"
    best_val_loss = float("inf")
    ckpt_val_loss = float("inf")  # val_loss at last checkpoint, save only when improved

    # Initial eval
    log.info("Running initial validation...")
    val_images = val_loader.next_batch().to(cfg.device)
    val_loss = eval_and_log(exp, 0, avp, teacher, get_targets, val_images)
    log.info(f"Initial val_loss: {val_loss:.4f}")
    save_checkpoint(avp, ckpt_path, exp, 0, val_loss)
    best_val_loss = val_loss
    ckpt_val_loss = val_loss

    # Initialize training state: teacher only sees fresh_count at a time
    n_init_batches = (cfg.batch_size + fresh_count - 1) // fresh_count
    init_imgs_list, init_targets_list = [], []
    with torch.no_grad():
        for _ in range(n_init_batches):
            batch = train_loader.next_batch().to(cfg.device)
            init_imgs_list.append(batch)
            init_targets_list.append(get_targets(batch))
    init_imgs = torch.cat(init_imgs_list, dim=0)[: cfg.batch_size]
    init_targets = torch.cat(init_targets_list, dim=0)[: cfg.batch_size]
    hidden_init_full = avp._init_hidden(cfg.batch_size, None)
    local_init_full = (
        avp.local_init.expand(cfg.batch_size, -1, -1)
        if avp.local_init is not None
        else None
    )
    state = TrainState.init(init_imgs, init_targets, hidden_init_full, local_init_full)

    ema_loss_t = torch.tensor(0.0, device=cfg.device)
    alpha = 2 / (cfg.log_every + 1)

    log.info("Starting training loop...")
    pbar = tqdm(range(cfg.n_steps), desc="Training", unit="step")

    for step in pbar:
        # Load only fresh_count images (the speedup!)
        fresh_imgs = train_loader.next_batch().to(cfg.device)
        with torch.no_grad():
            fresh_targets = get_targets(fresh_imgs)

        # Inner loop: multiple viewpoints per optimizer step
        viewpoints = [
            random_viewpoint(
                cfg.batch_size,
                cfg.device,
                cfg.min_viewpoint_scale,
                cfg.max_viewpoint_scale,
            )
            for _ in range(cfg.n_viewpoints_per_step)
        ]
        loss, final_hidden, final_local = avp.forward_loss(
            state.images,
            viewpoints,
            state.targets,
            state.hidden,
            state.local_prev,
            loss_fn=LOSS_FN,
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
            exp.log_metrics(
                {"train/loss": ema_loss, "train/grad_norm": grad_norm, "train/lr": lr},
                step=step,
            )
            pbar.set_postfix_str(
                f"loss={ema_loss:.2e} grad={grad_norm:.2e} lr={lr:.2e}"
            )

        if step > 0 and step % cfg.val_every == 0:
            # Training viz: actual state, viewpoints, targets (BEFORE state update)
            train_l1, train_mse = viz_and_log(
                exp,
                step,
                "train",
                avp,
                teacher,
                state.images,
                viewpoints,
                state.targets,
                state.hidden,
            )
            exp.log_metric("train/viz_l1", train_l1[-1], step=step)
            exp.log_metric("train/viz_mse", train_mse[-1], step=step)

            # Validation viz: fresh batch, fixed viewpoints
            val_images = val_loader.next_batch().to(cfg.device)
            val_loss = eval_and_log(exp, step, avp, teacher, get_targets, val_images)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            if step % cfg.ckpt_every == 0 and best_val_loss < ckpt_val_loss:
                save_checkpoint(avp, ckpt_path, exp, step, best_val_loss)
                ckpt_val_loss = best_val_loss
            trial.report(val_loss, step)
            if trial.should_prune():
                exp.end()
                raise optuna.TrialPruned()

        # Fresh ratio survival: permute batch, replace first K with fresh
        hidden_init = avp._init_hidden(fresh_count, None)
        local_init = (
            avp.local_init.expand(fresh_count, -1, -1)
            if avp.local_init is not None
            else None
        )
        state = state.step(
            fresh_imgs,
            fresh_targets,
            final_hidden,
            final_local,
            hidden_init,
            local_init,
        )

    val_images = val_loader.next_batch().to(cfg.device)
    val_loss = eval_and_log(exp, cfg.n_steps, avp, teacher, get_targets, val_images)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
    if best_val_loss < ckpt_val_loss:
        save_checkpoint(avp, ckpt_path, exp, cfg.n_steps, best_val_loss)
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
