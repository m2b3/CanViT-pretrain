"""Train AVP scene representation to match frozen teacher backbone patches."""

import copy
import io
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

import comet_ml
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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


@dataclass
class Config:
    # Data
    train_dir: Path = Path("/datasets/ILSVRC/Data/CLS-LOC/train")
    val_dir: Path = Path("/datasets/ILSVRC/Data/CLS-LOC/val")
    # Model
    scene_grid_size: int = 32
    glimpse_grid_size: int = 7
    gate_init: float = 1e-5
    use_output_proj: bool = True
    use_scene_registers: bool = True
    freeze_inner_backbone: bool = False
    # Viewpoints
    use_policy: bool = True  # Use learned policy for viewpoint selection
    n_policy_viewpoints: int = 2  # Number of policy-selected viewpoints (after first imposed)
    policy_noise_std: float = 0.05  # Gaussian noise std on pol_out during training (DDPG-style)
    n_random_viewpoints: int = 2  # For non-policy mode: random viewpoints after full scene
    # For imposed first viewpoint:
    min_viewpoint_scale: float = 0.3
    max_viewpoint_scale: float = 1.0
    # Training
    n_steps: int = 2000
    batch_size: int = 32
    num_workers: int = 8
    ref_lr: float = 1e-5
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    # Logging
    log_every: int = 20
    viz_every: int = 100
    val_every: int = 200
    ckpt_every: int = 1000
    ckpt_dir: Path = Path("checkpoints")
    # Optuna
    n_trials: int = 100
    # Runtime
    device: torch.device = field(default_factory=get_sensible_device)


def random_viewpoint(
    B: int, device: torch.device, min_scale: float, max_scale: float
) -> Viewpoint:
    """Random viewpoint with scale in [min_scale, max_scale], center constrained to stay in bounds."""
    scales = torch.rand(B, device=device) * (max_scale - min_scale) + min_scale
    # Center must satisfy |c| <= 1 - s to keep glimpse in [-1, 1]
    max_offset = (1 - scales).unsqueeze(1)  # [B, 1]
    centers = (
        torch.rand(B, 2, device=device) * 2 - 1
    ) * max_offset  # [B, 2] in [-max_offset, max_offset]
    return Viewpoint(name="random", centers=centers, scales=scales)


def make_viewpoints(B: int, cfg: Config) -> list[Viewpoint]:
    """Full scene + n_random random viewpoints."""
    vps = [Viewpoint.full_scene(B, cfg.device)]
    for _ in range(cfg.n_random_viewpoints):
        vps.append(
            random_viewpoint(
                B, cfg.device, cfg.min_viewpoint_scale, cfg.max_viewpoint_scale
            )
        )
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
        use_policy=cfg.use_policy,
    )
    return AVPViT(backbone_copy, avp_cfg).to(cfg.device)


def policy_scale(cfg: Config) -> float:
    """Fixed scale for policy viewpoints = finest meaningful scale."""
    return cfg.glimpse_grid_size / cfg.scene_grid_size


def make_train_loader(
    cfg: Config, scene_size: int
) -> DataLoader[tuple[Tensor, Tensor]]:
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(scene_size, scale=(0.4, 1.0)),
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


def make_val_loader(cfg: Config, scene_size: int) -> DataLoader[tuple[Tensor, Tensor]]:
    transform = transforms.Compose(
        [
            transforms.Resize(scene_size),
            transforms.CenterCrop(scene_size),
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


def train_step_fixed(
    avp: AVPViT,
    teacher: DINOv3Backbone,
    images: Tensor,
    viewpoints: list[Viewpoint],
) -> Tensor:
    """Train with fixed (pre-defined) viewpoints."""
    with torch.no_grad():
        target = teacher.forward_norm_patches(images)
    return avp.forward_with_loss(
        images, viewpoints, lambda scene: nn.functional.mse_loss(scene, target)
    )


def train_step_policy(
    avp: AVPViT,
    teacher: DINOv3Backbone,
    images: Tensor,
    first_vp: Viewpoint,
    n_policy_steps: int,
    pol_scale: float,
    noise_std: float = 0.0,
) -> Tensor:
    """Train with policy-selected viewpoints.

    First viewpoint is imposed, subsequent viewpoints selected by policy.
    If noise_std > 0, adds Gaussian noise to pol_out before tanh (DDPG-style exploration).
    """
    with torch.no_grad():
        target = teacher.forward_norm_patches(images)
    n_total = 1 + n_policy_steps

    scene: Tensor | None = None
    vp = first_vp
    loss_sum = torch.tensor(0.0, device=images.device)

    for i in range(n_total):
        out = avp.forward_step(images, vp, scene)
        scene = out.scene
        loss_sum = loss_sum + nn.functional.mse_loss(avp.output_proj(scene), target)

        # Generate next viewpoint from policy (except on last step)
        if i < n_total - 1:
            assert out.pol_out is not None, "Policy must be enabled for train_step_policy"
            pol_out = out.pol_out
            if noise_std > 0:
                pol_out = pol_out + torch.randn_like(pol_out) * noise_std
            vp = avp.policy_to_viewpoint(pol_out, pol_scale)

    return loss_sum / n_total


def fit_pca(features: Tensor) -> PCA:
    pca = PCA(n_components=3, whiten=True)
    pca.fit(features.float().cpu().numpy())
    return pca


def pca_rgb(pca: PCA, features: Tensor, H: int, W: int) -> Tensor:
    proj = pca.transform(features.float().cpu().numpy())
    return torch.sigmoid(torch.from_numpy(proj).view(H, W, 3) * 2.0)


def tensor_to_img(t: Tensor) -> Tensor:
    """Convert [3, H, W] normalized tensor to displayable [H, W, 3] in [0, 1].

    Assumes ImageNet normalization, reverses it for display.
    """
    t = t.detach().cpu().float()
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    t = t * std + mean
    t = t.clamp(0, 1)
    return t.permute(1, 2, 0)


def _timestep_colors(n: int) -> list[tuple[float, float, float, float]]:
    """Generate n colors from viridis colormap for timestep visualization."""
    cmap = plt.get_cmap("viridis")
    return [cmap(i / max(1, n - 1)) for i in range(n)]


def plot_viewpoint_trajectory(
    image: Tensor,
    viewpoints: list[Viewpoint],
    batch_idx: int = 0,
) -> Figure:
    """Plot image with viewpoint boxes and scatter of ALL batch centers.

    Left: single sample's image with viewpoint boxes
    Right: scatter of ALL batch items' centers per timestep (shows distribution)

    Coordinate convention: centers[:, 0] = y, centers[:, 1] = x, in [-1, 1].
    """
    img = tensor_to_img(image).numpy()
    H, W = img.shape[:2]
    n_views = len(viewpoints)
    colors = _timestep_colors(n_views)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: image with boxes for batch_idx sample
    axes[0].imshow(img)
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
        axes[0].add_patch(rect)
        axes[0].plot(px_cx, px_cy, "o", color=colors[i], markersize=6)
        axes[0].text(px_cx + 3, px_cy - 3, str(i), color=colors[i], fontsize=10, fontweight="bold")
    axes[0].set_title(f"Trajectory (sample {batch_idx})")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].axis("off")

    # Right: scatter ALL batch items' centers per timestep with trajectory lines
    ax = axes[1]
    B = viewpoints[0].centers.shape[0]
    # Gather all centers: [n_views, B, 2]
    all_centers = [vp.centers.cpu().numpy() for vp in viewpoints]
    # Draw lines for each batch item
    for b in range(B):
        xs = [all_centers[t][b, 1] for t in range(n_views)]
        ys = [all_centers[t][b, 0] for t in range(n_views)]
        ax.plot(xs, ys, color="gray", alpha=0.3, linewidth=1, zorder=1)
    # Scatter points per timestep
    for i, vp in enumerate(viewpoints):
        centers = vp.centers.cpu().numpy()
        cx_all, cy_all = centers[:, 1], centers[:, 0]
        ax.scatter(cx_all, cy_all, c=[colors[i]], s=40, alpha=0.6, label=f"t={i}", zorder=2)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"All Centers (B={B})")
    ax.legend(loc="upper right", fontsize=8)
    ax.invert_yaxis()

    plt.tight_layout()
    return fig


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
            out = avp.forward_step(images, vp, scene)
            scene = out.scene
            # Strip prefix tokens and apply teacher's output norm for fair comparison
            local_patches = teacher.output_norm(out.local[:, n_prefix:])
            locals_list.append(local_patches)
            scene_proj = avp.output_proj(scene)
            scenes.append(scene_proj)
            mse = nn.functional.mse_loss(scene_proj, teacher_patches).item()
            mses.append(mse)

    return MultistepResult(
        teacher_patches, scenes, locals_list, glimpse_imgs, mses, viewpoints
    )


def run_multistep_inference_policy(
    avp: AVPViT,
    teacher: DINOv3Backbone,
    images: Tensor,
    first_vp: Viewpoint,
    n_policy_steps: int,
    pol_scale: float,
    noise_std: float = 0.0,
) -> MultistepResult:
    """Run multi-step inference with policy-selected viewpoints.

    If noise_std > 0, adds Gaussian noise to pol_out before tanh.
    """
    n_prefix = teacher.n_prefix_tokens
    n_total = 1 + n_policy_steps

    with torch.inference_mode():
        teacher_patches = teacher.forward_norm_patches(images)
        scene: Tensor | None = None
        scenes: list[Tensor] = []
        locals_list: list[Tensor] = []
        glimpse_imgs: list[Tensor] = []
        mses: list[float] = []
        viewpoints: list[Viewpoint] = []

        vp = first_vp
        for i in range(n_total):
            viewpoints.append(vp)
            glimpse_imgs.append(extract_glimpse(images, vp, avp.glimpse_size))
            out = avp.forward_step(images, vp, scene)
            scene = out.scene
            local_patches = teacher.output_norm(out.local[:, n_prefix:])
            locals_list.append(local_patches)
            scene_proj = avp.output_proj(scene)
            scenes.append(scene_proj)
            mse = nn.functional.mse_loss(scene_proj, teacher_patches).item()
            mses.append(mse)

            # Generate next viewpoint from policy (except on last step)
            if i < n_total - 1:
                assert out.pol_out is not None
                pol_out = out.pol_out
                if noise_std > 0:
                    pol_out = pol_out + torch.randn_like(pol_out) * noise_std
                vp = avp.policy_to_viewpoint(pol_out, pol_scale)

    return MultistepResult(
        teacher_patches, scenes, locals_list, glimpse_imgs, mses, viewpoints
    )


def _vp_to_pixel_box(
    vp: Viewpoint, sample_idx: int, H: int, W: int
) -> tuple[float, float, float, float]:
    """Convert viewpoint to pixel coordinates (cx, cy, half_w, half_h).

    Coordinate convention: centers[:, 0] = y, centers[:, 1] = x, in [-1, 1].
    """
    cy, cx = vp.centers[sample_idx].cpu().numpy()
    s = vp.scales[sample_idx].item()
    px_cx = (cx + 1) / 2 * W
    px_cy = (cy + 1) / 2 * H
    return px_cx, px_cy, s * W / 2, s * H / 2


def plot_multistep_pca(
    result: MultistepResult,
    full_image: Tensor,
    scene_grid_size: int,
    glimpse_grid_size: int,
    sample_idx: int = 0,
) -> Figure:
    """Create multi-row PCA visualization figure.

    Columns: Trajectory | Glimpse | Teacher | Scene | Local | Delta | Error
    - Trajectory: full image with cumulative viewpoint boxes and path
    - Local: glimpse features using same PCA as scene
    - Delta: spatial change in scene from previous timestep

    Args:
        full_image: [3, H, W] the full input image for this sample
        sample_idx: Which batch sample to visualize
    """
    S = scene_grid_size
    G = glimpse_grid_size
    n_views = len(result.viewpoints)

    teacher_i = result.teacher_patches[sample_idx]
    pca = fit_pca(teacher_i)
    teacher_pca = pca_rgb(pca, teacher_i, S, S)
    full_img_np = tensor_to_img(full_image).numpy()
    H, W = full_img_np.shape[:2]

    # Precompute viewpoint pixel coords
    vp_coords = [_vp_to_pixel_box(vp, sample_idx, H, W) for vp in result.viewpoints]

    # Precompute error maps for consistent colorbar
    error_maps = []
    for row in range(n_views):
        scene_i = result.scenes[row][sample_idx]
        error_map = (scene_i - teacher_i).pow(2).mean(dim=-1).view(S, S).cpu()
        error_maps.append(error_map)
    error_vmax = max(e.max().item() for e in error_maps)

    colors = _timestep_colors(n_views)
    fig, axes = plt.subplots(n_views, 7, figsize=(28, 4 * n_views))

    for row, vp in enumerate(result.viewpoints):
        glimpse_i = result.glimpse_imgs[row][sample_idx]
        scene_i = result.scenes[row][sample_idx]
        local_i = result.locals[row][sample_idx]
        scene_pca = pca_rgb(pca, scene_i, S, S)
        local_pca = pca_rgb(pca, local_i, G, G)

        # Delta
        if row > 0:
            prev_scene_i = result.scenes[row - 1][sample_idx]
            delta_map = (scene_i - prev_scene_i).pow(2).mean(dim=-1).view(S, S).cpu()
        else:
            delta_map = torch.zeros(S, S)

        # Col 0: Trajectory - full image with cumulative boxes and path
        ax = axes[row, 0]
        ax.imshow(full_img_np)
        # Draw all boxes up to current row
        for t in range(row + 1):
            px_cx, px_cy, hw, hh = vp_coords[t]
            alpha = 1.0 if t == row else 0.4
            rect = Rectangle(
                (px_cx - hw, px_cy - hh), 2 * hw, 2 * hh,
                linewidth=2, edgecolor=colors[t], facecolor="none", alpha=alpha
            )
            ax.add_patch(rect)
            ax.plot(px_cx, px_cy, "o", color=colors[t], markersize=5, alpha=alpha)
            ax.text(px_cx + 3, px_cy - 3, str(t), color=colors[t], fontsize=8, alpha=alpha)
        # Draw path lines
        for t in range(1, row + 1):
            x0, y0 = vp_coords[t - 1][:2]
            x1, y1 = vp_coords[t][:2]
            ax.plot([x0, x1], [y0, y1], "-", color=colors[t], linewidth=1.5, alpha=0.7)
        ax.set_title(f"t={row}" if row == 0 else f"t={row}")
        ax.axis("off")

        # Col 1: Glimpse
        axes[row, 1].imshow(tensor_to_img(glimpse_i).numpy())
        axes[row, 1].set_title(f"Glimpse ({vp.name})")
        axes[row, 1].axis("off")

        # Col 2: Teacher PCA
        axes[row, 2].imshow(teacher_pca.numpy())
        axes[row, 2].set_title("Teacher" if row == 0 else "")
        axes[row, 2].axis("off")

        # Col 3: Scene PCA
        axes[row, 3].imshow(scene_pca.numpy())
        axes[row, 3].set_title(f"Scene t={row}")
        axes[row, 3].axis("off")

        # Col 4: Local PCA
        axes[row, 4].imshow(local_pca.numpy())
        axes[row, 4].set_title(f"Local t={row}")
        axes[row, 4].axis("off")

        # Col 5: Delta
        im_delta = axes[row, 5].imshow(delta_map.numpy(), cmap="viridis")
        axes[row, 5].set_title("Δ Scene" if row > 0 else "Δ (n/a)")
        axes[row, 5].axis("off")
        fig.colorbar(im_delta, ax=axes[row, 5], fraction=0.046, pad=0.04)

        # Col 6: Error
        sample_mse = error_maps[row].mean().item()
        im_err = axes[row, 6].imshow(error_maps[row].numpy(), cmap="hot", vmin=0, vmax=error_vmax)
        axes[row, 6].set_title(f"Err ({sample_mse:.3f})")
        axes[row, 6].axis("off")
        fig.colorbar(im_err, ax=axes[row, 6], fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


def log_train_pca(
    exp: comet_ml.Experiment,
    step: int,
    avp: AVPViT,
    teacher: DINOv3Backbone,
    images: Tensor,
    cfg: Config,
) -> None:
    """Log multi-row train PCA viz to Comet."""
    B = images.shape[0]
    if cfg.use_policy:
        first_vp = random_viewpoint(B, cfg.device, cfg.min_viewpoint_scale, cfg.max_viewpoint_scale)
        result = run_multistep_inference_policy(avp, teacher, images, first_vp, cfg.n_policy_viewpoints, policy_scale(cfg), cfg.policy_noise_std)
    else:
        viewpoints = make_viewpoints(B, cfg)
        result = run_multistep_inference(avp, teacher, images, viewpoints)
    fig = plot_multistep_pca(result, images[0], cfg.scene_grid_size, cfg.glimpse_grid_size, sample_idx=0)
    with io.BytesIO() as buf:
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
    B = images.shape[0]
    sample_idx = 0  # Consistent sample for all visualizations

    if cfg.use_policy:
        first_vp = random_viewpoint(B, cfg.device, cfg.min_viewpoint_scale, cfg.max_viewpoint_scale)
        result = run_multistep_inference_policy(avp, teacher, images, first_vp, cfg.n_policy_viewpoints, policy_scale(cfg))  # No noise for eval
    else:
        viewpoints = make_viewpoints(B, cfg)
        result = run_multistep_inference(avp, teacher, images, viewpoints)

    fig = plot_multistep_pca(result, images[sample_idx], cfg.scene_grid_size, cfg.glimpse_grid_size, sample_idx=sample_idx)
    with io.BytesIO() as buf:
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        exp.log_image(buf, name="val/pca", step=step)
    plt.close(fig)

    # Log viewpoint trajectory (same sample as PCA)
    fig_traj = plot_viewpoint_trajectory(images[sample_idx], result.viewpoints, batch_idx=sample_idx)
    with io.BytesIO() as buf:
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        exp.log_image(buf, name="val/trajectory", step=step)
    plt.close(fig_traj)

    # Log MSE evolution (batch-averaged)
    for t, mse in enumerate(result.mses):
        exp.log_metric(f"val/mse_t{t}", mse, step=step)

    # Log viewpoint stats for visualized sample
    for t, vp in enumerate(result.viewpoints):
        exp.log_metric(f"val/vp_scale_t{t}", vp.scales[sample_idx].item(), step=step)
        exp.log_metric(f"val/vp_cx_t{t}", vp.centers[sample_idx, 1].item(), step=step)
        exp.log_metric(f"val/vp_cy_t{t}", vp.centers[sample_idx, 0].item(), step=step)

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
    scene_size = teacher.patch_size * cfg.scene_grid_size

    train_loader = make_train_loader(cfg, scene_size)
    val_iter = ValIter(make_val_loader(cfg, scene_size))
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
        B = teacher_img.shape[0]

        optimizer.zero_grad()
        if cfg.use_policy:
            first_vp = random_viewpoint(B, cfg.device, cfg.min_viewpoint_scale, cfg.max_viewpoint_scale)
            loss = train_step_policy(avp, teacher, teacher_img, first_vp, cfg.n_policy_viewpoints, policy_scale(cfg), cfg.policy_noise_std)
        else:
            viewpoints = make_viewpoints(B, cfg)
            loss = train_step_fixed(avp, teacher, teacher_img, viewpoints)
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
            exp.log_metrics(
                {"train/loss": ema_loss, "train/grad_norm": grad_norm, "train/lr": lr},
                step=step,
            )
            pbar.set_postfix_str(
                f"loss={ema_loss:.2e} grad={grad_norm:.2e} lr={lr:.2e}"
            )

        if step > 0 and step % cfg.viz_every == 0:
            log_train_pca(exp, step, avp, teacher, teacher_img, cfg)

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
