"""Train AVP scene representation to match frozen teacher backbone patches."""

import copy
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path

import comet_ml
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from dinov3.hub.backbones import dinov3_vits16
from sklearn.decomposition import PCA
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from ymc.lr import get_linear_scaled_lr
from ytch.correctness import assert_shape
from ytch.device import get_sensible_device
from ytch.model import count_parameters

from avp_vit import AVPConfig, AVPViT
from avp_vit.backbone.dinov3 import DINOv3Backbone

matplotlib.use("Agg")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

CKPT_PATH = Path("dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PATCH_SIZE = 16


@dataclass
class Config:
    train_dir: Path = Path("/datasets/ILSVRC/Data/CLS-LOC/train")
    val_dir: Path = Path("/datasets/ILSVRC/Data/CLS-LOC/val")
    scene_grid_size: int = 16
    glimpse_grid_size: int = 7
    gate_init: float = 1e-4
    use_output_proj: bool = True
    freeze_inner_backbone: bool = True
    n_steps: int = 50000
    batch_size: int = 256
    num_workers: int = 8
    ref_lr: float = 1e-5
    weight_decay: float = 1e-3
    warmup_ratio: float = 0.04
    grad_clip: float = 1.0
    log_every: int = 20
    viz_every: int = 200
    val_every: int = 200
    ckpt_every: int = 1000
    ckpt_dir: Path = Path("checkpoints")
    device: torch.device = field(default_factory=get_sensible_device)

    @property
    def scene_size(self) -> int:
        return self.scene_grid_size * PATCH_SIZE

    @property
    def glimpse_size(self) -> int:
        return self.glimpse_grid_size * PATCH_SIZE


class TrainSample:
    """A training sample. Glimpse is always derived from scene via downsampling."""

    teacher_img: Tensor  # [B, 3, scene_size, scene_size]
    glimpse_img: Tensor  # [B, 3, glimpse_size, glimpse_size]
    centers: Tensor  # [B, 2]
    scales: Tensor  # [B]

    def __init__(self, scene: Tensor, glimpse_size: int, device: torch.device) -> None:
        self.teacher_img = scene.to(device)
        self.glimpse_img = nn.functional.interpolate(
            self.teacher_img, (glimpse_size, glimpse_size), mode="bilinear"
        )
        B = scene.shape[0]
        self.centers = torch.zeros(B, 2, device=device)
        self.scales = torch.ones(B, device=device)


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
    )
    return AVPViT(backbone_copy, avp_cfg).to(cfg.device)


def make_train_loader(cfg: Config) -> DataLoader[tuple[Tensor, Tensor]]:
    transform = transforms.Compose([
        transforms.RandomResizedCrop(cfg.scene_size, scale=(0.4, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
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
    transform = transforms.Compose([
        transforms.Resize(cfg.scene_size),
        transforms.CenterCrop(cfg.scene_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    dataset = ImageFolder(str(cfg.val_dir), transform=transform)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )


def init_scene_tokens(avp: AVPViT, embed_dim: int) -> None:
    """Initialize scene_tokens to randn / sqrt(embed_dim)."""
    with torch.no_grad():
        avp.scene_tokens.data.normal_(0, 1.0 / (embed_dim**0.5))
    log.info(f"scene_tokens initialized to randn / sqrt({embed_dim})")


def teacher_forward(teacher: DINOv3Backbone, img: Tensor) -> Tensor:
    """Teacher forward: img -> patch features [B, H*W, D]."""
    with torch.no_grad():
        out = teacher._backbone.forward_features(img)
    return out["x_norm_patchtokens"]


def tokenize_glimpse(teacher: DINOv3Backbone, img: Tensor) -> Tensor:
    """Tokenize glimpse image for AVP input."""
    tokens, _ = teacher._backbone.prepare_tokens_with_masks(img, masks=None)
    return tokens


def train_step(
    avp: AVPViT,
    teacher: DINOv3Backbone,
    sample: TrainSample,
    cfg: Config,
) -> Tensor:
    """Train step: teacher processes full-res img, AVP processes glimpse img."""
    B = sample.teacher_img.shape[0]
    S = cfg.scene_grid_size**2
    D = teacher.embed_dim

    teacher_patches = teacher_forward(teacher, sample.teacher_img)
    assert_shape(teacher_patches, (B, S, D))

    glimpse_tokens = tokenize_glimpse(teacher, sample.glimpse_img)
    _, scene = avp(glimpse_tokens, sample.centers, sample.scales)
    assert_shape(scene, (B, S, D))

    return nn.functional.mse_loss(scene, teacher_patches)


def normalize_local(avp: AVPViT, local: Tensor) -> Tensor:
    """Normalize local stream and strip prefix tokens -> [B, G*G, D]."""
    local_norm = avp.backbone.norm(local)
    return local_norm[:, avp.backbone.n_prefix_tokens :]


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


def log_train_pca(
    exp: comet_ml.Experiment,
    step: int,
    sample: TrainSample,
    teacher_patches: Tensor,
    local_patches: Tensor,
    scene: Tensor,
    cfg: Config,
) -> None:
    """Log train PCA viz to Comet (random sample from batch)."""
    S = cfg.scene_grid_size
    G = cfg.glimpse_grid_size
    idx = torch.randint(teacher_patches.shape[0], (1,)).item()
    teacher_i, local_i, scene_i = teacher_patches[idx], local_patches[idx], scene[idx]
    pca = fit_pca(teacher_i)
    teacher_rgb = pca_rgb(pca, teacher_i, S, S)
    local_rgb = pca_rgb(pca, local_i, G, G)
    scene_rgb = pca_rgb(pca, scene_i, S, S)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(tensor_to_img(sample.teacher_img[idx]).numpy())
    axes[0].set_title("Input")
    axes[0].axis("off")
    axes[1].imshow(teacher_rgb.numpy())
    axes[1].set_title("Teacher")
    axes[1].axis("off")
    axes[2].imshow(local_rgb.numpy())
    axes[2].set_title("Local")
    axes[2].axis("off")
    axes[3].imshow(scene_rgb.numpy())
    axes[3].set_title("Scene")
    axes[3].axis("off")
    plt.tight_layout()
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
    val_loader: DataLoader[tuple[Tensor, Tensor]],
    cfg: Config,
) -> float:
    """Eval on val set, log PCA plots to Comet."""
    S = cfg.scene_grid_size
    G = cfg.glimpse_grid_size

    total_loss = 0.0
    n_batches = 0
    first_sample: TrainSample | None = None
    first_teacher: Tensor | None = None
    first_local: Tensor | None = None
    first_scene: Tensor | None = None

    with torch.inference_mode():
        for imgs, _ in val_loader:
            sample = TrainSample(imgs, cfg.glimpse_size, cfg.device)
            teacher_patches = teacher_forward(teacher, sample.teacher_img)
            glimpse_tokens = tokenize_glimpse(teacher, sample.glimpse_img)
            local, scene = avp(glimpse_tokens, sample.centers, sample.scales)
            loss = nn.functional.mse_loss(scene, teacher_patches)
            total_loss += loss.item()
            n_batches += 1

            if first_sample is None:
                first_sample = sample
                first_teacher = teacher_patches
                first_local = normalize_local(avp, local)
                first_scene = scene

            if n_batches >= 10:
                break

    val_loss = total_loss / n_batches

    # PCA viz from first batch
    assert first_sample is not None
    assert first_teacher is not None
    assert first_local is not None
    assert first_scene is not None

    teacher_0, local_0, scene_0 = first_teacher[0], first_local[0], first_scene[0]
    pca = fit_pca(teacher_0)
    teacher_pca = pca_rgb(pca, teacher_0, S, S)
    local_pca = pca_rgb(pca, local_0, G, G)
    scene_pca = pca_rgb(pca, scene_0, S, S)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(tensor_to_img(first_sample.teacher_img[0]).numpy())
    axes[0].set_title("Input")
    axes[0].axis("off")
    axes[1].imshow(teacher_pca.numpy())
    axes[1].set_title("Teacher")
    axes[1].axis("off")
    axes[2].imshow(local_pca.numpy())
    axes[2].set_title("Local")
    axes[2].axis("off")
    axes[3].imshow(scene_pca.numpy())
    axes[3].set_title("Scene")
    axes[3].axis("off")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    exp.log_image(buf, name="val/pca", step=step)
    plt.close(fig)

    exp.log_metric("val/loss", val_loss, step=step)
    return val_loss


def save_checkpoint(
    avp: AVPViT, ckpt_path: Path, exp: comet_ml.Experiment, step: int, val_loss: float
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(avp.state_dict(), ckpt_path)
    size_mb = ckpt_path.stat().st_size / (1024 * 1024)
    log.info(f"Saved checkpoint: {ckpt_path} ({size_mb:.1f} MB), val_loss={val_loss:.4f}")
    exp.log_metric("ckpt/val_loss", val_loss, step=step)


def main() -> None:
    import tyro

    torch.set_float32_matmul_precision("high")
    cfg = tyro.cli(Config)
    log.info(f"Config: {cfg}")

    exp = comet_ml.Experiment(
        project_name="avp-vit-scene-match", auto_metric_logging=False
    )
    exp.log_parameters(
        {
            k: str(v) if isinstance(v, (torch.device, Path)) else v
            for k, v in cfg.__dict__.items()
        }
    )

    teacher = load_teacher(cfg.device)
    avp = create_avp(teacher, cfg)
    init_scene_tokens(avp, teacher.embed_dim)

    train_loader = make_train_loader(cfg)
    val_loader = make_val_loader(cfg)
    train_iter = iter(train_loader)

    trainable = [p for p in avp.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_teacher = count_parameters(teacher)
    n_train = len(train_loader.dataset)  # type: ignore[arg-type]
    n_val = len(val_loader.dataset)  # type: ignore[arg-type]
    log.info(f"Trainable: {n_trainable:,}, Teacher: {n_teacher:,}")
    log.info(f"Train: {n_train:,}, Val: {n_val:,}")
    exp.log_parameters({
        "trainable_params": n_trainable,
        "teacher_params": n_teacher,
        "train_size": n_train,
        "val_size": n_val,
    })

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

    val_loss = eval_and_log(exp, 0, avp, teacher, val_loader, cfg)
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

        sample = TrainSample(imgs, cfg.glimpse_size, cfg.device)

        optimizer.zero_grad()
        loss = train_step(avp, teacher, sample, cfg)
        loss.backward()
        grad_norm_t = torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        ema_loss_t = alpha * loss.detach() + (1 - alpha) * ema_loss_t if step > 0 else loss.detach()

        if step % cfg.log_every == 0:
            # Sync only when logging
            ema_loss = ema_loss_t.item()
            grad_norm = grad_norm_t.item()
            lr = scheduler.get_last_lr()[0]
            sps = pbar.format_dict["rate"] * cfg.batch_size if pbar.format_dict["rate"] else 0
            exp.log_metrics(
                {"train/loss": ema_loss, "train/grad_norm": grad_norm, "train/lr": lr},
                step=step,
            )
            pbar.set_postfix_str(
                f"loss={ema_loss:.2e} grad={grad_norm:.2e} lr={lr:.2e} sps={sps:.0f}"
            )

        if step > 0 and step % cfg.viz_every == 0:
            with torch.inference_mode():
                teacher_patches = teacher_forward(teacher, sample.teacher_img)
                glimpse_tokens = tokenize_glimpse(teacher, sample.glimpse_img)
                local, scene = avp(glimpse_tokens, sample.centers, sample.scales)
                local_patches = normalize_local(avp, local)
            log_train_pca(exp, step, sample, teacher_patches, local_patches, scene, cfg)

        if step > 0 and step % cfg.val_every == 0:
            val_loss = eval_and_log(exp, step, avp, teacher, val_loader, cfg)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if step % cfg.ckpt_every == 0:
                    save_checkpoint(avp, ckpt_path, exp, step, val_loss)

    val_loss = eval_and_log(exp, cfg.n_steps, avp, teacher, val_loader, cfg)
    if val_loss < best_val_loss:
        save_checkpoint(avp, ckpt_path, exp, cfg.n_steps, val_loss)
    log.info(f"Final: train_ema={ema_loss_t.item():.4f}, val={val_loss:.4f}, best={best_val_loss:.4f}")
    exp.end()


if __name__ == "__main__":
    main()
