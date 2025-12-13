"""Train AVP scene representation to match frozen teacher backbone patches."""

import copy
import io
import logging
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import comet_ml
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import tyro
from dinov3.hub.backbones import dinov3_vits16
from PIL import Image
from sklearn.decomposition import PCA
from torch import Tensor
from tqdm import tqdm
from ymc.lr import get_linear_scaled_lr
from ytch.correctness import assert_shape
from ytch.device import get_sensible_device
from ytch.lr.warmup import get_linear_warmup_scheduler
from ytch.model import count_parameters

from avp_vit import AVPConfig, AVPViT
from avp_vit.backbone.dinov3 import DINOv3Backbone

matplotlib.use("Agg")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

CKPT_PATH = Path("dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
TRAIN_IMAGE_URLS = [
    "https://dl.fbaipublicfiles.com/dinov3/notebooks/pca/test_image.jpg",  # dog
    "https://dl.fbaipublicfiles.com/dino/img.png",  # bird
]
VAL_IMAGE_URL = "https://dl.fbaipublicfiles.com/dinov2/images/example.jpg"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class Config:
    scene_grid_size: int = 8
    glimpse_grid_size: int = 7
    freeze_inner_backbone: bool = False
    n_steps: int = 5000
    batch_size: int = 8
    use_real_image: bool = False
    ref_lr: float = 1e-4
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    log_every: int = 10
    viz_every: int = 20
    val_every: int = 50
    n_init_samples: int = 64
    device: torch.device = field(default_factory=get_sensible_device)


@dataclass
class TrainSample:
    """A training sample with images (not tokens - tokenization happens in forward)."""

    teacher_img: Tensor  # [B, 3, scene_size, scene_size]
    glimpse_img: Tensor  # [B, 3, glimpse_size, glimpse_size]
    centers: Tensor  # [B, 2]
    scales: Tensor  # [B]
    img_pil: Image.Image | None = None  # for visualization


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
        scene_grid_size=cfg.scene_grid_size, glimpse_grid_size=cfg.glimpse_grid_size
    )
    return AVPViT(backbone_copy, avp_cfg).to(cfg.device)


def load_image_sample(
    url: str,
    scene_grid: int,
    glimpse_grid: int,
    device: torch.device,
) -> TrainSample:
    """Load image and create teacher/glimpse image tensors."""
    scene_size = scene_grid * 16
    glimpse_size = glimpse_grid * 16

    img_pil = (
        Image.open(urllib.request.urlopen(url))
        .convert("RGB")
        .resize((scene_size, scene_size))
    )
    glimpse_pil = img_pil.resize((glimpse_size, glimpse_size), Image.BILINEAR)

    teacher_img = (
        TF.normalize(TF.to_tensor(img_pil), mean=IMAGENET_MEAN, std=IMAGENET_STD)
        .unsqueeze(0)
        .to(device)
    )
    glimpse_img = (
        TF.normalize(TF.to_tensor(glimpse_pil), mean=IMAGENET_MEAN, std=IMAGENET_STD)
        .unsqueeze(0)
        .to(device)
    )

    return TrainSample(
        teacher_img=teacher_img,
        glimpse_img=glimpse_img,
        centers=torch.zeros(1, 2, device=device),
        scales=torch.ones(1, device=device),
        img_pil=img_pil,
    )


def random_sample(cfg: Config) -> TrainSample:
    """Create random image tensors."""
    B = cfg.batch_size
    scene_size = cfg.scene_grid_size * 16
    glimpse_size = cfg.glimpse_grid_size * 16

    return TrainSample(
        teacher_img=torch.randn(B, 3, scene_size, scene_size, device=cfg.device),
        glimpse_img=torch.randn(B, 3, glimpse_size, glimpse_size, device=cfg.device),
        centers=torch.zeros(B, 2, device=cfg.device),
        scales=torch.ones(B, device=cfg.device),
        img_pil=None,
    )


def teacher_forward(teacher: DINOv3Backbone, img: Tensor) -> Tensor:
    """Teacher forward: img -> patch features [B, H*W, D]."""
    with torch.no_grad():
        out = teacher._backbone.forward_features(img)
    return out["x_norm_patchtokens"]


def tokenize_glimpse(teacher: DINOv3Backbone, img: Tensor) -> Tensor:
    """Tokenize glimpse image for AVP input."""
    tokens, _ = teacher._backbone.prepare_tokens_with_masks(img, masks=None)
    return tokens


def init_scene_tokens(
    avp: AVPViT, teacher: DINOv3Backbone, cfg: Config
) -> None:
    """Initialize scene_tokens to average latents from random images."""
    S = cfg.scene_grid_size**2
    D = teacher.embed_dim
    scene_size = cfg.scene_grid_size * 16

    log.info(f"Initializing scene_tokens from {cfg.n_init_samples} random images...")
    all_patches = []
    batch_size = min(cfg.n_init_samples, 16)
    for i in range(0, cfg.n_init_samples, batch_size):
        B = min(batch_size, cfg.n_init_samples - i)
        imgs = torch.randn(B, 3, scene_size, scene_size, device=cfg.device)
        patches = teacher_forward(teacher, imgs)
        all_patches.append(patches)

    all_patches = torch.cat(all_patches, dim=0)  # [N, S, D]
    avg_patches = all_patches.mean(dim=0, keepdim=True)  # [1, S, D]
    assert_shape(avg_patches, (1, S, D))

    avp.scene_tokens.data.copy_(avg_patches)
    log.info(f"scene_tokens initialized to mean of {cfg.n_init_samples} samples")


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

    # Teacher: standard forward on full-res image
    teacher_patches = teacher_forward(teacher, sample.teacher_img)
    assert_shape(teacher_patches, (B, S, D))

    # AVP: tokenize glimpse, then forward with custom positions
    glimpse_tokens = tokenize_glimpse(teacher, sample.glimpse_img)
    _, scene = avp(glimpse_tokens, sample.centers, sample.scales)
    assert_shape(scene, (B, S, D))

    return nn.functional.mse_loss(scene, teacher_patches)


def fit_pca(features: Tensor) -> PCA:
    """Fit PCA. Expects [N, D] input."""
    pca = PCA(n_components=3, whiten=True)
    pca.fit(features.float().cpu().numpy())
    return pca


def pca_rgb(pca: PCA, features: Tensor, H: int, W: int) -> Tensor:
    """Apply PCA. Expects [N, D] input."""
    proj = pca.transform(features.float().cpu().numpy())
    return torch.sigmoid(torch.from_numpy(proj).view(H, W, 3) * 2.0)


def log_train_pca(
    exp: comet_ml.Experiment,
    step: int,
    sample: TrainSample,
    teacher_patches: Tensor,
    scene: Tensor,
    cfg: Config,
) -> None:
    """Log train PCA viz to Comet (first sample only)."""
    S = cfg.scene_grid_size
    teacher_0, scene_0 = teacher_patches[0], scene[0]  # [S*S, D]
    pca = fit_pca(teacher_0)
    teacher_rgb = pca_rgb(pca, teacher_0, S, S)
    scene_rgb = pca_rgb(pca, scene_0, S, S)

    n_plots = 3 if sample.img_pil is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
    i = 0
    if sample.img_pil is not None:
        axes[i].imshow(sample.img_pil)
        axes[i].set_title("Input")
        axes[i].axis("off")
        i += 1
    axes[i].imshow(teacher_rgb.numpy())
    axes[i].set_title("Teacher PCA")
    axes[i].axis("off")
    i += 1
    axes[i].imshow(scene_rgb.numpy())
    axes[i].set_title("Scene PCA")
    axes[i].axis("off")
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
    sample: TrainSample,
    cfg: Config,
) -> float:
    """Eval on sample, log PCA plots to Comet."""
    B = sample.teacher_img.shape[0]
    S = cfg.scene_grid_size
    D = teacher.embed_dim

    with torch.inference_mode():
        teacher_patches = teacher_forward(teacher, sample.teacher_img)
        assert_shape(teacher_patches, (B, S * S, D))
        glimpse_tokens = tokenize_glimpse(teacher, sample.glimpse_img)
        _, scene = avp(glimpse_tokens, sample.centers, sample.scales)
        assert_shape(scene, (B, S * S, D))
        val_loss = nn.functional.mse_loss(scene, teacher_patches).item()

    # PCA viz (first sample only)
    teacher_0, scene_0 = teacher_patches[0], scene[0]  # [S*S, D]
    pca = fit_pca(teacher_0)
    teacher_pca = pca_rgb(pca, teacher_0, S, S)
    scene_pca = pca_rgb(pca, scene_0, S, S)

    n_plots = 3 if sample.img_pil is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
    i = 0
    if sample.img_pil is not None:
        axes[i].imshow(sample.img_pil)
        axes[i].set_title("Input")
        axes[i].axis("off")
        i += 1
    axes[i].imshow(teacher_pca.numpy())
    axes[i].set_title("Teacher PCA")
    axes[i].axis("off")
    i += 1
    axes[i].imshow(scene_pca.numpy())
    axes[i].set_title("Scene PCA")
    axes[i].axis("off")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    exp.log_image(buf, name="val/pca", step=step)
    plt.close(fig)

    exp.log_metric("val/loss", val_loss, step=step)
    return val_loss


def main() -> None:
    cfg = tyro.cli(Config)
    log.info(f"Config: {cfg}")

    exp = comet_ml.Experiment(
        project_name="avp-vit-scene-match", auto_metric_logging=False
    )
    exp.log_parameters(
        {
            k: str(v) if isinstance(v, torch.device) else v
            for k, v in cfg.__dict__.items()
        }
    )

    # Load models
    teacher = load_teacher(cfg.device)
    avp = create_avp(teacher, cfg)

    # Initialize scene_tokens to average latents
    init_scene_tokens(avp, teacher, cfg)

    # Load train samples (real images or will generate random)
    train_samples: list[TrainSample] = []
    if cfg.use_real_image:
        for url in TRAIN_IMAGE_URLS:
            sample = load_image_sample(
                url, cfg.scene_grid_size, cfg.glimpse_grid_size, cfg.device
            )
            train_samples.append(sample)

    # Load val sample
    val_sample = load_image_sample(
        VAL_IMAGE_URL, cfg.scene_grid_size, cfg.glimpse_grid_size, cfg.device
    )

    # Trainable params
    trainable = list(avp.read_attn.parameters()) + list(avp.write_attn.parameters())
    trainable += list(avp.read_gate) + list(avp.write_gate) + [avp.scene_tokens]
    if not cfg.freeze_inner_backbone:
        trainable += list(avp.backbone.parameters())

    n_trainable = sum(p.numel() for p in trainable)
    n_teacher = count_parameters(teacher)
    log.info(f"Trainable: {n_trainable:,}, Teacher: {n_teacher:,}")
    exp.log_parameters({"trainable_params": n_trainable, "teacher_params": n_teacher})

    # Optimizer
    peak_lr = get_linear_scaled_lr(cfg.ref_lr, cfg.batch_size)
    optimizer = torch.optim.AdamW(trainable, lr=peak_lr)
    warmup_steps = int(cfg.n_steps * cfg.warmup_ratio)
    scheduler = get_linear_warmup_scheduler(optimizer, warmup_steps)

    # Initial eval
    eval_and_log(exp, 0, avp, teacher, val_sample, cfg)

    # Training
    ema_loss, alpha = 0.0, 2 / (cfg.log_every + 1)
    pbar = tqdm(range(cfg.n_steps), desc="Training")
    for step in pbar:
        if cfg.use_real_image:
            sample = train_samples[step % len(train_samples)]
        else:
            sample = random_sample(cfg)

        optimizer.zero_grad()
        loss = train_step(avp, teacher, sample, cfg)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip).item()
        optimizer.step()
        scheduler.step()
        ema_loss = (
            alpha * loss.item() + (1 - alpha) * ema_loss if step > 0 else loss.item()
        )

        if step % cfg.log_every == 0:
            lr = scheduler.get_last_lr()[0]
            exp.log_metrics(
                {"train/loss": ema_loss, "train/grad_norm": grad_norm, "train/lr": lr},
                step=step,
            )
            pbar.set_postfix(
                loss=f"{ema_loss:.4f}", grad=f"{grad_norm:.2f}", lr=f"{lr:.2e}"
            )

        if step > 0 and step % cfg.viz_every == 0:
            with torch.inference_mode():
                teacher_patches = teacher_forward(teacher, sample.teacher_img)
                glimpse_tokens = tokenize_glimpse(teacher, sample.glimpse_img)
                _, scene = avp(glimpse_tokens, sample.centers, sample.scales)
            log_train_pca(exp, step, sample, teacher_patches, scene, cfg)

        if step > 0 and step % cfg.val_every == 0:
            eval_and_log(exp, step, avp, teacher, val_sample, cfg)

    # Final eval
    val_loss = eval_and_log(exp, cfg.n_steps, avp, teacher, val_sample, cfg)
    log.info(f"Final: train_ema={ema_loss:.4f}, val={val_loss:.4f}")
    exp.end()


if __name__ == "__main__":
    main()
