#!/usr/bin/env python3
"""Train ADE20K segmentation probes on AVP features.

Recurrent probes: ONE probe per feature type, trained on features AVERAGED across timesteps.
Static probes (baselines): teacher_full, teacher_glimpse.

Logs mIoU to compare feature quality.
"""

import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TypedDict

import albumentations as A
import comet_ml
import dacite
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from canvit.backbone.dinov3 import DINOv3Backbone
from canvit.hub import create_backbone
from PIL import Image
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import MulticlassJaccardIndex
from tqdm import tqdm

from avp_vit import ActiveCanViT, ActiveCanViTConfig
from avp_vit.checkpoint import load as load_ckpt
from avp_vit.train.viewpoint import Viewpoint  # Use this, NOT canvit.viewpoint - has safe-box sampling

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

NUM_CLASSES = 150
IGNORE_LABEL = 255
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])

FeatureType = Literal["hidden", "predicted_norm", "teacher_full", "teacher_glimpse"]


@dataclass
class Config:
    avp_ckpt: Path
    ade20k_root: Path = Path("/datasets/ADE20k/ADEChallengeData2016")
    teacher_ckpt: Path | None = None

    features: list[FeatureType] = field(default_factory=lambda: ["hidden", "predicted_norm", "teacher_glimpse"])
    n_timesteps: int = 5

    image_size: int = 512
    batch_size: int = 64
    eval_batch_size: int = 32
    num_workers: int = 4

    peak_lr: float = 1e-4
    min_lr: float = 1e-7
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.1
    max_steps: int = 5000
    grad_clip: float = 1.0
    focal_gamma: float = 2.0

    log_every: int = 20
    val_every: int = 500
    viz_every: int = 1000
    ema_alpha: float = 0.1

    comet_project: str = "avp-ade20k-probe"
    comet_workspace: str = "m2b3-ava"
    device: str | None = None
    amp: bool = True
    probe_ckpt_dir: Path | None = None


class ProbeHead(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ln = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, NUM_CLASSES)

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 4, f"Expected [B,H,W,D], got {x.shape}"
        assert x.shape[-1] == self.embed_dim, f"Expected D={self.embed_dim}, got {x.shape[-1]}"
        return self.linear(self.ln(x)).permute(0, 3, 1, 2)


@dataclass
class Probe:
    name: str
    head: ProbeHead
    optimizer: AdamW
    scheduler: SequentialLR
    loss_ema: float = 0.0
    grad_norm_ema: float = 0.0
    best_miou: float = 0.0

    def update(self, loss: float, grad_norm: float, alpha: float) -> None:
        if self.loss_ema == 0.0:
            self.loss_ema, self.grad_norm_ema = loss, grad_norm
        else:
            self.loss_ema = alpha * loss + (1 - alpha) * self.loss_ema
            self.grad_norm_ema = alpha * grad_norm + (1 - alpha) * self.grad_norm_ema


@dataclass
class Features:
    """Extracted features. All are single tensors [B, H, W, D].

    Recurrent features (hidden, predicted_norm) are AVERAGED across timesteps.
    This is the correct approach: ONE probe trained on averaged features.
    """
    hidden: Tensor           # canvas hidden state, averaged across timesteps
    predicted_norm: Tensor   # predicted teacher features, averaged across timesteps
    teacher_full: Tensor     # teacher features on full image (baseline)
    teacher_glimpse: Tensor  # teacher features on small image (baseline)

    def get(self, feat_type: FeatureType) -> Tensor:
        if feat_type == "hidden":
            return self.hidden
        elif feat_type == "predicted_norm":
            return self.predicted_norm
        elif feat_type == "teacher_full":
            return self.teacher_full
        elif feat_type == "teacher_glimpse":
            return self.teacher_glimpse
        raise ValueError(f"Unknown feature type: {feat_type}")


def focal_loss(logits: Tensor, masks: Tensor, gamma: float) -> Tensor:
    """Pixel-wise focal loss. Upsamples logits to mask resolution."""
    C, H, W = logits.shape[1], masks.shape[1], masks.shape[2]
    logits_up = F.interpolate(logits, (H, W), mode="bilinear", align_corners=False)
    log_probs = F.log_softmax(logits_up, dim=1)  # [B, C, H, W]
    probs = log_probs.exp()

    valid = masks != IGNORE_LABEL  # [B, H, W]
    targets = masks.clamp(0, C - 1).long()  # [B, H, W]

    log_p = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B, H, W]
    p = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B, H, W]
    return -((1 - p) ** gamma * log_p * valid).sum() / valid.sum().clamp(min=1)


class ADE20kDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, root: Path, split: str, size: int, augment: bool = False) -> None:
        self.size = size
        self.load_size = size * 2 if augment else size
        self.transform = A.Compose([A.HorizontalFlip(p=0.5), A.RandomCrop(size, size)]) if augment else None
        img_dir, ann_dir = root / "images" / split, root / "annotations" / split
        self.imgs = sorted(img_dir.glob("*.jpg"))
        self.anns = [ann_dir / (p.stem + ".png") for p in self.imgs]
        assert len(self.imgs) > 0, f"No images found in {img_dir}"
        assert len(self.imgs) == len(self.anns)
        log.info(f"ADE20k {split}: {len(self)} images from {root}")

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        img = np.array(Image.open(self.imgs[i]).convert("RGB").resize((self.load_size, self.load_size), Image.Resampling.BILINEAR))
        mask = np.array(Image.open(self.anns[i]).resize((self.load_size, self.load_size), Image.Resampling.NEAREST))
        if self.transform:
            out = self.transform(image=img, mask=mask)
            img, mask = out["image"], out["mask"]
        img_t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        img_t = (img_t - IMAGENET_MEAN.view(3, 1, 1)) / IMAGENET_STD.view(3, 1, 1)
        mask_t = torch.from_numpy(mask.astype(np.int64))
        valid = (mask_t >= 1) & (mask_t <= 150)
        return img_t, torch.where(valid, mask_t - 1, IGNORE_LABEL)


def extract_features(
    model: ActiveCanViT,
    teacher: DINOv3Backbone,
    images: Tensor,
    n_timesteps: int,
    canvas_grid: int,
    glimpse_grid: int,
    glimpse_px: int,
    teacher_dim: int,
    device: torch.device,
) -> Features:
    """Extract features from model and teacher.

    Recurrent features are AVERAGED across all timesteps - this is the correct
    approach for probe training (ONE probe on averaged features, not N probes).

    IMPORTANT: Uses Viewpoint.random() for proper safe-box-area sampling.
    DO NOT use raw torch.rand() - viewpoints can go out of bounds!
    """
    B, C, H, W = images.shape
    assert C == 3 and H == W, f"Expected square RGB images, got {images.shape}"

    hidden_sum = torch.zeros(B, canvas_grid, canvas_grid, model.canvas_dim, device=device)
    predicted_sum = torch.zeros(B, canvas_grid, canvas_grid, teacher_dim, device=device)
    state = model.init_state(batch_size=B, canvas_grid_size=canvas_grid)

    for t in range(n_timesteps):
        # t=0: full scene view. t>0: random viewpoints with PROPER safe-box sampling.
        if t == 0:
            vp = Viewpoint.full_scene(batch_size=B, device=device)
        else:
            # MUST use Viewpoint.random() - ensures |center| + scale <= 1
            # DO NOT use raw torch.rand() - viewpoints can go out of bounds!
            vp = Viewpoint.random(batch_size=B, device=device, min_scale=0.1, max_scale=0.5)

        out = model.forward_step(image=images, state=state, viewpoint=vp, glimpse_size_px=glimpse_px)
        state = out.state

        hidden_sum += model.get_spatial(state.canvas).view(B, canvas_grid, canvas_grid, -1)
        predicted_sum += model.predict_teacher_scene(state.canvas).view(B, canvas_grid, canvas_grid, -1)

    # Average across timesteps
    hidden_avg = hidden_sum / n_timesteps
    predicted_avg = predicted_sum / n_timesteps

    # Teacher baselines (no recurrence)
    teacher_full = teacher.forward_norm_features(images).patches.view(B, canvas_grid, canvas_grid, -1)
    sz = glimpse_grid * teacher.patch_size_px
    small = F.interpolate(images, (sz, sz), mode="bilinear", align_corners=False)
    teacher_glimpse = teacher.forward_norm_features(small).patches.view(B, glimpse_grid, glimpse_grid, -1)

    return Features(hidden_avg, predicted_avg, teacher_full, teacher_glimpse)


def upsample_preds(logits: Tensor, target_size: int) -> Tensor:
    """Upsample logits to mask resolution, then argmax."""
    return F.interpolate(logits, (target_size, target_size), mode="bilinear", align_corners=False).argmax(1)


def train_probe(probe: Probe, feat: Tensor, masks: Tensor, *, grad_clip: float, ema_alpha: float, focal_gamma: float) -> None:
    """Single probe training step."""
    probe.head.train()
    probe.optimizer.zero_grad()
    loss = focal_loss(probe.head(feat.detach().float()), masks, gamma=focal_gamma)
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(probe.head.parameters(), grad_clip).item()
    probe.optimizer.step()
    probe.scheduler.step()
    probe.update(loss.item(), grad_norm, ema_alpha)


def eval_probe(probe: Probe, feat: Tensor, masks: Tensor, iou_metric: MulticlassJaccardIndex) -> None:
    """Single probe evaluation step."""
    preds = upsample_preds(probe.head(feat.float()), masks.shape[1])
    iou_metric.update(preds, masks)


def log_viz(
    exp: comet_ml.Experiment,
    step: int,
    probes: dict[str, Probe],
    feats: Features,
    images: Tensor,
    masks: Tensor,
    cfg: Config,
) -> None:
    """Log segmentation visualization to Comet."""
    n = min(4, images.shape[0])
    palette = np.random.RandomState(42).randint(0, 255, (NUM_CLASSES + 1, 3), dtype=np.uint8)
    palette[NUM_CLASSES] = 0

    def colorize(m: np.ndarray) -> np.ndarray:
        return palette[np.where(m == IGNORE_LABEL, NUM_CLASSES, m)]

    def denorm(t: Tensor) -> np.ndarray:
        t = t * IMAGENET_STD.view(3, 1, 1).to(t.device) + IMAGENET_MEAN.view(3, 1, 1).to(t.device)
        return (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    preds: dict[str, np.ndarray] = {}
    for feat_type in cfg.features:
        feat = feats.get(feat_type)
        pred = upsample_preds(probes[feat_type].head(feat.float()), masks.shape[1])[:n].cpu().numpy()
        preds[feat_type] = pred

    cols = 2 + len(preds)
    fig, axes = plt.subplots(n, cols, figsize=(2.5 * cols, 2.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        col = 0
        axes[i, col].imshow(denorm(images[i]))
        axes[i, col].set_title("Image")
        col += 1
        axes[i, col].imshow(colorize(masks[i].cpu().numpy()))
        axes[i, col].set_title("GT")
        col += 1
        for name, pred in preds.items():
            axes[i, col].imshow(colorize(pred[i]))
            axes[i, col].set_title(name[:8])
            col += 1
        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    exp.log_figure(figure_name=f"predictions_{step}", figure=fig, step=step)
    plt.close(fig)


def make_probe(name: str, dim: int, cfg: Config, device: torch.device) -> Probe:
    warmup_steps = int(cfg.warmup_ratio * cfg.max_steps)
    head = ProbeHead(dim).to(device)
    opt = AdamW(head.parameters(), lr=cfg.peak_lr, weight_decay=cfg.weight_decay)
    warmup = LinearLR(opt, cfg.min_lr / cfg.peak_lr, 1.0, max(1, warmup_steps))
    cosine = CosineAnnealingLR(opt, cfg.max_steps - warmup_steps, eta_min=cfg.min_lr)
    return Probe(name, head, opt, SequentialLR(opt, [warmup, cosine], [warmup_steps]))


class ProbeCheckpoint(TypedDict):
    """Checkpoint for probe heads."""
    probe_state_dicts: dict[str, dict[str, Tensor]]
    best_mious: dict[str, float]
    step: int
    config: dict
    avp_ckpt: str
    timestamp: str


def save_probes(path: Path, probes: dict[str, Probe], step: int, cfg: Config) -> None:
    """Save all probe heads atomically."""
    data: ProbeCheckpoint = {
        "probe_state_dicts": {name: p.head.state_dict() for name, p in probes.items()},
        "best_mious": {name: p.best_miou for name, p in probes.items()},
        "step": step,
        "config": asdict(cfg),
        "avp_ckpt": str(cfg.avp_ckpt),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".pt.tmp", dir=path.parent)
    try:
        os.close(fd)
        torch.save(data, tmp)
        Path(tmp).rename(path)
        size_mb = path.stat().st_size / (1024 * 1024)
        log.info(f"Saved probes: {path} ({size_mb:.1f} MB, step={step})")
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


def main(cfg: Config) -> None:
    torch.set_float32_matmul_precision("high")
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    log.info("=" * 60)
    log.info("ADE20K Probe Training")
    log.info("=" * 60)
    log.info(f"Device: {device}, AMP: {cfg.amp}")
    log.info(f"AVP checkpoint: {cfg.avp_ckpt}")
    log.info(f"Features: {cfg.features}")
    log.info(f"Timesteps: {cfg.n_timesteps}")
    log.info(f"Batch size: {cfg.batch_size}, Max steps: {cfg.max_steps}")

    amp_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if cfg.amp else torch.autocast(device_type=device.type, enabled=False)

    # Load AVP model
    log.info("Loading AVP model...")
    ckpt = load_ckpt(cfg.avp_ckpt, device)
    model_cfg = dacite.from_dict(ActiveCanViTConfig, {**ckpt["model_config"], "teacher_dim": ckpt["teacher_dim"]})
    bb = create_backbone(ckpt["backbone"], pretrained=False)
    model = ActiveCanViT(backbone=bb, cfg=model_cfg, policy=None)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing:
        log.warning(f"  Missing keys: {missing}")
    if unexpected:
        log.info(f"  Unexpected keys (ignored): {unexpected}")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    avp_params = sum(p.numel() for p in model.parameters())
    log.info(f"  Backbone: {ckpt['backbone']}, AVP params: {avp_params:,}")

    # Load teacher
    log.info("Loading teacher...")
    weights = str(cfg.teacher_ckpt) if cfg.teacher_ckpt else None
    teacher = create_backbone(ckpt["backbone"], pretrained=weights is None, weights=weights)
    assert isinstance(teacher, DINOv3Backbone)
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    log.info(f"  Teacher embed_dim: {teacher.embed_dim}, patch_size: {teacher.patch_size_px}")

    # Compute grid sizes
    patch_size = model.backbone.patch_size_px
    assert teacher.patch_size_px == patch_size, f"Teacher/model patch size mismatch: {teacher.patch_size_px} vs {patch_size}"
    assert cfg.image_size % patch_size == 0, f"image_size {cfg.image_size} not divisible by patch_size {patch_size}"

    # glimpse_grid_size is in TRAINING config (not model config)
    from avp_vit.train.config import Config as TrainConfig
    train_hist = ckpt.get("training_config_history") or {}
    train_cfg = list(train_hist.values())[-1] if train_hist else {}
    if "glimpse_grid_size" in train_cfg:
        glimpse_grid = train_cfg["glimpse_grid_size"]
    else:
        # FALLBACK: old checkpoint without training_config_history
        glimpse_grid = TrainConfig.glimpse_grid_size
        log.warning(f"glimpse_grid_size not in checkpoint, using default: {glimpse_grid}")

    canvas_grid = cfg.image_size // patch_size
    glimpse_px = glimpse_grid * patch_size
    log.info(f"  Canvas grid: {canvas_grid}x{canvas_grid}, Glimpse grid: {glimpse_grid}x{glimpse_grid}, Patch: {patch_size}px")

    dims: dict[FeatureType, int] = {
        "hidden": model.canvas_dim,
        "predicted_norm": ckpt["teacher_dim"],
        "teacher_full": teacher.embed_dim,
        "teacher_glimpse": teacher.embed_dim,
    }

    # Create probes - ONE probe per feature type
    # Recurrent features (hidden, predicted_norm) are averaged across timesteps in extract_features()
    log.info("Creating probes...")
    probes: dict[str, Probe] = {}
    for feat in cfg.features:
        probes[feat] = make_probe(feat, dims[feat], cfg, device)

    probe_params = sum(sum(p.numel() for p in probe.head.parameters()) for probe in probes.values())
    log.info(f"  {len(probes)} probes, {probe_params:,} trainable params total")
    log.info(f"  Probe names: {list(probes.keys())}")

    # Create IoU metrics ONCE here - NOT in validation loop (avoids GPU sync on .to(device))
    iou_metrics: dict[str, MulticlassJaccardIndex] = {
        feat: MulticlassJaccardIndex(NUM_CLASSES, ignore_index=IGNORE_LABEL, average="macro").to(device)
        for feat in cfg.features
    }

    # Data
    log.info("Loading datasets...")
    train_ds = ADE20kDataset(cfg.ade20k_root, "training", cfg.image_size, augment=True)
    val_ds = ADE20kDataset(cfg.ade20k_root, "validation", cfg.image_size)
    train_loader = DataLoader(train_ds, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, cfg.eval_batch_size, num_workers=cfg.num_workers, pin_memory=True)

    # Comet
    log.info(f"Initializing Comet: {cfg.comet_workspace}/{cfg.comet_project}")
    exp = comet_ml.Experiment(project_name=cfg.comet_project, workspace=cfg.comet_workspace)
    exp.log_parameters(asdict(cfg))
    exp.log_parameter("avp_params", avp_params)
    exp.log_parameter("probe_params", probe_params)

    log.info("=" * 60)
    log.info("Starting training loop")
    log.info("=" * 60)

    step = 0
    train_iter = iter(train_loader)
    pbar = tqdm(total=cfg.max_steps, desc="Training")

    while step < cfg.max_steps:
        try:
            images, masks = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, masks = next(train_iter)
        images, masks = images.to(device), masks.to(device)

        # Validation
        if step % cfg.val_every == 0:
            for p in probes.values():
                p.head.eval()
            # Reset metrics (already on device - NO .to(device) here, that causes GPU sync!)
            for m in iou_metrics.values():
                m.reset()

            with torch.no_grad():
                for vi, vm in val_loader:
                    vi, vm = vi.to(device), vm.to(device)
                    with amp_ctx:
                        feats = extract_features(model, teacher, vi, cfg.n_timesteps, canvas_grid, glimpse_grid, glimpse_px, ckpt["teacher_dim"], device)
                    for feat_type in cfg.features:
                        eval_probe(probes[feat_type], feats.get(feat_type), vm, iou_metrics[feat_type])

            any_improved = False
            for name, iou in iou_metrics.items():
                miou = iou.compute().item()
                exp.log_metric(f"{name}/val_miou", miou, step=step)
                if miou > probes[name].best_miou:
                    probes[name].best_miou = miou
                    any_improved = True

            if any_improved and cfg.probe_ckpt_dir:
                save_probes(cfg.probe_ckpt_dir / "best.pt", probes, step, cfg)

            postfix = {feat[:3]: f"{iou_metrics[feat].compute().item():.3f}" for feat in cfg.features}
            pbar.set_postfix(postfix)

        # Visualization
        if step % cfg.viz_every == 0:
            with torch.no_grad(), amp_ctx:
                viz_feats = extract_features(model, teacher, images, cfg.n_timesteps, canvas_grid, glimpse_grid, glimpse_px, ckpt["teacher_dim"], device)
            log_viz(exp, step, probes, viz_feats, images, masks, cfg)

        # Train step
        with amp_ctx:
            feats = extract_features(model, teacher, images, cfg.n_timesteps, canvas_grid, glimpse_grid, glimpse_px, ckpt["teacher_dim"], device)

        for feat_type in cfg.features:
            train_probe(probes[feat_type], feats.get(feat_type), masks, grad_clip=cfg.grad_clip, ema_alpha=cfg.ema_alpha, focal_gamma=cfg.focal_gamma)

        step += 1
        pbar.update(1)

        if step % cfg.log_every == 0:
            log_dict: dict[str, float] = {"lr": list(probes.values())[0].scheduler.get_last_lr()[0]}
            for name, p in probes.items():
                log_dict[f"{name}/loss"] = p.loss_ema
                log_dict[f"{name}/grad_norm"] = p.grad_norm_ema
            exp.log_metrics(log_dict, step=step)

    pbar.close()
    log.info("=" * 60)
    log.info("Training complete. Best mIoU:")
    for name, p in probes.items():
        log.info(f"  {name}: {p.best_miou:.4f}")
        exp.log_metric(f"best/{name}", p.best_miou)
    log.info("=" * 60)


if __name__ == "__main__":
    main(tyro.cli(Config))
