#!/usr/bin/env python3
"""Train ADE20K segmentation probes on AVP features.

ONE probe per feature type with SHARED weights across timesteps (anytime decoding).
Training: loss averaged across timesteps, single backward pass.
Eval: mIoU computed per timestep → logged as curves.

Same policy (t=0 full, t>0 random) in train and val.
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
from avp_vit.train.config import Config as TrainConfig
from avp_vit.train.viewpoint import Viewpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

NUM_CLASSES = 150
IGNORE_LABEL = 255
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])

FeatureType = Literal["hidden", "predicted_norm", "teacher_glimpse"]


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
        return self.linear(self.ln(x)).permute(0, 3, 1, 2)


@dataclass
class Probe:
    name: str
    head: ProbeHead
    optimizer: AdamW
    scheduler: SequentialLR
    best_mean_miou: float = 0.0
    # Accumulators - NO .item() until log time
    _loss_sum: Tensor | None = None
    _grad_norm_sum: Tensor | None = None
    _count: int = 0

    def accumulate(self, loss: Tensor, grad_norm: Tensor) -> None:
        """Accumulate loss/grad_norm tensors. NO GPU SYNC."""
        if self._loss_sum is None:
            self._loss_sum = loss.detach().clone()
            self._grad_norm_sum = grad_norm.detach().clone()
        else:
            self._loss_sum += loss.detach()
            assert self._grad_norm_sum is not None
            self._grad_norm_sum += grad_norm.detach()
        self._count += 1

    def get_and_reset_stats(self) -> tuple[float, float]:
        """Get averaged stats and reset. SYNCS HERE (at log intervals only)."""
        assert self._loss_sum is not None and self._grad_norm_sum is not None
        avg_loss = (self._loss_sum / self._count).item()
        avg_grad = (self._grad_norm_sum / self._count).item()
        self._loss_sum = None
        self._grad_norm_sum = None
        self._count = 0
        return avg_loss, avg_grad


# Per-timestep features: {feature_type: [feat_t0, feat_t1, ...]}
PerTimestepFeatures = dict[FeatureType, list[Tensor]]


def downsample_masks(masks: Tensor, target_h: int, target_w: int) -> Tensor:
    """Downsample masks using nearest neighbor (INT_MAX workaround)."""
    if masks.shape[1:] == (target_h, target_w):
        return masks
    return F.interpolate(masks.unsqueeze(1).float(), (target_h, target_w), mode="nearest").squeeze(1).long()


def focal_loss(logits: Tensor, masks: Tensor, gamma: float) -> Tensor:
    """Focal loss. Downsamples masks to logits resolution."""
    B, C, Hl, Wl = logits.shape
    masks = downsample_masks(masks, Hl, Wl)
    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp()
    valid = masks != IGNORE_LABEL
    targets = masks.clamp(0, C - 1)
    log_p = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    p = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    return -((1 - p) ** gamma * log_p * valid).sum() / valid.sum().clamp(min=1)


class ADE20kDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, root: Path, split: str, size: int, augment: bool = False) -> None:
        self.size = size
        self.load_size = size * 2 if augment else size
        self.transform = A.Compose([A.HorizontalFlip(p=0.5), A.RandomCrop(size, size)]) if augment else None
        img_dir = root / "images" / split
        ann_dir = root / "annotations" / split
        self.imgs = sorted(img_dir.glob("*.jpg"))
        self.anns = [ann_dir / (p.stem + ".png") for p in self.imgs]
        assert len(self.imgs) > 0, f"No images in {img_dir.resolve()} (root={root.resolve()})"
        log.info(f"ADE20k {split}: {len(self)} images")

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
    device: torch.device,
    min_vp_scale: float,
) -> PerTimestepFeatures:
    """Extract per-timestep features. Returns {feat_type: [t0, t1, ...]}."""
    B, C, H, W = images.shape
    assert C == 3 and H == W, f"Expected square RGB images, got {images.shape}"

    feats: PerTimestepFeatures = {"hidden": [], "predicted_norm": [], "teacher_glimpse": []}
    state = model.init_state(batch_size=B, canvas_grid_size=canvas_grid)

    # Teacher glimpse baseline (static, same at all timesteps)
    sz = glimpse_grid * teacher.patch_size_px
    small = F.interpolate(images, size=(sz, sz), mode="bilinear", align_corners=False)
    teacher_glimpse_feat = teacher.forward_norm_features(small).patches.view(B, glimpse_grid, glimpse_grid, -1)
    assert teacher_glimpse_feat.shape == (B, glimpse_grid, glimpse_grid, teacher.embed_dim)

    for t in range(n_timesteps):
        if t == 0:
            vp = Viewpoint.full_scene(batch_size=B, device=device)
        else:
            vp = Viewpoint.random(batch_size=B, device=device, min_scale=min_vp_scale, max_scale=1.0)

        out = model.forward_step(image=images, state=state, viewpoint=vp, glimpse_size_px=glimpse_px)
        state = out.state

        hidden = model.get_spatial(state.canvas).view(B, canvas_grid, canvas_grid, -1)
        predicted = model.predict_teacher_scene(state.canvas).view(B, canvas_grid, canvas_grid, -1)

        feats["hidden"].append(hidden)
        feats["predicted_norm"].append(predicted)
        feats["teacher_glimpse"].append(teacher_glimpse_feat)  # same each timestep (baseline)

    return feats


def make_probe(name: str, dim: int, cfg: Config, device: torch.device) -> Probe:
    warmup_steps = int(cfg.warmup_ratio * cfg.max_steps)
    head = ProbeHead(dim).to(device)
    opt = AdamW(head.parameters(), lr=cfg.peak_lr, weight_decay=cfg.weight_decay)
    warmup = LinearLR(opt, cfg.min_lr / cfg.peak_lr, 1.0, max(1, warmup_steps))
    cosine = CosineAnnealingLR(opt, cfg.max_steps - warmup_steps, eta_min=cfg.min_lr)
    return Probe(name, head, opt, SequentialLR(opt, [warmup, cosine], [warmup_steps]))


class ProbeCheckpoint(TypedDict):
    probe_state_dicts: dict[str, dict[str, Tensor]]
    best_mean_mious: dict[str, float]
    step: int
    config: dict
    avp_ckpt: str
    timestamp: str


def save_probes(path: Path, probes: dict[str, Probe], step: int, cfg: Config) -> None:
    data: ProbeCheckpoint = {
        "probe_state_dicts": {n: p.head.state_dict() for n, p in probes.items()},
        "best_mean_mious": {n: p.best_mean_miou for n, p in probes.items()},
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
        log.info(f"Saved probes: {path} ({path.stat().st_size / 1e6:.1f} MB, step={step})")
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


def main(cfg: Config) -> None:
    torch.set_float32_matmul_precision("high")
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    log.info("=" * 60)
    log.info("ADE20K Probe Training (anytime decoding)")
    log.info("=" * 60)
    log.info(f"Device: {device}, AMP: {cfg.amp}")
    log.info(f"AVP checkpoint: {cfg.avp_ckpt}")
    log.info(f"Features: {cfg.features}, Timesteps: {cfg.n_timesteps}")
    log.info(f"Batch size: {cfg.batch_size}, Max steps: {cfg.max_steps}")

    amp_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if cfg.amp else torch.autocast(device_type=device.type, enabled=False)

    # Load AVP model
    log.info("Loading AVP model...")
    ckpt = load_ckpt(cfg.avp_ckpt, device)
    model_cfg = dacite.from_dict(ActiveCanViTConfig, {**ckpt["model_config"], "teacher_dim": ckpt["teacher_dim"]})
    bb = create_backbone(ckpt["backbone"], pretrained=False)
    model = ActiveCanViT(backbone=bb, cfg=model_cfg, policy=None)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    log.info(f"  Backbone: {ckpt['backbone']}, params: {sum(p.numel() for p in model.parameters()):,}")

    # Load teacher
    log.info("Loading teacher...")
    weights = str(cfg.teacher_ckpt) if cfg.teacher_ckpt else None
    teacher = create_backbone(ckpt["backbone"], pretrained=weights is None, weights=weights)
    assert isinstance(teacher, DINOv3Backbone)
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Grid sizes and viewpoint config from training checkpoint
    patch_size = model.backbone.patch_size_px
    assert cfg.image_size % patch_size == 0, f"image_size {cfg.image_size} not divisible by patch_size {patch_size}"
    canvas_grid = cfg.image_size // patch_size
    train_hist = ckpt.get("training_config_history") or {}
    train_cfg = list(train_hist.values())[-1] if train_hist else {}
    glimpse_grid = train_cfg.get("glimpse_grid_size", TrainConfig.glimpse_grid_size)
    glimpse_px = glimpse_grid * patch_size
    min_vp_scale = train_cfg.get("min_viewpoint_scale", TrainConfig.min_viewpoint_scale)
    log.info(f"  Canvas: {canvas_grid}x{canvas_grid}, Glimpse: {glimpse_grid}x{glimpse_grid}")
    log.info(f"  Viewpoint: t=0 full, t>0 random (min_scale={min_vp_scale}, max_scale=1.0)")
    exp_params = {"glimpse_grid": glimpse_grid, "min_vp_scale": min_vp_scale, "canvas_grid": canvas_grid}

    dims: dict[FeatureType, int] = {
        "hidden": model.canvas_dim,
        "predicted_norm": ckpt["teacher_dim"],
        "teacher_glimpse": teacher.embed_dim,
    }

    # Create probes - ONE per feature type, shared weights across timesteps
    log.info("Creating probes...")
    probes: dict[str, Probe] = {feat: make_probe(feat, dims[feat], cfg, device) for feat in cfg.features}
    log.info(f"  {len(probes)} probes: {list(probes.keys())}")

    # Per-timestep IoU metrics for val
    val_iou: dict[str, list[MulticlassJaccardIndex]] = {
        feat: [MulticlassJaccardIndex(NUM_CLASSES, ignore_index=IGNORE_LABEL, average="macro").to(device) for _ in range(cfg.n_timesteps)]
        for feat in cfg.features
    }
    # Per-timestep IoU metrics for train
    train_iou: dict[str, list[MulticlassJaccardIndex]] = {
        feat: [MulticlassJaccardIndex(NUM_CLASSES, ignore_index=IGNORE_LABEL, average="macro").to(device) for _ in range(cfg.n_timesteps)]
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
    exp.log_parameters(exp_params)  # glimpse_grid, min_vp_scale, canvas_grid from ckpt

    log.info("=" * 60)
    log.info("Starting training")
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

        # === Validation ===
        if step % cfg.val_every == 0:
            for p in probes.values():
                p.head.eval()
            for feat in cfg.features:
                for m in val_iou[feat]:
                    m.reset()
            any_improved = False

            with torch.no_grad():
                for vi, vm in val_loader:
                    vi, vm = vi.to(device), vm.to(device)
                    with amp_ctx:
                        feats = extract_features(model, teacher, vi, cfg.n_timesteps, canvas_grid, glimpse_grid, glimpse_px, device, min_vp_scale)
                    for feat_type in cfg.features:
                        for t in range(cfg.n_timesteps):
                            logits = probes[feat_type].head(feats[feat_type][t].float())
                            preds = F.interpolate(logits, vm.shape[1:], mode="bilinear", align_corners=False).argmax(1)
                            val_iou[feat_type][t].update(preds, vm)

            # Log per-timestep mIoU curves
            for feat_type in cfg.features:
                mious = [val_iou[feat_type][t].compute().item() for t in range(cfg.n_timesteps)]
                mean_miou = sum(mious) / len(mious)
                for t, miou in enumerate(mious):
                    exp.log_metric(f"{feat_type}/val_miou_t{t}", miou, step=step)
                exp.log_metric(f"{feat_type}/val_miou_mean", mean_miou, step=step)
                exp.log_curve(f"{feat_type}/val_miou_curve", x=list(range(cfg.n_timesteps)), y=mious, step=step)

                if mean_miou > probes[feat_type].best_mean_miou:
                    probes[feat_type].best_mean_miou = mean_miou
                    any_improved = True

            if any_improved and cfg.probe_ckpt_dir:
                save_probes(cfg.probe_ckpt_dir / "best.pt", probes, step, cfg)

            pbar.set_postfix({f[:3]: f"{sum(val_iou[f][t].compute().item() for t in range(cfg.n_timesteps)) / cfg.n_timesteps:.3f}" for f in cfg.features})

        # === Training step ===
        for p in probes.values():
            p.head.train()

        with amp_ctx:
            feats = extract_features(model, teacher, images, cfg.n_timesteps, canvas_grid, glimpse_grid, glimpse_px, device, min_vp_scale)

        for feat_type in cfg.features:
            probe = probes[feat_type]
            probe.optimizer.zero_grad()

            # Compute logits BEFORE optimizer step (for both loss and metrics)
            logits_list = [probe.head(feats[feat_type][t].float()) for t in range(cfg.n_timesteps)]
            losses = [focal_loss(logits, masks, cfg.focal_gamma) for logits in logits_list]
            loss = torch.stack(losses).mean()
            loss.backward()

            grad_norm = nn.utils.clip_grad_norm_(probe.head.parameters(), cfg.grad_clip)
            probe.optimizer.step()
            probe.scheduler.step()
            probe.accumulate(loss, grad_norm)  # NO .item() - stays on GPU

            # Update train IoU metrics (using pre-update logits)
            with torch.no_grad():
                for t, logits in enumerate(logits_list):
                    preds = F.interpolate(logits.detach(), masks.shape[1:], mode="bilinear", align_corners=False).argmax(1)
                    train_iou[feat_type][t].update(preds, masks)

        step += 1
        pbar.update(1)

        # === Logging (GPU sync happens here) ===
        if step % cfg.log_every == 0:
            log_dict: dict[str, float] = {"lr": list(probes.values())[0].scheduler.get_last_lr()[0]}
            for name, p in probes.items():
                avg_loss, avg_grad = p.get_and_reset_stats()
                log_dict[f"{name}/loss"] = avg_loss
                log_dict[f"{name}/grad_norm"] = avg_grad

            # Log train mIoU curves
            for feat_type in cfg.features:
                mious = [train_iou[feat_type][t].compute().item() for t in range(cfg.n_timesteps)]
                for t, miou in enumerate(mious):
                    log_dict[f"{feat_type}/train_miou_t{t}"] = miou
                log_dict[f"{feat_type}/train_miou_mean"] = sum(mious) / len(mious)
                exp.log_curve(f"{feat_type}/train_miou_curve", x=list(range(cfg.n_timesteps)), y=mious, step=step)
                # Reset for next interval
                for m in train_iou[feat_type]:
                    m.reset()

            exp.log_metrics(log_dict, step=step)

    pbar.close()
    log.info("=" * 60)
    log.info("Training complete. Best mean mIoU:")
    for name, p in probes.items():
        log.info(f"  {name}: {p.best_mean_miou:.4f}")
        exp.log_metric(f"best/{name}", p.best_mean_miou)
    log.info("=" * 60)


if __name__ == "__main__":
    main(tyro.cli(Config))
