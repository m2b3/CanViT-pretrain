#!/usr/bin/env python3
"""Train ADE20K segmentation probes across multiple timesteps.

Probes: hidden, predicted_norm, teacher_glimpse (frozen backbone only).
Logs mIoU curves vs timestep to see feature quality evolution.
"""

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

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
from canvit.viewpoint import Viewpoint
from PIL import Image
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import MulticlassJaccardIndex
from tqdm import tqdm

from avp_vit import ActiveCanViT, ActiveCanViTConfig
from avp_vit.checkpoint import load as load_ckpt

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
    n_timesteps: int = 5  # t=0 full, t=1..4 random

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

    log_every: int = 20
    val_every: int = 500

    comet_project: str = "avp-ade20k-probe"
    comet_workspace: str = "m2b3-ava"
    device: str | None = None
    amp: bool = True


def focal_loss(logits: Tensor, masks: Tensor, scale: int, gamma: float = 2.0) -> Tensor:
    B, C, h, w = logits.shape
    log_probs = F.log_softmax(logits, dim=1).permute(0, 2, 3, 1).reshape(-1, C)
    probs = log_probs.exp()
    mask_patches = masks.reshape(B, h, scale, w, scale).permute(0, 1, 3, 2, 4).reshape(-1, scale * scale)
    valid = mask_patches != IGNORE_LABEL
    targets = mask_patches.clamp(0, C - 1).long()
    log_p = log_probs.gather(1, targets)
    p = probs.gather(1, targets)
    return -((1 - p) ** gamma * log_p * valid).sum() / valid.sum().clamp(min=1)


class ProbeHead(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, NUM_CLASSES)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(self.ln(x)).permute(0, 3, 1, 2)


class ADE20kDataset(Dataset):
    def __init__(self, root: Path, split: str, size: int, augment: bool = False) -> None:
        self.size = size
        self.load_size = size * 2 if augment else size
        self.transform = A.Compose([A.HorizontalFlip(p=0.5), A.RandomCrop(size, size)]) if augment else None
        img_dir, ann_dir = root / "images" / split, root / "annotations" / split
        self.imgs = sorted(img_dir.glob("*.jpg"))
        self.anns = [ann_dir / (p.stem + ".png") for p in self.imgs]
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


@dataclass
class Probe:
    name: str
    head: ProbeHead
    optimizer: AdamW
    scheduler: SequentialLR
    loss_sum: float = 0.0
    loss_count: int = 0
    best_miou: float = 0.0

    def accumulate(self, loss: Tensor) -> None:
        self.loss_sum += loss.item()
        self.loss_count += 1

    def reset(self) -> float:
        avg = self.loss_sum / max(self.loss_count, 1)
        self.loss_sum, self.loss_count = 0.0, 0
        return avg


def extract_features(
    model: ActiveCanViT,
    teacher: DINOv3Backbone,
    images: Tensor,
    n_timesteps: int,
    canvas_grid: int,
    glimpse_grid: int,
    glimpse_px: int,
    device: torch.device,
) -> dict[str, list[Tensor]]:
    """Extract features at each timestep. Returns {feature_type: [t0, t1, ...]}."""
    B = images.shape[0]
    out: dict[str, list[Tensor]] = {"hidden": [], "predicted_norm": [], "teacher_glimpse": []}
    state = model.init_state(batch_size=B, canvas_grid_size=canvas_grid)

    for t in range(n_timesteps):
        # t=0: full view, t>0: random
        if t == 0:
            vp = Viewpoint(torch.zeros(B, 2, device=device), torch.ones(B, device=device))
        else:
            vp = Viewpoint(torch.rand(B, 2, device=device) * 2 - 1, torch.rand(B, device=device) * 0.4 + 0.1)

        step_out = model.forward_step(image=images, state=state, viewpoint=vp, glimpse_size_px=glimpse_px)
        state = step_out.state

        out["hidden"].append(model.get_spatial(state.canvas).view(B, canvas_grid, canvas_grid, -1))
        out["predicted_norm"].append(model.predict_teacher_scene(state.canvas).view(B, canvas_grid, canvas_grid, -1))

        # Teacher on glimpse-sized input (always same - baseline)
        sz = glimpse_grid * teacher.patch_size_px
        small = F.interpolate(images, (sz, sz), mode="bilinear", align_corners=False)
        out["teacher_glimpse"].append(teacher.forward_norm_features(small).patches.view(B, glimpse_grid, glimpse_grid, -1))

    return out


def main(cfg: Config) -> None:
    torch.set_float32_matmul_precision("high")
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    log.info(f"Device: {device}, AMP: {cfg.amp}")

    amp_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if cfg.amp else torch.autocast(device_type=device.type, enabled=False)

    # Load AVP model
    ckpt = load_ckpt(cfg.avp_ckpt, device)
    model_cfg = dacite.from_dict(ActiveCanViTConfig, {**ckpt["model_config"], "teacher_dim": ckpt["teacher_dim"]})
    bb = create_backbone(ckpt["backbone"], pretrained=False)
    model = ActiveCanViT(backbone=bb, cfg=model_cfg, policy=None)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Teacher
    weights = str(cfg.teacher_ckpt) if cfg.teacher_ckpt else None
    teacher = create_backbone(ckpt["backbone"], pretrained=weights is None, weights=weights)
    assert isinstance(teacher, DINOv3Backbone)
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Dimensions
    canvas_grid = cfg.image_size // model.backbone.patch_size_px
    glimpse_grid = ckpt["model_config"].get("glimpse_grid_size", 8)
    glimpse_px = glimpse_grid * model.backbone.patch_size_px
    dims = {"hidden": model.canvas_dim, "predicted_norm": ckpt["teacher_dim"], "teacher_glimpse": teacher.embed_dim}

    # Create probes: one per (feature, timestep)
    probes: dict[str, Probe] = {}
    warmup_steps = int(cfg.warmup_ratio * cfg.max_steps)
    for feat in cfg.features:
        for t in range(cfg.n_timesteps):
            name = f"{feat}/t{t}"
            head = ProbeHead(dims[feat]).to(device)
            opt = AdamW(head.parameters(), lr=cfg.peak_lr, weight_decay=cfg.weight_decay)
            warmup = LinearLR(opt, cfg.min_lr / cfg.peak_lr, 1.0, max(1, warmup_steps))
            cosine = CosineAnnealingLR(opt, cfg.max_steps - warmup_steps, eta_min=cfg.min_lr)
            probes[name] = Probe(name, head, opt, SequentialLR(opt, [warmup, cosine], [warmup_steps]))
    log.info(f"Probes: {len(probes)} ({cfg.features} x {cfg.n_timesteps} timesteps)")

    # Data
    train_ds = ADE20kDataset(cfg.ade20k_root, "training", cfg.image_size, augment=True)
    val_ds = ADE20kDataset(cfg.ade20k_root, "validation", cfg.image_size)
    train_loader = DataLoader(train_ds, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, cfg.eval_batch_size, num_workers=cfg.num_workers, pin_memory=True)

    # Comet
    exp = comet_ml.Experiment(project_name=cfg.comet_project, workspace=cfg.comet_workspace)
    exp.log_parameters(asdict(cfg))

    # Training
    step, train_iter = 0, iter(train_loader)
    pbar = tqdm(total=cfg.max_steps, desc="Training")

    while step < cfg.max_steps:
        try:
            images, masks = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, masks = next(train_iter)
        images, masks = images.to(device), masks.to(device)
        H_mask = masks.shape[1]

        # Validation
        if step % cfg.val_every == 0:
            for p in probes.values():
                p.head.eval()
            ious = {n: MulticlassJaccardIndex(NUM_CLASSES, ignore_index=IGNORE_LABEL, average="macro").to(device) for n in probes}

            with torch.no_grad():
                for vi, vm in val_loader:
                    vi, vm = vi.to(device), vm.to(device)
                    with amp_ctx:
                        feats = extract_features(model, teacher, vi, cfg.n_timesteps, canvas_grid, glimpse_grid, glimpse_px, device)
                    for feat_type, feat_list in feats.items():
                        if feat_type not in cfg.features:
                            continue
                        for t, feat in enumerate(feat_list):
                            name = f"{feat_type}/t{t}"
                            scale = H_mask // feat.shape[1]
                            preds = probes[name].head(feat).argmax(1).repeat_interleave(scale, 1).repeat_interleave(scale, 2)
                            ious[name].update(preds, vm)

            # Log metrics + curves
            for name, iou in ious.items():
                miou = iou.compute().item()
                exp.log_metric(f"{name}/val_miou", miou, step=step)
                if miou > probes[name].best_miou:
                    probes[name].best_miou = miou

            for feat in cfg.features:
                curve_y = [ious[f"{feat}/t{t}"].compute().item() for t in range(cfg.n_timesteps)]
                exp.log_curve(f"{feat}/miou_vs_t", x=list(range(cfg.n_timesteps)), y=curve_y, step=step)

            pbar.set_postfix({f[:3]: f"{ious[f'{f}/t0'].compute().item():.3f}" for f in cfg.features})

        # Train
        with amp_ctx:
            feats = extract_features(model, teacher, images, cfg.n_timesteps, canvas_grid, glimpse_grid, glimpse_px, device)

        for feat_type, feat_list in feats.items():
            if feat_type not in cfg.features:
                continue
            for t, feat in enumerate(feat_list):
                name = f"{feat_type}/t{t}"
                p = probes[name]
                p.head.train()
                p.optimizer.zero_grad()
                scale = H_mask // feat.shape[1]
                with amp_ctx:
                    loss = focal_loss(p.head(feat.detach()), masks, scale)
                loss.backward()
                nn.utils.clip_grad_norm_(p.head.parameters(), cfg.grad_clip)
                p.optimizer.step()
                p.scheduler.step()
                p.accumulate(loss)

        step += 1
        pbar.update(1)

        if step % cfg.log_every == 0:
            log_dict = {"lr": list(probes.values())[0].scheduler.get_last_lr()[0]}
            for name, p in probes.items():
                log_dict[f"{name}/loss"] = p.reset()
            exp.log_metrics(log_dict, step=step)

    pbar.close()
    log.info("Best mIoU:")
    for name, p in probes.items():
        log.info(f"  {name}: {p.best_miou:.4f}")
        exp.log_metric(f"best/{name}", p.best_miou)


if __name__ == "__main__":
    main(tyro.cli(Config))
