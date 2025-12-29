#!/usr/bin/env python3
"""Train ADE20K segmentation probes on AVP and/or teacher features.

Frozen probes: backbone frozen, only probe head trained.
Finetune probes: backbone gradients enabled (separate model copy).

All probes enabled by default. Single-pass validation for efficiency.
"""

import logging
from contextlib import nullcontext
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
from avp_vit.train.norm import PositionAwareNorm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

# === Constants ===
NUM_CLASSES = 150
IGNORE_LABEL = 255
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])

FeatureType = Literal[
    "hidden", "predicted_norm", "predicted_denorm", "teacher_full", "teacher_glimpse"
]
ABBREV = {
    "hidden": "hid",
    "predicted_norm": "prd_n",
    "predicted_denorm": "prd_d",
    "teacher_full": "tch_f",
    "teacher_glimpse": "tch_g",
}


# === Config ===
@dataclass
class Config:
    avp_ckpt: Path
    ade20k_root: Path = Path("/datasets/ADE20k/ADEChallengeData2016")

    # Which feature types to train probes on
    frozen_features: list[FeatureType] = field(
        default_factory=lambda: [
            "hidden",
            "predicted_norm",
            "predicted_denorm",
            "teacher_full",
            "teacher_glimpse",
        ]
    )
    finetune_features: list[FeatureType] = field(default_factory=lambda: ["hidden"])

    image_size: int = 512
    batch_size: int = 128
    eval_batch_size: int = 32
    num_workers: int = 4

    peak_lr: float = 1e-4
    min_lr: float = 1e-7
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.1
    max_steps: int = 20_000
    grad_clip: float = 1.0

    log_every: int = 10
    val_every: int = 200
    boundary_width: int = 3  # pixels from class boundary for boundary mIoU

    comet_project: str = "avp-ade20k-probe"
    comet_workspace: str = "m2b3-ava"
    device: str | None = None
    amp: bool = True
    resume_ckpt: Path | None = None  # resume from combined checkpoint


# === Loss ===
FOCAL_GAMMA = 2.0


def get_boundary_mask(masks: Tensor, width: int, ignore_label: int = IGNORE_LABEL) -> Tensor:
    """Bool tensor: True for pixels within `width` of class boundary. Fully vectorized."""
    B, H, W = masks.shape
    valid = (masks != ignore_label).unsqueeze(1).float()  # (B, 1, H, W)
    m = masks.unsqueeze(1).float()  # (B, 1, H, W)

    # Pad and unfold to get 3x3 neighborhoods
    m_padded = F.pad(m, (1, 1, 1, 1), mode='replicate')
    valid_padded = F.pad(valid, (1, 1, 1, 1), mode='constant', value=0)

    # (B, 9, H, W) - 3x3 patches centered at each pixel
    patches = F.unfold(m_padded, 3).view(B, 9, H, W)
    valid_patches = F.unfold(valid_padded, 3).view(B, 9, H, W)

    center = patches[:, 4:5]  # center of 3x3
    center_valid = valid_patches[:, 4:5]

    # Edge: any valid neighbor differs from valid center
    differs = (patches != center) & (valid_patches > 0) & (center_valid > 0)
    edges = differs.any(dim=1, keepdim=True).float()  # (B, 1, H, W)

    # Dilate by `width`
    if width > 0:
        kernel = 2 * width + 1
        edges = F.max_pool2d(edges, kernel, stride=1, padding=width)

    return (edges.squeeze(1) > 0) & (valid.squeeze(1) > 0)


def implicit_upsample_focal(logits: Tensor, masks: Tensor, scale: int) -> Tensor:
    """Focal loss with implicit nearest-neighbor upsampling. ~95x memory reduction."""
    B, C, h, w = logits.shape
    log_probs = F.log_softmax(logits, dim=1).permute(0, 2, 3, 1).reshape(-1, C)
    probs = log_probs.exp()
    mask_patches = (
        masks.reshape(B, h, scale, w, scale)
        .permute(0, 1, 3, 2, 4)
        .reshape(-1, scale * scale)
    )
    valid = mask_patches != IGNORE_LABEL
    targets = mask_patches.clamp(0, C - 1).long()
    log_p = log_probs.gather(1, targets)
    p = probs.gather(1, targets)
    focal_weight = (1 - p) ** FOCAL_GAMMA
    return -(focal_weight * log_p * valid).sum() / valid.sum().clamp(min=1)


# === Probe head ===
def save_checkpoint(
    path: str | Path,
    step: int,
    probes: list["Probe"],
    ft_model: "ActiveCanViT | None",
) -> None:
    """Save combined checkpoint with all probe states."""
    path = Path(path)
    probe_states = {}
    for p in probes:
        probe_states[p.name] = {
            "head": p.head.state_dict(),
            "optimizer": p.optimizer.state_dict(),
            "scheduler": p.scheduler.state_dict(),
            "best_miou": p.best_miou,
        }
    save_dict: dict[str, object] = {"step": step, "probes": probe_states}
    if ft_model is not None:
        save_dict["ft_model"] = ft_model.state_dict()
    torch.save(save_dict, path)
    log.info(f"Saved checkpoint to {path} (step {step}, {path.stat().st_size / 1e6:.1f}MB)")


class ProbeHead(nn.Module):
    """LN + linear."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, NUM_CLASSES)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, H, W, D) -> (B, C, H, W)
        return self.linear(self.ln(x)).permute(0, 3, 1, 2)


# === Dataset ===
def make_train_transform(size: int) -> A.Compose:
    """Random crop + flip. Images are loaded at 2x size, cropped to target."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomCrop(size, size),
        ]
    )


class ADE20kDataset(Dataset):
    def __init__(
        self, root: Path, split: str, size: int, augment: bool = False
    ) -> None:
        self.size = size
        # Load at 2x for training (random crop = zoom in only), 1x for val
        self.load_size = size * 2 if augment else size
        self.transform = make_train_transform(size) if augment else None
        img_dir = root / "images" / split
        ann_dir = root / "annotations" / split
        self.imgs = sorted(img_dir.glob("*.jpg"))
        self.anns = [ann_dir / (p.stem + ".png") for p in self.imgs]
        log.info(
            f"ADE20k {split}: {len(self)} images, load@{self.load_size} -> {size}, augment={augment}"
        )

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        img = np.array(
            Image.open(self.imgs[i])
            .convert("RGB")
            .resize((self.load_size, self.load_size), Image.Resampling.BILINEAR)
        )
        mask = np.array(
            Image.open(self.anns[i]).resize(
                (self.load_size, self.load_size), Image.Resampling.NEAREST
            )
        )

        if self.transform:
            out = self.transform(image=img, mask=mask)
            img, mask = out["image"], out["mask"]

        img_t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        img_t = (img_t - IMAGENET_MEAN.view(3, 1, 1)) / IMAGENET_STD.view(3, 1, 1)

        mask_t = torch.from_numpy(mask.astype(np.int64))
        # Remap: 0 (background) and 255 (padding) → IGNORE_LABEL, classes 1-150 → 0-149
        valid_class = (mask_t >= 1) & (mask_t <= 150)
        mask_t = torch.where(valid_class, mask_t - 1, IGNORE_LABEL)
        return img_t, mask_t


# === Probe state ===
@dataclass
class Probe:
    name: str
    feature: FeatureType
    finetune: bool
    head: ProbeHead
    optimizer: AdamW
    scheduler: SequentialLR
    loss_sum: Tensor | None = None
    loss_count: int = 0
    best_miou: float = 0.0

    def accumulate_loss(self, loss: Tensor) -> None:
        if self.loss_sum is None:
            self.loss_sum = loss.detach()
        else:
            self.loss_sum = self.loss_sum + loss.detach()
        self.loss_count += 1

    def get_and_reset_loss(self) -> float | None:
        if self.loss_count == 0 or self.loss_sum is None:
            return None
        avg = (self.loss_sum / self.loss_count).item()
        self.loss_sum = None
        self.loss_count = 0
        return avg


# === Feature extraction ===
class FeatureExtractor:
    def __init__(
        self,
        model: ActiveCanViT,
        scene_norm: PositionAwareNorm | None,
        teacher: DINOv3Backbone | None,
        canvas_grid: int,
        glimpse_grid: int,
        glimpse_px: int,
        teacher_patch_size: int,
        device: torch.device,
    ):
        self.model = model
        self.scene_norm = scene_norm
        self.teacher = teacher
        self.canvas_grid = canvas_grid
        self.glimpse_grid = glimpse_grid
        self.glimpse_px = glimpse_px
        self.teacher_patch_size = teacher_patch_size
        self.device = device

    def extract(
        self, images: Tensor, features: set[FeatureType], with_grad: bool
    ) -> dict[FeatureType, Tensor]:
        B = images.shape[0]
        result: dict[FeatureType, Tensor] = {}
        ctx = torch.enable_grad() if with_grad else torch.no_grad()

        avp_features = {"hidden", "predicted_norm", "predicted_denorm"}
        if features & avp_features:
            with ctx:
                canvas = self.model.init_canvas(
                    batch_size=B, canvas_grid_size=self.canvas_grid
                )
                cls = self.model.init_cls(batch_size=B)
                vp = Viewpoint(
                    torch.zeros(B, 2, device=self.device),
                    torch.ones(B, device=self.device),
                )
                out = self.model.forward_step(
                    image=images,
                    canvas=canvas,
                    cls=cls,
                    viewpoint=vp,
                    glimpse_size_px=self.glimpse_px,
                )

                if "hidden" in features:
                    result["hidden"] = self.model.get_spatial(out.canvas).view(
                        B, self.canvas_grid, self.canvas_grid, -1
                    )
                if "predicted_norm" in features or "predicted_denorm" in features:
                    pred = self.model.predict_teacher_scene(out.canvas)
                    if "predicted_norm" in features:
                        result["predicted_norm"] = pred.view(
                            B, self.canvas_grid, self.canvas_grid, -1
                        )
                    if "predicted_denorm" in features:
                        denorm = (
                            self.scene_norm.denormalize(pred)
                            if self.scene_norm
                            else pred
                        )
                        result["predicted_denorm"] = denorm.view(
                            B, self.canvas_grid, self.canvas_grid, -1
                        )

        if self.teacher is not None:
            with (
                torch.no_grad()
            ):  # not inference_mode - tensors must be usable by probe heads
                if "teacher_full" in features:
                    feats = self.teacher.forward_norm_features(images)
                    result["teacher_full"] = feats.patches.view(
                        B, self.canvas_grid, self.canvas_grid, -1
                    )
                if "teacher_glimpse" in features:
                    sz = self.glimpse_grid * self.teacher_patch_size
                    small = F.interpolate(
                        images, (sz, sz), mode="bilinear", align_corners=False
                    )
                    feats = self.teacher.forward_norm_features(small)
                    result["teacher_glimpse"] = feats.patches.view(
                        B, self.glimpse_grid, self.glimpse_grid, -1
                    )

        return result


# === Training ===
class ProbeTrainer:
    def __init__(
        self,
        probes: list[Probe],
        frozen_ext: FeatureExtractor,
        ft_ext: FeatureExtractor | None,
        ft_model: ActiveCanViT | None,
        device: torch.device,
        grad_clip: float,
        amp_ctx: torch.autocast | nullcontext,
        boundary_width: int = 3,
    ):
        self.probes = probes
        self.frozen_ext = frozen_ext
        self.ft_ext = ft_ext
        self.ft_model = ft_model
        self.device = device
        self.grad_clip = grad_clip
        self.amp_ctx = amp_ctx
        self.boundary_width = boundary_width
        self.frozen_probes = [p for p in probes if not p.finetune]
        self.ft_probes = [p for p in probes if p.finetune]

    def train_step(self, images: Tensor, masks: Tensor) -> dict[str, Tensor]:
        H_mask = masks.shape[1]
        grad_norms: dict[str, Tensor] = {}

        # Frozen probes
        frozen_features: set[FeatureType] = {p.feature for p in self.frozen_probes}
        if frozen_features:
            with self.amp_ctx:
                feats = self.frozen_ext.extract(images, frozen_features, with_grad=False)
            for p in self.frozen_probes:
                feat = feats[p.feature]
                scale = H_mask // feat.shape[1]
                p.head.train()
                p.optimizer.zero_grad()
                with self.amp_ctx:
                    loss = implicit_upsample_focal(p.head(feat.detach()), masks, scale)
                loss.backward()
                grad_norms[p.name] = nn.utils.clip_grad_norm_(
                    p.head.parameters(), self.grad_clip
                ).detach()
                p.optimizer.step()
                p.scheduler.step()
                p.accumulate_loss(loss)

        # Finetune probes
        if self.ft_ext is not None and self.ft_model is not None:
            for p in self.ft_probes:
                p.head.train()
                p.optimizer.zero_grad()
                with self.amp_ctx:
                    feat = self.ft_ext.extract(images, {p.feature}, with_grad=True)[
                        p.feature
                    ]
                    scale = H_mask // feat.shape[1]
                    loss = implicit_upsample_focal(p.head(feat), masks, scale)
                loss.backward()
                params = list(self.ft_model.parameters()) + list(p.head.parameters())
                grad_norms[p.name] = nn.utils.clip_grad_norm_(
                    params, self.grad_clip
                ).detach()
                p.optimizer.step()
                p.scheduler.step()
                p.accumulate_loss(loss)

        return grad_norms

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict[str, float]:
        for p in self.probes:
            p.head.eval()

        ious = {
            p.name: MulticlassJaccardIndex(
                NUM_CLASSES, ignore_index=IGNORE_LABEL, average="macro"
            ).to(self.device)
            for p in self.probes
        }
        boundary_ious = {
            p.name: MulticlassJaccardIndex(
                NUM_CLASSES, ignore_index=IGNORE_LABEL, average="macro"
            ).to(self.device)
            for p in self.probes
        }
        losses = {p.name: 0.0 for p in self.probes}

        # Cross-eval: teacher_full probe on predicted features
        teacher_full_probe = next(
            (p for p in self.probes if p.name == "teacher_full"), None
        )
        cross_features: list[FeatureType] = (
            ["predicted_norm", "predicted_denorm"] if teacher_full_probe else []
        )
        cross_ious = {
            f: MulticlassJaccardIndex(
                NUM_CLASSES, ignore_index=IGNORE_LABEL, average="macro"
            ).to(self.device)
            for f in cross_features
        }
        cross_boundary_ious = {
            f: MulticlassJaccardIndex(
                NUM_CLASSES, ignore_index=IGNORE_LABEL, average="macro"
            ).to(self.device)
            for f in cross_features
        }
        cross_losses = {f: 0.0 for f in cross_features}

        # Precompute feature sets (avoid recomputing per batch)
        frozen_feature_set: set[FeatureType] = {
            *(p.feature for p in self.frozen_probes),
            *cross_features,
        }
        ft_feature_set: set[FeatureType] = {p.feature for p in self.ft_probes}

        n = 0
        for images, masks in loader:
            images, masks = images.to(self.device), masks.to(self.device)
            B, H_mask = images.shape[0], masks.shape[1]

            # Boundary mask for this batch (torch.where avoids clone + indexed assign)
            boundary = get_boundary_mask(masks, self.boundary_width)
            masks_boundary = torch.where(boundary, masks, IGNORE_LABEL)

            with self.amp_ctx:
                frozen_feats = (
                    self.frozen_ext.extract(images, frozen_feature_set, with_grad=False)
                    if frozen_feature_set
                    else {}
                )

                ft_feats: dict[FeatureType, Tensor] = {}
                if self.ft_ext is not None and ft_feature_set:
                    ft_feats = self.ft_ext.extract(images, ft_feature_set, with_grad=False)

                for p in self.probes:
                    feat = (
                        ft_feats.get(p.feature)
                        if p.finetune
                        else frozen_feats.get(p.feature)
                    )
                    if feat is None:
                        continue
                    scale = H_mask // feat.shape[1]
                    logits = p.head(feat)
                    losses[p.name] += implicit_upsample_focal(logits, masks, scale).item() * B
                    preds = (
                        logits.argmax(1)
                        .repeat_interleave(scale, 1)
                        .repeat_interleave(scale, 2)
                    )
                    ious[p.name].update(preds, masks)
                    boundary_ious[p.name].update(preds, masks_boundary)

                # Cross-eval: apply teacher_full probe to predicted features
                if teacher_full_probe:
                    for f in cross_features:
                        feat = frozen_feats.get(f)
                        if feat is None:
                            continue
                        scale = H_mask // feat.shape[1]
                        logits = teacher_full_probe.head(feat)
                        cross_losses[f] += (
                            implicit_upsample_focal(logits, masks, scale).item() * B
                        )
                        preds = (
                            logits.argmax(1)
                            .repeat_interleave(scale, 1)
                            .repeat_interleave(scale, 2)
                        )
                        cross_ious[f].update(preds, masks)
                        cross_boundary_ious[f].update(preds, masks_boundary)

            n += B

        result = {f"{name}/val_loss": losses[name] / n for name in losses}
        result |= {f"{name}/val_miou": ious[name].compute().item() for name in ious}
        result |= {f"{name}/val_boundary_miou": boundary_ious[name].compute().item() for name in boundary_ious}
        for f in cross_features:
            result[f"cross/tch_full_on_{ABBREV[f]}/val_loss"] = cross_losses[f] / n
            result[f"cross/tch_full_on_{ABBREV[f]}/val_miou"] = (
                cross_ious[f].compute().item()
            )
            result[f"cross/tch_full_on_{ABBREV[f]}/val_boundary_miou"] = (
                cross_boundary_ious[f].compute().item()
            )
        return result


# === Visualization ===
def log_viz(
    exp: comet_ml.Experiment,
    step: int,
    trainer: ProbeTrainer,
    images: Tensor,
    masks: Tensor,
    n_samples: int = 4,
) -> None:
    import matplotlib.pyplot as plt

    n = min(n_samples, images.shape[0])
    H_mask = masks.shape[1]

    # Get all logits
    frozen_feats = trainer.frozen_ext.extract(
        images, {p.feature for p in trainer.frozen_probes}, False
    )
    ft_feats = (
        trainer.ft_ext.extract(images, {p.feature for p in trainer.ft_probes}, False)
        if trainer.ft_ext
        else {}
    )

    logits_dict = {}
    for p in trainer.probes:
        feat = ft_feats.get(p.feature) if p.finetune else frozen_feats.get(p.feature)
        if feat is not None:
            p.head.eval()
            with torch.no_grad():
                logits_dict[p.name] = p.head(feat)

    # Plot
    n_probes = len(logits_dict)
    fig, axes = plt.subplots(n, 2 + n_probes, figsize=(2.5 * (2 + n_probes), 2.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    palette = np.random.RandomState(42).randint(
        0, 255, (NUM_CLASSES + 1, 3), dtype=np.uint8
    )
    palette[NUM_CLASSES] = 0

    def colorize(m: np.ndarray) -> np.ndarray:
        return palette[np.where(m == IGNORE_LABEL, NUM_CLASSES, m)]

    def denorm(t: Tensor) -> np.ndarray:
        t = t * IMAGENET_STD.view(3, 1, 1).to(t.device) + IMAGENET_MEAN.view(
            3, 1, 1
        ).to(t.device)
        return (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    for i in range(n):
        axes[i, 0].imshow(denorm(images[i]))
        axes[i, 0].set_title("Image")
        axes[i, 1].imshow(colorize(masks[i].cpu().numpy()))
        axes[i, 1].set_title("GT")

        for j, (name, logits) in enumerate(logits_dict.items()):
            scale = H_mask // logits.shape[2]
            pred = (
                logits[i]
                .argmax(0)
                .repeat_interleave(scale, 0)
                .repeat_interleave(scale, 1)
                .cpu()
                .numpy()
            )
            axes[i, 2 + j].imshow(colorize(pred))
            abbrev = ABBREV.get(name.replace("ft_", ""), name[:6])
            axes[i, 2 + j].set_title(
                f"{'ft_' if name.startswith('ft_') else ''}{abbrev}"
            )

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    exp.log_figure(figure_name=f"predictions_{step}", figure=fig, step=step)
    plt.close(fig)


# === Main ===
def main(cfg: Config) -> None:
    torch.set_float32_matmul_precision("high")
    device = torch.device(
        cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    log.info(f"Device: {device}")

    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if cfg.amp else nullcontext()
    )
    log.info(f"AMP: {'bfloat16' if cfg.amp else 'disabled'}")

    # Load checkpoint
    ckpt = load_ckpt(cfg.avp_ckpt, device)
    backbone_name = ckpt["backbone"]
    model_cfg = dacite.from_dict(
        ActiveCanViTConfig, {**ckpt["model_config"], "teacher_dim": ckpt["teacher_dim"]}
    )

    def make_model(trainable: bool = False) -> ActiveCanViT:
        bb = create_backbone(backbone_name, pretrained=False)
        # No policy needed for t=0 full-view probing
        m = ActiveCanViT(backbone=bb, cfg=model_cfg, policy=None)
        m.load_state_dict(ckpt["state_dict"], strict=False)
        m = m.to(device)
        if trainable:
            m.train()
        else:
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
        return m

    frozen_model = make_model(trainable=False)
    ft_model = make_model(trainable=True) if cfg.finetune_features else None

    # Scene norm
    scene_norm = None
    if (ns := ckpt.get("scene_norm_state")) is not None:
        n, d = ns["mean"].shape
        scene_norm = PositionAwareNorm(n, d, int(n**0.5))
        scene_norm.load_state_dict(ns)
        scene_norm = scene_norm.eval().to(device)

    # Teacher
    need_teacher = bool(set(cfg.frozen_features) & {"teacher_full", "teacher_glimpse"})
    teacher = None
    if need_teacher:
        teacher = create_backbone(backbone_name, pretrained=True)
        assert isinstance(teacher, DINOv3Backbone)
        teacher = teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    # Dimensions
    patch_size = frozen_model.backbone.patch_size_px
    teacher_patch_size = teacher.patch_size_px if teacher else patch_size
    canvas_grid = cfg.image_size // patch_size
    glimpse_grid = ckpt["model_config"].get("glimpse_grid_size", 8)
    glimpse_px = glimpse_grid * patch_size
    hidden_dim = frozen_model.canvas_dim
    teacher_dim = ckpt["teacher_dim"]

    def get_dim(f: FeatureType) -> int:
        if f == "hidden":
            return hidden_dim
        if f in ("predicted_norm", "predicted_denorm"):
            return teacher_dim
        return teacher.embed_dim if teacher else teacher_dim

    # Extractors (ft_ext has no teacher - finetune only supports AVP features)
    frozen_ext = FeatureExtractor(
        frozen_model,
        scene_norm,
        teacher,
        canvas_grid,
        glimpse_grid,
        glimpse_px,
        teacher_patch_size,
        device,
    )
    ft_ext = (
        FeatureExtractor(
            ft_model,
            scene_norm,
            None,
            canvas_grid,
            glimpse_grid,
            glimpse_px,
            teacher_patch_size,
            device,
        )
        if ft_model
        else None
    )

    # Create probes
    peak_lr = cfg.peak_lr
    warmup_steps = int(cfg.warmup_ratio * cfg.max_steps)

    def make_probe(name: str, feature: FeatureType, finetune: bool) -> Probe:
        head = ProbeHead(get_dim(feature)).to(device)
        if finetune and ft_model is not None:
            params = list(ft_model.parameters()) + list(head.parameters())
        else:
            params = list(head.parameters())
        opt = AdamW(params, lr=peak_lr, weight_decay=cfg.weight_decay)
        warmup = LinearLR(opt, cfg.min_lr / peak_lr, 1.0, max(1, warmup_steps))
        cosine = CosineAnnealingLR(
            opt, cfg.max_steps - warmup_steps, eta_min=cfg.min_lr
        )
        sched = SequentialLR(opt, [warmup, cosine], [warmup_steps])
        return Probe(name, feature, finetune, head, opt, sched)

    probes = [make_probe(f, f, False) for f in cfg.frozen_features]
    probes += [make_probe(f"ft_{f}", f, True) for f in cfg.finetune_features]

    # Resume from checkpoint
    start_step = 0
    if cfg.resume_ckpt is not None:
        resume = torch.load(cfg.resume_ckpt, map_location=device, weights_only=False)
        start_step = resume["step"]
        for p in probes:
            if p.name in resume["probes"]:
                state = resume["probes"][p.name]
                p.head.load_state_dict(state["head"])
                p.optimizer.load_state_dict(state["optimizer"])
                p.scheduler.load_state_dict(state["scheduler"])
                p.best_miou = state["best_miou"]
                log.info(f"Resumed {p.name}: best_miou={p.best_miou:.4f}")
            else:
                log.warning(f"Probe {p.name} not found in checkpoint, starting fresh")
        if ft_model is not None and "ft_model" in resume:
            ft_model.load_state_dict(resume["ft_model"])
            log.info("Resumed ft_model weights")
        log.info(f"Resuming from step {start_step}")

    trainer = ProbeTrainer(probes, frozen_ext, ft_ext, ft_model, device, cfg.grad_clip, amp_ctx, cfg.boundary_width)

    log.info(f"Probes: {[p.name for p in probes]}")

    # Data
    train_ds = ADE20kDataset(cfg.ade20k_root, "training", cfg.image_size, augment=True)
    val_ds = ADE20kDataset(cfg.ade20k_root, "validation", cfg.image_size, augment=False)
    train_loader = DataLoader(
        train_ds,
        cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, cfg.eval_batch_size, num_workers=cfg.num_workers, pin_memory=True
    )

    # Comet
    exp = comet_ml.Experiment(
        project_name=cfg.comet_project, workspace=cfg.comet_workspace
    )
    exp.log_parameters(asdict(cfg))
    exp.log_parameters(
        {
            "peak_lr": peak_lr,
            "warmup_steps": warmup_steps,
            "canvas_grid": canvas_grid,
            "hidden_dim": hidden_dim,
            "teacher_dim": teacher_dim,
            "backbone": backbone_name,
        }
    )

    # Training loop
    step = start_step
    train_iter = iter(train_loader)
    pbar = tqdm(total=cfg.max_steps, initial=start_step, desc="Training")
    ckpt_path = Path(f"probe_ckpt_{exp.id}.pt")

    while step < cfg.max_steps:
        try:
            images, masks = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, masks = next(train_iter)

        images, masks = (
            images.to(device, non_blocking=True),
            masks.to(device, non_blocking=True),
        )

        # Validate before training (step 0 = baseline)
        if step % cfg.val_every == 0:
            metrics = trainer.validate(val_loader)
            exp.log_metrics(metrics, step=step)
            if step == 0:
                for k, v in metrics.items():
                    log.info(f"  {k}: {v:.4f}")

            # Track best, save checkpoints
            postfix = {}
            for p in probes:
                miou = metrics[f"{p.name}/val_miou"]
                if miou > p.best_miou:
                    p.best_miou = miou
                    ckpt_path = f"probe_{p.name}_best_{exp.id}.pt"
                    save_dict = {"head": p.head.state_dict()}
                    if p.finetune and ft_model:
                        save_dict["backbone"] = ft_model.state_dict()
                    torch.save(save_dict, ckpt_path)
                    log.info(f"Step {step}: new best {p.name} mIoU: {miou:.4f}")
                key = (
                    f"ft_{ABBREV[p.feature]}"
                    if p.finetune
                    else ABBREV.get(p.name, p.name[:6])
                )
                postfix[key] = f"{miou:.3f}"
            pbar.set_postfix(postfix)

            # Save combined checkpoint
            save_checkpoint(ckpt_path, step, probes, ft_model)

            # Viz
            log_viz(exp, step, trainer, images, masks)

        # Train
        grad_norms = trainer.train_step(images, masks)

        step += 1
        pbar.update(1)

        # Log
        if step % cfg.log_every == 0:
            log_dict = {"lr": probes[0].scheduler.get_last_lr()[0]}
            for p in probes:
                if (loss := p.get_and_reset_loss()) is not None:
                    log_dict[f"{p.name}/train_loss"] = loss
                if p.name in grad_norms:
                    log_dict[f"{p.name}/grad_norm"] = grad_norms[p.name].item()
            exp.log_metrics(log_dict, step=step)

    pbar.close()
    log.info("=" * 60)
    log.info("Best mIoU per probe:")
    for p in probes:
        log.info(f"  {p.name}: {p.best_miou:.4f}")
        exp.log_metric(f"best_{p.name}_miou", p.best_miou)


if __name__ == "__main__":
    main(tyro.cli(Config))
