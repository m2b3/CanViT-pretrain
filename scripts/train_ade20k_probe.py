#!/usr/bin/env python3
"""Train ADE20K segmentation probes on AVP and/or teacher features.

AVP probes (require --avp-ckpt):
- probe_hidden: model.get_spatial(canvas) - internal representation
- probe_predicted_norm: model.predict_teacher_scene(canvas) - normalized DINOv3 space
- probe_predicted_denorm: denorm(predict_teacher_scene) - raw DINOv3 space

Teacher probes (baseline comparison, raw DINOv3 output):
- probe_teacher_full: teacher at full canvas resolution
- probe_teacher_glimpse: teacher at glimpse resolution

Usage:
    COMET_API_KEY=$(cat ~/comet_api_key.txt) uv run python scripts/train_ade20k_probe.py \
        --avp-ckpt path/to/checkpoint.pt \
        --ade20k-root /datasets/ADE20k/ADEChallengeData2016 \
        --probe-hidden --probe-predicted-norm --probe-teacher-full
"""

import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import comet_ml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import MulticlassJaccardIndex
from tqdm import tqdm
import tyro

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# === Constants ===
NUM_CLASSES = 150
IGNORE_LABEL = 255
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])
COMET_WORKSPACE = "m2b3-ava"


def implicit_upsample_ce(
    logits: Tensor,
    masks: Tensor,
    scale: int,
    ignore_index: int = IGNORE_LABEL,
) -> Tensor:
    """Cross-entropy with implicit nearest-neighbor upsampling of logits.

    Computes CE as if logits were upsampled to mask resolution, but without
    materializing the upsampled tensor. ~95x memory reduction vs naive.

    Args:
        logits: (B, C, h, w) predictions at patch resolution
        masks: (B, H, W) targets at pixel resolution, H=h*scale, W=w*scale
        scale: upsampling factor
        ignore_index: label to ignore (default: 255)
    """
    B, C, h, w = logits.shape
    H, W = masks.shape[1], masks.shape[2]
    assert H == h * scale and W == w * scale, f"{H}!={h}*{scale} or {W}!={w}*{scale}"

    # log_softmax at low res, flatten to (n_positions, C)
    log_probs = F.log_softmax(logits, dim=1).permute(0, 2, 3, 1).reshape(-1, C)

    # Group scale² high-res targets per low-res position
    mask_patches = (
        masks.reshape(B, h, scale, w, scale)
        .permute(0, 1, 3, 2, 4)
        .reshape(-1, scale * scale)
    )

    # Gather log probs for target classes, masking ignored pixels
    valid = mask_patches != ignore_index
    target_indices = mask_patches.clamp(0, C - 1).long()
    gathered = log_probs.gather(1, target_indices)

    return -(gathered * valid).sum() / valid.sum().clamp(min=1)


@dataclass
class Config:
    avp_ckpt: Path
    ade20k_root: Path = Path("/datasets/ADE20k/ADEChallengeData2016")

    # AVP probes (at least one probe must be enabled)
    probe_hidden: bool = True
    probe_predicted_norm: bool = True
    probe_predicted_denorm: bool = True

    # Teacher probes (baseline comparison)
    probe_teacher_full: bool = True
    probe_teacher_glimpse: bool = True

    # Finetuning: backprop through backbone (not just probe head)
    finetune: bool = False

    # Image/grid settings - will be validated against checkpoint
    image_size: int = 512

    # Training HPs (matched to ava_dv3/train_seg_probe)
    batch_size: int = 128
    eval_batch_size: int = 32
    num_workers: int = 4

    ref_lr: float = 2.5e-5
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.1
    max_steps: int = 20_000
    grad_clip: float = 1.0

    log_every: int = 10
    val_every: int = 200
    viz_every: int = 500
    n_viz_samples: int = 4
    ema_alpha: float = 0.1

    comet_project: str = "avp-ade20k-probe"
    device: str | None = None
    max_train: int | None = None
    max_val: int | None = None


# === Probe ===
class LinearSegmentationHead(nn.Module):
    """BN + 1x1 conv: (B, H, W, D) → (B, C, H, W)."""

    def __init__(self, embed_dim: int, num_classes: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.bn = nn.BatchNorm2d(embed_dim)
        self.conv = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        B, H, W, D = x.shape
        assert D == self.embed_dim, f"Expected D={self.embed_dim}, got {D}"
        x = x.permute(0, 3, 1, 2)  # (B, D, H, W)
        return self.conv(self.bn(x))


# === Dataset ===
def remap_mask(mask: Tensor) -> Tensor:
    """ADE20K: 0→255 (ignore), 1-150→0-149."""
    mask = mask.clone().to(torch.int64)
    mask[mask == 0] = IGNORE_LABEL
    mask[mask != IGNORE_LABEL] -= 1
    return mask


class ADE20kDataset(Dataset):
    def __init__(self, root: Path, split: str, image_size: int) -> None:
        self.image_size = image_size
        images_dir = root / "images" / split
        annotations_dir = root / "annotations" / split
        assert images_dir.exists(), f"Not found: {images_dir}"
        self.image_paths = sorted(images_dir.glob("*.jpg"))
        self.annotation_paths = [annotations_dir / (p.stem + ".png") for p in self.image_paths]
        log.info(f"ADE20k {split}: {len(self)} images at {image_size}x{image_size}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        img_t = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
        mean, std = IMAGENET_MEAN.view(3, 1, 1), IMAGENET_STD.view(3, 1, 1)
        img_t = (img_t - mean) / std

        mask = Image.open(self.annotation_paths[idx])
        mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        mask_t = torch.from_numpy(np.array(mask, dtype=np.int64))
        return img_t, remap_mask(mask_t)


# === Visualization ===
def make_palette(num_classes: int = 150) -> np.ndarray:
    """Generate deterministic color palette."""
    rng = np.random.RandomState(42)
    palette = rng.randint(0, 255, (num_classes + 1, 3), dtype=np.uint8)
    palette[num_classes] = [0, 0, 0]
    return palette


PALETTE = make_palette()


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Class mask → RGB."""
    mask_safe = np.where(mask == IGNORE_LABEL, NUM_CLASSES, mask)
    return PALETTE[mask_safe]


def imagenet_denorm(t: Tensor) -> np.ndarray:
    """(3, H, W) tensor → (H, W, 3) uint8 numpy."""
    mean = IMAGENET_MEAN.view(3, 1, 1).to(t.device)
    std = IMAGENET_STD.view(3, 1, 1).to(t.device)
    t = t * std + mean
    t = t.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (t * 255).astype(np.uint8)


def compute_entropy(logits: Tensor, target_size: int) -> Tensor:
    """Compute per-pixel entropy of categorical distribution."""
    logits_up = F.interpolate(logits, size=(target_size, target_size), mode="bilinear", align_corners=False)
    log_probs = F.log_softmax(logits_up, dim=1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=1)


MAX_ENTROPY = math.log(NUM_CLASSES)


def log_viz(
    exp: comet_ml.Experiment,
    step: int,
    images: Tensor,
    masks: Tensor,
    logits_dict: dict[str, Tensor],
    n_samples: int,
    image_size: int,
) -> None:
    """Log sample predictions and entropy maps to Comet."""
    import matplotlib.pyplot as plt

    n = min(n_samples, images.shape[0])
    n_probes = len(logits_dict)
    # Columns: image, GT, then for each probe: pred + entropy
    n_cols = 2 + 2 * n_probes
    fig, axes = plt.subplots(n, n_cols, figsize=(2.5 * n_cols, 2.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    probe_names = list(logits_dict.keys())
    for i in range(n):
        img_np = imagenet_denorm(images[i])
        mask_np = masks[i].cpu().numpy()

        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title("Image")
        axes[i, 1].imshow(colorize_mask(mask_np))
        axes[i, 1].set_title("GT")

        col = 2
        for name in probe_names:
            logits = logits_dict[name]
            H_mask = masks.shape[1]

            # Prediction
            pred_up = pred_nearest(logits[i:i+1], H_mask)[0].cpu().numpy()
            axes[i, col].imshow(colorize_mask(pred_up))
            axes[i, col].set_title(f"{name[:8]}")
            col += 1

            # Entropy
            entropy = compute_entropy(logits[i:i+1], H_mask)[0].cpu().numpy()
            axes[i, col].imshow(entropy, cmap="magma", vmin=0, vmax=MAX_ENTROPY)
            axes[i, col].set_title(f"{name[:8]}_H")
            col += 1

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    exp.log_figure(figure_name=f"predictions_step_{step}", figure=fig, step=step)
    plt.close(fig)


# === Feature extraction ===
class FeatureExtractor:
    """Extracts features from AVP model and optionally teacher."""

    def __init__(
        self,
        model: nn.Module,
        scene_norm: nn.Module | None,
        canvas_grid: int,
        glimpse_grid: int,
        glimpse_px: int,
        patch_size: int,
        device: torch.device,
        teacher: nn.Module | None = None,
    ):
        self.model = model
        self.scene_norm = scene_norm
        self.canvas_grid = canvas_grid
        self.glimpse_grid = glimpse_grid
        self.glimpse_px = glimpse_px
        self.patch_size = patch_size
        self.device = device
        self.teacher = teacher

        from canvit.viewpoint import Viewpoint
        self.Viewpoint = Viewpoint

    def extract(self, images: Tensor, enabled: set[str]) -> dict[str, Tensor]:
        """Extract requested feature types at t=0 (full scene).

        Returns dict with keys from enabled set.
        Note: teacher_glimpse has different spatial resolution than others!
        Shapes: (B, H, W, D) where H=canvas_grid for most, H=glimpse_grid for teacher_glimpse
        """
        B = images.shape[0]
        result: dict[str, Tensor] = {}

        # AVP features (hidden, predicted_norm, predicted_denorm)
        need_avp = bool(enabled & {"hidden", "predicted_norm", "predicted_denorm"})
        if need_avp:
            canvas = self.model.init_canvas(batch_size=B, canvas_grid_size=self.canvas_grid)
            cls = self.model.init_cls(batch_size=B)
            vp = self.Viewpoint(
                centers=torch.zeros(B, 2, device=self.device),
                scales=torch.ones(B, device=self.device),
            )

            with torch.inference_mode():
                out = self.model.forward_step(
                    image=images, canvas=canvas, cls=cls,
                    viewpoint=vp, glimpse_size_px=self.glimpse_px,
                )

            if "hidden" in enabled:
                hidden_flat = self.model.get_spatial(out.canvas)
                result["hidden"] = hidden_flat.view(B, self.canvas_grid, self.canvas_grid, -1)

            if "predicted_norm" in enabled or "predicted_denorm" in enabled:
                predicted_norm_flat = self.model.predict_teacher_scene(out.canvas)
                if "predicted_norm" in enabled:
                    result["predicted_norm"] = predicted_norm_flat.view(B, self.canvas_grid, self.canvas_grid, -1)
                if "predicted_denorm" in enabled:
                    if self.scene_norm is not None:
                        predicted_denorm_flat = self.scene_norm.denormalize(predicted_norm_flat)
                    else:
                        predicted_denorm_flat = predicted_norm_flat
                    result["predicted_denorm"] = predicted_denorm_flat.view(B, self.canvas_grid, self.canvas_grid, -1)

        # Teacher features
        if self.teacher is not None:
            if "teacher_full" in enabled:
                with torch.inference_mode():
                    feats = self.teacher.forward_norm_features(images)
                result["teacher_full"] = feats.patches.view(B, self.canvas_grid, self.canvas_grid, -1)

            if "teacher_glimpse" in enabled:
                # Resize to glimpse resolution
                glimpse_size = self.glimpse_grid * self.patch_size
                images_small = F.interpolate(images, size=(glimpse_size, glimpse_size), mode="bilinear", align_corners=False)
                with torch.inference_mode():
                    feats = self.teacher.forward_norm_features(images_small)
                result["teacher_glimpse"] = feats.patches.view(B, self.glimpse_grid, self.glimpse_grid, -1)

        return result


# === Probe manager ===
@dataclass
class ProbeState:
    """State for a single probe."""
    probe: LinearSegmentationHead
    optimizer: AdamW
    scheduler: SequentialLR
    loss_ema: float = 0.0
    best_miou: float = 0.0


class ProbeManager:
    """Manages multiple probes with shared training logic."""

    def __init__(
        self,
        probe_configs: dict[str, int],  # name -> embed_dim
        num_classes: int,
        peak_lr: float,
        weight_decay: float,
        warmup_steps: int,
        max_steps: int,
        device: torch.device,
    ):
        self.probes: dict[str, ProbeState] = {}
        self.device = device

        for name, embed_dim in probe_configs.items():
            probe = LinearSegmentationHead(embed_dim, num_classes).to(device)
            optimizer = AdamW(probe.parameters(), lr=peak_lr, weight_decay=weight_decay)

            warmup = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=max(1, warmup_steps))
            cosine = CosineAnnealingLR(optimizer, T_max=max_steps - warmup_steps)
            scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])

            self.probes[name] = ProbeState(probe=probe, optimizer=optimizer, scheduler=scheduler)
            log.info(f"Probe '{name}': embed_dim={embed_dim}, params={sum(p.numel() for p in probe.parameters()):,}")

    def train_step(
        self,
        features: dict[str, Tensor],
        masks: Tensor,
        grad_clip: float,
        ema_alpha: float,
    ) -> dict[str, tuple[Tensor, Tensor]]:
        """Train all probes on given features. Returns (loss, grad_norm) tensors per probe.

        Args:
            features: Dict of feature tensors, each (B, H, W, D) - resolutions may differ
            masks: Full-resolution masks (B, H_img, W_img) - upsampled implicitly

        NOTE: Returns tensors to avoid GPU sync. Call .item() only at log intervals.
        """
        metrics: dict[str, tuple[Tensor, Tensor]] = {}
        H_mask = masks.shape[1]

        for name, state in self.probes.items():
            if name not in features:
                continue

            feat = features[name].detach()
            H_feat = feat.shape[1]
            scale = H_mask // H_feat
            assert H_mask == H_feat * scale, f"mask {H_mask} not divisible by feat {H_feat}"

            state.probe.train()
            state.optimizer.zero_grad()

            logits = state.probe(feat)  # (B, C, H_feat, W_feat)
            loss = implicit_upsample_ce(logits, masks, scale)
            loss.backward()

            grad_norm = nn.utils.clip_grad_norm_(state.probe.parameters(), grad_clip)
            state.optimizer.step()
            state.scheduler.step()

            # Store tensors - NO .item() here to avoid sync
            metrics[name] = (loss.detach(), grad_norm.detach())

        return metrics

    def update_ema(self, metrics: dict[str, tuple[Tensor, Tensor]], ema_alpha: float) -> None:
        """Update EMA loss from metrics. Call at log intervals (syncs GPU)."""
        for name, (loss, _) in metrics.items():
            loss_val = loss.item()  # sync here is OK - only at log intervals
            state = self.probes[name]
            state.loss_ema = ema_alpha * loss_val + (1 - ema_alpha) * state.loss_ema if state.loss_ema > 0 else loss_val

    def get_lr(self) -> float:
        """Get current LR (same for all probes)."""
        first = next(iter(self.probes.values()))
        return first.scheduler.get_last_lr()[0]


# === Validation ===
def pred_nearest(logits: Tensor, target_size: int) -> Tensor:
    """Argmax at low-res then nearest-neighbor upsample to target_size."""
    _, _, h, _ = logits.shape
    scale = target_size // h
    return (
        logits.argmax(dim=1)
        .repeat_interleave(scale, dim=1)
        .repeat_interleave(scale, dim=2)
    )


@torch.no_grad()
def validate(
    extractor: FeatureExtractor,
    probes: ProbeManager,
    loader: DataLoader,
    device: torch.device,
    enabled_probes: set[str],
    image_size: int,
) -> dict[str, float]:
    """Compute val metrics for all probes (upsample preds, not downsample masks)."""
    for state in probes.probes.values():
        state.probe.eval()

    metrics_iou = {name: MulticlassJaccardIndex(NUM_CLASSES, ignore_index=IGNORE_LABEL, average="macro").to(device)
                   for name in enabled_probes}
    loss_sums = {name: 0.0 for name in enabled_probes}
    n = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        B = images.shape[0]
        H_mask = masks.shape[1]

        features = extractor.extract(images, enabled_probes)

        for name in enabled_probes:
            if name not in features:
                continue
            feat = features[name]
            H_feat = feat.shape[1]
            scale = H_mask // H_feat

            state = probes.probes[name]
            logits = state.probe(feat)

            # Loss: implicit upsample CE
            loss_sums[name] += implicit_upsample_ce(logits, masks, scale).item() * B

            # mIoU: upsample predictions to mask resolution
            preds_up = pred_nearest(logits, H_mask)
            metrics_iou[name].update(preds_up, masks)

        n += B

    results = {}
    for name in enabled_probes:
        results[f"{name}/val_loss"] = loss_sums[name] / n
        results[f"{name}/val_miou"] = metrics_iou[name].compute().item()

    return results


# === Main ===
def main(cfg: Config) -> None:
    # Validate at least one probe enabled
    enabled = set()
    if cfg.probe_hidden:
        enabled.add("hidden")
    if cfg.probe_predicted_norm:
        enabled.add("predicted_norm")
    if cfg.probe_predicted_denorm:
        enabled.add("predicted_denorm")
    if cfg.probe_teacher_full:
        enabled.add("teacher_full")
    if cfg.probe_teacher_glimpse:
        enabled.add("teacher_glimpse")

    if not enabled:
        raise ValueError("At least one probe must be enabled")

    log.info(f"Enabled probes: {enabled}")
    need_teacher = bool(enabled & {"teacher_full", "teacher_glimpse"})

    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    log.info(f"Device: {device}")

    # === Load AVP model ===
    from avp_vit.checkpoint import load as load_ckpt
    from avp_vit.train.norm import PositionAwareNorm
    from avp_vit import ActiveCanViT, ActiveCanViTConfig
    from canvit.hub import create_backbone
    from canvit.policy import PolicyConfig, PolicyHead
    import dacite

    log.info(f"Loading AVP checkpoint: {cfg.avp_ckpt}")
    ckpt = load_ckpt(cfg.avp_ckpt, device)

    # Create model from checkpoint (inlined to avoid double load)
    backbone = create_backbone(ckpt["backbone"], pretrained=False)
    model_cfg_dict = ckpt["model_config"]
    if "teacher_dim" not in model_cfg_dict:
        model_cfg_dict = {**model_cfg_dict, "teacher_dim": ckpt["teacher_dim"]}
    model_cfg = dacite.from_dict(ActiveCanViTConfig, model_cfg_dict)

    policy: PolicyHead | None = None
    if (pc := ckpt.get("policy_config")) is not None:
        policy_cfg = dacite.from_dict(PolicyConfig, pc)
        policy = PolicyHead(embed_dim=backbone.embed_dim, cfg=policy_cfg)

    model = ActiveCanViT(backbone=backbone, cfg=model_cfg, policy=policy)
    result = model.load_state_dict(ckpt["state_dict"], strict=False)
    if result.missing_keys:
        log.warning(f"Missing keys: {result.missing_keys}")
    if result.unexpected_keys:
        log.warning(f"Unexpected keys: {result.unexpected_keys}")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # === Extract and validate settings from checkpoint ===
    patch_size = model.backbone.patch_size_px
    hidden_dim = model.canvas_dim
    teacher_dim = ckpt["teacher_dim"]
    backbone_name = ckpt["backbone"]
    train_step = ckpt.get("step", "unknown")

    log.info("Checkpoint info:")
    log.info(f"  backbone: {backbone_name}")
    log.info(f"  train_step: {train_step}")
    log.info(f"  patch_size: {patch_size}")
    log.info(f"  hidden_dim (canvas_dim): {hidden_dim}")
    log.info(f"  teacher_dim: {teacher_dim}")

    # Load scene normalizer from checkpoint
    scene_norm: PositionAwareNorm | None = None
    if ckpt.get("scene_norm_state") is not None:
        norm_state = ckpt["scene_norm_state"]
        n_tokens, embed_dim = norm_state["mean"].shape
        grid_from_norm = int(n_tokens ** 0.5)
        scene_norm = PositionAwareNorm(n_tokens, embed_dim, grid_from_norm)
        scene_norm.load_state_dict(norm_state)
        scene_norm = scene_norm.eval().to(device)
        log.info(f"Loaded scene_norm: grid={grid_from_norm}, embed_dim={embed_dim}")
    else:
        log.warning("No scene_norm_state in checkpoint - denormalization will be identity!")

    # Compute and validate grid sizes
    canvas_grid = cfg.image_size // patch_size
    log.info(f"Computed canvas_grid: {cfg.image_size} // {patch_size} = {canvas_grid}")

    if scene_norm is not None and canvas_grid != scene_norm.grid_size:
        log.warning(f"⚠️  GRID SIZE MISMATCH: canvas_grid={canvas_grid} but scene_norm.grid_size={scene_norm.grid_size}")
        log.warning(f"    This may cause issues! Consider using image_size={scene_norm.grid_size * patch_size}")

    # Get glimpse settings from model config
    model_config = ckpt["model_config"]
    glimpse_grid = model_config.get("glimpse_grid_size", 8)
    if "glimpse_grid_size" not in model_config:
        log.warning(f"glimpse_grid_size not in checkpoint, defaulting to {glimpse_grid}")
    glimpse_px = glimpse_grid * patch_size
    log.info(f"Glimpse: grid={glimpse_grid}, px={glimpse_px}")

    # === Load teacher if needed ===
    from canvit.backbone.dinov3 import DINOv3Backbone
    teacher: DINOv3Backbone | None = None
    if need_teacher:
        log.info(f"Loading teacher: {backbone_name}")
        teacher = create_backbone(backbone_name, pretrained=True)
        assert isinstance(teacher, DINOv3Backbone)
        teacher = teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        log.info(f"Teacher loaded: embed_dim={teacher.embed_dim}, patch_size={teacher.patch_size_px}")

    # === Create feature extractor ===
    extractor = FeatureExtractor(
        model=model,
        scene_norm=scene_norm,
        canvas_grid=canvas_grid,
        glimpse_grid=glimpse_grid,
        glimpse_px=glimpse_px,
        patch_size=patch_size,
        device=device,
        teacher=teacher,
    )

    # === Create probes ===
    probe_dims = {}
    if "hidden" in enabled:
        probe_dims["hidden"] = hidden_dim
    if "predicted_norm" in enabled:
        probe_dims["predicted_norm"] = teacher_dim
    if "predicted_denorm" in enabled:
        probe_dims["predicted_denorm"] = teacher_dim
    if "teacher_full" in enabled:
        assert teacher is not None
        probe_dims["teacher_full"] = teacher.embed_dim
    if "teacher_glimpse" in enabled:
        assert teacher is not None
        probe_dims["teacher_glimpse"] = teacher.embed_dim

    peak_lr = cfg.ref_lr * cfg.batch_size
    warmup_steps = int(cfg.warmup_ratio * cfg.max_steps)

    probes = ProbeManager(
        probe_configs=probe_dims,
        num_classes=NUM_CLASSES,
        peak_lr=peak_lr,
        weight_decay=cfg.weight_decay,
        warmup_steps=warmup_steps,
        max_steps=cfg.max_steps,
        device=device,
    )

    # === Data ===
    train_ds = ADE20kDataset(cfg.ade20k_root.expanduser(), "training", cfg.image_size)
    val_ds = ADE20kDataset(cfg.ade20k_root.expanduser(), "validation", cfg.image_size)
    if cfg.max_train:
        train_ds = torch.utils.data.Subset(train_ds, range(min(cfg.max_train, len(train_ds))))
        log.info(f"Limited training to {len(train_ds)} images")
    if cfg.max_val:
        val_ds = torch.utils.data.Subset(val_ds, range(min(cfg.max_val, len(val_ds))))
        log.info(f"Limited validation to {len(val_ds)} images")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.eval_batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)

    # === Comet ===
    exp = comet_ml.Experiment(project_name=cfg.comet_project, workspace=COMET_WORKSPACE)
    exp.log_parameters(asdict(cfg))
    exp.log_parameter("peak_lr", peak_lr)
    exp.log_parameter("warmup_steps", warmup_steps)
    exp.log_parameter("hidden_dim", hidden_dim)
    exp.log_parameter("teacher_dim", teacher_dim)
    exp.log_parameter("canvas_grid", canvas_grid)
    exp.log_parameter("glimpse_grid", glimpse_grid)
    exp.log_parameter("patch_size", patch_size)
    exp.log_parameter("backbone", backbone_name)
    exp.log_parameter("avp_train_step", train_step)
    exp.log_parameter("train_size", len(train_ds))
    exp.log_parameter("val_size", len(val_ds))
    exp.log_parameter("enabled_probes", list(enabled))

    # === Training ===
    step = 0
    log.info("Starting training...")
    pbar = tqdm(total=cfg.max_steps, desc="Training")

    while step < cfg.max_steps:
        for images, masks in train_loader:
            if step >= cfg.max_steps:
                break

            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # Extract features (needed for viz before training)
            features = extractor.extract(images, enabled)

            # Validation BEFORE training (step 0 = untrained baseline)
            if step % cfg.val_every == 0:
                val_metrics = validate(extractor, probes, val_loader, device, enabled, cfg.image_size)
                for k, v in val_metrics.items():
                    exp.log_metric(k, v, step=step)
                    if step == 0:
                        log.info(f"  {k}: {v:.4f}")

                postfix = {}
                for name in enabled:
                    miou = val_metrics[f"{name}/val_miou"]
                    state = probes.probes[name]
                    if miou > state.best_miou:
                        state.best_miou = miou
                        ckpt_path = f"probe_{name}_best_{exp.id}.pt"
                        torch.save(state.probe.state_dict(), ckpt_path)
                        log.info(f"Step {step}: new best {name} mIoU: {miou:.4f} -> {ckpt_path}")
                    postfix[f"{name[:3]}"] = f"{miou:.3f}"
                pbar.set_postfix(postfix)

            # Visualization BEFORE training
            if step % cfg.viz_every == 0:
                for state in probes.probes.values():
                    state.probe.eval()
                with torch.no_grad():
                    logits_dict = {}
                    for name in enabled:
                        logits_dict[name] = probes.probes[name].probe(features[name])
                log_viz(exp, step, images, masks, logits_dict, cfg.n_viz_samples, cfg.image_size)

            # Train step - pass full-res masks, resize per-probe inside
            train_metrics = probes.train_step(features, masks, cfg.grad_clip, cfg.ema_alpha)

            step += 1
            pbar.update(1)

            # Log training metrics
            if step % cfg.log_every == 0:
                probes.update_ema(train_metrics, cfg.ema_alpha)
                log_dict = {"lr": probes.get_lr()}
                for name, (loss, grad_norm) in train_metrics.items():
                    log_dict[f"{name}/train_loss"] = probes.probes[name].loss_ema
                    log_dict[f"{name}/grad_norm"] = grad_norm.item()
                exp.log_metrics(log_dict, step=step)

    pbar.close()

    # Final summary
    log.info("=" * 60)
    log.info("Training complete. Best mIoU per probe:")
    for name in enabled:
        log.info(f"  {name}: {probes.probes[name].best_miou:.4f}")
        exp.log_metric(f"best_{name}_miou", probes.probes[name].best_miou)
    log.info("=" * 60)


if __name__ == "__main__":
    main(tyro.cli(Config))
