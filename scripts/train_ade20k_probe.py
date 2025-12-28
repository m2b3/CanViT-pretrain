#!/usr/bin/env python3
"""Train ADE20K segmentation probes on AVP features.

Three probe types (enable via flags):
- probe_hidden: model.get_spatial(canvas) - internal representation
- probe_predicted_norm: model.predict_teacher_scene(canvas) - normalized DINOv3 space
- probe_predicted_denorm: denorm(predict_teacher_scene) - raw DINOv3 space

Usage:
    COMET_API_KEY=$(cat ~/comet_api_key.txt) uv run python scripts/train_ade20k_probe.py \
        --avp-ckpt path/to/checkpoint.pt \
        --ade20k-root /datasets/ADE20k/ADEChallengeData2016 \
        --probe-hidden --probe-predicted-norm --probe-predicted-denorm
"""

import logging
import time
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


@dataclass
class Config:
    avp_ckpt: Path
    ade20k_root: Path = Path("/datasets/ADE20k/ADEChallengeData2016")

    # Which probes to train (at least one must be True)
    probe_hidden: bool = True
    probe_predicted_norm: bool = True
    probe_predicted_denorm: bool = True

    # Image/grid settings - will be validated against checkpoint
    image_size: int = 512

    batch_size: int = 32
    num_workers: int = 4

    ref_lr: float = 1e-5
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.1
    max_steps: int = 5000
    grad_clip: float = 1.0

    log_every: int = 10
    val_every: int = 100
    viz_every: int = 200
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


def log_viz(
    exp: comet_ml.Experiment,
    step: int,
    images: Tensor,
    masks: Tensor,
    predictions: dict[str, Tensor],
    n_samples: int,
) -> None:
    """Log sample predictions to Comet."""
    import matplotlib.pyplot as plt

    n = min(n_samples, images.shape[0])
    n_preds = len(predictions)
    fig, axes = plt.subplots(n, 2 + n_preds, figsize=(3 * (2 + n_preds), 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    pred_names = list(predictions.keys())
    for i in range(n):
        img_np = imagenet_denorm(images[i])
        mask_np = masks[i].cpu().numpy()

        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title("Image")
        axes[i, 1].imshow(colorize_mask(mask_np))
        axes[i, 1].set_title("GT")

        for j, name in enumerate(pred_names):
            pred_np = predictions[name][i].cpu().numpy()
            axes[i, 2 + j].imshow(colorize_mask(pred_np))
            axes[i, 2 + j].set_title(name)

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    exp.log_figure(figure_name=f"predictions_step_{step}", figure=fig, step=step)
    plt.close(fig)


# === Feature extraction ===
class FeatureExtractor:
    """Extracts features from AVP model with proper normalization handling."""

    def __init__(
        self,
        model: nn.Module,
        scene_norm: nn.Module | None,
        canvas_grid: int,
        glimpse_px: int,
        device: torch.device,
    ):
        self.model = model
        self.scene_norm = scene_norm
        self.canvas_grid = canvas_grid
        self.glimpse_px = glimpse_px
        self.device = device

        # Import here to avoid circular imports
        from canvit.viewpoint import Viewpoint
        self.Viewpoint = Viewpoint

    def extract(self, images: Tensor) -> dict[str, Tensor]:
        """Extract all feature types at t=0 (full scene).

        Returns dict with keys: 'hidden', 'predicted_norm', 'predicted_denorm'
        All shapes: (B, H, W, D)
        """
        B = images.shape[0]

        canvas = self.model.init_canvas(batch_size=B, canvas_grid_size=self.canvas_grid)
        cls = self.model.init_cls(batch_size=B)

        # Full scene viewpoint: center=(0,0), scale=1.0
        vp = self.Viewpoint(
            centers=torch.zeros(B, 2, device=self.device),
            scales=torch.ones(B, device=self.device),
        )

        with torch.inference_mode():
            out = self.model.forward_step(
                image=images,
                canvas=canvas,
                cls=cls,
                viewpoint=vp,
                glimpse_size_px=self.glimpse_px,
            )

        # Extract features
        hidden_flat = self.model.get_spatial(out.canvas)  # (B, N, D_hidden)
        predicted_norm_flat = self.model.predict_teacher_scene(out.canvas)  # (B, N, D_teacher)

        # Denormalize if we have normalizer
        if self.scene_norm is not None:
            predicted_denorm_flat = self.scene_norm.denormalize(predicted_norm_flat)
        else:
            log.warning("No scene_norm available - predicted_denorm will equal predicted_norm")
            predicted_denorm_flat = predicted_norm_flat

        # Reshape to spatial
        N = hidden_flat.shape[1]
        H = W = int(N ** 0.5)
        assert H * W == N, f"N={N} is not a perfect square"

        return {
            "hidden": hidden_flat.view(B, H, W, -1),
            "predicted_norm": predicted_norm_flat.view(B, H, W, -1),
            "predicted_denorm": predicted_denorm_flat.view(B, H, W, -1),
        }


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
    ) -> dict[str, dict[str, float]]:
        """Train all probes on given features. Returns metrics per probe."""
        metrics = {}

        for name, state in self.probes.items():
            if name not in features:
                continue

            feat = features[name].detach()
            state.probe.train()
            state.optimizer.zero_grad()

            logits = state.probe(feat)
            loss = F.cross_entropy(logits, masks, ignore_index=IGNORE_LABEL)
            loss.backward()

            grad_norm = nn.utils.clip_grad_norm_(state.probe.parameters(), grad_clip)
            state.optimizer.step()
            state.scheduler.step()

            # EMA loss
            loss_val = loss.item()
            state.loss_ema = ema_alpha * loss_val + (1 - ema_alpha) * state.loss_ema if state.loss_ema > 0 else loss_val

            metrics[name] = {
                "train_loss": state.loss_ema,
                "grad_norm": grad_norm.item(),
            }

        return metrics

    def get_lr(self) -> float:
        """Get current LR (same for all probes)."""
        first = next(iter(self.probes.values()))
        return first.scheduler.get_last_lr()[0]


# === Validation ===
@torch.no_grad()
def validate(
    extractor: FeatureExtractor,
    probes: ProbeManager,
    loader: DataLoader,
    device: torch.device,
    enabled_probes: set[str],
) -> dict[str, float]:
    """Compute val metrics for all probes."""
    for state in probes.probes.values():
        state.probe.eval()

    # Metrics per probe
    metrics_iou = {name: MulticlassJaccardIndex(NUM_CLASSES, ignore_index=IGNORE_LABEL, average="macro").to(device)
                   for name in enabled_probes}
    loss_sums = {name: 0.0 for name in enabled_probes}
    n = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        B = images.shape[0]

        features = extractor.extract(images)

        # Downsample masks to feature grid
        H_feat = features["hidden"].shape[1]
        masks_down = F.interpolate(
            masks.unsqueeze(1).float(), size=(H_feat, H_feat), mode="nearest"
        ).squeeze(1).long()

        for name in enabled_probes:
            if name not in features:
                continue
            state = probes.probes[name]
            logits = state.probe(features[name])
            loss_sums[name] += F.cross_entropy(logits, masks_down, ignore_index=IGNORE_LABEL).item() * B
            metrics_iou[name].update(logits.argmax(dim=1), masks_down)

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

    if not enabled:
        raise ValueError("At least one probe must be enabled (--probe-hidden, --probe-predicted-norm, --probe-predicted-denorm)")

    log.info(f"Enabled probes: {enabled}")

    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    log.info(f"Device: {device}")

    # === Load AVP model ===
    from avp_vit.checkpoint import load_model, load as load_ckpt
    from avp_vit.train.norm import PositionAwareNorm
    from avp_vit.train.config import Config as TrainConfig

    log.info(f"Loading AVP checkpoint: {cfg.avp_ckpt}")
    model = load_model(cfg.avp_ckpt, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    ckpt = load_ckpt(cfg.avp_ckpt, device)

    # === Extract and validate settings from checkpoint ===
    patch_size = model.backbone.patch_size_px
    hidden_dim = model.canvas_dim
    teacher_dim = ckpt["teacher_dim"]
    backbone_name = ckpt["backbone"]
    train_step = ckpt.get("step", "unknown")

    log.info(f"Checkpoint info:")
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
    model_config = ckpt.get("model_config", {})
    glimpse_grid = model_config.get("glimpse_grid_size", 8)
    glimpse_px = glimpse_grid * patch_size
    log.info(f"Glimpse: grid={glimpse_grid}, px={glimpse_px}")

    # === Create feature extractor ===
    extractor = FeatureExtractor(model, scene_norm, canvas_grid, glimpse_px, device)

    # === Create probes ===
    probe_dims = {}
    if "hidden" in enabled:
        probe_dims["hidden"] = hidden_dim
    if "predicted_norm" in enabled:
        probe_dims["predicted_norm"] = teacher_dim
    if "predicted_denorm" in enabled:
        probe_dims["predicted_denorm"] = teacher_dim

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
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
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

    # Val at step 0
    log.info("Running validation at step 0...")
    val_metrics = validate(extractor, probes, val_loader, device, enabled)
    for k, v in val_metrics.items():
        exp.log_metric(k, v, step=0)
        log.info(f"  {k}: {v:.4f}")

    while step < cfg.max_steps:
        for images, masks in train_loader:
            if step >= cfg.max_steps:
                break

            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # Extract features
            features = extractor.extract(images)

            # Downsample masks
            H_feat = features["hidden"].shape[1]
            masks_down = F.interpolate(
                masks.unsqueeze(1).float(), size=(H_feat, H_feat), mode="nearest"
            ).squeeze(1).long()

            # Train step
            train_metrics = probes.train_step(features, masks_down, cfg.grad_clip, cfg.ema_alpha)

            step += 1
            pbar.update(1)

            # Logging
            if step % cfg.log_every == 0:
                log_dict = {"lr": probes.get_lr()}
                for name, m in train_metrics.items():
                    log_dict[f"{name}/train_loss"] = m["train_loss"]
                    log_dict[f"{name}/grad_norm"] = m["grad_norm"]
                exp.log_metrics(log_dict, step=step)

            # Validation
            if step % cfg.val_every == 0:
                val_metrics = validate(extractor, probes, val_loader, device, enabled)
                for k, v in val_metrics.items():
                    exp.log_metric(k, v, step=step)

                # Check for best and save
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

            # Visualization
            if step % cfg.viz_every == 0:
                for state in probes.probes.values():
                    state.probe.eval()

                with torch.no_grad():
                    predictions = {}
                    for name in enabled:
                        logits = probes.probes[name].probe(features[name])
                        pred = logits.argmax(dim=1)
                        # Upsample for viz
                        pred_up = F.interpolate(pred.unsqueeze(1).float(), size=cfg.image_size, mode="nearest").squeeze(1).long()
                        predictions[name] = pred_up

                log_viz(exp, step, images, masks, predictions, cfg.n_viz_samples)

    pbar.close()

    # Final summary
    log.info("=" * 60)
    log.info("Training complete. Best mIoU per probe:")
    for name in enabled:
        log.info(f"  {name}: {probes.probes[name].best_miou:.4f}")
        exp.log_metric(f"best_{name}_miou", probes.probes[name].best_miou)
    log.info("=" * 60)


if __name__ == "__main__":
    tyro.cli(main)
