"""ADE20K probe training for CanViT.

Trains segmentation probes on frozen CanViT features:
- ONE probe per feature type, shared weights across timesteps (anytime decoding)
- Training: loss averaged across timesteps, single backward pass
- Eval: mIoU computed per timestep, logged as curves
"""

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import comet_ml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from canvit import CanViTForPretrainingHFHub, sample_at_viewpoint
from canvit.backbone.dinov3 import DINOv3Backbone
from canvit_utils.policies import random_viewpoints
from canvit_utils.teacher import load_teacher
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassJaccardIndex
from tqdm import tqdm

from canvit_eval.ade20k.dataset import IGNORE_LABEL, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES, ADE20kDataset
from canvit_eval.ade20k.probe import ProbeHead

log = logging.getLogger(__name__)

FeatureType = Literal["hidden", "predicted_norm", "teacher_glimpse"]
STATIC_FEATURES: set[FeatureType] = {"teacher_glimpse"}


@dataclass
class Config:
    """ADE20K probe training configuration."""

    model_repo: str
    ade20k_root: Path
    features: list[FeatureType] = field(default_factory=lambda: ["hidden", "predicted_norm", "teacher_glimpse"])
    n_timesteps: int = 10
    image_size: int = 512
    glimpse_px: int = 128  # Glimpse size in pixels
    min_vp_scale: float = 0.25  # Minimum viewpoint scale for random sampling
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
    viz_every: int = 500
    viz_samples: int = 4
    comet_project: str = "canvit-ade20k-probe"
    comet_workspace: str = "m2b3-ava"
    device: str = "cuda"
    amp: bool = True
    probe_ckpt_dir: Path | None = None
    resume_from: Path | None = None


@dataclass
class ProbeState:
    """Training state for one probe."""

    name: str
    head: ProbeHead
    optimizer: AdamW
    scheduler: SequentialLR
    best_mean_miou: float = 0.0
    _loss_sum: Tensor | None = None
    _grad_norm_sum: Tensor | None = None
    _count: int = 0

    def accumulate(self, loss: Tensor, grad_norm: Tensor) -> None:
        if self._loss_sum is None:
            self._loss_sum = loss.detach().clone()
            self._grad_norm_sum = grad_norm.detach().clone()
        else:
            self._loss_sum += loss.detach()
            assert self._grad_norm_sum is not None
            self._grad_norm_sum += grad_norm.detach()
        self._count += 1

    def get_and_reset(self) -> tuple[float, float]:
        assert self._loss_sum is not None and self._grad_norm_sum is not None
        avg_loss = (self._loss_sum / self._count).item()
        avg_grad = (self._grad_norm_sum / self._count).item()
        self._loss_sum = self._grad_norm_sum = None
        self._count = 0
        return avg_loss, avg_grad


def _make_probe(name: str, dim: int, cfg: Config, device: torch.device) -> ProbeState:
    warmup_steps = int(cfg.warmup_ratio * cfg.max_steps)
    head = ProbeHead(dim).to(device)
    opt = AdamW(head.parameters(), lr=cfg.peak_lr, weight_decay=cfg.weight_decay)
    warmup = LinearLR(opt, cfg.min_lr / cfg.peak_lr, 1.0, max(1, warmup_steps))
    cosine = CosineAnnealingLR(opt, cfg.max_steps - warmup_steps, eta_min=cfg.min_lr)
    return ProbeState(name, head, opt, SequentialLR(opt, [warmup, cosine], [warmup_steps]))


def _focal_loss(logits: Tensor, masks: Tensor, gamma: float) -> Tensor:
    B, C, Hl, Wl = logits.shape
    if masks.shape[1:] != (Hl, Wl):
        masks = F.interpolate(masks.unsqueeze(1).float(), (Hl, Wl), mode="nearest").squeeze(1).long()
    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp()
    valid = masks != IGNORE_LABEL
    targets = masks.clamp(0, C - 1)
    log_p = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    p = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    return -((1 - p) ** gamma * log_p * valid).sum() / valid.sum().clamp(min=1)


def _upsample_preds(preds: Tensor, H: int, W: int) -> Tensor:
    if preds.shape[1:] == (H, W):
        return preds
    return F.interpolate(preds.unsqueeze(1).float(), (H, W), mode="nearest").squeeze(1).long()


def _imagenet_denormalize(img: Tensor) -> np.ndarray:
    mean = IMAGENET_MEAN.to(img.device).view(3, 1, 1)
    std = IMAGENET_STD.to(img.device).view(3, 1, 1)
    img = (img * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


_PALETTE = np.random.RandomState(42).randint(0, 255, (NUM_CLASSES + 1, 3), dtype=np.uint8)
_PALETTE[NUM_CLASSES] = 0


def _colorize(mask: np.ndarray) -> np.ndarray:
    return _PALETTE[np.where(mask == IGNORE_LABEL, NUM_CLASSES, mask)]


def _extract_features(
    model: CanViTForPretrainingHFHub,
    teacher: DINOv3Backbone,
    images: Tensor,
    n_timesteps: int,
    canvas_grid: int,
    glimpse_px: int,
    device: torch.device,
    min_vp_scale: float,
) -> dict[FeatureType, list[Tensor]]:
    B = images.shape[0]
    feats: dict[FeatureType, list[Tensor]] = {"hidden": [], "predicted_norm": [], "teacher_glimpse": []}
    state = model.init_state(batch_size=B, canvas_grid_size=canvas_grid)

    glimpse_grid = glimpse_px // teacher.patch_size_px
    sz = glimpse_grid * teacher.patch_size_px
    small = F.interpolate(images, size=(sz, sz), mode="bilinear", align_corners=False)
    teacher_feat = teacher.forward_norm_features(small).patches.view(B, glimpse_grid, glimpse_grid, -1)

    # Generate all viewpoints upfront using shared policy
    viewpoints = random_viewpoints(
        B, device, n_timesteps, min_scale=min_vp_scale, max_scale=1.0, start_with_full_scene=True
    )

    for vp in viewpoints:
        glimpse = sample_at_viewpoint(spatial=images, viewpoint=vp, glimpse_size_px=glimpse_px)
        out = model(glimpse=glimpse, state=state, viewpoint=vp)
        state = out.state

        hidden = model.get_spatial(state.canvas).view(B, canvas_grid, canvas_grid, -1)
        predicted = model.predict_teacher_scene(state.canvas).view(B, canvas_grid, canvas_grid, -1)

        feats["hidden"].append(hidden)
        feats["predicted_norm"].append(predicted)
        feats["teacher_glimpse"].append(teacher_feat)

    return feats


def train(cfg: Config) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_float32_matmul_precision("high")
    device = torch.device(cfg.device)

    log.info("=" * 60)
    log.info("ADE20K Probe Training")
    log.info("=" * 60)

    log.info(f"Loading model from {cfg.model_repo}...")
    model = CanViTForPretrainingHFHub.from_pretrained(cfg.model_repo).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    log.info(f"  params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Load PRETRAINED teacher for baseline (NOT model.backbone which is trained from random init)
    teacher = load_teacher(model.backbone_name, device)

    patch_size = model.backbone.patch_size_px
    canvas_grid = cfg.image_size // patch_size
    glimpse_px = cfg.glimpse_px
    min_vp_scale = cfg.min_vp_scale

    dims: dict[FeatureType, int] = {
        "hidden": model.canvas_dim,
        "predicted_norm": teacher.embed_dim,
        "teacher_glimpse": teacher.embed_dim,
    }

    probes = {feat: _make_probe(feat, dims[feat], cfg, device) for feat in cfg.features}

    val_iou = {
        feat: [
            MulticlassJaccardIndex(NUM_CLASSES, ignore_index=IGNORE_LABEL, average="macro").to(device)
            for _ in range(cfg.n_timesteps)
        ]
        for feat in cfg.features
    }

    train_ds = ADE20kDataset(cfg.ade20k_root, "training", cfg.image_size, augment=True)
    val_ds = ADE20kDataset(cfg.ade20k_root, "validation", cfg.image_size, augment=False)
    train_loader = DataLoader(
        train_ds, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(val_ds, cfg.eval_batch_size, num_workers=cfg.num_workers, pin_memory=True)
    log.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    exp = comet_ml.Experiment(project_name=cfg.comet_project, workspace=cfg.comet_workspace)
    exp.log_parameters(asdict(cfg))

    if cfg.amp:
        amp_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    else:
        amp_ctx = torch.autocast(device_type=device.type, enabled=False)

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
            for feat in cfg.features:
                for m in val_iou[feat]:
                    m.reset()

            with torch.no_grad():
                for vi, vm in val_loader:
                    vi, vm = vi.to(device), vm.to(device)
                    with amp_ctx:
                        feats = _extract_features(
                            model, teacher, vi, cfg.n_timesteps, canvas_grid, glimpse_px, device, min_vp_scale
                        )
                    for feat_type in cfg.features:
                        t_range = [0] if feat_type in STATIC_FEATURES else range(cfg.n_timesteps)
                        for t in t_range:
                            logits = probes[feat_type].head(feats[feat_type][t].float())
                            preds_up = _upsample_preds(logits.argmax(1), vm.shape[1], vm.shape[2])
                            val_iou[feat_type][t].update(preds_up, vm)

            for feat_type in cfg.features:
                if feat_type in STATIC_FEATURES:
                    miou = val_iou[feat_type][0].compute().item()
                    exp.log_metric(f"{feat_type}/val_miou", miou, step=step)
                    mean_miou = miou
                else:
                    mious = [val_iou[feat_type][t].compute().item() for t in range(cfg.n_timesteps)]
                    mean_miou = sum(mious) / len(mious)
                    for t, miou in enumerate(mious):
                        exp.log_metric(f"{feat_type}/val_miou_t{t}", miou, step=step)
                    exp.log_metric(f"{feat_type}/val_miou_mean", mean_miou, step=step)
                if mean_miou > probes[feat_type].best_mean_miou:
                    probes[feat_type].best_mean_miou = mean_miou

        # Training step
        for p in probes.values():
            p.head.train()

        with amp_ctx:
            feats = _extract_features(
                model, teacher, images, cfg.n_timesteps, canvas_grid, glimpse_px, device, min_vp_scale
            )

        for feat_type in cfg.features:
            probe = probes[feat_type]
            probe.optimizer.zero_grad()
            t_range = [0] if feat_type in STATIC_FEATURES else range(cfg.n_timesteps)
            logits_list = [probe.head(feats[feat_type][t].float()) for t in t_range]
            losses = [_focal_loss(logits, masks, cfg.focal_gamma) for logits in logits_list]
            loss = torch.stack(losses).mean()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(probe.head.parameters(), cfg.grad_clip)
            probe.optimizer.step()
            probe.scheduler.step()
            probe.accumulate(loss, grad_norm)

        step += 1
        pbar.update(1)

        if step % cfg.log_every == 0:
            log_dict = {"lr": list(probes.values())[0].scheduler.get_last_lr()[0]}
            for name, p in probes.items():
                avg_loss, avg_grad = p.get_and_reset()
                log_dict[f"{name}/loss"] = avg_loss
                log_dict[f"{name}/grad_norm"] = avg_grad
            exp.log_metrics(log_dict, step=step)

    pbar.close()
    log.info("=" * 60)
    log.info("Training complete. Best mean mIoU:")
    for name, p in probes.items():
        log.info(f"  {name}: {p.best_mean_miou:.4f}")
        exp.log_metric(f"best/{name}", p.best_mean_miou)


def main() -> None:
    cfg = tyro.cli(Config)
    train(cfg)
