"""Evaluate reconstruction quality on held-out images.

Computes per-timestep cosine similarity between CanViT canvas reconstruction
and DINOv3 teacher features. No probes, no labels — just forward passes.

Usage:
    uv run python -m canvit_eval.reconstruction \
        --model-repo canvit/canvitb16-abl-baseline-2026-03-02 \
        --image-dir /datasets/ADE20k/ADEChallengeData2016/images/validation \
        --output results/abl_baseline.pt
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from canvit import CanViTOutput, RecurrentState, Viewpoint, sample_at_viewpoint
from canvit_utils.teacher import DINOv3Teacher, load_teacher
from canvit_utils.transforms import preprocess
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from canvit import CanViTForPretrainingHFHub

from canvit_eval.utils import collect_metadata, make_viewpoints

log = logging.getLogger(__name__)

TEACHER_REPO = "facebook/dinov3-vitb16-pretrain-lvd1689m"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


class FlatImageDataset(Dataset):
    """Load images from a flat directory (no subdirectories required)."""

    def __init__(self, root: Path, transform: object = None) -> None:
        self.paths = sorted(
            p for p in root.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        assert len(self.paths) > 0, f"No images found in {root}"
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img


@dataclass
class ReconstructionEvalConfig:
    model_repo: str
    image_dir: Path
    output: Path

    policy: Literal["coarse_to_fine", "random", "full_then_random"] = "random"
    scene_size: int = 512
    canvas_grid: int = 32
    glimpse_px: int = 128
    n_timesteps: int = 10
    batch_size: int = 16
    num_workers: int = 4
    device: str = "cuda"
    teacher_cache: Path | None = None


@dataclass
class TimestepMetrics:
    """Accumulated cosine similarities for one timestep (raw + normalized)."""
    scene_raw_sum: float = 0.0
    cls_raw_sum: float = 0.0
    scene_norm_sum: float = 0.0
    cls_norm_sum: float = 0.0
    n_images: int = 0

    def update(self, *, scene_raw: float, cls_raw: float,
               scene_norm: float, cls_norm: float, batch_size: int) -> None:
        self.scene_raw_sum += scene_raw * batch_size
        self.cls_raw_sum += cls_raw * batch_size
        self.scene_norm_sum += scene_norm * batch_size
        self.cls_norm_sum += cls_norm * batch_size
        self.n_images += batch_size


def _cache_teacher_features(
    teacher: DINOv3Teacher,
    loader: DataLoader,
    scene_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Extract teacher features for entire dataset. Returns (patches, cls)."""
    all_patches: list[Tensor] = []
    all_cls: list[Tensor] = []
    log.info("Caching teacher features for %d batches...", len(loader))
    with torch.inference_mode():
        for images in tqdm(loader, desc="Teacher features"):
            images = images.to(device)
            feats = teacher.forward_norm_features(images)
            all_patches.append(feats.patches.cpu().half())
            all_cls.append(feats.cls.cpu().half())
    patches = torch.cat(all_patches)
    cls = torch.cat(all_cls)
    log.info("Cached: patches %s, cls %s (%.2f GB)",
             patches.shape, cls.shape,
             (patches.nelement() + cls.nelement()) * 2 / 1e9)
    return patches, cls


def evaluate(cfg: ReconstructionEvalConfig) -> dict:
    device = torch.device(cfg.device)

    # Load model from HuggingFace Hub
    log.info("Loading model from %s", cfg.model_repo)
    model = CanViTForPretrainingHFHub.from_pretrained(cfg.model_repo).to(device).eval()

    canvas_grid = cfg.canvas_grid
    has_cls = model.scene_cls_head is not None
    cls_std, scene_std = model.standardizers(canvas_grid)
    assert scene_std.initialized, (
        f"Standardizer not initialized for grid {canvas_grid}. "
        "Checkpoint may not have standardizer stats for this grid size."
    )

    # Dataset — just images, no labels needed
    transform = preprocess(cfg.scene_size)
    dataset = FlatImageDataset(cfg.image_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        shuffle=False,
    )
    log.info("Dataset: %d images from %s", len(dataset), cfg.image_dir)

    # Teacher features — cache or compute
    teacher = load_teacher(TEACHER_REPO, device)
    if cfg.teacher_cache is not None and cfg.teacher_cache.exists():
        log.info("Loading cached teacher features from %s", cfg.teacher_cache)
        cached = torch.load(cfg.teacher_cache, map_location="cpu", weights_only=True)
        teacher_patches = cached["patches"]
        teacher_cls = cached["cls"]
    else:
        teacher_patches, teacher_cls = _cache_teacher_features(
            teacher, loader, cfg.scene_size, device,
        )
        if cfg.teacher_cache is not None:
            cfg.teacher_cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"patches": teacher_patches, "cls": teacher_cls}, cfg.teacher_cache)
            log.info("Saved teacher cache to %s", cfg.teacher_cache)

    del teacher
    torch.cuda.empty_cache()

    # Evaluate reconstruction quality
    T = cfg.n_timesteps
    metrics = [TimestepMetrics() for _ in range(T)]

    start_time = time.perf_counter()
    img_idx = 0

    with torch.inference_mode():
        for images in tqdm(loader, desc="Reconstruction eval"):
            B = images.shape[0]
            images = images.to(device)

            # Slice pre-cached teacher features for this batch
            raw_patches = teacher_patches[img_idx:img_idx + B].to(device).float()
            raw_cls = teacher_cls[img_idx:img_idx + B].to(device).float()
            img_idx += B

            # Standardized targets (for normalized cosine sim)
            norm_patches = scene_std(raw_patches)
            norm_cls = cls_std(raw_cls.unsqueeze(1)).squeeze(1)

            viewpoints = make_viewpoints(
                cfg.policy, B, device, T,
            )

            # Step-by-step recurrent forward pass
            state = model.init_state(batch_size=B, canvas_grid_size=canvas_grid)
            for t, vp in enumerate(viewpoints):
                glimpse = sample_at_viewpoint(
                    spatial=images, viewpoint=vp, glimpse_size_px=cfg.glimpse_px,
                )
                out = model.forward(glimpse=glimpse, state=state, viewpoint=vp)
                state = out.state

                pred_scene = model.predict_teacher_scene(state.canvas)
                scene_raw = F.cosine_similarity(pred_scene, raw_patches, dim=-1).mean().item()
                scene_norm = F.cosine_similarity(pred_scene, norm_patches, dim=-1).mean().item()

                cls_raw = cls_norm_val = 0.0
                if has_cls:
                    pred_cls = model.predict_scene_teacher_cls(state.recurrent_cls)
                    cls_raw = F.cosine_similarity(pred_cls, raw_cls, dim=-1).mean().item()
                    cls_norm_val = F.cosine_similarity(pred_cls, norm_cls, dim=-1).mean().item()

                metrics[t].update(
                    scene_raw=scene_raw, cls_raw=cls_raw,
                    scene_norm=scene_norm, cls_norm=cls_norm_val,
                    batch_size=B,
                )

    elapsed = time.perf_counter() - start_time
    log.info("Evaluation done: %d images in %.1fs (%.1f img/s)",
             img_idx, elapsed, img_idx / elapsed)

    # Build results — store full precision, let consumers round
    per_timestep = [
        {
            "t": t,
            "scene_cos_raw": m.scene_raw_sum / m.n_images,
            "cls_cos_raw": m.cls_raw_sum / m.n_images,
            "scene_cos_norm": m.scene_norm_sum / m.n_images,
            "cls_cos_norm": m.cls_norm_sum / m.n_images,
        }
        for t, m in enumerate(metrics)
    ]

    result = {
        "per_timestep": per_timestep,
        "n_images": img_idx,
        "elapsed_s": round(elapsed, 1),
        "metadata": {
            "model_repo": cfg.model_repo,
            "backbone_name": model.backbone_name,
            "scene_size": cfg.scene_size,
            "canvas_grid": cfg.canvas_grid,
            "glimpse_px": cfg.glimpse_px,
            "n_timesteps": T,
            "dataset": str(cfg.image_dir),
            **collect_metadata(cfg),
        },
    }

    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, cfg.output)
    log.info("Saved to %s", cfg.output)

    # Print summary
    for p in per_timestep:
        log.info("  t=%d  scene_raw=%.4f  cls_raw=%.4f  scene_norm=%.4f  cls_norm=%.4f",
                 p["t"], p["scene_cos_raw"], p["cls_cos_raw"],
                 p["scene_cos_norm"], p["cls_cos_norm"])

    return result
