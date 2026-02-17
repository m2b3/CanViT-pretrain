"""ADE20K canvas probe evaluation with configurable policies."""

import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from canvit import CanViTForPretrainingHFHub
from canvit_utils.teacher import load_teacher
from torch.utils.data import DataLoader
from tqdm import tqdm

from canvit_eval.ade20k.dataset import IGNORE_LABEL, NUM_CLASSES, ADE20kDataset, ResizeMode, make_val_transform
from canvit_eval.ade20k.probe import ProbeHead, eval_probe_on_batch
from canvit_eval.ade20k.train_probe.config import (
    CANVAS_FEATURES,
    CanvasFeatureType,
    _default_ade20k_root,
    get_feature_dims,
)
from canvit_eval.ade20k.train_probe.features import extract_canvas_features
from canvit_eval.metrics import IoUAccumulator
from canvit_eval.utils import PolicyName, collect_metadata, make_viewpoints

log = logging.getLogger(__name__)


def _default_output() -> Path:
    job_id = os.environ.get("SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID", "local"))
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    ts = time.strftime("%Y%m%d_%H%M%S")
    return Path(f"outputs/ade20k_canvas_eval_{ts}_{job_id}_{task_id}.pt")


@dataclass
class EvalConfig:
    """ADE20K canvas probe evaluation configuration."""

    probe_ckpt: Path
    ade20k_root: Path = field(default_factory=_default_ade20k_root)
    output: Path = field(default_factory=_default_output)

    model_repo: str = "canvit/canvit-vitb16-pretrain-512px-in21k"
    teacher_repo: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    policy: PolicyName = "coarse_to_fine"
    resize_mode: ResizeMode = "squish"
    n_timesteps: int = 16
    image_size: int = 512
    glimpse_px: int = 128

    min_scale: float = 0.05
    max_scale: float = 1.0
    start_full: bool = True

    batch_size: int = 32
    num_workers: int = 8
    device: str = "cuda"
    amp: bool = True


def load_probes(
    ckpt_path: Path,
    device: torch.device,
    canvas_dim: int,
    teacher_dim: int,
) -> dict[CanvasFeatureType, ProbeHead]:
    log.info(f"Loading probes from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    dims = get_feature_dims(canvas_dim, teacher_dim)
    dropout = ckpt.get("config", {}).get("dropout", 0.0)

    probes: dict[CanvasFeatureType, ProbeHead] = {}
    for name, state_dict in ckpt["probe_state_dicts"].items():
        probe = ProbeHead(dims[name], dropout=dropout, use_ln=CANVAS_FEATURES[name].needs_ln).to(device)
        probe.load_state_dict(state_dict)
        probe.eval()
        probes[name] = probe
        best_per_t = ckpt.get("best_mious_per_t", {}).get(name)
        if best_per_t:
            best_str = f"best per-t={best_per_t}"
        else:
            old = ckpt.get("best_mean_mious", {}).get(name, "N/A")
            best_str = f"best(legacy)={old}"
        log.info(f"  {name}: loaded ({best_str})")

    return probes


@torch.inference_mode()
def evaluate(cfg: EvalConfig) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_float32_matmul_precision("high")

    device = torch.device(cfg.device)
    T = cfg.n_timesteps

    log.info("=" * 70)
    log.info("ADE20K Canvas Probe Evaluation")
    log.info("=" * 70)
    log.info(f"  policy={cfg.policy}, resize_mode={cfg.resize_mode}, n_timesteps={T}")
    for k, v in asdict(cfg).items():
        log.info(f"  {k}: {v}")

    # Load model
    log.info(f"Loading model: {cfg.model_repo}")
    model = CanViTForPretrainingHFHub.from_pretrained(cfg.model_repo).to(device).eval()
    teacher = load_teacher(cfg.teacher_repo, device)
    log.info(f"  canvas_dim={model.canvas_dim}, teacher_dim={teacher.embed_dim}")

    patch_size = model.backbone.patch_size_px
    canvas_grid = cfg.image_size // patch_size

    # Load probes
    probes = load_probes(cfg.probe_ckpt, device, model.canvas_dim, teacher.embed_dim)
    feature_types = list(probes.keys())
    log.info(f"Features: {feature_types}")

    # Dataset
    dataset = ADE20kDataset(
        root=cfg.ade20k_root, split="validation",
        transform=make_val_transform(cfg.image_size, cfg.resize_mode),
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    # IoU metrics
    iou_metrics: dict[CanvasFeatureType, list[IoUAccumulator]] = {
        feat: [IoUAccumulator(NUM_CLASSES, IGNORE_LABEL, device) for _ in range(T)]
        for feat in feature_types
    }

    amp_dtype = torch.bfloat16 if cfg.amp else torch.float32
    amp_ctx = torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=cfg.amp)

    log.info(f"Evaluating with policy={cfg.policy}, {T} timesteps...")

    for images, masks in tqdm(loader, desc="Evaluating", unit="batch"):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        B = images.shape[0]

        viewpoints = make_viewpoints(
            cfg.policy, B, device, T,
            min_scale=cfg.min_scale, max_scale=cfg.max_scale,
            start_with_full_scene=cfg.start_full,
        )

        with amp_ctx:
            feats = extract_canvas_features(
                model=model, images=images,
                canvas_grid=canvas_grid, glimpse_px=cfg.glimpse_px,
                viewpoints=viewpoints,
            )

        for feat_type in feature_types:
            for t in range(T):
                eval_probe_on_batch(probes[feat_type], feats.get(feat_type, t), masks, iou_metrics[feat_type][t])

    # Results
    log.info("")
    log.info("=" * 70)
    log.info("RESULTS (global mIoU)")
    log.info("=" * 70)

    results: dict = {"mious": {}, "metadata": collect_metadata(cfg)}

    for feat_type in feature_types:
        mious = [iou_metrics[feat_type][t].compute() for t in range(T)]
        mean_miou = sum(mious) / len(mious)
        results["mious"][feat_type] = {
            **{f"t{t}": m for t, m in enumerate(mious)},
            "mean": mean_miou,
        }
        log.info(f"  {feat_type}:")
        for t, m in enumerate(mious):
            log.info(f"    t{t}: {100*m:.2f}%")
        log.info(f"    mean: {100*mean_miou:.2f}%")

    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, cfg.output)
    log.info(f"Saved to {cfg.output} ({cfg.output.stat().st_size / 1024:.1f} KB)")

    return cfg.output


def main() -> None:
    import tyro
    cfg = tyro.cli(EvalConfig)
    evaluate(cfg)
