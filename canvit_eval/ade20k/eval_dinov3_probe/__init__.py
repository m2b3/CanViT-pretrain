"""DINOv3 baseline probe evaluation on ADE20K.

Deterministic — no viewpoint policy, no variance. Single run is sufficient.
"""

import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from canvit_utils.teacher import load_teacher
from torch.utils.data import DataLoader
from tqdm import tqdm

from canvit_eval.ade20k.dataset import IGNORE_LABEL, NUM_CLASSES, ADE20kDataset, ResizeMode, make_val_transform
from canvit_eval.ade20k.probe import eval_probe_on_batch
from canvit_utils.probes import SegmentationProbe
from canvit_eval.ade20k.train_probe.config import _default_ade20k_root
from canvit_eval.metrics import IoUAccumulator
from canvit_eval.utils import collect_metadata

log = logging.getLogger(__name__)


def _default_output() -> Path:
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    ts = time.strftime("%Y%m%d_%H%M%S")
    return Path(f"outputs/ade20k_dinov3_eval_{ts}_{job_id}.pt")


@dataclass
class DINOv3ProbeEvalConfig:
    """DINOv3 baseline probe evaluation configuration."""

    probe_repo: str
    ade20k_root: Path = field(default_factory=_default_ade20k_root)
    output: Path = field(default_factory=_default_output)
    resize_mode: ResizeMode = "squish"
    scene_size: int = 512
    batch_size: int = 32
    num_workers: int = 8
    device: str = "cuda"
    amp: bool = True


@torch.inference_mode()
def evaluate(cfg: DINOv3ProbeEvalConfig) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_float32_matmul_precision("high")
    device = torch.device(cfg.device)

    log.info("=" * 70)
    log.info("DINOv3 Baseline Probe Evaluation (ADE20K)")
    log.info("=" * 70)
    for k, v in asdict(cfg).items():
        log.info(f"  {k}: {v}")

    # Load probe from HuggingFace Hub
    log.info(f"Loading probe: {cfg.probe_repo}")
    probe = SegmentationProbe.from_pretrained(cfg.probe_repo).to(device).eval()

    # Extract teacher model + resolution from probe's HF config metadata
    import json
    from huggingface_hub import hf_hub_download
    config_path = hf_hub_download(cfg.probe_repo, "config.json")
    probe_config = json.loads(Path(config_path).read_text())
    probe_meta = probe_config.get("metadata", {})
    probe_train_config = probe_meta.get("config", {})
    model_name = probe_train_config.get("model", probe_meta.get("model"))
    resolution = probe_train_config.get("resolution", probe_meta.get("resolution"))
    assert model_name is not None, f"No teacher model in probe config metadata for {cfg.probe_repo}"
    assert resolution is not None, f"No resolution in probe config metadata for {cfg.probe_repo}"
    log.info(f"  teacher={model_name}, resolution={resolution}px, embed_dim={probe.embed_dim}")

    # Load teacher
    teacher = load_teacher(model_name, device)
    patch_size = teacher.model.config.patch_size
    grid = resolution // patch_size
    assert grid > 0, f"resolution={resolution} too small for patch_size={patch_size}"
    assert teacher.embed_dim == probe.embed_dim, (
        f"Teacher embed_dim={teacher.embed_dim} != probe embed_dim={probe.embed_dim}"
    )
    log.info(f"  patch_size={patch_size}, grid={grid}x{grid}")

    # Dataset
    dataset = ADE20kDataset(
        root=cfg.ade20k_root, split="validation",
        transform=make_val_transform(cfg.scene_size, cfg.resize_mode),
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    iou = IoUAccumulator(NUM_CLASSES, IGNORE_LABEL, device)

    amp_dtype = torch.bfloat16 if cfg.amp else torch.float32
    amp_ctx = torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=cfg.amp)

    log.info(f"Evaluating at {resolution}px...")

    for images, masks in tqdm(loader, desc="Evaluating", unit="batch"):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with amp_ctx:
            sz = grid * patch_size
            resized = F.interpolate(images, size=(sz, sz), mode="bilinear", align_corners=False)
            feats = teacher.forward_norm_features(resized).patches
            feats = feats.view(images.shape[0], grid, grid, -1)

        eval_probe_on_batch(probe, feats, masks, iou)

    miou = iou.compute()

    # Derive probe name from model repo + resolution
    model_short = model_name.split("/")[-1].replace("-pretrain-lvd1689m", "").replace("-pretrain", "")
    probe_name = f"{model_short}_{resolution}px"

    log.info("")
    log.info("=" * 70)
    log.info(f"RESULT: {probe_name} mIoU = {100*miou:.2f}%")
    log.info("=" * 70)

    results = {
        "mious": {probe_name: {"t0": miou, "mean": miou}},
        "metadata": {
            **collect_metadata(cfg),
            "resolution": resolution,
            "model": model_name,
            "resize_mode": cfg.resize_mode,
        },
    }

    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, cfg.output)
    log.info(f"Saved to {cfg.output} ({cfg.output.stat().st_size / 1024:.1f} KB)")

    return cfg.output


def main() -> None:
    import tyro
    cfg = tyro.cli(DINOv3ProbeEvalConfig)
    evaluate(cfg)
