"""ADE20K canvas probe evaluation with configurable policies.

Evaluates a frozen CanViT checkpoint using a trained segmentation probe
on the ADE20K-SceneParse150 validation set (2000 images, 150 classes).

Feature type: always canvas_hidden (LayerNorm'd spatial canvas tokens).
Metric: global mIoU (intersection/union summed across all images, DINOv3-style).

Output .pt file contains:
    mious: dict[str, float] — per-timestep mIoU ("t0", "t1", ..., "mean")
    viewpoints: list[dict] — per-timestep viewpoint metadata for first batch
    metadata: dict — full config, timing, git commit, hardware info
"""

import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from canvit import CanViTForPretrainingHFHub, Viewpoint, sample_at_viewpoint
from torch.utils.data import DataLoader
from tqdm import tqdm

from canvit_eval.ade20k.dataset import IGNORE_LABEL, NUM_CLASSES, ADE20kDataset, ResizeMode, make_val_transform
from canvit_eval.ade20k.probe import ProbeHead, eval_probe_on_batch
from canvit_eval.ade20k.train_probe.config import CANVAS_FEATURES, _default_ade20k_root
from canvit_eval.metrics import IoUAccumulator
from canvit_eval.policies import PolicyName, make_eval_policy
from canvit_eval.utils import collect_metadata

log = logging.getLogger(__name__)

FEATURE_TYPE = "canvas_hidden"


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
    policy: PolicyName = "coarse_to_fine"
    resize_mode: ResizeMode = "squish"
    n_timesteps: int = 16
    scene_size: int = 512
    glimpse_px: int = 128
    canvas_grid: int | None = None

    min_scale: float = 0.05
    max_scale: float = 1.0
    start_full: bool = True

    batch_size: int = 32
    num_workers: int = 8
    device: str = "cuda"
    amp: bool = True


def load_probe(ckpt_path: Path, device: torch.device, embed_dim: int) -> ProbeHead:
    """Load a canvas_hidden probe from checkpoint."""
    log.info(f"Loading probe from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    feat_spec = CANVAS_FEATURES[FEATURE_TYPE]
    dropout = ckpt.get("config", {}).get("dropout", 0.0)

    if "feat_type" in ckpt:
        assert ckpt["feat_type"] == FEATURE_TYPE, (
            f"Probe is for {ckpt['feat_type']!r}, expected {FEATURE_TYPE!r}"
        )
        state_dict = ckpt["probe_state_dict"]
    else:
        assert FEATURE_TYPE in ckpt["probe_state_dicts"], (
            f"Legacy checkpoint missing {FEATURE_TYPE!r}"
        )
        state_dict = ckpt["probe_state_dicts"][FEATURE_TYPE]

    probe = ProbeHead(embed_dim, dropout=dropout, use_ln=feat_spec.needs_ln).to(device)
    probe.load_state_dict(state_dict)
    probe.eval()

    best_per_t = ckpt.get("best_mious_per_t")
    log.info(f"  {FEATURE_TYPE}: loaded (best per-t mIoU: {best_per_t})")
    return probe


def _viewpoint_to_dict(vp: "Viewpoint", t: int) -> dict:
    """Serialize first batch element's viewpoint for logging."""
    return {
        "t": t,
        "center_y": vp.centers[0, 0].item(),
        "center_x": vp.centers[0, 1].item(),
        "scale": vp.scales[0].item(),
    }


@torch.inference_mode()
def evaluate(cfg: EvalConfig) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_float32_matmul_precision("high")

    device = torch.device(cfg.device)
    T = cfg.n_timesteps

    log.info("=" * 70)
    log.info("ADE20K Canvas Probe Evaluation")
    log.info("=" * 70)
    for k, v in asdict(cfg).items():
        log.info(f"  {k}: {v}")

    # Load model
    model = CanViTForPretrainingHFHub.from_pretrained(cfg.model_repo).to(device).eval()
    log.info(f"  canvas_dim={model.canvas_dim}")

    patch_size = model.backbone.patch_size_px
    canvas_grid = cfg.canvas_grid if cfg.canvas_grid is not None else cfg.scene_size // patch_size
    cfg.canvas_grid = canvas_grid

    # Load probe
    probe = load_probe(cfg.probe_ckpt, device, model.canvas_dim)

    # Dataset
    dataset = ADE20kDataset(
        root=cfg.ade20k_root, split="validation",
        transform=make_val_transform(cfg.scene_size, cfg.resize_mode),
    )
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    iou_per_t = [IoUAccumulator(NUM_CLASSES, IGNORE_LABEL, device) for _ in range(T)]

    amp_dtype = torch.bfloat16 if cfg.amp else torch.float32
    amp_ctx = torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=cfg.amp)

    log.info(f"Evaluating: policy={cfg.policy}, T={T}, scene={cfg.scene_size}, "
             f"canvas={canvas_grid}, glimpse={cfg.glimpse_px}")

    viewpoint_log: list[dict] = []
    t_start = time.monotonic()

    for batch_idx, (images, masks) in enumerate(tqdm(loader, desc="Evaluating", unit="batch")):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        B = images.shape[0]

        policy = make_eval_policy(
            cfg.policy, B, device, T,
            canvas_grid=canvas_grid,
            min_scale=cfg.min_scale, max_scale=cfg.max_scale,
            start_with_full_scene=cfg.start_full,
            probe=probe, get_spatial_fn=model.get_spatial,
        )

        state = model.init_state(batch_size=B, canvas_grid_size=canvas_grid)

        with amp_ctx:
            for t in range(T):
                vp = policy.step(t, state)

                # Log viewpoints from first batch for reproducibility
                if batch_idx == 0:
                    viewpoint_log.append(_viewpoint_to_dict(vp, t))

                glimpse = sample_at_viewpoint(
                    spatial=images, viewpoint=vp, glimpse_size_px=cfg.glimpse_px,
                )
                out = model(glimpse=glimpse, state=state, viewpoint=vp)
                state = out.state

                features = model.get_spatial(state.canvas).view(B, canvas_grid, canvas_grid, -1)
                eval_probe_on_batch(probe, features, masks, iou_per_t[t])

    wall_time = time.monotonic() - t_start

    # Results
    mious = [iou_per_t[t].compute() for t in range(T)]

    log.info("")
    log.info("=" * 70)
    log.info("RESULTS (global mIoU)")
    log.info("=" * 70)
    for t, m in enumerate(mious):
        log.info(f"  t{t}: {100*m:.2f}%")
    log.info(f"  mean: {100*sum(mious)/len(mious):.2f}%")
    log.info(f"  wall_time: {wall_time:.1f}s")

    results = {
        "mious": {f"t{t}": m for t, m in enumerate(mious)},
        "viewpoints": viewpoint_log,
        "metadata": {
            **collect_metadata(cfg),
            "wall_time_seconds": wall_time,
            "n_images": len(dataset),
            "feature_type": FEATURE_TYPE,
        },
    }

    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, cfg.output)
    log.info(f"Saved to {cfg.output} ({cfg.output.stat().st_size / 1024:.1f} KB)")

    return cfg.output


def main() -> None:
    import tyro
    cfg = tyro.cli(EvalConfig)
    evaluate(cfg)
