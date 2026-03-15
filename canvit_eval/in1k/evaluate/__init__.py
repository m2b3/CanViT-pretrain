"""IN1k evaluation: run single trajectory, save tensors for post-hoc analysis.

Design:
- One policy, one rollout per job (parallelize via SLURM job arrays)
- Pre-allocated tensors, no dict appends, no pandas
- Save full logits + CLS for maximum flexibility
- Rich metadata for sanity checks
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn.functional as F
import tyro
from canvit import CanViTForPretrainingHFHub, Viewpoint, sample_at_viewpoint
from canvit_utils.transforms import preprocess
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from canvit_eval.policies import PolicyName, make_eval_policy
from canvit_eval.utils import collect_metadata

log = logging.getLogger(__name__)
TOP_K = 5


# === Config ===


def _default_val_dir() -> Path:
    return Path(os.environ.get("IN1K_VAL_IMAGE_DIR", "/datashare/imagenet/ILSVRC2012/val"))


def _default_output() -> Path:
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    return Path(f"outputs/in1k_{job_id}.pt")


@dataclass
class Config:
    val_dir: Path = field(default_factory=_default_val_dir)
    output: Path = field(default_factory=_default_output)
    model_repo: str = "canvit/canvitb16-pretrain-g128px-s512px-in21k-dv3b16"
    probe_repo: str = "yberreby/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe"
    policy: PolicyName = "coarse_to_fine"
    n_viewpoints: int = 21
    canvas_grid: int = 32
    glimpse_px: int = 128
    batch_size: int = 64
    num_workers: int = 8
    device: str = "cuda"
    save_logits: bool = True
    save_cls: bool = True
    # Random policy params (ignored for coarse_to_fine)
    random_min_scale: float = 0.05
    random_max_scale: float = 1.0
    random_start_full: bool = True


# === Pure helpers ===


def load_probe(repo: str, device: torch.device) -> nn.Linear:
    with open(hf_hub_download(repo, "config.json")) as f:
        probe_cfg = json.load(f)
    probe = nn.Linear(probe_cfg["in_features"], probe_cfg["out_features"])
    probe.load_state_dict(load_file(hf_hub_download(repo, "model.safetensors")))
    return probe.to(device).eval()


class OutputTensors(NamedTuple):
    labels: Tensor          # [N] int16
    top_k_preds: Tensor     # [N, T, K] int16
    top_k_probs: Tensor     # [N, T, K] float16
    viewpoints: Tensor      # [N, T, 3] float16 (cx, cy, scale)
    logits: Tensor | None   # [N, T, C] float16
    cls_tokens: Tensor | None  # [N, T, D] float16


def allocate_outputs(
    n_images: int,
    n_timesteps: int,
    n_classes: int,
    cls_dim: int,
    save_logits: bool,
    save_cls: bool,
) -> OutputTensors:
    return OutputTensors(
        labels=torch.zeros(n_images, dtype=torch.int16),
        top_k_preds=torch.zeros(n_images, n_timesteps, TOP_K, dtype=torch.int16),
        top_k_probs=torch.zeros(n_images, n_timesteps, TOP_K, dtype=torch.float16),
        viewpoints=torch.zeros(n_images, n_timesteps, 3, dtype=torch.float16),
        logits=torch.zeros(n_images, n_timesteps, n_classes, dtype=torch.float16) if save_logits else None,
        cls_tokens=torch.zeros(n_images, n_timesteps, cls_dim, dtype=torch.float16) if save_cls else None,
    )


def extract_viewpoint_coords(vp: Viewpoint) -> Tensor:
    """Extract (cx, cy, scale) from viewpoint. Returns [B, 3] float16 on CPU."""
    # vp.centers is [B, 2] with (y, x) ordering, vp.scales is [B]
    cx = vp.centers[:, 1]  # x coord
    cy = vp.centers[:, 0]  # y coord
    return torch.stack([cx, cy, vp.scales], dim=1).to(torch.float16).cpu()


class StepOutput(NamedTuple):
    top_k_classes: Tensor  # [B, K] int64 on GPU
    top_k_probs: Tensor    # [B, K] float32 on GPU
    logits: Tensor         # [B, C] float32 on GPU
    cls_raw: Tensor        # [B, D] float32 on GPU


def run_timestep(
    model: CanViTForPretrainingHFHub,
    probe: nn.Linear,
    cls_std,
    state,
    images: Tensor,
    vp: Viewpoint,
    glimpse_px: int,
) -> tuple[StepOutput, any]:
    """Run single timestep, return predictions and new state."""
    glimpse = sample_at_viewpoint(spatial=images, viewpoint=vp, glimpse_size_px=glimpse_px)
    out = model(glimpse=glimpse, state=state, viewpoint=vp)
    state = out.state

    cls_pred = model.predict_scene_teacher_cls(out.state.recurrent_cls)
    cls_raw = cls_std.destandardize(cls_pred.unsqueeze(1)).squeeze(1)
    logits = probe(cls_raw)
    probs = F.softmax(logits, dim=-1)
    top_probs, top_classes = probs.topk(TOP_K, dim=-1)

    return StepOutput(top_classes, top_probs, logits, cls_raw), state


def store_timestep(
    out: OutputTensors,
    step: StepOutput,
    vp: Viewpoint,
    start: int,
    end: int,
    t: int,
) -> None:
    """Store timestep results to pre-allocated CPU tensors."""
    out.top_k_preds[start:end, t] = step.top_k_classes.to(torch.int16).cpu()
    out.top_k_probs[start:end, t] = step.top_k_probs.to(torch.float16).cpu()
    out.viewpoints[start:end, t] = extract_viewpoint_coords(vp)
    if out.logits is not None:
        out.logits[start:end, t] = step.logits.to(torch.float16).cpu()
    if out.cls_tokens is not None:
        out.cls_tokens[start:end, t] = step.cls_raw.to(torch.float16).cpu()


def update_accuracy(
    correct_top1: Tensor,
    correct_top5: Tensor,
    top_k_classes: Tensor,
    labels: Tensor,
    t: int,
) -> None:
    """Update accuracy accumulators on GPU."""
    correct_top1[t] += (top_k_classes[:, 0] == labels).sum()
    correct_top5[t] += (top_k_classes == labels.unsqueeze(1)).any(dim=1).sum()


def build_output_dict(
    out: OutputTensors,
    paths: list[str],
    correct_top1: Tensor,
    correct_top5: Tensor,
    n_images: int,
    cfg: Config,
) -> dict:
    """Build final output dictionary for saving."""
    c1 = correct_top1.tolist()
    c5 = correct_top5.tolist()
    T = cfg.n_viewpoints

    result = {
        "paths": paths,  # List of image paths, index-aligned with tensors
        "labels": out.labels,
        "top_k_preds": out.top_k_preds,
        "top_k_probs": out.top_k_probs,
        "viewpoints": out.viewpoints,
        "accuracy_top1": torch.tensor([c1[t] / n_images for t in range(T)]),
        "accuracy_top5": torch.tensor([c5[t] / n_images for t in range(T)]),
        "metadata": collect_metadata(cfg),
    }
    if out.logits is not None:
        result["logits"] = out.logits
    if out.cls_tokens is not None:
        result["cls_tokens"] = out.cls_tokens
    return result


# === Main evaluation ===


@torch.inference_mode()
def evaluate(cfg: Config) -> Path:
    """Run evaluation, save tensors, return output path."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Validate inputs
    assert cfg.val_dir.exists(), f"val_dir does not exist: {cfg.val_dir}"

    device = torch.device(cfg.device)
    T = cfg.n_viewpoints

    # Log config
    log.info("=" * 70)
    log.info("ImageNet-1k Evaluation")
    log.info("=" * 70)
    for k, v in asdict(cfg).items():
        log.info(f"  {k}: {v}")

    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name()}")
        log.info(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")

    # Load model and probe
    log.info(f"Loading model: {cfg.model_repo}")
    model = CanViTForPretrainingHFHub.from_pretrained(cfg.model_repo).to(device).eval()
    log.info(f"  params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    log.info(f"Loading probe: {cfg.probe_repo}")
    probe = load_probe(cfg.probe_repo, device)
    n_classes = probe.out_features
    cls_dim = model.backbone.embed_dim
    log.info(f"  probe: {probe.in_features} -> {n_classes}")

    cls_std, _ = model.standardizers(cfg.canvas_grid)
    assert cls_std.initialized, "CLS standardizer not initialized"
    assert probe.in_features == cls_dim, f"Probe input {probe.in_features} != model dim {cls_dim}"

    # Dataset
    patch_size = model.backbone.patch_size_px
    img_size = cfg.canvas_grid * patch_size
    # CRITICAL: must use the canonical preprocess() (Resize shortest edge + CenterCrop).
    # A previous bug used Resize((H,W)) which squishes non-square images and costs ~2% accuracy.
    transform = preprocess(img_size)
    dataset = ImageFolder(str(cfg.val_dir), transform=transform)
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True
    )
    N = len(dataset)
    assert N > 0, f"Empty dataset at {cfg.val_dir}"
    paths = [p for p, _ in dataset.samples]  # From ImageFolder metadata, no I/O
    log.info(f"Dataset: {N} images, {len(loader)} batches")

    # Allocate outputs
    out = allocate_outputs(N, T, n_classes, cls_dim, cfg.save_logits, cfg.save_cls)

    # Accuracy accumulators on GPU
    correct_top1 = torch.zeros(T, device=device, dtype=torch.long)
    correct_top5 = torch.zeros(T, device=device, dtype=torch.long)

    # Run evaluation
    log.info(f"Running {cfg.policy} policy, {T} viewpoints...")
    pbar = tqdm(loader, desc="Evaluating", unit="batch")
    total_processed = 0
    pbar_sync_interval = 20

    for batch_idx, (images, batch_labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels_gpu = batch_labels.to(device, non_blocking=True)
        B = images.shape[0]
        start = batch_idx * cfg.batch_size
        end = start + B

        # Store labels
        out.labels[start:end] = batch_labels.to(torch.int16)

        # Run trajectory
        policy = make_eval_policy(
            cfg.policy, B, device, T,
            canvas_grid=cfg.canvas_grid,
            min_scale=cfg.random_min_scale,
            max_scale=cfg.random_max_scale,
            start_with_full_scene=cfg.random_start_full,
        )
        state = model.init_state(batch_size=B, canvas_grid_size=cfg.canvas_grid)

        for t in range(T):
            vp = policy.step(t, state)
            step, state = run_timestep(model, probe, cls_std, state, images, vp, cfg.glimpse_px)
            store_timestep(out, step, vp, start, end, t)
            update_accuracy(correct_top1, correct_top5, step.top_k_classes, labels_gpu, t)

        total_processed += B

        # Progress update (periodic sync)
        if batch_idx % pbar_sync_interval == 0:
            c1 = correct_top1.tolist()
            ts = " ".join(f"t{t}={100*c1[t]/total_processed:.1f}" for t in range(T))
            pbar.set_postfix_str(ts)

    # Final results
    log.info("")
    log.info("=" * 70)
    log.info("RESULTS")
    log.info("=" * 70)
    c1 = correct_top1.tolist()
    c5 = correct_top5.tolist()
    for t in range(T):
        log.info(f"  t{t}: top1={100*c1[t]/N:.2f}%, top5={100*c5[t]/N:.2f}%")

    # Save
    output_dict = build_output_dict(out, paths, correct_top1, correct_top5, N, cfg)
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_dict, cfg.output)
    size_mb = cfg.output.stat().st_size / 1e6
    log.info(f"Saved to {cfg.output} ({size_mb:.1f} MB)")

    return cfg.output


def main() -> None:
    cfg = tyro.cli(Config)
    evaluate(cfg)
