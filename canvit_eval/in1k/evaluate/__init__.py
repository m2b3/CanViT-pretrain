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
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
import tyro
from canvit import CanViTForPretrainingHFHub, Viewpoint, sample_at_viewpoint
from canvit_utils.policies import coarse_to_fine_viewpoints, random_viewpoints
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

log = logging.getLogger(__name__)

PolicyName = Literal["coarse_to_fine", "random"]


def _default_val_dir() -> Path:
    return Path(os.environ.get("IN1K_VAL_DIR", "/datashare/imagenet/ILSVRC2012/val"))


def _default_output() -> Path:
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    return Path(f"outputs/in1k_{job_id}.pt")


@dataclass
class Config:
    """IN1k evaluation configuration.

    Defaults read from environment for cluster compatibility.
    One policy per run - use SLURM job arrays for multiple policies/rollouts.
    """

    val_dir: Path = field(default_factory=_default_val_dir)
    output: Path = field(default_factory=_default_output)
    model_repo: str = "canvit/canvit-vitb16-pretrain-512px-in21k"
    probe_repo: str = "yberreby/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe"
    policy: PolicyName = "coarse_to_fine"
    n_viewpoints: int = 10
    canvas_grid: int = 32
    glimpse_px: int = 128
    batch_size: int = 64
    num_workers: int = 8
    device: str = "cuda"
    save_logits: bool = True
    save_cls: bool = True


def _get_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def _collect_metadata(cfg: Config) -> dict:
    return {
        "config": asdict(cfg),
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": _get_git_commit(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": os.environ.get("HOSTNAME") or os.environ.get("SLURMD_NODENAME"),
        "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }


def _load_probe(repo: str, device: torch.device) -> nn.Linear:
    with open(hf_hub_download(repo, "config.json")) as f:
        probe_cfg = json.load(f)
    probe = nn.Linear(probe_cfg["in_features"], probe_cfg["out_features"])
    probe.load_state_dict(load_file(hf_hub_download(repo, "model.safetensors")))
    return probe.to(device).eval()


def _make_viewpoints(policy: PolicyName, batch_size: int, device: torch.device, n: int) -> list[Viewpoint]:
    if policy == "coarse_to_fine":
        return coarse_to_fine_viewpoints(batch_size, device, n)
    return random_viewpoints(batch_size, device, n, min_scale=0.25, max_scale=1.0, start_with_full_scene=True)


@torch.inference_mode()
def evaluate(cfg: Config) -> Path:
    """Run evaluation, save tensors, return output path."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device(cfg.device)
    T = cfg.n_viewpoints

    log.info("=" * 70)
    log.info("ImageNet-1k Evaluation")
    log.info("=" * 70)
    for k, v in asdict(cfg).items():
        log.info(f"  {k}: {v}")
    log.info("")

    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name()}")
        log.info(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")

    # Load model
    log.info(f"Loading model: {cfg.model_repo}")
    model = CanViTForPretrainingHFHub.from_pretrained(cfg.model_repo).to(device).eval()
    log.info(f"  params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Load probe
    log.info(f"Loading probe: {cfg.probe_repo}")
    probe = _load_probe(cfg.probe_repo, device)

    cls_std, _ = model.standardizers(cfg.canvas_grid)
    assert cls_std.initialized, "CLS standardizer not initialized"

    # Dataset (no shuffle - deterministic ordering)
    patch_size = model.backbone.patch_size_px
    img_size = cfg.canvas_grid * patch_size
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])
    dataset = ImageFolder(str(cfg.val_dir), transform=transform)
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True
    )
    N = len(dataset)
    log.info(f"Dataset: {N} images, {len(loader)} batches")

    # Pre-allocate output tensors on CPU
    D_cls = model.backbone.embed_dim
    D_logits = 1000

    labels = torch.zeros(N, dtype=torch.int16)
    top5_preds = torch.zeros(N, T, 5, dtype=torch.int16)
    top5_probs = torch.zeros(N, T, 5, dtype=torch.float16)
    viewpoints = torch.zeros(N, T, 3, dtype=torch.float16)

    logits_out = torch.zeros(N, T, D_logits, dtype=torch.float16) if cfg.save_logits else None
    cls_out = torch.zeros(N, T, D_cls, dtype=torch.float16) if cfg.save_cls else None

    # Accuracy accumulators on GPU
    correct_top1 = torch.zeros(T, device=device, dtype=torch.long)
    correct_top5 = torch.zeros(T, device=device, dtype=torch.long)
    pbar_sync_interval = 20

    log.info(f"Running {cfg.policy} policy, {T} viewpoints...")
    pbar = tqdm(loader, desc="Evaluating", unit="batch")

    for batch_idx, (images, batch_labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        batch_labels_gpu = batch_labels.to(device, non_blocking=True)
        B = images.shape[0]
        start = batch_idx * cfg.batch_size
        end = start + B

        # Store labels
        labels[start:end] = batch_labels.to(torch.int16)

        # Generate viewpoints and run trajectory
        vps = _make_viewpoints(cfg.policy, B, device, T)
        state = model.init_state(batch_size=B, canvas_grid_size=cfg.canvas_grid)

        for t, vp in enumerate(vps):
            glimpse = sample_at_viewpoint(spatial=images, viewpoint=vp, glimpse_size_px=cfg.glimpse_px)
            out = model(glimpse=glimpse, state=state, viewpoint=vp)
            state = out.state

            # Predict CLS
            cls_pred = model.predict_scene_teacher_cls(out.state.recurrent_cls)
            cls_raw = cls_std.destandardize(cls_pred.unsqueeze(1)).squeeze(1)
            logits = probe(cls_raw)
            probs = F.softmax(logits, dim=-1)
            top_probs, top_classes = probs.topk(5, dim=-1)

            # Update accuracy on GPU
            correct_top1[t] += (top_classes[:, 0] == batch_labels_gpu).sum()
            correct_top5[t] += (top_classes == batch_labels_gpu.unsqueeze(1)).any(dim=1).sum()

            # Store to CPU tensors (single copy per timestep)
            top5_preds[start:end, t] = top_classes.to(torch.int16).cpu()
            top5_probs[start:end, t] = top_probs.to(torch.float16).cpu()
            viewpoints[start:end, t, 0] = vp.centers[:, 1].to(torch.float16).cpu()
            viewpoints[start:end, t, 1] = vp.centers[:, 0].to(torch.float16).cpu()
            viewpoints[start:end, t, 2] = vp.scales.to(torch.float16).cpu()

            if logits_out is not None:
                logits_out[start:end, t] = logits.to(torch.float16).cpu()
            if cls_out is not None:
                cls_out[start:end, t] = cls_raw.to(torch.float16).cpu()

        # Progress update (sync only periodically)
        if batch_idx % pbar_sync_interval == 0:
            total = end
            c1 = correct_top1.tolist()
            ts = " ".join(f"t{t}={100*c1[t]/total:.1f}" for t in range(T))
            pbar.set_postfix_str(ts)

    # Final accuracy
    log.info("")
    log.info("=" * 70)
    log.info("RESULTS")
    log.info("=" * 70)
    c1 = correct_top1.tolist()
    c5 = correct_top5.tolist()
    for t in range(T):
        log.info(f"  t{t}: top1={100*c1[t]/N:.2f}%, top5={100*c5[t]/N:.2f}%")

    # Build output dict
    output = {
        "labels": labels,
        "top5_preds": top5_preds,
        "top5_probs": top5_probs,
        "viewpoints": viewpoints,
        "accuracy_top1": torch.tensor([c1[t] / N for t in range(T)]),
        "accuracy_top5": torch.tensor([c5[t] / N for t in range(T)]),
        "metadata": _collect_metadata(cfg),
    }
    if logits_out is not None:
        output["logits"] = logits_out
    if cls_out is not None:
        output["cls_tokens"] = cls_out

    # Save
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, cfg.output)
    size_mb = cfg.output.stat().st_size / 1e6
    log.info(f"Saved to {cfg.output} ({size_mb:.1f} MB)")

    return cfg.output


def main() -> None:
    cfg = tyro.cli(Config)
    evaluate(cfg)
