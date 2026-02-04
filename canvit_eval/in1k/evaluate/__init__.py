"""IN1k evaluation: run trajectories on validation set, save predictions to parquet."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
import torch.nn.functional as F
import tyro
from canvit import CanViTForPretrainingHFHub, Viewpoint, sample_at_viewpoint
from canvit.backbone.dinov3 import DINOv3Backbone
from canvit_utils.policies import coarse_to_fine_viewpoints, random_viewpoints
from canvit_utils.teacher import load_teacher
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

log = logging.getLogger(__name__)

PolicyName = Literal["coarse_to_fine", "random"]


@dataclass
class Config:
    """IN1k evaluation configuration."""

    val_dir: Path
    output: Path = Path("outputs/in1k_predictions.parquet")
    model_repo: str = "canvit/canvit-vitb16-pretrain-512px-in21k"
    probe_repo: str = "yberreby/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe"
    policies: list[PolicyName] = field(default_factory=lambda: ["coarse_to_fine"])
    n_trajectories: int = 1
    n_viewpoints: int = 10
    canvas_grid: int = 32
    glimpse_px: int = 128
    batch_size: int = 64
    num_workers: int = 8
    device: str = "cuda"
    shuffle: bool = True  # Shuffle for representative partial results
    teacher_baseline: bool = False  # Compare against teacher's full-image features


def _load_probe(repo: str, device: torch.device) -> nn.Linear:
    with open(hf_hub_download(repo, "config.json")) as f:
        cfg = json.load(f)
    probe = nn.Linear(cfg["in_features"], cfg["out_features"])
    probe.load_state_dict(load_file(hf_hub_download(repo, "model.safetensors")))
    return probe.to(device).eval()


def _make_viewpoints(
    policy: PolicyName, batch_size: int, device: torch.device, n_viewpoints: int
) -> list[Viewpoint]:
    if policy == "coarse_to_fine":
        return coarse_to_fine_viewpoints(batch_size, device, n_viewpoints)
    return random_viewpoints(
        batch_size, device, n_viewpoints, min_scale=0.25, max_scale=1.0, start_with_full_scene=True
    )


def _run_trajectory(
    model: CanViTForPretrainingHFHub,
    cls_std,
    probe: nn.Linear,
    images: Tensor,
    viewpoints: list[Viewpoint],
    canvas_grid: int,
    glimpse_px: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run trajectory, return stacked top-5 predictions and viewpoints.

    Returns:
        classes: [T, B, 5] class indices
        probs: [T, B, 5] probabilities
        vp_data: [T, B, 3] viewpoint (cx, cy, scale)
    """
    B = images.shape[0]
    state = model.init_state(batch_size=B, canvas_grid_size=canvas_grid)

    all_classes, all_probs, all_vp = [], [], []
    for vp in viewpoints:
        glimpse = sample_at_viewpoint(spatial=images, viewpoint=vp, glimpse_size_px=glimpse_px)
        out = model(glimpse=glimpse, state=state, viewpoint=vp)
        state = out.state

        cls_pred = model.predict_scene_teacher_cls(out.state.recurrent_cls)
        cls_raw = cls_std.destandardize(cls_pred.unsqueeze(1)).squeeze(1)
        logits = probe(cls_raw)
        probs = F.softmax(logits, dim=-1)
        top_probs, top_classes = probs.topk(5, dim=-1)

        all_classes.append(top_classes)
        all_probs.append(top_probs)
        all_vp.append(torch.stack([vp.centers[:, 1], vp.centers[:, 0], vp.scales], dim=1))

    return torch.stack(all_classes), torch.stack(all_probs), torch.stack(all_vp)


@torch.inference_mode()
def evaluate(cfg: Config) -> pd.DataFrame:
    """Run evaluation, return DataFrame."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device(cfg.device)

    # Detailed startup logging
    log.info("=" * 70)
    log.info("ImageNet-1k Validation")
    log.info("=" * 70)
    log.info("")
    log.info("Configuration:")
    log.info(f"  model_repo:    {cfg.model_repo}")
    log.info(f"  probe_repo:    {cfg.probe_repo}")
    log.info(f"  val_dir:       {cfg.val_dir}")
    log.info(f"  output:        {cfg.output}")
    log.info(f"  batch_size:    {cfg.batch_size}")
    log.info(f"  num_workers:   {cfg.num_workers}")
    log.info(f"  canvas_grid:   {cfg.canvas_grid}")
    log.info(f"  glimpse_px:    {cfg.glimpse_px}")
    log.info(f"  n_viewpoints:  {cfg.n_viewpoints}")
    log.info(f"  policies:      {cfg.policies}")
    log.info(f"  n_trajectories:{cfg.n_trajectories}")
    log.info("")

    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"  GPU: {torch.cuda.get_device_name()}")
        log.info(f"  Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    log.info("")

    log.info(f"Loading model from {cfg.model_repo}...")
    model = CanViTForPretrainingHFHub.from_pretrained(cfg.model_repo).to(device).eval()
    log.info(f"  params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    log.info("")

    log.info(f"Loading probe from {cfg.probe_repo}...")
    probe = _load_probe(cfg.probe_repo, device)
    log.info("")

    cls_std, _ = model.standardizers(cfg.canvas_grid)
    assert cls_std.initialized, "CLS standardizer not initialized"

    patch_size = model.backbone.patch_size_px
    img_size = cfg.canvas_grid * patch_size
    log.info(f"  patch_size: {patch_size}px")
    log.info(f"  canvas: {cfg.canvas_grid}x{cfg.canvas_grid} = {img_size}px")
    log.info("")

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])

    dataset = ImageFolder(str(cfg.val_dir), transform=transform)
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle, num_workers=cfg.num_workers, pin_memory=True
    )
    log.info(f"Dataset: {len(dataset)} images, {len(loader)} batches")
    log.info("")

    # Teacher baseline (optional) - uses pretrained DINOv3, not model.backbone
    teacher: DINOv3Backbone | None = None
    if cfg.teacher_baseline:
        teacher = load_teacher(model.backbone_name, device)

    records: list[dict] = []
    T = cfg.n_viewpoints

    # Real-time accuracy tracking on GPU
    correct_top1 = torch.zeros(T, device=device, dtype=torch.long)
    correct_top5 = torch.zeros(T, device=device, dtype=torch.long)
    teacher_correct_top1 = torch.tensor(0, device=device, dtype=torch.long)
    teacher_correct_top5 = torch.tensor(0, device=device, dtype=torch.long)
    total = 0
    pbar_sync_interval = 20

    pbar = tqdm(loader, desc="Evaluating", unit="batch")
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        B = images.shape[0]
        start_idx = batch_idx * cfg.batch_size
        paths = [dataset.samples[start_idx + i][0] for i in range(B)]

        for policy in cfg.policies:
            for traj_idx in range(cfg.n_trajectories):
                viewpoints = _make_viewpoints(policy, B, device, T)
                classes, probs, vp = _run_trajectory(
                    model, cls_std, probe, images, viewpoints, cfg.canvas_grid, cfg.glimpse_px
                )

                # Update accuracy accumulators on GPU
                for t in range(T):
                    top1_pred = classes[t, :, 0]
                    top5_pred = classes[t, :, :5]
                    correct_top1[t] += (top1_pred == labels).sum()
                    correct_top5[t] += (top5_pred == labels.unsqueeze(1)).any(dim=1).sum()

                # Move to CPU once per trajectory for parquet
                classes_np = classes.cpu().numpy()
                probs_np = probs.cpu().numpy()
                vp_np = vp.cpu().numpy()
                labels_np = labels.cpu().numpy()

                for t in range(T):
                    for b in range(B):
                        records.append({
                            "image_path": paths[b],
                            "label": int(labels_np[b]),
                            "policy": policy,
                            "trajectory_idx": traj_idx,
                            "timestep": t,
                            "top1_class": int(classes_np[t, b, 0]),
                            "top1_prob": float(probs_np[t, b, 0]),
                            "top2_class": int(classes_np[t, b, 1]),
                            "top2_prob": float(probs_np[t, b, 1]),
                            "top3_class": int(classes_np[t, b, 2]),
                            "top3_prob": float(probs_np[t, b, 2]),
                            "top4_class": int(classes_np[t, b, 3]),
                            "top4_prob": float(probs_np[t, b, 3]),
                            "top5_class": int(classes_np[t, b, 4]),
                            "top5_prob": float(probs_np[t, b, 4]),
                            "viewpoint_cx": float(vp_np[t, b, 0]),
                            "viewpoint_cy": float(vp_np[t, b, 1]),
                            "viewpoint_scale": float(vp_np[t, b, 2]),
                        })

        # Teacher baseline (once per batch, outside policy/trajectory loops)
        if teacher is not None:
            teacher_cls = teacher.forward_norm_features(images).cls
            teacher_logits = probe(teacher_cls)
            teacher_probs = F.softmax(teacher_logits, dim=-1)
            _, teacher_top_classes = teacher_probs.topk(5, dim=-1)
            teacher_correct_top1 += (teacher_top_classes[:, 0] == labels).sum()
            teacher_correct_top5 += (teacher_top_classes == labels.unsqueeze(1)).any(dim=1).sum()

        total += B

        # Sync only every N batches for pbar update
        if batch_idx % pbar_sync_interval == 0 and total > 0:
            c1 = correct_top1.tolist()
            ts = " ".join(f"t{t}={100*c1[t]/total:.1f}" for t in range(T))
            if teacher is not None:
                ts += f" teacher={100*teacher_correct_top1.item()/total:.1f}"
            pbar.set_postfix_str(ts)

    # Final results
    log.info("")
    log.info("=" * 70)
    log.info("RESULTS")
    log.info("=" * 70)
    log.info(f"Total samples: {total}")
    log.info("")
    log.info(f"Accuracy by timestep ({T} viewpoints):")
    c1_list = correct_top1.tolist()
    c5_list = correct_top5.tolist()
    for t in range(T):
        acc1 = 100 * c1_list[t] / total
        acc5 = 100 * c5_list[t] / total
        log.info(f"  t{t}: top1={acc1:.2f}%, top5={acc5:.2f}%")

    if teacher is not None:
        t1 = 100 * teacher_correct_top1.item() / total
        t5 = 100 * teacher_correct_top5.item() / total
        log.info(f"Teacher baseline: top1={t1:.2f}%, top5={t5:.2f}%")

    df = pd.DataFrame(records)
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cfg.output, index=False)
    log.info(f"Saved {len(df)} rows to {cfg.output}")
    return df


def main() -> None:
    cfg = tyro.cli(Config)
    evaluate(cfg)
