"""IN1k classification probe utilities for DINOv3 features."""

import torch
from dinov3_probes import DINOv3LinearClassificationHead
from torch import Tensor

PROBE_REPOS = {
    "dinov3_vits16": "yberreby/dinov3-vits16-lvd1689m-in1k-512x512-linear-clf-probe",
    "dinov3_vitb16": "yberreby/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe",
    "dinov3_vitl16": "yberreby/dinov3-vitl16-lvd1689m-in1k-512x512-linear-clf-probe",
}


def load_probe(backbone: str, device: torch.device) -> DINOv3LinearClassificationHead | None:
    """Load IN1k classification probe from HF Hub. Returns None if backbone unsupported."""
    if backbone not in PROBE_REPOS:
        return None
    probe = DINOv3LinearClassificationHead.from_pretrained(PROBE_REPOS[backbone])
    return probe.to(device).eval()


def compute_in1k_top1(logits: Tensor, labels: Tensor) -> float:
    """Compute top-1 accuracy as percentage."""
    preds = logits.argmax(dim=-1)
    correct = (preds == labels).sum().item()
    return 100.0 * correct / labels.shape[0]
