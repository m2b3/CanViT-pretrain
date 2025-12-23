"""Checkpoint serialization for ActiveCanViT models."""

import logging
import subprocess
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import torch
from torch import Tensor

from avp_vit import ActiveCanViT, ActiveCanViTConfig
import dacite

log = logging.getLogger(__name__)


class CheckpointData(TypedDict):
    """Checkpoint structure. All fields required for model reconstruction."""

    state_dict: dict[str, Tensor]
    model_config: dict
    teacher_dim: int
    backbone: str
    timestamp: str
    git_commit: str | None
    git_dirty: bool
    step: int | None
    train_loss: float | None
    comet_id: str | None
    # Normalizer stats (required for correct inference)
    scene_norm_state: dict[str, Tensor] | None
    cls_norm_state: dict[str, Tensor] | None


def _git_info() -> tuple[str | None, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], stderr=subprocess.DEVNULL) != 0
        return commit, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, False


def _strip_orig_mod(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    """Strip '_orig_mod.' from keys (torch.compile artifact)."""
    result = {}
    n = 0
    for k, v in state_dict.items():
        if "._orig_mod" in k:
            k = k.replace("._orig_mod", "")
            n += 1
        result[k] = v
    if n:
        log.warning(f"Stripped _orig_mod from {n} keys (legacy checkpoint)")
    return result


def save(
    path: Path,
    model: ActiveCanViT,
    backbone: str,
    *,
    step: int | None = None,
    train_loss: float | None = None,
    comet_id: str | None = None,
    scene_norm_state: dict[str, Tensor] | None = None,
    cls_norm_state: dict[str, Tensor] | None = None,
) -> None:
    """Save checkpoint with all info needed to reconstruct model."""
    assert isinstance(model.cfg, ActiveCanViTConfig)
    path.parent.mkdir(parents=True, exist_ok=True)
    git_commit, git_dirty = _git_info()

    data: CheckpointData = {
        "state_dict": model.state_dict(),
        "model_config": asdict(model.cfg),
        "teacher_dim": model.cfg.teacher_dim,
        "backbone": backbone,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "step": step,
        "train_loss": train_loss,
        "comet_id": comet_id,
        "scene_norm_state": scene_norm_state,
        "cls_norm_state": cls_norm_state,
    }

    torch.save(data, path)
    size_mb = path.stat().st_size / (1024 * 1024)

    log.info(f"Checkpoint saved: {path} ({size_mb:.1f} MB)")
    log.info(f"  backbone={backbone}, teacher_dim={model.cfg.teacher_dim}")
    if step is not None:
        log.info(f"  step={step}, train_loss={train_loss:.4e}" if train_loss else f"  step={step}")
    if git_commit:
        log.info(f"  git={git_commit[:8]}{'*' if git_dirty else ''}")


def load(path: Path, device: torch.device | str = "cpu") -> CheckpointData:
    """Load checkpoint data. Caller creates model with returned config."""
    log.info(f"Loading checkpoint: {path}")

    raw = torch.load(path, weights_only=False, map_location=device)

    data: CheckpointData = {
        "state_dict": _strip_orig_mod(raw["state_dict"]),
        "model_config": raw["model_config"],
        "teacher_dim": raw["teacher_dim"],
        "backbone": raw["backbone"],
        "timestamp": raw.get("timestamp", "unknown"),
        "git_commit": raw.get("git_commit"),
        "git_dirty": raw.get("git_dirty", False),
        "step": raw.get("step"),
        "train_loss": raw.get("train_loss"),
        "comet_id": raw.get("comet_id"),
        "scene_norm_state": raw.get("scene_norm_state"),
        "cls_norm_state": raw.get("cls_norm_state"),
    }

    log.info(f"  backbone={data['backbone']}, teacher_dim={data['teacher_dim']}")
    if data["step"] is not None:
        log.info(f"  step={data['step']}, train_loss={data['train_loss']:.4e}" if data["train_loss"] else f"  step={data['step']}")
    if data["git_commit"]:
        log.info(f"  git={data['git_commit'][:8]}{'*' if data['git_dirty'] else ''}")
    if data["comet_id"]:
        log.info(f"  comet={data['comet_id']}")

    return data


BACKBONE_REGISTRY: dict[str, str] = {
    "dinov3_vits16": "dinov3.hub.backbones.dinov3_vits16",
    "dinov3_vits16plus": "dinov3.hub.backbones.dinov3_vits16plus",
    "dinov3_vitb16": "dinov3.hub.backbones.dinov3_vitb16",
    "dinov3_vitl16": "dinov3.hub.backbones.dinov3_vitl16",
    "dinov3_vitl16plus": "dinov3.hub.backbones.dinov3_vitl16plus",
}


def _get_backbone_factory(name: str) -> Callable[..., torch.nn.Module]:
    """Get backbone factory by name. Raises ValueError if unknown."""
    if name not in BACKBONE_REGISTRY:
        available = ", ".join(sorted(BACKBONE_REGISTRY.keys()))
        raise ValueError(f"Unknown backbone: {name!r}. Available: {available}")

    module_path = BACKBONE_REGISTRY[name]
    module_name, func_name = module_path.rsplit(".", 1)

    import importlib
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def load_model(path: Path, device: torch.device | str = "cpu", strict: bool = True) -> ActiveCanViT:
    """Load ActiveCanViT from checkpoint.

    Args:
        path: Checkpoint file path.
        device: Target device.
        strict: If True (default), state_dict must match exactly. If False, ignores missing/extra keys.

    Returns:
        Model in eval mode on target device.

    Raises:
        ValueError: Unknown backbone or checkpoint format issues.
        RuntimeError: State dict mismatch (if strict=True).
    """
    from canvit.backbone.dinov3 import DINOv3Backbone
    from dinov3.models.vision_transformer import DinoVisionTransformer

    ckpt = load(path, device)

    # Validate required fields
    for key in ("backbone", "model_config", "teacher_dim", "state_dict"):
        if key not in ckpt:
            raise ValueError(f"Checkpoint missing required field: {key!r}")

    factory = _get_backbone_factory(ckpt["backbone"])
    raw_backbone = factory(pretrained=False)
    assert isinstance(raw_backbone, DinoVisionTransformer)
    backbone = DINOv3Backbone(raw_backbone)

    model_config = ckpt["model_config"]
    if "teacher_dim" not in model_config:
        model_config = {**model_config, "teacher_dim": ckpt["teacher_dim"]}
    cfg = dacite.from_dict(ActiveCanViTConfig, model_config)
    model = ActiveCanViT(backbone=backbone, cfg=cfg)

    result = model.load_state_dict(ckpt["state_dict"], strict=strict)
    if strict and (result.missing_keys or result.unexpected_keys):
        raise RuntimeError(
            f"State dict mismatch. Missing: {result.missing_keys}, Unexpected: {result.unexpected_keys}"
        )
    if result.missing_keys:
        log.warning(f"Missing keys (ignored): {result.missing_keys}")
    if result.unexpected_keys:
        log.warning(f"Unexpected keys (ignored): {result.unexpected_keys}")

    if isinstance(device, str):
        device = torch.device(device)
    return model.to(device).eval()
