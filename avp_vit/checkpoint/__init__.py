"""Checkpoint serialization for ActiveCanViT models."""

import logging
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import dacite
import torch
from torch import Tensor

from avp_vit import ActiveCanViT, ActiveCanViTConfig

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
    # Policy config (None if no policy, or legacy checkpoint before policy support)
    policy_config: dict | None
    # Optimizer/scheduler state for resuming training
    optimizer_state: dict | None
    scheduler_state: dict | None


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
    optimizer_state: dict | None = None,
    scheduler_state: dict | None = None,
) -> None:
    """Save checkpoint with all info needed to reconstruct model."""
    from canvit.policy import PolicyHead

    assert isinstance(model.cfg, ActiveCanViTConfig)
    path.parent.mkdir(parents=True, exist_ok=True)
    git_commit, git_dirty = _git_info()

    # Save policy config if model has policy
    policy_config: dict | None = None
    if isinstance(model.policy, PolicyHead):
        policy_config = asdict(model.policy.cfg)

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
        "policy_config": policy_config,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
    }

    torch.save(data, path)
    size_mb = path.stat().st_size / (1024 * 1024)

    log.info(f"Checkpoint saved: {path} ({size_mb:.1f} MB)")
    log.info(f"  backbone={backbone}, teacher_dim={model.cfg.teacher_dim}, policy={policy_config is not None}")
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
        "policy_config": raw.get("policy_config"),
        "optimizer_state": raw.get("optimizer_state"),
        "scheduler_state": raw.get("scheduler_state"),
    }

    has_policy = data["policy_config"] is not None
    log.info(f"  backbone={data['backbone']}, teacher_dim={data['teacher_dim']}, policy={has_policy}")
    if data["step"] is not None:
        log.info(f"  step={data['step']}, train_loss={data['train_loss']:.4e}" if data["train_loss"] else f"  step={data['step']}")
    if data["git_commit"]:
        log.info(f"  git={data['git_commit'][:8]}{'*' if data['git_dirty'] else ''}")
    if data["comet_id"]:
        log.info(f"  comet={data['comet_id']}")

    return data


def _has_policy_keys(state_dict: dict[str, Tensor]) -> bool:
    """Check if state_dict contains policy.* keys."""
    return any(k.startswith("policy.") for k in state_dict)


def load_model(path: Path, device: torch.device | str = "cpu", strict: bool = False) -> ActiveCanViT:
    """Load ActiveCanViT from checkpoint.

    Policy handling:
    - If checkpoint has policy_config: create PolicyHead with that config
    - If checkpoint has policy.* keys but no policy_config (legacy): use default PolicyConfig + warn
    - Otherwise: no policy
    """
    from canvit.hub import create_backbone
    from canvit.policy import PolicyConfig, PolicyHead

    ckpt = load(path, device)

    for key in ("backbone", "model_config", "teacher_dim", "state_dict"):
        if key not in ckpt:
            raise ValueError(f"Checkpoint missing required field: {key!r}")

    backbone = create_backbone(ckpt["backbone"], pretrained=False)

    model_config = ckpt["model_config"]
    if "teacher_dim" not in model_config:
        model_config = {**model_config, "teacher_dim": ckpt["teacher_dim"]}
    cfg = dacite.from_dict(ActiveCanViTConfig, model_config)

    # Create policy if checkpoint has policy config or policy weights
    policy: PolicyHead | None = None
    policy_config = ckpt.get("policy_config")
    has_policy_weights = _has_policy_keys(ckpt["state_dict"])

    if policy_config is not None:
        # New checkpoint with explicit policy config
        policy_cfg = dacite.from_dict(PolicyConfig, policy_config)
        policy = PolicyHead(embed_dim=backbone.embed_dim, cfg=policy_cfg)
        log.info(f"Policy created from checkpoint config: min_scale={policy_cfg.min_scale}")
    elif has_policy_weights:
        # Legacy checkpoint: has policy weights but no config (use defaults)
        policy_cfg = PolicyConfig()
        policy = PolicyHead(embed_dim=backbone.embed_dim, cfg=policy_cfg)
        log.warning(
            f"Checkpoint has policy weights but no policy_config (legacy). "
            f"Using default PolicyConfig: min_scale={policy_cfg.min_scale}. "
            f"Re-save checkpoint to fix."
        )

    model = ActiveCanViT(backbone=backbone, cfg=cfg, policy=policy)

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
