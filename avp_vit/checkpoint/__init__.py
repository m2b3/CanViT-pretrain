"""Checkpoint serialization for AVP models."""

import logging
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import torch
from torch import Tensor

from avp_vit import AVPConfig, AVPViT
from avp_vit.attention import CrossAttentionConfig

log = logging.getLogger(__name__)


class CheckpointData(TypedDict):
    """Checkpoint structure. All fields required for model reconstruction."""

    state_dict: dict[str, Tensor]
    avp_config: dict
    teacher_dim: int
    backbone: str
    timestamp: str
    git_commit: str | None
    git_dirty: bool
    step: int | None
    train_loss: float | None
    comet_id: str | None


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
    avp: AVPViT,
    backbone: str,
    *,
    step: int | None = None,
    train_loss: float | None = None,
    comet_id: str | None = None,
) -> None:
    """Save checkpoint with all info needed to reconstruct model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    git_commit, git_dirty = _git_info()

    data: CheckpointData = {
        "state_dict": avp.state_dict(),
        "avp_config": asdict(avp.cfg),
        "teacher_dim": avp.teacher_dim,
        "backbone": backbone,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "step": step,
        "train_loss": train_loss,
        "comet_id": comet_id,
    }

    torch.save(data, path)
    size_mb = path.stat().st_size / (1024 * 1024)

    log.info(f"Checkpoint saved: {path} ({size_mb:.1f} MB)")
    log.info(f"  backbone={backbone}, teacher_dim={avp.teacher_dim}")
    log.info(f"  gating={avp.cfg.gating}, use_recurrence_ln={avp.cfg.use_recurrence_ln}")
    if step is not None:
        log.info(f"  step={step}, train_loss={train_loss:.4e}" if train_loss else f"  step={step}")
    if git_commit:
        log.info(f"  git={git_commit[:8]}{'*' if git_dirty else ''}")


def load(path: Path, device: torch.device | str = "cpu") -> CheckpointData:
    """Load checkpoint data. Caller creates model with returned config."""
    log.info(f"Loading checkpoint: {path}")

    raw = torch.load(path, weights_only=False, map_location=device)

    # Detect old format (missing avp_config)
    if "avp_config" not in raw:
        raise ValueError(f"Legacy checkpoint without config. Keys: {list(raw.keys())}")

    # Handle key rename: old="avp", new="state_dict"
    if "avp" in raw and "state_dict" not in raw:
        raw["state_dict"] = raw.pop("avp")

    # Reconstruct nested dataclasses in avp_config
    avp_config = raw["avp_config"].copy()
    if isinstance(avp_config.get("read_attention"), dict):
        avp_config["read_attention"] = CrossAttentionConfig(**avp_config["read_attention"])
    if isinstance(avp_config.get("write_attention"), dict):
        avp_config["write_attention"] = CrossAttentionConfig(**avp_config["write_attention"])

    data: CheckpointData = {
        "state_dict": _strip_orig_mod(raw["state_dict"]),
        "avp_config": avp_config,
        "teacher_dim": raw["teacher_dim"],
        "backbone": raw["backbone"],
        "timestamp": raw.get("timestamp", "unknown"),
        "git_commit": raw.get("git_commit"),
        "git_dirty": raw.get("git_dirty", False),
        "step": raw.get("step"),
        "train_loss": raw.get("train_loss"),
        "comet_id": raw.get("comet_id"),
    }

    cfg = AVPConfig(**data["avp_config"])
    log.info(f"  backbone={data['backbone']}, teacher_dim={data['teacher_dim']}")
    log.info(f"  gating={cfg.gating}, use_recurrence_ln={cfg.use_recurrence_ln}, registers={cfg.n_scene_registers}")
    if data["step"] is not None:
        log.info(f"  step={data['step']}, train_loss={data['train_loss']:.4e}" if data["train_loss"] else f"  step={data['step']}")
    if data["git_commit"]:
        log.info(f"  git={data['git_commit'][:8]}{'*' if data['git_dirty'] else ''}")
    if data["comet_id"]:
        log.info(f"  comet={data['comet_id']}")

    return data
