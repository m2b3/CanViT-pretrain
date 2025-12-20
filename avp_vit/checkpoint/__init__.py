"""Checkpoint serialization for ActiveCanViT models."""

import logging
import subprocess
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import torch
from torch import Tensor

from avp_vit import ActiveCanViT
from canvit import CanViTConfig
from canvit.attention import CrossAttentionConfig

# Field renames for legacy checkpoint migration
_CROSS_ATTN_RENAMES = {
    "normalize_q": "pre_proj_q_ln",
    "normalize_k": "pre_proj_k_ln",
    "normalize_v": "pre_proj_v_ln",
}
_CROSS_ATTN_FIELDS = {f.name for f in fields(CrossAttentionConfig)}

# Backbone embed_dim for canvas_dim_mult migration
_BACKBONE_EMBED_DIM = {
    "dinov3_vits16": 384,
    "dinov3_vitb16": 768,
    "dinov3_vitl16": 1024,
}

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
) -> None:
    """Save checkpoint with all info needed to reconstruct model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    git_commit, git_dirty = _git_info()

    data: CheckpointData = {
        "state_dict": model.state_dict(),
        "model_config": asdict(model.cfg),
        "teacher_dim": model.teacher_dim,
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
    log.info(f"  backbone={backbone}, teacher_dim={model.teacher_dim}")
    log.info(f"  glimpse_grid_size={model.cfg.glimpse_grid_size}, n_canvas_registers={model.n_canvas_registers}")
    if step is not None:
        log.info(f"  step={step}, train_loss={train_loss:.4e}" if train_loss else f"  step={step}")
    if git_commit:
        log.info(f"  git={git_commit[:8]}{'*' if git_dirty else ''}")


def load(path: Path, device: torch.device | str = "cpu") -> CheckpointData:
    """Load checkpoint data. Caller creates model with returned config."""
    log.info(f"Loading checkpoint: {path}")

    raw = torch.load(path, weights_only=False, map_location=device)

    # Detect old format (missing model_config and avp_config)
    if "model_config" not in raw and "avp_config" not in raw:
        raise ValueError(f"Legacy checkpoint without config. Keys: {list(raw.keys())}")

    # Handle key rename: old="avp", new="state_dict"
    if "avp" in raw and "state_dict" not in raw:
        raw["state_dict"] = raw.pop("avp")

    # Handle config key rename: old="avp_config", new="model_config"
    config_key = "model_config" if "model_config" in raw else "avp_config"
    model_config = raw[config_key].copy()

    # Reconstruct nested dataclasses in canvit config (migrate renamed fields)
    canvit_config = model_config.get("canvit", {})
    for key in ("read_attention", "write_attention"):
        if isinstance(canvit_config.get(key), dict):
            old_dict = canvit_config[key]
            migrated = {_CROSS_ATTN_RENAMES.get(k, k): v for k, v in old_dict.items()}
            filtered = {k: v for k, v in migrated.items() if k in _CROSS_ATTN_FIELDS}
            canvit_config[key] = CrossAttentionConfig(**filtered)
    # Migrate canvas_dim_mult → canvas_head_dim
    if "canvas_dim_mult" in canvit_config:
        embed_dim = _BACKBONE_EMBED_DIM.get(raw["backbone"])
        assert embed_dim is not None, f"Unknown backbone for migration: {raw['backbone']}"
        canvas_dim = embed_dim * canvit_config.pop("canvas_dim_mult")
        canvas_num_heads = canvit_config.get("canvas_num_heads", 2)
        canvit_config["canvas_head_dim"] = canvas_dim // canvas_num_heads
        log.info(f"  Migrated canvas_dim_mult → canvas_head_dim={canvit_config['canvas_head_dim']}")
    model_config["canvit"] = CanViTConfig(**canvit_config)

    data: CheckpointData = {
        "state_dict": _strip_orig_mod(raw["state_dict"]),
        "model_config": model_config,
        "teacher_dim": raw["teacher_dim"],
        "backbone": raw["backbone"],
        "timestamp": raw.get("timestamp", "unknown"),
        "git_commit": raw.get("git_commit"),
        "git_dirty": raw.get("git_dirty", False),
        "step": raw.get("step"),
        "train_loss": raw.get("train_loss"),
        "comet_id": raw.get("comet_id"),
    }

    log.info(f"  backbone={data['backbone']}, teacher_dim={data['teacher_dim']}")
    log.info(f"  glimpse_grid_size={model_config.get('glimpse_grid_size')}")
    if data["step"] is not None:
        log.info(f"  step={data['step']}, train_loss={data['train_loss']:.4e}" if data["train_loss"] else f"  step={data['step']}")
    if data["git_commit"]:
        log.info(f"  git={data['git_commit'][:8]}{'*' if data['git_dirty'] else ''}")
    if data["comet_id"]:
        log.info(f"  comet={data['comet_id']}")

    return data
