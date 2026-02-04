"""Checkpoint serialization for CanViTForPretraining models."""

import logging
import os
import socket
import subprocess
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import dacite
import torch
from torch import Tensor

from avp_vit import CanViTForPretraining, CanViTForPretrainingConfig

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
    # Training config history: {timestamp: config_dict} - tracks config across resumes
    training_config_history: dict[str, dict] | None
    # Environment metadata for debugging
    hostname: str | None
    slurm_job_id: str | None
    slurm_array_task_id: str | None
    cmdline: list[str] | None


def _git_info() -> tuple[str | None, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], stderr=subprocess.DEVNULL) != 0
        return commit, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, False


def get_env_metadata() -> tuple[str | None, str | None, str | None, list[str] | None]:
    """Collect (hostname, slurm_job_id, slurm_array_task_id, cmdline). Best-effort."""
    hostname: str | None = None
    cmdline: list[str] | None = None
    try:
        hostname = socket.gethostname()
    except Exception as e:
        log.warning(f"Failed to get hostname: {e}")
    try:
        cmdline = sys.argv.copy()
    except Exception as e:
        log.warning(f"Failed to get cmdline: {e}")
    return (
        hostname,
        os.environ.get("SLURM_JOB_ID"),
        os.environ.get("SLURM_ARRAY_TASK_ID"),
        cmdline,
    )


def atomic_torch_save(data: CheckpointData, path: Path) -> None:
    """Save data to path atomically using tmp file + rename."""
    log.info(f"Saving checkpoint to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path_str = tempfile.mkstemp(
        suffix=".pt.tmp", prefix=path.stem, dir=path.parent
    )
    tmp_path = Path(tmp_path_str)
    log.debug(f"Writing to tmp file: {tmp_path}")
    try:
        os.close(fd)
        torch.save(data, tmp_path)
        tmp_path.rename(path)
    except Exception:
        log.exception(f"Failed to save checkpoint to {path}")
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def update_symlink(symlink_path: Path, target: Path) -> None:
    """Atomically update symlink to point to target."""
    log.info(f"Updating symlink {symlink_path} -> {target.name}")
    tmp_link = symlink_path.parent / f".{symlink_path.name}.tmp.{os.getpid()}"
    if tmp_link.is_symlink() or tmp_link.exists():
        tmp_link.unlink()
    tmp_link.symlink_to(target.name)
    tmp_link.rename(symlink_path)


def find_latest(run_dir: Path) -> Path | None:
    """Find latest.pt symlink in run_dir, return resolved path or None."""
    latest = run_dir / "latest.pt"
    if latest.is_symlink():
        resolved = latest.resolve()
        if resolved.exists():
            return resolved
        log.warning(f"latest.pt symlink broken: {latest} -> {resolved}")
    return None


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
    model: CanViTForPretraining,
    backbone: str,
    *,
    step: int | None = None,
    train_loss: float | None = None,
    comet_id: str | None = None,
    scene_norm_state: dict[str, Tensor] | None = None,
    cls_norm_state: dict[str, Tensor] | None = None,
    optimizer_state: dict | None = None,
    scheduler_state: dict | None = None,
    training_config_history: dict[str, dict] | None = None,
) -> None:
    """Save checkpoint with all info needed to reconstruct model.

    Uses atomic write (tmp file + rename) to prevent corruption.
    """
    from canvit.policy import PolicyHead

    assert isinstance(model.cfg, CanViTForPretrainingConfig)
    git_commit, git_dirty = _git_info()
    hostname, slurm_job_id, slurm_array_task_id, cmdline = get_env_metadata()

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
        "training_config_history": training_config_history,
        "hostname": hostname,
        "slurm_job_id": slurm_job_id,
        "slurm_array_task_id": slurm_array_task_id,
        "cmdline": cmdline,
    }

    atomic_torch_save(data, path)
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
        "training_config_history": raw.get("training_config_history"),
        "hostname": raw.get("hostname"),
        "slurm_job_id": raw.get("slurm_job_id"),
        "slurm_array_task_id": raw.get("slurm_array_task_id"),
        "cmdline": raw.get("cmdline"),
    }

    has_policy = data["policy_config"] is not None
    log.info(f"  backbone={data['backbone']}, teacher_dim={data['teacher_dim']}, policy={has_policy}")
    if data["step"] is not None:
        log.info(f"  step={data['step']}, train_loss={data['train_loss']:.4e}" if data["train_loss"] else f"  step={data['step']}")
    if data["git_commit"]:
        log.info(f"  git={data['git_commit'][:8]}{'*' if data['git_dirty'] else ''}")
    if data["comet_id"]:
        log.info(f"  comet={data['comet_id']}")
    if data["hostname"]:
        log.info(f"  host={data['hostname']}")

    return data


def _has_policy_keys(state_dict: dict[str, Tensor]) -> bool:
    """Check if state_dict contains policy.* keys."""
    return any(k.startswith("policy.") for k in state_dict)


def load_model(path: Path, device: torch.device | str = "cpu", strict: bool = False) -> CanViTForPretraining:
    """Load CanViTForPretraining from checkpoint.

    Policy handling:
    - If checkpoint has policy_config: create PolicyHead with that config
    - If checkpoint has policy.* keys but no policy_config (legacy): use default PolicyConfig + warn
    - Otherwise: no policy
    """
    from canvit import create_backbone
    from canvit.policy import PolicyConfig, PolicyHead

    ckpt = load(path, device)

    for key in ("backbone", "model_config", "teacher_dim", "state_dict"):
        if key not in ckpt:
            raise ValueError(f"Checkpoint missing required field: {key!r}")

    backbone = create_backbone(ckpt["backbone"], pretrained=False)

    model_config = ckpt["model_config"]
    if "teacher_dim" not in model_config:
        model_config = {**model_config, "teacher_dim": ckpt["teacher_dim"]}
    cfg = dacite.from_dict(CanViTForPretrainingConfig, model_config)

    # Create policy if checkpoint has policy config or policy weights
    policy: PolicyHead | None = None
    policy_config = ckpt.get("policy_config")
    has_policy_weights = _has_policy_keys(ckpt["state_dict"])

    if policy_config is not None:
        # New checkpoint with explicit policy config
        policy_cfg = dacite.from_dict(PolicyConfig, policy_config)
        policy = PolicyHead(embed_dim=backbone.embed_dim, cfg=policy_cfg)
        log.info("Policy created from checkpoint config")
    elif has_policy_weights:
        # Legacy checkpoint: has policy weights but no config (use defaults)
        policy_cfg = PolicyConfig()
        policy = PolicyHead(embed_dim=backbone.embed_dim, cfg=policy_cfg)
        log.warning(
            "Checkpoint has policy weights but no policy_config (legacy). "
            "Using default PolicyConfig. Re-save checkpoint to fix."
        )

    model = CanViTForPretraining(backbone=backbone, cfg=cfg, policy=policy)

    # Filter state_dict to avoid size mismatches (e.g., architectural changes)
    state_dict = ckpt["state_dict"]
    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape != v.shape:
            skipped.append(f"{k}: ckpt {tuple(v.shape)} vs model {tuple(model_state[k].shape)}")
        else:
            filtered[k] = v
    if skipped:
        log.warning(f"Skipped {len(skipped)} keys with shape mismatch: {skipped}")

    result = model.load_state_dict(filtered, strict=strict)
    if strict and (result.missing_keys or result.unexpected_keys):
        raise RuntimeError(
            f"State dict mismatch. Missing: {result.missing_keys}, Unexpected: {result.unexpected_keys}"
        )
    if result.missing_keys:
        log.warning(f"Missing keys (freshly initialized): {result.missing_keys}")
    if result.unexpected_keys:
        log.warning(f"Unexpected keys (ignored): {result.unexpected_keys}")

    if isinstance(device, str):
        device = torch.device(device)
    return model.to(device).eval()
