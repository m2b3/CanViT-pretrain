"""Checkpoint serialization for CanViTForPretraining models."""

import logging
import os
import socket
import subprocess
import sys
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

import dacite
import torch
from torch import Tensor

from canvit_pretrain import CanViTForPretraining, CanViTForPretrainingConfig

log = logging.getLogger(__name__)

# Old backbone names from pre-refactor checkpoints → new registry names
BACKBONE_NAME_MAP: dict[str, str] = {
    "dinov3_vits16": "canvits16",
    "dinov3_vitb16": "canvitb16",
    "dinov3_vitl16": "canvitl16",
}


class CheckpointData(TypedDict):
    """Checkpoint structure. All fields present; None where not applicable."""

    # --- Model reconstruction (required) ---
    state_dict: dict[str, Tensor]
    model_config: dict
    backbone: str
    grid_sizes: list[int]
    teacher_dim: int
    teacher_repo_id: str

    # --- Training context ---
    glimpse_grid_size: int
    image_resolution: int
    step: int | None
    train_loss: float | None

    # --- Normalizer stats (required for correct inference) ---
    scene_norm_state: dict[str, Tensor] | None
    cls_norm_state: dict[str, Tensor] | None

    # --- Optimizer/scheduler state for resuming training ---
    optimizer_state: dict | None
    scheduler_state: dict | None
    training_config_history: dict[str, dict] | None

    # --- Provenance ---
    timestamp: str
    git_commit: str | None
    git_dirty: bool
    comet_id: str | None
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


def _map_backbone_name(name: str) -> str:
    """Map old backbone names to new registry names for backward compat."""
    mapped = BACKBONE_NAME_MAP.get(name, name)
    if mapped != name:
        log.info(f"Mapped backbone name: {name!r} -> {mapped!r}")
    return mapped


def save(
    path: Path,
    model: CanViTForPretraining,
    backbone: str,
    *,
    teacher_repo_id: str,
    glimpse_grid_size: int,
    image_resolution: int,
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
    assert isinstance(model.cfg, CanViTForPretrainingConfig)
    git_commit, git_dirty = _git_info()
    hostname, slurm_job_id, slurm_array_task_id, cmdline = get_env_metadata()

    data: CheckpointData = {
        "state_dict": model.state_dict(),
        "model_config": asdict(model.cfg),
        "backbone": backbone,
        "grid_sizes": model.grid_sizes,
        "teacher_dim": model.cfg.teacher_dim,
        "teacher_repo_id": teacher_repo_id,
        "glimpse_grid_size": glimpse_grid_size,
        "image_resolution": image_resolution,
        "step": step,
        "train_loss": train_loss,
        "scene_norm_state": scene_norm_state,
        "cls_norm_state": cls_norm_state,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "training_config_history": training_config_history,
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "comet_id": comet_id,
        "hostname": hostname,
        "slurm_job_id": slurm_job_id,
        "slurm_array_task_id": slurm_array_task_id,
        "cmdline": cmdline,
    }

    atomic_torch_save(data, path)
    size_mb = path.stat().st_size / (1024 * 1024)

    log.info(f"Checkpoint saved: {path} ({size_mb:.1f} MB)")
    log.info(
        f"  backbone={backbone}, grid_sizes={model.grid_sizes},"
        f" teacher={teacher_repo_id}, glimpse={glimpse_grid_size}, res={image_resolution}px"
    )
    if step is not None:
        log.info(f"  step={step}, train_loss={train_loss:.4e}" if train_loss else f"  step={step}")
    if git_commit:
        log.info(f"  git={git_commit[:8]}{'*' if git_dirty else ''}")


def load(path: Path, device: torch.device | str = "cpu") -> CheckpointData:
    """Load checkpoint data. All fields are required — no silent fallbacks."""
    log.info(f"Loading checkpoint: {path}")
    raw = torch.load(path, weights_only=False, map_location=device)

    # Required model fields — fail loudly if missing
    for key in ("state_dict", "model_config", "backbone", "grid_sizes", "teacher_dim", "teacher_repo_id"):
        assert key in raw, f"Checkpoint {path.name} missing required field: {key!r}"

    data: CheckpointData = {
        "state_dict": _strip_orig_mod(raw["state_dict"]),
        "model_config": raw["model_config"],
        "backbone": _map_backbone_name(raw["backbone"]),
        "grid_sizes": raw["grid_sizes"],
        "teacher_dim": raw["teacher_dim"],
        "teacher_repo_id": raw["teacher_repo_id"],
        "glimpse_grid_size": raw["glimpse_grid_size"],
        "image_resolution": raw["image_resolution"],
        "step": raw["step"],
        "train_loss": raw["train_loss"],
        "scene_norm_state": raw["scene_norm_state"],
        "cls_norm_state": raw["cls_norm_state"],
        "optimizer_state": raw["optimizer_state"],
        "scheduler_state": raw["scheduler_state"],
        "training_config_history": raw["training_config_history"],
        "timestamp": raw["timestamp"],
        "git_commit": raw["git_commit"],
        "git_dirty": raw["git_dirty"],
        "comet_id": raw["comet_id"],
        "hostname": raw["hostname"],
        "slurm_job_id": raw["slurm_job_id"],
        "slurm_array_task_id": raw["slurm_array_task_id"],
        "cmdline": raw["cmdline"],
    }

    log.info(
        f"  backbone={data['backbone']}, grid_sizes={data['grid_sizes']},"
        f" teacher={data['teacher_repo_id']}, res={data['image_resolution']}px"
    )
    if data["step"] is not None:
        step = data["step"]
        msg = f"  step={step}, train_loss={data['train_loss']:.4e}" if data["train_loss"] else f"  step={step}"
        log.info(msg)
    if data["git_commit"]:
        log.info(f"  git={data['git_commit'][:8]}{'*' if data['git_dirty'] else ''}")

    return data


def load_model(path: Path, device: torch.device | str = "cpu", strict: bool = False) -> CanViTForPretraining:
    """Load CanViTForPretraining from checkpoint."""
    from canvit import create_backbone

    ckpt = load(path, device)

    backbone_name = ckpt["backbone"]
    backbone = create_backbone(backbone_name)

    model_config = ckpt["model_config"]
    if "teacher_dim" not in model_config:
        model_config = {**model_config, "teacher_dim": ckpt["teacher_dim"]}
    cfg = dacite.from_dict(CanViTForPretrainingConfig, model_config)

    model = CanViTForPretraining(
        backbone=backbone,
        cfg=cfg,
        backbone_name=backbone_name,
        grid_sizes=ckpt["grid_sizes"],
    )

    # Strip legacy policy keys and filter shape mismatches
    state_dict = {k: v for k, v in ckpt["state_dict"].items() if not k.startswith("policy.")}
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
