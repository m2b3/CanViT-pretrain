"""Tests for checkpoint serialization."""

import tempfile
from pathlib import Path

import torch
from canvit import create_backbone

from canvit_pretrain import CanViTForPretraining, CanViTForPretrainingConfig
from canvit_pretrain.checkpoint import CheckpointData, load, save

_TEACHER_REPO = "facebook/dinov3-vits16-pretrain"


def _make_tiny_model(device: torch.device) -> CanViTForPretraining:
    """Create minimal CanViTForPretraining for testing (no pretrained weights needed)."""
    backbone = create_backbone("canvits16").to(device)
    cfg = CanViTForPretrainingConfig(teacher_dim=384)
    return CanViTForPretraining(
        backbone=backbone,
        cfg=cfg,
        backbone_name="canvits16",
        grid_sizes=[8, 16, 32],
    ).to(device)


def test_save_load_roundtrip() -> None:
    device = torch.device("cpu")
    model = _make_tiny_model(device)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.pt"
        save(
            path, model, backbone="canvits16",
            teacher_repo_id=_TEACHER_REPO, glimpse_grid_size=8, image_resolution=512,
            step=100, train_loss=0.5,
        )

        data = load(path, device)

        assert data["backbone"] == "canvits16"
        assert data["grid_sizes"] == [8, 16, 32]
        assert data["teacher_dim"] == 384
        assert data["teacher_repo_id"] == _TEACHER_REPO
        assert data["glimpse_grid_size"] == 8
        assert data["image_resolution"] == 512
        assert data["step"] == 100
        assert data["train_loss"] == 0.5

        # Verify state_dict loads into fresh model
        model2 = _make_tiny_model(device)
        model2.load_state_dict(data["state_dict"])


def test_strips_orig_mod() -> None:
    """Verify _orig_mod prefix stripping works."""
    device = torch.device("cpu")
    model = _make_tiny_model(device)

    # Manually add _orig_mod prefix to simulate old checkpoint
    state_dict = {k.replace("canvit", "canvit._orig_mod"): v for k, v in model.state_dict().items()}

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.pt"
        raw: CheckpointData = {
            "state_dict": state_dict,
            "model_config": {},
            "backbone": "canvits16",
            "grid_sizes": [8, 16, 32],
            "teacher_dim": 384,
            "teacher_repo_id": _TEACHER_REPO,
            "glimpse_grid_size": 8,
            "image_resolution": 512,
            "step": None,
            "train_loss": None,
            "scene_norm_state": None,
            "cls_norm_state": None,
            "optimizer_state": None,
            "scheduler_state": None,
            "training_config_history": None,
            "timestamp": "test",
            "git_commit": None,
            "git_dirty": False,
            "comet_id": None,
            "hostname": None,
            "slurm_job_id": None,
            "slurm_array_task_id": None,
            "cmdline": None,
        }
        torch.save(raw, path)

        data = load(path, device)
        # Keys should be stripped
        assert not any("_orig_mod" in k for k in data["state_dict"])
