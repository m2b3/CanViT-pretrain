"""Tests for checkpoint serialization."""

import tempfile
from pathlib import Path

import torch

from avp_vit import ActiveCanViT, ActiveCanViTConfig
from canvit.backbone.dinov3 import DINOv3Backbone
from avp_vit.checkpoint import CheckpointData, load, save


def _make_tiny_model(device: torch.device) -> ActiveCanViT:
    """Create minimal ActiveCanViT for testing (no pretrained weights needed)."""
    from dinov3.hub.backbones import dinov3_vits16

    backbone = DINOv3Backbone(dinov3_vits16(pretrained=False).to(device))
    cfg = ActiveCanViTConfig(teacher_dim=384)
    return ActiveCanViT(backbone=backbone, cfg=cfg).to(device)


def test_save_load_roundtrip() -> None:
    device = torch.device("cpu")
    model = _make_tiny_model(device)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.pt"
        save(path, model, backbone="dinov3_vits16", step=100, train_loss=0.5)

        data = load(path, device)

        assert data["backbone"] == "dinov3_vits16"
        assert data["teacher_dim"] == 384
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
            "teacher_dim": 384,
            "backbone": "dinov3_vits16",
            "timestamp": "test",
            "git_commit": None,
            "git_dirty": False,
            "step": None,
            "train_loss": None,
            "comet_id": None,
            "scene_norm_state": None,
            "cls_norm_state": None,
        }
        torch.save(raw, path)

        data = load(path, device)
        # Keys should be stripped
        assert not any("_orig_mod" in k for k in data["state_dict"])
