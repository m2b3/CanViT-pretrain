"""Tests for checkpoint serialization."""

import tempfile
from pathlib import Path

import torch

from avp_vit import AVPConfig, AVPViT
from avp_vit.backbone.dinov3 import DINOv3Backbone
from avp_vit.checkpoint import CheckpointData, load, save


def _make_tiny_avp(device: torch.device) -> AVPViT:
    """Create minimal AVP for testing (no pretrained weights needed)."""
    from dinov3.hub.backbones import dinov3_vits16

    backbone = DINOv3Backbone(dinov3_vits16(pretrained=False).to(device))
    cfg = AVPConfig(use_recurrence_ln=True, gating="cheap")
    return AVPViT(backbone, cfg, teacher_dim=384).to(device)


def test_save_load_roundtrip() -> None:
    device = torch.device("cpu")
    avp = _make_tiny_avp(device)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.pt"
        save(path, avp, backbone="dinov3_vits16", step=100, train_loss=0.5)

        data = load(path, device)

        assert data["backbone"] == "dinov3_vits16"
        assert data["teacher_dim"] == 384
        assert data["step"] == 100
        assert data["train_loss"] == 0.5
        assert data["avp_config"]["use_recurrence_ln"] is True
        assert data["avp_config"]["gating"] == "cheap"

        # Verify state_dict loads into fresh model
        avp2 = _make_tiny_avp(device)
        avp2.load_state_dict(data["state_dict"])


def test_strips_orig_mod() -> None:
    """Verify _orig_mod prefix stripping works."""
    device = torch.device("cpu")
    avp = _make_tiny_avp(device)

    # Manually add _orig_mod prefix to simulate old checkpoint
    state_dict = {k.replace("backbone", "backbone._orig_mod"): v for k, v in avp.state_dict().items()}

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.pt"
        raw: CheckpointData = {
            "state_dict": state_dict,
            "avp_config": {"use_recurrence_ln": True, "gating": "cheap"},
            "teacher_dim": 384,
            "backbone": "dinov3_vits16",
            "timestamp": "test",
            "git_commit": None,
            "git_dirty": False,
            "step": None,
            "train_loss": None,
            "comet_id": None,
        }
        torch.save(raw, path)

        data = load(path, device)
        # Keys should be stripped
        assert not any("_orig_mod" in k for k in data["state_dict"])
