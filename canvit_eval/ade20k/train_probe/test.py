"""Tests for ADE20K probe training."""

from pathlib import Path

from canvit_eval.ade20k.train_probe import Config, FeatureType


def test_config_defaults_match_dinov3() -> None:
    """Test that defaults align with DINOv3's linear probing protocol."""
    cfg = Config(model_repo="canvit/model", ade20k_root=Path("/data/ade20k"))
    assert cfg.n_timesteps == 10
    assert cfg.max_steps == 40000  # DINOv3: 40k iterations
    assert cfg.batch_size == 16  # DINOv3: 2 * 8 GPUs
    assert cfg.peak_lr == 1e-3  # DINOv3: 1e-3
    assert cfg.weight_decay == 1e-3  # DINOv3: 1e-3
    assert cfg.dropout == 0.1  # DINOv3: 0.1
    assert cfg.loss_type == "ce"  # DINOv3: cross-entropy
    assert cfg.grad_clip == float("inf")  # DINOv3: no clipping


def test_feature_types() -> None:
    features: list[FeatureType] = ["hidden", "predicted_norm", "teacher_glimpse"]
    assert len(features) == 3
