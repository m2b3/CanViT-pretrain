"""Tests for IN1k evaluation."""

from pathlib import Path

from canvit_eval.in1k.evaluate import Config, PolicyName


def test_config_requires_paths() -> None:
    """Config requires val_dir, model_repo, probe_repo."""
    cfg = Config(
        val_dir=Path("/tmp/val"),
        model_repo="canvit/model",
        probe_repo="user/probe",
    )
    assert cfg.n_viewpoints == 16
    assert cfg.canvas_grid == 32


def test_policy_names() -> None:
    policies: list[PolicyName] = ["coarse_to_fine", "random"]
    assert len(policies) == 2
