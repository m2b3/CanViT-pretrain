"""Smoke tests for scene matching training module."""

from .config import Config


def test_config_defaults() -> None:
    """Config has sensible defaults."""
    cfg = Config()
    assert cfg.grid_size == 32
    assert cfg.batch_size == 128
    assert cfg.n_viewpoints_per_step >= 1
