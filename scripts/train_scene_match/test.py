"""Smoke tests for curriculum training module."""

from avp_vit import AVPConfig

from .config import Config
from .data import create_curriculum_stages


def test_config_defaults() -> None:
    """Config defaults are valid."""
    cfg = Config()
    assert cfg.max_grid_size == 64
    assert cfg.warmup_steps == int(cfg.n_steps * cfg.warmup_ratio)
    assert cfg.main_training_steps == cfg.n_steps - cfg.warmup_steps


def test_curriculum_stages_creation() -> None:
    """Can create curriculum stages."""
    cfg = Config(grid_sizes=(16, 32))
    stages = create_curriculum_stages(cfg, patch_size=14)

    assert 16 in stages
    assert 32 in stages
    assert stages[16].scene_grid_size == 16
    assert stages[32].scene_grid_size == 32
    assert stages[16].batch_size > stages[32].batch_size  # Smaller grid = larger batch


def test_config_min_scale() -> None:
    """Min viewpoint scale computed correctly from curriculum stages."""
    cfg = Config(
        avp=AVPConfig(scene_grid_size=64, glimpse_grid_size=7),
        grid_sizes=(16, 32, 64),
    )
    stages = create_curriculum_stages(cfg, patch_size=14)

    # min_scale = glimpse_grid / scene_grid
    assert abs(stages[16].min_viewpoint_scale - 7 / 16) < 1e-6
    assert abs(stages[32].min_viewpoint_scale - 7 / 32) < 1e-6
    assert abs(stages[64].min_viewpoint_scale - 7 / 64) < 1e-6


def test_batch_sizes_scale_quadratically() -> None:
    """Batch sizes scale quadratically with inverse grid size."""
    cfg = Config(batch_size=32, grid_sizes=(16, 32, 64))
    stages = create_curriculum_stages(cfg, patch_size=14)

    # bs_max=32 at G_max=64
    # G=32: 32 * (64/32)² = 128
    # G=16: 32 * (64/16)² = 512
    assert stages[64].batch_size == 32
    assert stages[32].batch_size == 128
    assert stages[16].batch_size == 512
