"""Smoke tests for scene matching training module."""

from avp_vit import AVPConfig

from .config import Config
from .data import create_resolution_stages


def test_resolution_stages_creation() -> None:
    """Can create resolution stages."""
    cfg = Config(grid_sizes=(16, 32), batch_size=16, batch_size_at_min_grid=64)
    stages = create_resolution_stages(cfg, patch_size=14)

    assert 16 in stages
    assert 32 in stages
    assert stages[16].scene_grid_size == 16
    assert stages[32].scene_grid_size == 32
    assert stages[16].batch_size > stages[32].batch_size  # Smaller grid = larger batch


def test_config_min_scale() -> None:
    """Min viewpoint scale computed correctly from resolution stages."""
    cfg = Config(
        avp=AVPConfig(glimpse_grid_size=7),
        grid_sizes=(16, 32, 64),
    )
    stages = create_resolution_stages(cfg, patch_size=14)

    assert abs(stages[16].min_viewpoint_scale - 7 / 16) < 1e-6
    assert abs(stages[32].min_viewpoint_scale - 7 / 32) < 1e-6
    assert abs(stages[64].min_viewpoint_scale - 7 / 64) < 1e-6


def test_batch_sizes_linear_in_tokens() -> None:
    """Batch sizes interpolate linearly in token count (G²)."""
    cfg = Config(batch_size=16, batch_size_at_min_grid=64, grid_sizes=(16, 32, 64))
    stages = create_resolution_stages(cfg, patch_size=14)

    # Linear in tokens: t = (G² - G_min²) / (G_max² - G_min²)
    # G=32: T=1024, t = (1024-256)/(4096-256) = 768/3840 = 0.2
    # bs = 64 + 0.2*(16-64) = 64 - 9.6 = 54
    assert stages[64].batch_size == 16
    assert stages[16].batch_size == 64
    assert stages[32].batch_size == 54


def test_batch_sizes_default_quadratic_endpoints() -> None:
    """Without batch_size_at_min_grid, endpoints use quadratic formula."""
    cfg = Config(batch_size=32, grid_sizes=(16, 64))
    stages = create_resolution_stages(cfg, patch_size=14)

    # bs_at_min = 32 * (64/16)² = 512
    assert stages[64].batch_size == 32
    assert stages[16].batch_size == 512
