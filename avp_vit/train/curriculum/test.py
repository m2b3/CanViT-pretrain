"""Tests for curriculum learning utilities."""

from avp_vit.train.curriculum import (
    CurriculumStage,
    batch_size_for_grid,
    create_curriculum_stage,
    fresh_ratio_for_grid,
    n_eval_viewpoints,
)


class TestBatchSizeForGrid:
    def test_quadratic_scaling(self):
        """Batch size scales quadratically with inverse grid size."""
        bs_max = 32
        G_max = 64

        bs_64 = batch_size_for_grid(64, G_max, bs_max)
        bs_32 = batch_size_for_grid(32, G_max, bs_max)
        bs_16 = batch_size_for_grid(16, G_max, bs_max)

        assert bs_64 == 32  # 32 * (64/64)² = 32
        assert bs_32 == 128  # 32 * (64/32)² = 32 * 4 = 128
        assert bs_16 == 512  # 32 * (64/16)² = 32 * 16 = 512

    def test_minimum_one(self):
        """Batch size is at least 1."""
        assert batch_size_for_grid(128, 64, 1) >= 1


class TestNEvalViewpoints:
    def test_reference_values(self):
        """Check reference values: G=16→5, G=32→10, G=64→20."""
        assert n_eval_viewpoints(16, 7) == 5
        assert n_eval_viewpoints(32, 7) == 10
        assert n_eval_viewpoints(64, 7) == 20

    def test_minimum_two(self):
        """At least 2 viewpoints."""
        assert n_eval_viewpoints(1, 7) >= 2


class TestFreshRatioForGrid:
    def test_formula(self):
        """fresh_ratio = n_viewpoints_per_step / n_eval."""
        n_vp = 2
        # G=16: n_eval=5, ratio=2/5=0.4
        assert abs(fresh_ratio_for_grid(16, 7, n_vp) - 0.4) < 1e-6
        # G=32: n_eval=10, ratio=2/10=0.2
        assert abs(fresh_ratio_for_grid(32, 7, n_vp) - 0.2) < 1e-6
        # G=64: n_eval=20, ratio=2/20=0.1
        assert abs(fresh_ratio_for_grid(64, 7, n_vp) - 0.1) < 1e-6


class TestCurriculumStage:
    def test_computed_properties(self):
        """Test that computed properties work correctly."""
        stage = CurriculumStage(
            scene_grid_size=32,
            glimpse_grid_size=7,
            patch_size=14,
            batch_size=128,
            fresh_count=26,
            n_viewpoints_per_step=2,
        )

        assert stage.scene_size_px == 32 * 14
        assert stage.n_scene_tokens == 32 * 32
        assert stage.n_eval == 10
        assert abs(stage.fresh_ratio_intended - 0.2) < 1e-6
        assert abs(stage.fresh_ratio_actual - 26 / 128) < 1e-6
        assert abs(stage.min_viewpoint_scale - 7 / 32) < 1e-6

    def test_expected_glimpses(self):
        """expected_glimpses = n_viewpoints_per_step / fresh_ratio."""
        stage = CurriculumStage(
            scene_grid_size=16,
            glimpse_grid_size=7,
            patch_size=14,
            batch_size=512,
            fresh_count=205,
            n_viewpoints_per_step=2,
        )

        # intended: 2 / 0.4 = 5
        assert abs(stage.expected_glimpses_intended - 5.0) < 1e-6
        # actual: 2 / (205/512) = 2 / 0.4004 ≈ 5.0
        assert abs(stage.expected_glimpses_actual - 5.0) < 0.1


class TestCreateCurriculumStage:
    def test_creates_valid_stage(self):
        """create_curriculum_stage produces valid stages."""
        stage = create_curriculum_stage(
            scene_grid_size=32,
            glimpse_grid_size=7,
            patch_size=14,
            max_grid_size=64,
            max_batch_size=32,
            n_viewpoints_per_step=2,
        )

        assert stage.scene_grid_size == 32
        assert stage.batch_size == 128  # 32 * (64/32)² = 128
        assert stage.fresh_count > 0

    def test_fresh_count_at_least_one(self):
        """Fresh count is at least 1 even for small batches."""
        stage = create_curriculum_stage(
            scene_grid_size=64,
            glimpse_grid_size=7,
            patch_size=14,
            max_grid_size=64,
            max_batch_size=2,  # Very small
            n_viewpoints_per_step=2,
        )

        assert stage.fresh_count >= 1
