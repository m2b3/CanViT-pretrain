"""Tests for viewpoint sampling."""

import torch

from avp_vit.train.viewpoint import (
    _quadrants_at_depth,
    make_curriculum_eval_viewpoints,
    make_eval_viewpoints,
    random_viewpoint,
)


class TestRandomViewpoint:
    def test_shapes(self) -> None:
        B = 8
        vp = random_viewpoint(B, torch.device("cpu"), min_scale=0.3, max_scale=1.0)
        assert vp.centers.shape == (B, 2)
        assert vp.scales.shape == (B,)

    def test_scale_bounds(self) -> None:
        B = 100
        vp = random_viewpoint(B, torch.device("cpu"), min_scale=0.3, max_scale=0.7)
        assert (vp.scales >= 0.3).all()
        assert (vp.scales <= 0.7).all()

    def test_center_bounds(self) -> None:
        B = 100
        vp = random_viewpoint(B, torch.device("cpu"), min_scale=0.5, max_scale=0.5)
        # With scale=0.5, max_offset = 0.5, so centers in [-0.5, 0.5]
        assert (vp.centers >= -0.5).all()
        assert (vp.centers <= 0.5).all()

    def test_full_scale_center_is_zero(self) -> None:
        B = 100
        vp = random_viewpoint(B, torch.device("cpu"), min_scale=1.0, max_scale=1.0)
        # With scale=1.0, max_offset = 0, so centers must be 0
        assert (vp.scales == 1.0).all()
        assert (vp.centers == 0.0).all()


class TestMakeEvalViewpoints:
    def test_count(self) -> None:
        vps = make_eval_viewpoints(4, torch.device("cpu"))
        assert len(vps) == 5  # full + 4 quadrants

    def test_first_is_full(self) -> None:
        vps = make_eval_viewpoints(4, torch.device("cpu"))
        assert vps[0].name == "full"
        assert (vps[0].scales == 1.0).all()
        assert (vps[0].centers == 0.0).all()

    def test_quadrants(self) -> None:
        vps = make_eval_viewpoints(4, torch.device("cpu"))
        quadrant_names = {vp.name for vp in vps[1:]}
        assert quadrant_names == {"TL", "TR", "BL", "BR"}

    def test_quadrant_scales(self) -> None:
        vps = make_eval_viewpoints(4, torch.device("cpu"))
        for vp in vps[1:]:
            assert (vp.scales == 0.5).all()


class TestQuadrantsAtDepth:
    def test_depth1_count(self) -> None:
        vps = _quadrants_at_depth(2, 1, torch.device("cpu"))
        assert len(vps) == 4  # 2^1 x 2^1 = 4

    def test_depth2_count(self) -> None:
        vps = _quadrants_at_depth(2, 2, torch.device("cpu"))
        assert len(vps) == 16  # 2^2 x 2^2 = 16

    def test_depth1_scale(self) -> None:
        vps = _quadrants_at_depth(2, 1, torch.device("cpu"))
        for vp in vps:
            assert (vp.scales == 0.5).all()

    def test_depth2_scale(self) -> None:
        vps = _quadrants_at_depth(2, 2, torch.device("cpu"))
        for vp in vps:
            assert (vp.scales == 0.25).all()

    def test_centers_cover_grid(self) -> None:
        vps = _quadrants_at_depth(1, 1, torch.device("cpu"))
        centers = [vp.centers[0].tolist() for vp in vps]
        # depth=1: scale=0.5, n=2, centers at -0.5 and 0.5 on each axis
        expected = {(-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5)}
        actual = {(round(c[0], 1), round(c[1], 1)) for c in centers}
        assert actual == expected


class TestMakeCurriculumEvalViewpoints:
    def test_g16_returns_5(self) -> None:
        vps = make_curriculum_eval_viewpoints(2, G=16, g=7, device=torch.device("cpu"))
        assert len(vps) == 5

    def test_g32_returns_10(self) -> None:
        vps = make_curriculum_eval_viewpoints(2, G=32, g=7, device=torch.device("cpu"))
        assert len(vps) == 10

    def test_g64_returns_20(self) -> None:
        vps = make_curriculum_eval_viewpoints(2, G=64, g=7, device=torch.device("cpu"))
        assert len(vps) == 20

    def test_first_is_full(self) -> None:
        vps = make_curriculum_eval_viewpoints(2, G=32, g=7, device=torch.device("cpu"))
        assert vps[0].name == "full"
        assert (vps[0].scales == 1.0).all()

    def test_shapes(self) -> None:
        B = 4
        vps = make_curriculum_eval_viewpoints(B, G=32, g=7, device=torch.device("cpu"))
        for vp in vps:
            assert vp.centers.shape == (B, 2)
            assert vp.scales.shape == (B,)
