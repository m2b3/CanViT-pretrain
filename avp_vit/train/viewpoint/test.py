"""Tests for viewpoint sampling."""

import torch

from avp_vit.train.viewpoint import make_eval_viewpoints, random_viewpoint


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
