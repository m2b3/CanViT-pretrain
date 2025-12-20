"""Tests for viewpoint sampling."""

import math

import torch

from avp_vit.train.viewpoint import ViewpointScaleConfig, make_eval_viewpoints, random_viewpoint


def test_random_viewpoint_shapes():
    B = 8
    cfg = ViewpointScaleConfig()
    vp = random_viewpoint(B, torch.device("cpu"), cfg)
    assert vp.centers.shape == (B, 2)
    assert vp.scales.shape == (B,)


def test_random_viewpoint_bounds():
    """Scale bounds derived from area bounds: scale = sqrt(area)."""
    B = 1000
    min_area, max_area = 0.1, 0.5
    cfg = ViewpointScaleConfig(min_area=min_area, max_area=max_area)
    vp = random_viewpoint(B, torch.device("cpu"), cfg)
    # scale = sqrt(area), so bounds are sqrt(min_area) to sqrt(max_area)
    assert (vp.scales >= math.sqrt(min_area) - 1e-6).all()
    assert (vp.scales <= math.sqrt(max_area) + 1e-6).all()


def test_eval_viewpoints():
    vps = make_eval_viewpoints(4, torch.device("cpu"))
    assert len(vps) == 5  # full + 4 quadrants
    assert vps[0].name == "full"
    assert {vp.name for vp in vps[1:]} == {"TL", "TR", "BL", "BR"}
