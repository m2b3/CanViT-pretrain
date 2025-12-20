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


def test_random_viewpoint_centers_uniform():
    """Centers sampled uniformly, constrained to allow min_scale."""
    B = 10000
    cfg = ViewpointScaleConfig()
    min_scale = cfg.min_area ** 0.5
    max_center = 1 - min_scale
    vp = random_viewpoint(B, torch.device("cpu"), cfg)
    # All centers in valid range
    assert (vp.centers >= -max_center - 1e-6).all()
    assert (vp.centers <= max_center + 1e-6).all()
    # Roughly uniform (mean ≈ 0)
    assert vp.centers.mean().abs() < 0.05


def test_random_viewpoint_fits():
    """Viewpoint always fits: |center| + scale ≤ 1."""
    B = 10000
    cfg = ViewpointScaleConfig()
    vp = random_viewpoint(B, torch.device("cpu"), cfg)
    # For each dim: |center| + scale ≤ 1
    margin = vp.centers.abs() + vp.scales.unsqueeze(1)
    assert (margin <= 1 + 1e-6).all()


def test_random_viewpoint_min_scale():
    """Scale never below sqrt(min_area)."""
    B = 10000
    min_area = 0.01
    cfg = ViewpointScaleConfig(min_area=min_area)
    vp = random_viewpoint(B, torch.device("cpu"), cfg)
    assert (vp.scales >= math.sqrt(min_area) - 1e-6).all()


def test_eval_viewpoints():
    vps = make_eval_viewpoints(4, torch.device("cpu"))
    assert len(vps) == 5  # full + 4 quadrants
    assert vps[0].name == "full"
    assert {vp.name for vp in vps[1:]} == {"TL", "TR", "BL", "BR"}
