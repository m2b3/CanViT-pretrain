"""Tests for viewpoint sampling."""

import torch

from avp_vit.train.viewpoint import make_eval_viewpoints, random_viewpoint


def test_random_viewpoint_shapes():
    B = 8
    vp = random_viewpoint(B, torch.device("cpu"))
    assert vp.centers.shape == (B, 2)
    assert vp.scales.shape == (B,)


def test_random_viewpoint_centers_symmetric():
    """Centers are symmetric around origin (mean ≈ 0)."""
    B = 10000
    vp = random_viewpoint(B, torch.device("cpu"))
    assert vp.centers.mean().abs() < 0.05


def test_random_viewpoint_centers_in_safe_box():
    """Each center lies within its scale's safe box: |center| ≤ 1 - scale."""
    B = 10000
    vp = random_viewpoint(B, torch.device("cpu"))
    L = 1 - vp.scales
    assert (vp.centers.abs() <= L.unsqueeze(1) + 1e-6).all()


def test_random_viewpoint_fits():
    """Viewpoint always fits: |center| + scale ≤ 1."""
    B = 10000
    vp = random_viewpoint(B, torch.device("cpu"))
    margin = vp.centers.abs() + vp.scales.unsqueeze(1)
    assert (margin <= 1 + 1e-6).all()


def test_random_viewpoint_min_scale():
    """Scale respects min_scale constraint."""
    B = 10000
    min_scale = 0.25
    vp = random_viewpoint(B, torch.device("cpu"), min_scale=min_scale)
    assert (vp.scales >= min_scale - 1e-6).all()
    # Still fits in scene
    margin = vp.centers.abs() + vp.scales.unsqueeze(1)
    assert (margin <= 1 + 1e-6).all()


def test_random_viewpoint_scale_range():
    """Scale respects both min and max constraints."""
    B = 10000
    min_scale, max_scale = 0.2, 0.5
    vp = random_viewpoint(B, torch.device("cpu"), min_scale=min_scale, max_scale=max_scale)
    assert (vp.scales >= min_scale - 1e-6).all()
    assert (vp.scales <= max_scale + 1e-6).all()


def test_eval_viewpoints():
    vps = make_eval_viewpoints(4, torch.device("cpu"))
    assert len(vps) == 5  # full + 4 quadrants
    assert vps[0].name == "full"
    assert {vp.name for vp in vps[1:]} == {"TL", "TR", "BL", "BR"}
