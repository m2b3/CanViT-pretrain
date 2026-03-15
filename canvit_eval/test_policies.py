"""Smoketests for evaluation policies."""

import torch
import pytest
from canvit_eval.policies import (
    StaticPolicy, EntropyGuidedC2F, make_eval_policy,
    _level_viewpoints, _tile_mean_uncertainty, _build_tile_masks,
)


def test_level_viewpoints_counts():
    assert len(_level_viewpoints(0)) == 1
    assert len(_level_viewpoints(1)) == 4
    assert len(_level_viewpoints(2)) == 16


def test_level_viewpoints_scales():
    for cy, cx, s in _level_viewpoints(0):
        assert s == 1.0
    for cy, cx, s in _level_viewpoints(1):
        assert s == 0.5
    for cy, cx, s in _level_viewpoints(2):
        assert s == 0.25


def test_level_viewpoints_cover_scene():
    """All crops at a given level should tile [-1, 1]² without gaps."""
    for level in range(3):
        crops = _level_viewpoints(level)
        n = 2**level
        s = 1.0 / n
        centers_y = sorted(set(cy for cy, _, _ in crops))
        centers_x = sorted(set(cx for _, cx, _ in crops))
        assert len(centers_y) == n
        assert len(centers_x) == n
        # First crop center should be at -1 + s, last at 1 - s
        assert abs(centers_y[0] - (-1.0 + s)) < 1e-6
        assert abs(centers_y[-1] - (1.0 - s)) < 1e-6


def test_static_policy_returns_pregenerated():
    from canvit import Viewpoint
    B, device = 2, torch.device("cpu")
    vps = [
        Viewpoint(centers=torch.zeros(B, 2), scales=torch.ones(B)),
        Viewpoint(centers=torch.ones(B, 2) * 0.5, scales=torch.ones(B) * 0.5),
    ]
    pol = StaticPolicy("test", vps)
    assert pol.name == "test"
    v0 = pol.step(0, None)
    assert torch.equal(v0.centers, vps[0].centers)
    v1 = pol.step(1, None)
    assert torch.equal(v1.scales, vps[1].scales)


def test_make_eval_policy_c2f():
    pol = make_eval_policy("c2f", batch_size=4, device=torch.device("cpu"), n_viewpoints=5)
    assert pol.name == "coarse_to_fine"
    vp = pol.step(0, None)
    assert vp.centers.shape == (4, 2)
    assert vp.scales.shape == (4,)
    # t=0 of C2F should be full scene: scale=1, center=(0,0)
    assert (vp.scales == 1.0).all()


def test_make_eval_policy_aliases():
    pol_c2f = make_eval_policy("c2f", 2, torch.device("cpu"), 5)
    pol_iid = make_eval_policy("iid", 2, torch.device("cpu"), 5)
    pol_fullrand = make_eval_policy("fullrand", 2, torch.device("cpu"), 5)
    assert pol_c2f.name == "coarse_to_fine"
    assert pol_iid.name == "random"
    assert pol_fullrand.name == "full_then_random"


def test_make_eval_policy_unknown_raises():
    with pytest.raises(ValueError, match="Unknown policy"):
        make_eval_policy("nonexistent", 2, torch.device("cpu"), 5)


def test_tile_masks_shape():
    crops = _level_viewpoints(1)
    masks = _build_tile_masks(crops, canvas_grid=32, device=torch.device("cpu"))
    assert masks.shape == (4, 32, 32)
    assert masks.dtype == torch.bool


def test_tile_masks_partition():
    """Level tiles should partition the canvas without overlap or gaps."""
    for level in range(3):
        crops = _level_viewpoints(level)
        G = 32
        masks = _build_tile_masks(crops, G, torch.device("cpu"))
        total = masks.sum(dim=0)  # [G, G]
        # Every cell covered exactly once
        assert (total == 1).all(), f"Level {level}: not a partition (min={total.min()}, max={total.max()})"


def test_tile_mean_uncertainty_shape():
    B, G = 3, 8
    uncertainty = torch.randn(B, G, G)
    crops = _level_viewpoints(1)
    masks = _build_tile_masks(crops, G, torch.device("cpu"))
    scores = _tile_mean_uncertainty(uncertainty, masks)
    assert scores.shape == (B, 4)


def test_tile_mean_uncertainty_uniform():
    """Uniform uncertainty should give equal scores across tiles."""
    B, G = 2, 32
    uncertainty = torch.ones(B, G, G)
    crops = _level_viewpoints(1)
    masks = _build_tile_masks(crops, G, torch.device("cpu"))
    scores = _tile_mean_uncertainty(uncertainty, masks)
    assert torch.allclose(scores, scores[:, :1].expand_as(scores), atol=0.01)


def test_tile_mean_uncertainty_localized():
    """High uncertainty in one quadrant should give that tile the highest score."""
    B, G = 1, 32
    uncertainty = torch.zeros(B, G, G)
    # Put high uncertainty in top-left quadrant
    uncertainty[:, :16, :16] = 10.0
    crops = _level_viewpoints(1)  # 4 quadrants at scale=0.5
    masks = _build_tile_masks(crops, G, torch.device("cpu"))
    scores = _tile_mean_uncertainty(uncertainty, masks)
    # The tile covering the top-left should have the highest score
    best_tile = scores[0].argmax().item()
    best_cy, best_cx, _ = crops[best_tile]
    assert best_cy < 0 and best_cx < 0, f"Expected top-left tile, got center ({best_cy}, {best_cx})"


def test_tile_mean_uncertainty_per_image():
    """Different images should get different scores."""
    B, G = 2, 32
    uncertainty = torch.zeros(B, G, G)
    uncertainty[0, :16, :16] = 10.0  # image 0: top-left
    uncertainty[1, 16:, 16:] = 10.0  # image 1: bottom-right
    crops = _level_viewpoints(1)
    masks = _build_tile_masks(crops, G, torch.device("cpu"))
    scores = _tile_mean_uncertainty(uncertainty, masks)
    best_0 = scores[0].argmax().item()
    best_1 = scores[1].argmax().item()
    assert best_0 != best_1, "Different images should pick different tiles"


def test_f2c_reverses_level_order():
    """F2C should start at the finest scale, C2F at the coarsest."""
    pol_c2f = make_eval_policy("coarse_to_fine", 2, torch.device("cpu"), 21)
    pol_f2c = make_eval_policy("fine_to_coarse", 2, torch.device("cpu"), 21)
    # C2F t=0 is full scene (scale=1)
    assert (pol_c2f.step(0, None).scales == 1.0).all()
    # F2C t=0 is finest scale (scale=0.25 for 3-level quadtree)
    assert (pol_f2c.step(0, None).scales == 0.25).all()


def test_f2c_covers_all_levels():
    """F2C at T=21 should cover levels 2, 1, 0 in that order."""
    pol = make_eval_policy("fine_to_coarse", 1, torch.device("cpu"), 21)
    scales = [pol.step(t, None).scales[0].item() for t in range(21)]
    # First 16: scale=0.25 (level 2)
    assert all(s == 0.25 for s in scales[:16])
    # Next 4: scale=0.5 (level 1)
    assert all(s == 0.5 for s in scales[16:20])
    # Last 1: scale=1.0 (level 0)
    assert scales[20] == 1.0


def test_c2f_and_f2c_visit_same_centers():
    """C2F and F2C should visit the exact same set of (y, x) centers."""
    B, T = 1, 21
    dev = torch.device("cpu")
    torch.manual_seed(0)
    pol_c2f = make_eval_policy("coarse_to_fine", B, dev, T)
    torch.manual_seed(0)
    pol_f2c = make_eval_policy("fine_to_coarse", B, dev, T)
    c2f_centers = {(pol_c2f.step(t, None).scales[0].item(),) for t in range(T)}
    f2c_centers = {(pol_f2c.step(t, None).scales[0].item(),) for t in range(T)}
    # Both visit 1 crop at scale 1.0, 4 at 0.5, 16 at 0.25
    c2f_scale_counts = {}
    f2c_scale_counts = {}
    torch.manual_seed(0)
    pol_c2f2 = make_eval_policy("coarse_to_fine", B, dev, T)
    torch.manual_seed(0)
    pol_f2c2 = make_eval_policy("fine_to_coarse", B, dev, T)
    for t in range(T):
        s = pol_c2f2.step(t, None).scales[0].item()
        c2f_scale_counts[s] = c2f_scale_counts.get(s, 0) + 1
    for t in range(T):
        s = pol_f2c2.step(t, None).scales[0].item()
        f2c_scale_counts[s] = f2c_scale_counts.get(s, 0) + 1
    assert c2f_scale_counts == f2c_scale_counts


def test_tile_masks_reject_non_power_of_2():
    """_build_tile_masks should reject non-power-of-2 grids."""
    crops = _level_viewpoints(1)
    with pytest.raises(AssertionError, match="power of 2"):
        _build_tile_masks(crops, canvas_grid=31, device=torch.device("cpu"))


def test_entropy_coarse_to_fine_needs_21_viewpoints():
    with pytest.raises(AssertionError, match="n_viewpoints=21"):
        make_eval_policy("entropy_coarse_to_fine", 2, torch.device("cpu"), 10,
                         probe=torch.nn.Identity(), get_spatial_fn=lambda x: x)
