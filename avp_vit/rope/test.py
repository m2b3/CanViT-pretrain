import torch

from . import (
    compute_rope,
    glimpse_positions,
    make_grid_positions,
    make_rope_periods,
    rope_apply,
    rope_apply_with_prefix,
    rope_rotate_half,
)


def test_make_rope_periods_shape():
    periods = make_rope_periods(64, dtype=torch.float32)
    assert periods.shape == (16,)


def test_make_rope_periods_dtype():
    periods = make_rope_periods(64, dtype=torch.bfloat16)
    assert periods.dtype == torch.bfloat16


def test_grid_positions_shape():
    pos = make_grid_positions(7, 7, torch.device("cpu"), dtype=torch.float32)
    assert pos.shape == (49, 2)


def test_grid_positions_range():
    pos = make_grid_positions(7, 7, torch.device("cpu"), dtype=torch.float32)
    assert pos.min() > -1 and pos.max() < 1


def test_glimpse_positions_shape():
    centers = torch.rand(4, 2)
    scales = torch.rand(4)
    pos = glimpse_positions(centers, scales, 7, 7, dtype=torch.float32)
    assert pos.shape == (4, 49, 2)


def test_glimpse_center_zero_scale_one_matches_grid():
    """center=0, scale=1 should produce same positions as make_grid_positions."""
    grid = make_grid_positions(7, 7, torch.device("cpu"), dtype=torch.float32)
    centers = torch.zeros(1, 2)
    scales = torch.ones(1)
    glimpse = glimpse_positions(centers, scales, 7, 7, dtype=torch.float32)
    assert torch.allclose(glimpse[0], grid)


def test_compute_rope_shape():
    positions = torch.randn(2, 49, 2)
    periods = make_rope_periods(64, dtype=torch.float32)
    sin, cos = compute_rope(positions, periods)
    assert sin.shape == (2, 1, 49, 64)
    assert cos.shape == (2, 1, 49, 64)


def test_rope_rotate_half():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    rotated = rope_rotate_half(x)
    expected = torch.tensor([-4.0, -5.0, -6.0, 1.0, 2.0, 3.0])
    assert torch.allclose(rotated, expected)


def test_rope_apply_identity():
    """sin=0, cos=1 should be identity."""
    x = torch.randn(2, 4, 8, 16)
    sin = torch.zeros_like(x)
    cos = torch.ones_like(x)
    out = rope_apply(x, sin, cos)
    assert torch.allclose(out, x)


def test_rope_apply_with_prefix_no_prefix():
    """When rope covers all tokens, behaves like rope_apply."""
    B, heads, N, D = 2, 4, 8, 16
    x = torch.randn(B, heads, N, D)
    sin = torch.randn(B, 1, N, D)
    cos = torch.randn(B, 1, N, D)
    out = rope_apply_with_prefix(x, (sin, cos))
    expected = rope_apply(x, sin, cos)
    assert torch.allclose(out, expected)


def test_rope_apply_with_prefix_preserves_prefix():
    """Prefix tokens should be unchanged."""
    B, heads, N, D = 2, 4, 10, 16
    n_prefix = 2
    n_rope = N - n_prefix
    x = torch.randn(B, heads, N, D)
    sin = torch.randn(B, 1, n_rope, D)
    cos = torch.randn(B, 1, n_rope, D)

    out = rope_apply_with_prefix(x, (sin, cos))

    assert torch.allclose(out[:, :, :n_prefix], x[:, :, :n_prefix])
    expected_rest = rope_apply(x[:, :, n_prefix:], sin, cos)
    assert torch.allclose(out[:, :, n_prefix:], expected_rest)


def test_stitched_quadrant_positions_match_full_grid():
    """4 quadrant positions stitched together should exactly match 14x14 grid.

    This catches center vs corner conventions, off-by-one errors, etc.
    """
    full_grid = make_grid_positions(14, 14, torch.device("cpu"), torch.float32)

    # Quadrants: scale=0.5, centers at (±0.5, ±0.5) in (y, x)
    quadrants = [
        ("TL", -0.5, -0.5, 0, 0),
        ("TR", -0.5, 0.5, 0, 7),
        ("BL", 0.5, -0.5, 7, 0),
        ("BR", 0.5, 0.5, 7, 7),
    ]

    stitched = torch.zeros(14, 14, 2)
    for _, cy, cx, r0, c0 in quadrants:
        centers = torch.tensor([[cy, cx]])
        scales = torch.tensor([0.5])
        pos = glimpse_positions(centers, scales, 7, 7, dtype=torch.float32).squeeze(0)
        stitched[r0 : r0 + 7, c0 : c0 + 7] = pos.reshape(7, 7, 2)

    assert torch.allclose(stitched.reshape(196, 2), full_grid)


def test_different_viewpoints_produce_different_positions():
    """Different center/scale should produce different positions."""
    centers_a = torch.zeros(1, 2)
    scales_a = torch.ones(1)
    pos_a = glimpse_positions(centers_a, scales_a, 7, 7, dtype=torch.float32)

    centers_b = torch.tensor([[0.3, -0.2]])
    scales_b = torch.tensor([0.5])
    pos_b = glimpse_positions(centers_b, scales_b, 7, 7, dtype=torch.float32)

    assert not torch.allclose(pos_a, pos_b)


def test_different_viewpoints_produce_different_rope():
    """Different viewpoints should produce different RoPE sin/cos."""
    periods = make_rope_periods(64, dtype=torch.float32)

    centers_a = torch.zeros(1, 2)
    scales_a = torch.ones(1)
    pos_a = glimpse_positions(centers_a, scales_a, 7, 7, dtype=torch.float32)
    sin_a, cos_a = compute_rope(pos_a, periods)

    centers_b = torch.tensor([[0.3, -0.2]])
    scales_b = torch.tensor([0.5])
    pos_b = glimpse_positions(centers_b, scales_b, 7, 7, dtype=torch.float32)
    sin_b, cos_b = compute_rope(pos_b, periods)

    assert not torch.allclose(sin_a, sin_b)
    assert not torch.allclose(cos_a, cos_b)
