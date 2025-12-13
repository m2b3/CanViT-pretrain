import torch

from . import (
    glimpse_positions,
    make_grid_positions,
    rope_apply,
    rope_apply_with_prefix,
    rope_rotate_half,
)


def test_grid_positions_shape():
    pos = make_grid_positions(7, 7, torch.device("cpu"))
    assert pos.shape == (49, 2)


def test_grid_positions_range():
    pos = make_grid_positions(7, 7, torch.device("cpu"))
    assert pos.min() > -1 and pos.max() < 1  # strictly within [-1, 1]


def test_glimpse_positions_shape():
    centers = torch.rand(4, 2)
    scales = torch.rand(4)
    pos = glimpse_positions(centers, scales, 7, 7)
    assert pos.shape == (4, 49, 2)


def test_glimpse_center_zero_scale_one_matches_grid():
    """center=0, scale=1 should produce same positions as make_grid_positions."""
    grid = make_grid_positions(7, 7, torch.device("cpu"))
    centers = torch.zeros(1, 2)
    scales = torch.ones(1)
    glimpse = glimpse_positions(centers, scales, 7, 7)
    assert torch.allclose(glimpse[0], grid)


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
