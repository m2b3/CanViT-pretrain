"""Tests for glimpse extraction - coordinate consistency is CRITICAL."""

import torch

from avp_vit.glimpse import Viewpoint, sample_at_viewpoint, normalized_to_pixel
from avp_vit.rope import glimpse_positions


class TestNormalizedToPixel:
    def test_center_maps_to_center(self) -> None:
        assert normalized_to_pixel(0.0, 100) == 50.0

    def test_left_maps_to_zero(self) -> None:
        assert normalized_to_pixel(-1.0, 100) == 0.0

    def test_right_maps_to_size(self) -> None:
        assert normalized_to_pixel(1.0, 100) == 100.0


class TestPixelBox:
    def test_full_scene_box(self) -> None:
        vp = Viewpoint.full_scene(1, torch.device("cpu"))
        box = vp.to_pixel_box(0, H=100, W=200)

        # Full scene: center at image center, covers entire image
        assert box.center_x == 100.0  # W/2
        assert box.center_y == 50.0  # H/2
        assert box.width == 200.0  # W
        assert box.height == 100.0  # H
        assert box.left == 0.0
        assert box.top == 0.0

    def test_quadrant_box(self) -> None:
        # Top-left quadrant: center at (-0.5, -0.5), scale=0.5
        vp = Viewpoint.quadrant(1, torch.device("cpu"), 0, 0)
        box = vp.to_pixel_box(0, H=100, W=100)

        # TL quadrant center: pixel (25, 25), size 50x50
        assert box.center_x == 25.0
        assert box.center_y == 25.0
        assert box.width == 50.0
        assert box.height == 50.0
        assert box.left == 0.0
        assert box.top == 0.0

    def test_bottom_right_quadrant(self) -> None:
        # BR quadrant: center at (0.5, 0.5), scale=0.5
        vp = Viewpoint.quadrant(1, torch.device("cpu"), 1, 1)
        box = vp.to_pixel_box(0, H=100, W=100)

        # BR quadrant center: pixel (75, 75), size 50x50
        assert box.center_x == 75.0
        assert box.center_y == 75.0
        assert box.width == 50.0
        assert box.height == 50.0
        assert box.left == 50.0
        assert box.top == 50.0

    def test_batch_indexing(self) -> None:
        # Different viewpoints per batch item
        centers = torch.tensor([[0.0, 0.0], [0.5, 0.5]])
        scales = torch.tensor([1.0, 0.5])
        vp = Viewpoint("mixed", centers, scales)

        box0 = vp.to_pixel_box(0, H=100, W=100)
        box1 = vp.to_pixel_box(1, H=100, W=100)

        # First: full scene
        assert box0.center_x == 50.0
        assert box0.width == 100.0

        # Second: BR quadrant
        assert box1.center_x == 75.0
        assert box1.width == 50.0


def test_viewpoint_full_scene() -> None:
    """Full scene viewpoint factory."""
    vp = Viewpoint.full_scene(4, torch.device("cpu"))
    assert vp.name == "full"
    assert vp.centers.shape == (4, 2)
    assert vp.scales.shape == (4,)
    assert (vp.centers == 0).all()
    assert (vp.scales == 1).all()


def test_viewpoint_quadrants() -> None:
    """Quadrant viewpoints have correct centers and names."""
    device = torch.device("cpu")
    tl = Viewpoint.quadrant(1, device, 0, 0)
    tr = Viewpoint.quadrant(1, device, 1, 0)
    bl = Viewpoint.quadrant(1, device, 0, 1)
    br = Viewpoint.quadrant(1, device, 1, 1)

    assert tl.name == "TL"
    assert tr.name == "TR"
    assert bl.name == "BL"
    assert br.name == "BR"

    # Centers are (y, x) order
    # TL=(y=-0.5, x=-0.5), TR=(y=-0.5, x=0.5), BL=(y=0.5, x=-0.5), BR=(y=0.5, x=0.5)
    assert torch.allclose(tl.centers, torch.tensor([[-0.5, -0.5]]))
    assert torch.allclose(tr.centers, torch.tensor([[-0.5, 0.5]]))
    assert torch.allclose(bl.centers, torch.tensor([[0.5, -0.5]]))
    assert torch.allclose(br.centers, torch.tensor([[0.5, 0.5]]))

    # All have scale=0.5
    assert (tl.scales == 0.5).all()


def test_sample_at_viewpoint_shape() -> None:
    """Output has correct shape."""
    img = torch.randn(4, 3, 224, 224)
    vp = Viewpoint.full_scene(4, torch.device("cpu"))
    out = sample_at_viewpoint(img, vp, out_size=112)
    assert out.shape == (4, 3, 112, 112)


def test_sample_at_viewpoint_br_quadrant() -> None:
    """center=(0.5, 0.5), scale=0.5 extracts bottom-right quadrant."""
    # Create image: BR quadrant = 1, rest = 0
    img = torch.zeros(1, 1, 8, 8)
    img[0, 0, 4:, 4:] = 1.0  # BR only

    vp = Viewpoint.quadrant(1, torch.device("cpu"), 1, 1)  # BR
    out = sample_at_viewpoint(img, vp, out_size=4)

    # Should get mostly 1s from BR quadrant
    assert out.mean() > 0.8, f"Expected high mean from BR quadrant, got {out.mean()}"


def test_sample_at_viewpoint_tl_quadrant() -> None:
    """center=(-0.5, -0.5), scale=0.5 extracts top-left quadrant."""
    # Create image: TL quadrant = 1, rest = 0
    img = torch.zeros(1, 1, 8, 8)
    img[0, 0, :4, :4] = 1.0  # TL only

    vp = Viewpoint.quadrant(1, torch.device("cpu"), 0, 0)  # TL
    out = sample_at_viewpoint(img, vp, out_size=4)

    # Should get mostly 1s from TL quadrant
    assert out.mean() > 0.8, f"Expected high mean from TL quadrant, got {out.mean()}"


def test_coordinate_consistency_with_rope() -> None:
    """CRITICAL: sample_at_viewpoint coordinates match glimpse_positions for RoPE.

    The pixel coordinates sampled by sample_at_viewpoint must correspond to the
    position embeddings computed by glimpse_positions. If these diverge,
    the model will have inconsistent spatial information.

    Both use grid_offsets as the single source of truth, so this test
    verifies the integration is correct.
    """
    B, H, W = 2, 7, 7

    # Create gradient image where pixel value = normalized coordinate
    # Y gradient (value = y coord in [-1, 1])
    img_y = torch.linspace(-1, 1, 112).view(1, 1, 112, 1).expand(B, 1, 112, 112)
    # X gradient (value = x coord in [-1, 1])
    img_x = torch.linspace(-1, 1, 112).view(1, 1, 1, 112).expand(B, 1, 112, 112)

    # Test with a random viewpoint - centers are (y, x) order
    centers = torch.tensor([[0.3, -0.2], [-0.1, 0.4]])
    scales = torch.tensor([0.4, 0.6])
    vp = Viewpoint("test", centers, scales)

    # Get RoPE positions - also (y, x) order
    rope_pos = glimpse_positions(centers, scales, H, W, dtype=torch.float32)
    assert rope_pos.shape == (B, H * W, 2)

    # Extract glimpse from gradient images
    glimpse_y = sample_at_viewpoint(img_y, vp, out_size=H * 16)
    glimpse_x = sample_at_viewpoint(img_x, vp, out_size=W * 16)

    # Sample at patch centers (stride 16, offset 8 for center)
    for b in range(B):
        for i in range(H):
            for j in range(W):
                patch_idx = i * W + j
                # Sample at patch center
                py, px = i * 16 + 8, j * 16 + 8
                sampled_y = glimpse_y[b, 0, py, px].item()
                sampled_x = glimpse_x[b, 0, py, px].item()

                expected_y = rope_pos[b, patch_idx, 0].item()  # y is dim 0
                expected_x = rope_pos[b, patch_idx, 1].item()  # x is dim 1

                # Allow small tolerance for bilinear interpolation
                assert abs(sampled_y - expected_y) < 0.1, (
                    f"Y mismatch at b={b}, patch=({i},{j}): "
                    f"sampled={sampled_y:.3f}, expected={expected_y:.3f}"
                )
                assert abs(sampled_x - expected_x) < 0.1, (
                    f"X mismatch at b={b}, patch=({i},{j}): "
                    f"sampled={sampled_x:.3f}, expected={expected_x:.3f}"
                )


def test_batch_independence() -> None:
    """Each batch item uses its own viewpoint."""
    img = torch.randn(2, 3, 64, 64)
    # Different viewpoints for each batch item
    centers = torch.tensor([[0.0, 0.0], [0.5, 0.5]])
    scales = torch.tensor([1.0, 0.5])
    vp = Viewpoint("mixed", centers, scales)

    out = sample_at_viewpoint(img, vp, out_size=32)
    assert out.shape == (2, 3, 32, 32)


def test_axis_order_asymmetric() -> None:
    """Test with asymmetric viewpoint to catch (x,y) vs (y,x) confusion.

    Uses a horizontal stripe (y=0.5, x=0) to verify y is the first axis.
    If axes were swapped, we'd extract a vertical stripe instead.
    """
    # Create image: top half = 0, bottom half = 1
    img = torch.zeros(1, 1, 8, 8)
    img[0, 0, 4:, :] = 1.0  # bottom half

    # Viewpoint centered on bottom half: y=0.5 (bottom), x=0 (center)
    # centers are (y, x) order
    vp = Viewpoint("bottom_center", torch.tensor([[0.5, 0.0]]), torch.tensor([0.5]))
    out = sample_at_viewpoint(img, vp, out_size=4)

    # Should get mostly 1s (from bottom half)
    assert out.mean() > 0.8, f"Expected mostly bottom half (1s), got mean={out.mean():.2f}"

    # Now test the opposite: top half should give 0s
    vp_top = Viewpoint("top_center", torch.tensor([[-0.5, 0.0]]), torch.tensor([0.5]))
    out_top = sample_at_viewpoint(img, vp_top, out_size=4)
    assert out_top.mean() < 0.2, f"Expected mostly top half (0s), got mean={out_top.mean():.2f}"


def test_axis_order_with_rope() -> None:
    """Verify axis order consistency between sample_at_viewpoint and glimpse_positions.

    Uses a non-square grid (3x5) to catch swapped axes. If axes were wrong,
    the position/pixel correspondence would break for non-square grids.
    """
    H, W = 3, 5  # Asymmetric!
    B = 1
    pixel_H, pixel_W = H * 16, W * 16

    # Create gradient image for Y axis (non-square scene)
    img_y = torch.linspace(-1, 1, pixel_H).view(1, 1, pixel_H, 1).expand(B, 1, pixel_H, pixel_W)

    # Asymmetric center to catch axis confusion
    centers = torch.tensor([[0.2, -0.3]])
    scales = torch.tensor([0.5])
    vp = Viewpoint("asymmetric", centers, scales)

    # Use 7x7 grid (square) but with asymmetric viewpoint
    G = 7
    rope_pos = glimpse_positions(centers, scales, G, G, dtype=torch.float32)

    # Extract from the non-square scene
    glimpse = sample_at_viewpoint(img_y.clone(), vp, out_size=G * 16)

    # Verify at a few patch centers
    for i in range(G):
        for j in range(G):
            patch_idx = i * G + j
            py, px = i * 16 + 8, j * 16 + 8
            sampled_y = glimpse[0, 0, py, px].item()
            expected_y = rope_pos[0, patch_idx, 0].item()

            assert abs(sampled_y - expected_y) < 0.15, (
                f"Y mismatch at ({i},{j}): sampled={sampled_y:.3f}, expected={expected_y:.3f}"
            )
