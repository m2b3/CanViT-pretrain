"""Tests for visualization utilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from avp_vit.train.viz import (
    PixelBox,
    fit_pca,
    imagenet_denormalize,
    pca_rgb,
    plot_pca_grid,
    plot_trajectory,
    timestep_colors,
)

TEST_OUTPUTS = Path(__file__).parent / "test_outputs"


class TestPCA:
    def test_fit_pca(self) -> None:
        features = np.random.randn(256, 384).astype(np.float32)
        pca = fit_pca(features)
        assert pca is not None
        assert pca.n_components == 12  # default for offset viewing

    def test_pca_rgb_shape(self) -> None:
        features = np.random.randn(256, 384).astype(np.float32)
        pca = fit_pca(features)
        rgb = pca_rgb(pca, features, 16, 16)
        assert rgb.shape == (16, 16, 3)

    def test_pca_rgb_bounds(self) -> None:
        features = np.random.randn(256, 384).astype(np.float32)
        pca = fit_pca(features)
        rgb = pca_rgb(pca, features, 16, 16)
        # Sigmoid output should be in (0, 1)
        assert (rgb >= 0).all()
        assert (rgb <= 1).all()


class TestImagenetDenormalize:
    def test_shape(self) -> None:
        img = torch.randn(3, 64, 64)
        result = imagenet_denormalize(img)
        assert result.shape == (64, 64, 3)

    def test_bounds(self) -> None:
        img = torch.randn(3, 64, 64)
        result = imagenet_denormalize(img)
        assert (result >= 0).all()
        assert (result <= 1).all()

    def test_zero_input(self) -> None:
        img = torch.zeros(3, 64, 64)
        result = imagenet_denormalize(img)
        # Zero normalized -> ImageNet mean
        expected = torch.tensor([0.485, 0.456, 0.406])
        assert torch.allclose(result[0, 0], expected, atol=1e-5)

    def test_same_device(self) -> None:
        img = torch.randn(3, 32, 32)
        result = imagenet_denormalize(img)
        assert result.device == img.device


class TestTimestepColors:
    def test_returns_correct_count(self) -> None:
        colors = timestep_colors(5)
        assert len(colors) == 5

    def test_single_color(self) -> None:
        colors = timestep_colors(1)
        assert len(colors) == 1

    def test_rgba_tuples(self) -> None:
        colors = timestep_colors(3)
        for c in colors:
            assert len(c) == 4  # RGBA


class TestPlotTrajectory:
    def test_returns_figure(self) -> None:
        img = np.random.rand(64, 64, 3).astype(np.float32)
        boxes = [
            PixelBox(left=0, top=0, width=64, height=64, center_x=32, center_y=32),
            PixelBox(left=16, top=16, width=32, height=32, center_x=32, center_y=32),
        ]
        names = ["full", "center"]
        fig = plot_trajectory(img, boxes, names)
        assert isinstance(fig, Figure)

    def test_empty_boxes(self) -> None:
        img = np.random.rand(64, 64, 3).astype(np.float32)
        fig = plot_trajectory(img, [], [])
        assert isinstance(fig, Figure)


class TestPlotPcaGrid:
    def test_returns_figure(self) -> None:
        reference = np.random.randn(16, 64).astype(np.float32)
        samples = [np.random.randn(16, 64).astype(np.float32) for _ in range(3)]
        pca = fit_pca(reference)
        titles = ["t=0", "t=1", "t=2"]
        fig = plot_pca_grid(pca, reference, samples, grid_size=4, titles=titles)
        assert isinstance(fig, Figure)

    def test_single_sample(self) -> None:
        reference = np.random.randn(16, 64).astype(np.float32)
        samples = [np.random.randn(16, 64).astype(np.float32)]
        pca = fit_pca(reference)
        fig = plot_pca_grid(pca, reference, samples, grid_size=4, titles=["t=0"])
        assert isinstance(fig, Figure)


# --- Visual smoke tests that save example outputs ---


def _make_gradient_image(H: int, W: int) -> np.ndarray:
    """Create a gradient image useful for visual verification."""
    y = np.linspace(0, 1, H)[:, None]
    x = np.linspace(0, 1, W)[None, :]
    r = y * np.ones_like(x)
    g = np.ones_like(y) * x
    b = 0.5 * np.ones((H, W))
    return np.stack([r, g, b], axis=-1).astype(np.float32)


class TestVisualSmokeTests:
    """Smoke tests that save example PNGs for visual inspection."""

    def test_trajectory_example(self) -> None:
        """Save example trajectory plot for visual verification.

        Expected output (verify visually):
        - "start" (purple): large box near top-left, wider than tall
        - "t1" (teal): small box upper-left, wider than tall
        - "t2" (green): medium box right side, TALLER than wide
        - "end" (yellow): small box bottom-left, wider than tall
        - White line connects centers: start -> t1 -> t2 -> end
        - Line goes: center -> up-left -> right-down -> left-down
        """
        img = _make_gradient_image(256, 256)
        boxes = [
            PixelBox(left=20, top=10, width=200, height=180, center_x=120, center_y=100),
            PixelBox(left=50, top=30, width=80, height=60, center_x=90, center_y=60),
            PixelBox(left=140, top=80, width=100, height=120, center_x=190, center_y=140),
            PixelBox(left=30, top=160, width=70, height=50, center_x=65, center_y=185),
        ]
        names = ["start", "t1", "t2", "end"]

        fig = plot_trajectory(img, boxes, names)
        TEST_OUTPUTS.mkdir(exist_ok=True)
        fig.savefig(TEST_OUTPUTS / "example_trajectory.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    def test_pca_grid_example(self) -> None:
        """Save example PCA grid plot for visual verification."""
        np.random.seed(42)
        G = 8
        D = 64
        reference = np.random.randn(G * G, D).astype(np.float32)
        # Samples that gradually approach reference
        samples = [
            (reference + np.random.randn(G * G, D) * scale).astype(np.float32)
            for scale in [2.0, 1.0, 0.5, 0.2]
        ]
        pca = fit_pca(reference)
        titles = ["t=0 (far)", "t=1", "t=2", "t=3 (close)"]

        fig = plot_pca_grid(pca, reference, samples, grid_size=G, titles=titles)
        TEST_OUTPUTS.mkdir(exist_ok=True)
        fig.savefig(TEST_OUTPUTS / "example_pca_grid.png", dpi=100, bbox_inches="tight")
        plt.close(fig)
