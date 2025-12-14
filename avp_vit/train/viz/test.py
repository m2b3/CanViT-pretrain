"""Tests for visualization utilities."""

import numpy as np
import torch

from avp_vit.train.viz import fit_pca, imagenet_denormalize, pca_rgb


class TestPCA:
    def test_fit_pca(self) -> None:
        features = np.random.randn(256, 384).astype(np.float32)
        pca = fit_pca(features)
        assert pca.n_components == 3

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
