"""Tests for data loading utilities."""

import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms

from avp_vit.train.data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    imagenet_normalize,
    train_transform,
    val_transform,
)


class TestImagenetNormalize:
    def test_returns_normalize_transform(self) -> None:
        norm = imagenet_normalize()
        assert isinstance(norm, transforms.Normalize)

    def test_uses_correct_constants(self) -> None:
        norm = imagenet_normalize()
        assert tuple(norm.mean) == IMAGENET_MEAN
        assert tuple(norm.std) == IMAGENET_STD


class TestTrainTransform:
    def test_output_shape(self) -> None:
        t = train_transform(224, crop_scale=(0.4, 1.0))
        img = Image.new("RGB", (256, 256))
        out = t(img)
        assert isinstance(out, Tensor)
        assert out.shape == (3, 224, 224)

    def test_output_dtype(self) -> None:
        t = train_transform(64, crop_scale=(0.5, 1.0))
        img = Image.new("RGB", (128, 128))
        out = t(img)
        assert isinstance(out, Tensor)
        assert out.dtype == torch.float32


class TestValTransform:
    def test_output_shape(self) -> None:
        t = val_transform(224)
        img = Image.new("RGB", (256, 256))
        out = t(img)
        assert isinstance(out, Tensor)
        assert out.shape == (3, 224, 224)

    def test_deterministic(self) -> None:
        t = val_transform(64)
        img = Image.new("RGB", (128, 128), color=(100, 150, 200))
        out1 = t(img)
        out2 = t(img)
        assert isinstance(out1, Tensor)
        assert isinstance(out2, Tensor)
        assert torch.allclose(out1, out2)
