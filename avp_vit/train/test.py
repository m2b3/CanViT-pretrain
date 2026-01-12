"""Tests for avp_vit.train module."""

import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms

from avp_vit.train.data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    MAX_CONSECUTIVE_FAILURES,
    InfiniteLoader,
    imagenet_normalize,
    train_transform,
    val_transform,
)
from avp_vit.train.norm import PositionAwareNorm
from avp_vit.train.scheduler import warmup_cosine_scheduler
from avp_vit.train.viewpoint import (
    PixelBox,
    random_viewpoint,
    make_eval_viewpoints,
    viewpoint_to_pixel_box,
)
from avp_vit.train.viz import imagenet_denormalize, timestep_colors


# === Data Tests ===

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
        assert isinstance(out1, Tensor) and isinstance(out2, Tensor)
        assert torch.allclose(out1, out2)


class _MockLoader:
    """Mock DataLoader that yields batches, optionally with errors."""

    def __init__(self, batches: list, error_at: int | None = None, error_type: type = ValueError):
        self._batches = batches
        self._error_at = error_at
        self._error_type = error_type

    def __iter__(self) -> "_MockLoaderIter":
        return _MockLoaderIter(self._batches, self._error_at, self._error_type)


class _MockLoaderIter:
    def __init__(self, batches: list, error_at: int | None, error_type: type):
        self._batches = batches
        self._error_at = error_at
        self._error_type = error_type
        self._idx = 0

    def __iter__(self) -> "_MockLoaderIter":
        return self

    def __next__(self) -> tuple[Tensor, Tensor]:
        if self._error_at is not None and self._idx == self._error_at:
            self._idx += 1
            raise self._error_type("simulated worker error")
        if self._idx >= len(self._batches):
            raise StopIteration
        batch = self._batches[self._idx]
        self._idx += 1
        return batch


class TestInfiniteLoader:
    def test_yields_batches(self) -> None:
        batches = [(torch.tensor([i]), torch.tensor([i])) for i in range(3)]
        loader = InfiniteLoader(_MockLoader(batches))  # type: ignore[arg-type]
        for i in range(3):
            imgs, labels = loader.next_batch_with_labels()
            assert imgs.item() == i
            assert labels.item() == i

    def test_wraps_around_on_epoch_end(self) -> None:
        batches = [(torch.tensor([i]), torch.tensor([i])) for i in range(2)]
        loader = InfiniteLoader(_MockLoader(batches))  # type: ignore[arg-type]
        # Get 5 batches - should wrap around
        results = [loader.next_batch_with_labels()[0].item() for _ in range(5)]
        assert results == [0, 1, 0, 1, 0]

    def test_recovers_from_single_error(self) -> None:
        batches = [(torch.tensor([i]), torch.tensor([i])) for i in range(3)]
        loader = InfiniteLoader(_MockLoader(batches, error_at=1))  # type: ignore[arg-type]
        # First batch OK
        assert loader.next_batch_with_labels()[0].item() == 0
        # Second batch errors, should retry and get batch from new epoch
        imgs, _ = loader.next_batch_with_labels()
        assert imgs.item() == 0  # New epoch starts from 0

    def test_raises_after_max_consecutive_failures(self) -> None:
        class AlwaysFailLoader:
            def __iter__(self) -> "AlwaysFailIter":
                return AlwaysFailIter()

        class AlwaysFailIter:
            def __iter__(self) -> "AlwaysFailIter":
                return self

            def __next__(self) -> tuple[Tensor, Tensor]:
                raise ValueError("always fails")

        loader = InfiniteLoader(AlwaysFailLoader())  # type: ignore[arg-type]
        import pytest
        with pytest.raises(RuntimeError, match=f"{MAX_CONSECUTIVE_FAILURES} consecutive"):
            loader.next_batch_with_labels()

    def test_next_batch_discards_labels(self) -> None:
        batches = [(torch.tensor([1, 2]), torch.tensor([99]))]
        loader = InfiniteLoader(_MockLoader(batches))  # type: ignore[arg-type]
        imgs = loader.next_batch()
        assert torch.equal(imgs, torch.tensor([1, 2]))


# === Norm Tests ===

class TestPositionAwareNorm:
    def test_basic_shapes(self) -> None:
        norm = PositionAwareNorm(n_tokens=16, embed_dim=32, grid_size=4)
        # Must set stats before forward
        data = torch.randn(100, 16, 32)
        norm.set_stats(data)
        x = torch.randn(2, 16, 32)
        out = norm(x)
        assert out.shape == x.shape

    def test_set_stats_initializes(self) -> None:
        norm = PositionAwareNorm(n_tokens=4, embed_dim=8, grid_size=2)
        assert not norm.initialized
        data = torch.randn(10, 4, 8)
        norm.set_stats(data)
        assert norm.initialized

    def test_denormalize_inverts(self) -> None:
        norm = PositionAwareNorm(n_tokens=4, embed_dim=8, grid_size=2)
        data = torch.randn(100, 4, 8)
        norm.set_stats(data)
        x = torch.randn(2, 4, 8)
        normalized = norm(x)
        recovered = norm.denormalize(normalized)
        # Should be exact inversion with fixed stats
        assert torch.allclose(recovered, x, atol=1e-5)


# === Scheduler Tests ===

class TestWarmupCosineScheduler:
    def test_warmup_with_explicit_lr(self) -> None:
        peak_lr = 1e-3
        start_lr = 1e-4
        end_lr = 1e-5
        optimizer = torch.optim.Adam([torch.zeros(1, requires_grad=True)], lr=peak_lr)
        scheduler = warmup_cosine_scheduler(
            optimizer, total_steps=100, warmup_steps=10, peak_lr=peak_lr,
            start_lr=start_lr, end_lr=end_lr,
        )
        # At step 0, lr = start_lr
        assert abs(scheduler.get_last_lr()[0] - start_lr) < 1e-6
        # Warmup
        for _ in range(10):
            scheduler.step()
        # After warmup, should be at peak
        assert abs(scheduler.get_last_lr()[0] - peak_lr) < 1e-5

    def test_warmup_with_none_defaults(self) -> None:
        peak_lr = 1e-3
        warmup_steps = 10
        optimizer = torch.optim.Adam([torch.zeros(1, requires_grad=True)], lr=peak_lr)
        scheduler = warmup_cosine_scheduler(
            optimizer, total_steps=100, warmup_steps=warmup_steps, peak_lr=peak_lr,
        )
        # At step 0, lr = peak_lr / warmup_steps (old behavior)
        expected_start = peak_lr / warmup_steps
        assert abs(scheduler.get_last_lr()[0] - expected_start) < 1e-7
        # Warmup to peak
        for _ in range(warmup_steps):
            scheduler.step()
        assert abs(scheduler.get_last_lr()[0] - peak_lr) < 1e-5
        # Decay to 0
        for _ in range(90):
            scheduler.step()
        assert scheduler.get_last_lr()[0] < 1e-6


# === Viewpoint Tests ===

class TestRandomViewpoint:
    def test_basic(self) -> None:
        vp = random_viewpoint(4, torch.device("cpu"))
        assert vp.centers.shape == (4, 2)
        assert vp.scales.shape == (4,)
        assert vp.name == "random"

    def test_scale_bounds(self) -> None:
        vp = random_viewpoint(100, torch.device("cpu"), min_scale=0.3, max_scale=0.7)
        assert (vp.scales >= 0.3).all()
        assert (vp.scales <= 0.7).all()


class TestMakeEvalViewpoints:
    def test_default_returns_10_viewpoints(self) -> None:
        vps = make_eval_viewpoints(2, torch.device("cpu"))
        assert len(vps) == 10
        assert vps[0].centers.shape == (2, 2)
        assert vps[0].scales.shape == (2,)

    def test_explicit_n_viewpoints(self) -> None:
        vps = make_eval_viewpoints(2, torch.device("cpu"), n_viewpoints=5)
        assert len(vps) == 5

    def test_different_ordering_per_batch_item(self) -> None:
        vps = make_eval_viewpoints(4, torch.device("cpu"), n_viewpoints=5)
        # t0 is always full view (same for all batch items)
        assert vps[0].name == "full"
        assert (vps[0].scales == 1.0).all()
        assert (vps[0].centers == 0.0).all()
        # t1-t4 are quadrants (scale=0.5) but different order per batch item
        for t in range(1, 5):
            assert (vps[t].scales == 0.5).all()
        # Different batch items should see different quadrant ordering
        all_same = all(
            torch.allclose(vps[t].centers[0], vps[t].centers[i])
            for t in range(1, 5)
            for i in range(1, 4)
        )
        assert not all_same, "All batch items have same quadrant order (should be random)"


class TestPixelBox:
    def test_viewpoint_to_pixel_box(self) -> None:
        # Pixel center convention: normalized [-1, 1] maps to pixel [0, W-1]
        centers = torch.tensor([[0.0, 0.0]])
        scales = torch.tensor([0.5])
        box = viewpoint_to_pixel_box(centers, scales, 0, H=100, W=100)
        assert isinstance(box, PixelBox)
        # center at norm (0,0) -> pixel (99/2, 99/2) = (49.5, 49.5)
        assert box.center_x == 49.5
        assert box.center_y == 49.5
        # scale 0.5 -> width/height = 0.5 * 99 = 49.5
        assert box.width == 49.5
        assert box.height == 49.5

    def test_full_image_box(self) -> None:
        # scale=1.0 should cover [0, W-1] and [0, H-1]
        centers = torch.tensor([[0.0, 0.0]])
        scales = torch.tensor([1.0])
        box = viewpoint_to_pixel_box(centers, scales, 0, H=100, W=100)
        assert box.left == 0.0
        assert box.top == 0.0
        # right = left + width = 0 + 99 = 99 (last pixel)
        assert box.left + box.width == 99.0
        assert box.top + box.height == 99.0


# === Viz Tests ===

class TestImagenetDenormalize:
    def test_output_range(self) -> None:
        # Input is [3, H, W], output is [H, W, 3]
        x = torch.zeros(3, 4, 4)
        out = imagenet_denormalize(x)
        assert out.shape == (4, 4, 3)


class TestTimestepColors:
    def test_returns_correct_count(self) -> None:
        colors = timestep_colors(5)
        assert len(colors) == 5

    def test_rgba_format(self) -> None:
        colors = timestep_colors(3)
        for c in colors:
            assert len(c) == 4  # RGBA


