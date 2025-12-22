"""Tests for fresh ratio survival state management."""

import torch

from avp_vit.train.state import SurvivalBatch


def _make_cls_targets(B: int, D: int) -> torch.Tensor:
    """Helper to create CLS targets for tests."""
    return torch.randn(B, D)


class TestSurvivalBatch:
    def test_init_shapes(self) -> None:
        B, C, H, W, D, G = 4, 3, 64, 64, 128, 16
        images = torch.randn(B, C, H, W)
        targets = torch.randn(B, G * G, D)
        cls_targets = _make_cls_targets(B, D)
        canvas = torch.randn(B, G * G, D)
        state = SurvivalBatch.init(images, targets, cls_targets, canvas)
        assert state.images.shape == images.shape
        assert state.targets.shape == targets.shape
        assert state.cls_targets.shape == cls_targets.shape
        assert state.canvas.shape == canvas.shape

    def test_init_permutes(self) -> None:
        """init() should permute to avoid first-step position bias."""
        B, C, H, W, D, G = 8, 3, 64, 64, 128, 16
        # Use distinct values per sample to detect permutation
        images = torch.arange(B).view(B, 1, 1, 1).expand(B, C, H, W).float()
        targets = torch.randn(B, G * G, D)
        cls_targets = _make_cls_targets(B, D)
        canvas = torch.randn(B, G * G, D)

        # Run multiple times - should get different orders
        orders = []
        for _ in range(5):
            state = SurvivalBatch.init(
                images.clone(), targets.clone(), cls_targets.clone(), canvas.clone()
            )
            order = state.images[:, 0, 0, 0].tolist()
            orders.append(tuple(order))

        assert len(set(orders)) > 1, "init should permute randomly"

    def test_init_preserves_alignment(self) -> None:
        """init() permutation must keep images/targets/cls_targets/canvas aligned."""
        B, C, H, W, D, G = 4, 3, 64, 64, 128, 16
        # Tag each sample with its index in all tensors
        images = torch.arange(B).view(B, 1, 1, 1).expand(B, C, H, W).float()
        targets = torch.arange(B).view(B, 1, 1).expand(B, G * G, D).float()
        cls_targets = torch.arange(B).view(B, 1).expand(B, D).float()
        canvas = torch.arange(B).view(B, 1, 1).expand(B, G * G, D).float()

        state = SurvivalBatch.init(images, targets, cls_targets, canvas)

        # All tensors should have same sample ordering
        img_ids = state.images[:, 0, 0, 0]
        tgt_ids = state.targets[:, 0, 0]
        cls_ids = state.cls_targets[:, 0]
        hid_ids = state.canvas[:, 0, 0]
        assert torch.equal(img_ids, tgt_ids)
        assert torch.equal(img_ids, cls_ids)
        assert torch.equal(img_ids, hid_ids)

    def test_step_shapes(self) -> None:
        B, K, C, H, W, D, G = 4, 2, 3, 64, 64, 128, 16
        state = SurvivalBatch.init(
            torch.randn(B, C, H, W),
            torch.randn(B, G * G, D),
            _make_cls_targets(B, D),
            torch.randn(B, G * G, D),
        )

        new_state = state.step(
            fresh_images=torch.randn(K, C, H, W),
            fresh_targets=torch.randn(K, G * G, D),
            fresh_cls_targets=_make_cls_targets(K, D),
            next_canvas=torch.randn(B, G * G, D),
            canvas_init=torch.randn(K, G * G, D),
        )
        assert new_state.images.shape == (B, C, H, W)
        assert new_state.targets.shape == (B, G * G, D)
        assert new_state.cls_targets.shape == (B, D)
        assert new_state.canvas.shape == (B, G * G, D)

    def test_fresh_count_equals_batch_resets_all(self) -> None:
        """K=B means all items are replaced (content matches, order may differ)."""
        B, C, H, W, D, G = 4, 3, 64, 64, 128, 16
        old_images = torch.randn(B, C, H, W)
        old_targets = torch.randn(B, G * G, D)
        state = SurvivalBatch.init(
            old_images, old_targets, _make_cls_targets(B, D), torch.randn(B, G * G, D)
        )

        fresh_images = torch.randn(B, C, H, W)
        fresh_targets = torch.randn(B, G * G, D)
        canvas_init = torch.randn(B, G * G, D)

        new_state = state.step(
            fresh_images=fresh_images,
            fresh_targets=fresh_targets,
            fresh_cls_targets=_make_cls_targets(B, D),
            next_canvas=torch.randn(B, G * G, D),
            canvas_init=canvas_init,
        )
        # All items replaced - check set equality (order may differ due to permutation)
        assert set(new_state.images.view(B, -1).sum(1).tolist()) == set(
            fresh_images.view(B, -1).sum(1).tolist()
        )

    def test_canvas_detached_for_survivors(self) -> None:
        """Surviving canvas states are detached to cut BPTT across optimizer steps."""
        B, K, C, H, W, D, G = 4, 1, 3, 64, 64, 128, 16
        state = SurvivalBatch.init(
            torch.randn(B, C, H, W),
            torch.randn(B, G * G, D),
            _make_cls_targets(B, D),
            torch.randn(B, G * G, D),
        )
        next_canvas = torch.randn(B, G * G, D, requires_grad=True)
        canvas_init = torch.randn(K, G * G, D, requires_grad=False)

        new_state = state.step(
            fresh_images=torch.randn(K, C, H, W),
            fresh_targets=torch.randn(K, G * G, D),
            fresh_cls_targets=_make_cls_targets(K, D),
            next_canvas=next_canvas,
            canvas_init=canvas_init,
        )
        # Result should not require grad (survivors detached, fresh was already detached)
        assert not new_state.canvas.requires_grad

    def test_canvas_init_grad_preserved(self) -> None:
        """canvas_init gradients should flow through (learnable spatial_canvas_init)."""
        B, K, C, H, W, D, G = 4, 2, 3, 64, 64, 128, 16
        state = SurvivalBatch.init(
            torch.randn(B, C, H, W),
            torch.randn(B, G * G, D),
            _make_cls_targets(B, D),
            torch.randn(B, G * G, D),
        )
        # Simulate learnable canvas_init (like spatial_canvas_init.expand())
        canvas_init = torch.randn(K, G * G, D, requires_grad=True)

        new_state = state.step(
            fresh_images=torch.randn(K, C, H, W),
            fresh_targets=torch.randn(K, G * G, D),
            fresh_cls_targets=_make_cls_targets(K, D),
            next_canvas=torch.randn(B, G * G, D),
            canvas_init=canvas_init,
        )
        # Should require grad because canvas_init does
        assert new_state.canvas.requires_grad

    def test_step_preserves_alignment(self) -> None:
        """step() must keep images/targets/cls_targets/canvas aligned after permutation."""
        B, K, C, H, W, D, G = 8, 2, 3, 64, 64, 128, 16

        # Tag survivors with indices 100+i, fresh with indices 0..K-1
        old_images = (100 + torch.arange(B)).view(B, 1, 1, 1).expand(B, C, H, W).float()
        old_targets = (100 + torch.arange(B)).view(B, 1, 1).expand(B, G * G, D).float()
        old_cls_targets = (100 + torch.arange(B)).view(B, 1).expand(B, D).float()
        old_canvas = (100 + torch.arange(B)).view(B, 1, 1).expand(B, G * G, D).float()
        state = SurvivalBatch(
            images=old_images,
            targets=old_targets,
            cls_targets=old_cls_targets,
            canvas=old_canvas,
        )

        fresh_images = torch.arange(K).view(K, 1, 1, 1).expand(K, C, H, W).float()
        fresh_targets = torch.arange(K).view(K, 1, 1).expand(K, G * G, D).float()
        fresh_cls_targets = torch.arange(K).view(K, 1).expand(K, D).float()
        canvas_init = torch.arange(K).view(K, 1, 1).expand(K, G * G, D).float()
        # Survivor canvas should use next_canvas[K:], tag those
        next_canvas = (200 + torch.arange(B)).view(B, 1, 1).expand(B, G * G, D).float()

        new_state = state.step(
            fresh_images=fresh_images,
            fresh_targets=fresh_targets,
            fresh_cls_targets=fresh_cls_targets,
            next_canvas=next_canvas,
            canvas_init=canvas_init,
        )

        # Extract IDs from each tensor
        img_ids = new_state.images[:, 0, 0, 0]
        tgt_ids = new_state.targets[:, 0, 0]
        cls_ids = new_state.cls_targets[:, 0]
        hid_ids = new_state.canvas[:, 0, 0]

        # images, targets, and cls_targets must be aligned
        assert torch.equal(img_ids, tgt_ids), "images/targets misaligned"
        assert torch.equal(img_ids, cls_ids), "images/cls_targets misaligned"

        # canvas alignment is trickier:
        # - fresh samples: img_id in [0, K), hid_id = img_id (from canvas_init)
        # - survivors: img_id in [100+K, 100+B), hid_id = 200 + (img_id - 100) from next_canvas[K:]
        for i in range(B):
            img_id = img_ids[i].item()
            hid_id = hid_ids[i].item()
            if img_id < 100:
                # Fresh sample
                assert hid_id == img_id, f"Fresh sample {i}: img={img_id}, hid={hid_id}"
            else:
                # Survivor: canvas comes from next_canvas, offset by 100
                expected_hid = 200 + (img_id - 100)
                assert hid_id == expected_hid, (
                    f"Survivor {i}: img={img_id}, hid={hid_id}, expected={expected_hid}"
                )

    def test_fresh_not_at_deterministic_positions(self) -> None:
        """CRITICAL: Fresh samples must NOT always be at indices [0:K].

        This was the original footgun - viz always showed sample 0, which was always fresh.
        """
        B, K, C, H, W, D, G = 8, 2, 3, 64, 64, 128, 16

        # Mark fresh with -1, survivors with their original index
        old_images = torch.arange(B).view(B, 1, 1, 1).expand(B, C, H, W).float()
        state = SurvivalBatch(
            images=old_images,
            targets=torch.randn(B, G * G, D),
            cls_targets=_make_cls_targets(B, D),
            canvas=torch.randn(B, G * G, D),
        )

        fresh_marker = -1.0
        fresh_images = torch.full((K, C, H, W), fresh_marker)

        # Run multiple steps, track which index has fresh samples
        fresh_at_idx0 = 0
        n_trials = 20
        for _ in range(n_trials):
            new_state = state.step(
                fresh_images=fresh_images,
                fresh_targets=torch.randn(K, G * G, D),
                fresh_cls_targets=_make_cls_targets(K, D),
                next_canvas=torch.randn(B, G * G, D),
                canvas_init=torch.randn(K, G * G, D),
            )
            if new_state.images[0, 0, 0, 0].item() == fresh_marker:
                fresh_at_idx0 += 1

        # Fresh should NOT always be at index 0 (would be n_trials if footgun present)
        # With random permutation, P(fresh at idx 0) = K/B = 2/8 = 0.25
        # So we expect ~5 out of 20, definitely not 20
        assert fresh_at_idx0 < n_trials, (
            f"Fresh always at index 0 ({fresh_at_idx0}/{n_trials}) - footgun not fixed!"
        )
        # Also shouldn't be 0 (would indicate broken permutation)
        # With 20 trials and p=0.25, P(zero successes) = 0.75^20 ≈ 0.003
        assert fresh_at_idx0 > 0, "Fresh never at index 0 - permutation may be broken"

    def test_survivor_sometimes_at_idx0(self) -> None:
        """Survivors should sometimes appear at index 0 (for viz to show carried-over canvas)."""
        B, K, C, H, W, D, G = 8, 2, 3, 64, 64, 128, 16

        survivor_marker = 999.0
        old_images = torch.full((B, C, H, W), survivor_marker)
        state = SurvivalBatch(
            images=old_images,
            targets=torch.randn(B, G * G, D),
            cls_targets=_make_cls_targets(B, D),
            canvas=torch.randn(B, G * G, D),
        )

        fresh_images = torch.zeros(K, C, H, W)  # Fresh = 0, survivor = 999

        survivor_at_idx0 = 0
        n_trials = 20
        for _ in range(n_trials):
            new_state = state.step(
                fresh_images=fresh_images,
                fresh_targets=torch.randn(K, G * G, D),
                fresh_cls_targets=_make_cls_targets(K, D),
                next_canvas=torch.randn(B, G * G, D),
                canvas_init=torch.randn(K, G * G, D),
            )
            if new_state.images[0, 0, 0, 0].item() == survivor_marker:
                survivor_at_idx0 += 1

        # Should have survivors at index 0 sometimes
        # P(survivor at idx 0) = (B-K)/B = 6/8 = 0.75
        assert survivor_at_idx0 > 0, "Survivor never at index 0"
        assert survivor_at_idx0 < n_trials, "Survivor always at index 0"

    def test_correct_survivor_count(self) -> None:
        """Each step should have exactly B-K survivors and K fresh samples."""
        B, K, C, H, W, D, G = 8, 2, 3, 2, 2, 4, 2

        survivor_marker = 100.0
        fresh_marker = -1.0

        state = SurvivalBatch(
            images=torch.full((B, C, H, W), survivor_marker),
            targets=torch.randn(B, G * G, D),
            cls_targets=_make_cls_targets(B, D),
            canvas=torch.randn(B, G * G, D),
        )

        new_state = state.step(
            fresh_images=torch.full((K, C, H, W), fresh_marker),
            fresh_targets=torch.randn(K, G * G, D),
            fresh_cls_targets=_make_cls_targets(K, D),
            next_canvas=torch.randn(B, G * G, D),
            canvas_init=torch.randn(K, G * G, D),
        )

        n_fresh = (new_state.images[:, 0, 0, 0] == fresh_marker).sum().item()
        n_survivors = (new_state.images[:, 0, 0, 0] == survivor_marker).sum().item()

        assert n_fresh == K, f"Expected {K} fresh, got {n_fresh}"
        assert n_survivors == B - K, f"Expected {B-K} survivors, got {n_survivors}"

    def test_shape_mismatch_canvas_raises(self) -> None:
        """Catch shape mismatch between next_canvas and canvas_init."""
        B, K, C, H, W, D, G = 4, 2, 3, 64, 64, 128, 16
        N_REGISTERS = 42
        state = SurvivalBatch.init(
            torch.randn(B, C, H, W),
            torch.randn(B, G * G, D),
            _make_cls_targets(B, D),
            torch.randn(B, N_REGISTERS + G * G, D),
        )

        # Matching shapes - should work
        new_state = state.step(
            fresh_images=torch.randn(K, C, H, W),
            fresh_targets=torch.randn(K, G * G, D),
            fresh_cls_targets=_make_cls_targets(K, D),
            next_canvas=torch.randn(B, N_REGISTERS + G * G, D),
            canvas_init=torch.randn(K, N_REGISTERS + G * G, D),
        )
        assert new_state.canvas.shape == (B, N_REGISTERS + G * G, D)

        # Mismatched - should fail
        try:
            state.step(
                fresh_images=torch.randn(K, C, H, W),
                fresh_targets=torch.randn(K, G * G, D),
                fresh_cls_targets=_make_cls_targets(K, D),
                next_canvas=torch.randn(B, N_REGISTERS + G * G, D),
                canvas_init=torch.randn(K, G * G, D),  # Wrong shape!
            )
            assert False, "Should have raised"
        except RuntimeError:
            pass
