"""Tests for fresh ratio survival state management."""

import torch

from avp_vit.train.state import TrainState


class TestTrainState:
    def test_init(self) -> None:
        B, C, H, W, D, G = 4, 3, 64, 64, 128, 16
        images = torch.randn(B, C, H, W)
        targets = torch.randn(B, G * G, D)
        hidden_init = torch.randn(B, G * G, D)
        state = TrainState.init(images, targets, hidden_init, None)
        assert state.images is images
        assert state.targets is targets
        assert state.hidden is hidden_init
        assert state.local_prev is None

    def test_step_shapes(self) -> None:
        B, K, C, H, W, D, G = 4, 2, 3, 64, 64, 128, 16
        state = TrainState.init(
            torch.randn(B, C, H, W),
            torch.randn(B, G * G, D),
            torch.randn(B, G * G, D),
            None,
        )

        new_state = state.step(
            fresh_images=torch.randn(K, C, H, W),
            fresh_targets=torch.randn(K, G * G, D),
            next_hidden=torch.randn(B, G * G, D),
            next_local_prev=None,
            hidden_init=torch.randn(K, G * G, D),
            local_init=None,
        )
        assert new_state.images.shape == (B, C, H, W)
        assert new_state.targets.shape == (B, G * G, D)
        assert new_state.hidden is not None
        assert new_state.hidden.shape == (B, G * G, D)
        assert new_state.local_prev is None

    def test_fresh_count_equals_batch_resets_all(self) -> None:
        """K=B means all items are replaced."""
        B, C, H, W, D, G = 4, 3, 64, 64, 128, 16
        old_images = torch.randn(B, C, H, W)
        old_targets = torch.randn(B, G * G, D)
        state = TrainState.init(old_images, old_targets, torch.randn(B, G * G, D), None)

        fresh_images = torch.randn(B, C, H, W)
        fresh_targets = torch.randn(B, G * G, D)
        hidden_init = torch.randn(B, G * G, D)

        new_state = state.step(
            fresh_images, fresh_targets,
            next_hidden=torch.randn(B, G * G, D),
            next_local_prev=None,
            hidden_init=hidden_init,
            local_init=None,
        )
        # All items replaced (though permuted)
        assert torch.equal(new_state.images, fresh_images)
        assert torch.equal(new_state.targets, fresh_targets)
        assert new_state.hidden is not None
        assert torch.equal(new_state.hidden, hidden_init)

    def test_hidden_detached(self) -> None:
        """Surviving hidden states are detached to cut BPTT across optimizer steps."""
        B, K, C, H, W, D, G = 4, 1, 3, 64, 64, 128, 16
        state = TrainState.init(
            torch.randn(B, C, H, W),
            torch.randn(B, G * G, D),
            torch.randn(B, G * G, D),
            None,
        )
        next_hidden = torch.randn(B, G * G, D, requires_grad=True)
        new_state = state.step(
            torch.randn(K, C, H, W),
            torch.randn(K, G * G, D),
            next_hidden,
            next_local_prev=None,
            hidden_init=torch.randn(K, G * G, D),
            local_init=None,
        )
        assert new_state.hidden is not None
        assert not new_state.hidden.requires_grad

    def test_local_prev_handled(self) -> None:
        """local_prev is properly handled when use_local_temporal=True."""
        B, K, C, H, W, D, G, N = 4, 2, 3, 64, 64, 128, 16, 10
        state = TrainState.init(
            torch.randn(B, C, H, W),
            torch.randn(B, G * G, D),
            torch.randn(B, G * G, D),
            torch.randn(B, N, D),
        )

        next_local_prev = torch.randn(B, N, D, requires_grad=True)
        local_init = torch.randn(K, N, D)

        new_state = state.step(
            torch.randn(K, C, H, W),
            torch.randn(K, G * G, D),
            torch.randn(B, G * G, D),
            next_local_prev=next_local_prev,
            hidden_init=torch.randn(K, G * G, D),
            local_init=local_init,
        )
        assert new_state.local_prev is not None
        assert new_state.local_prev.shape == (B, N, D)
        assert not new_state.local_prev.requires_grad  # detached

    def test_permutation_is_random(self) -> None:
        """Different calls produce different permutations."""
        B, K, C, H, W, D, G = 8, 2, 3, 64, 64, 128, 16
        images = torch.arange(B).view(B, 1, 1, 1).expand(B, C, H, W).float()
        state = TrainState.init(
            images,
            torch.randn(B, G * G, D),
            torch.randn(B, G * G, D),
            None,
        )

        results = []
        for _ in range(5):
            new_state = state.step(
                torch.zeros(K, C, H, W),
                torch.randn(K, G * G, D),
                torch.randn(B, G * G, D),
                None,
                torch.randn(K, G * G, D),
                None,
            )
            # Fresh images are zeros, survivors have original index values
            survivor_order = new_state.images[K:, 0, 0, 0].tolist()
            results.append(tuple(survivor_order))

        # Should have some variation (with overwhelming probability)
        assert len(set(results)) > 1

    def test_shape_mismatch_hidden_raises(self) -> None:
        """Catch shape mismatch between next_hidden and hidden_init."""
        B, K, C, H, W, D, G = 4, 2, 3, 64, 64, 128, 16
        N_REGISTERS = 42
        state = TrainState.init(
            torch.randn(B, C, H, W),
            torch.randn(B, G * G, D),
            torch.randn(B, N_REGISTERS + G * G, D),
            None,
        )

        # Matching shapes - should work
        new_state = state.step(
            torch.randn(K, C, H, W),
            torch.randn(K, G * G, D),
            torch.randn(B, N_REGISTERS + G * G, D),
            None,
            torch.randn(K, N_REGISTERS + G * G, D),
            None,
        )
        assert new_state.hidden is not None
        assert new_state.hidden.shape == (B, N_REGISTERS + G * G, D)

        # Mismatched - should fail
        try:
            state.step(
                torch.randn(K, C, H, W),
                torch.randn(K, G * G, D),
                torch.randn(B, N_REGISTERS + G * G, D),
                None,
                torch.randn(K, G * G, D),  # Wrong shape!
                None,
            )
            assert False, "Should have raised"
        except RuntimeError:
            pass
