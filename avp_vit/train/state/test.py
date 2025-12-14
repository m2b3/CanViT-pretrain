"""Tests for Bernoulli survival state management."""

import torch

from avp_vit.train.state import TrainState


class TestTrainState:
    def test_init(self) -> None:
        B, C, H, W, D, G = 4, 3, 64, 64, 128, 16
        images = torch.randn(B, C, H, W)
        targets = torch.randn(B, G * G, D)
        state = TrainState.init(images, targets)
        assert state.hidden is None
        assert state.local_prev is None
        assert state.images is images
        assert state.targets is targets

    def test_step_shapes(self) -> None:
        B, C, H, W, D, G = 4, 3, 64, 64, 128, 16
        state = TrainState.init(
            torch.randn(B, C, H, W),
            torch.randn(B, G * G, D),
        )

        new_state = state.step(
            fresh_images=torch.randn(B, C, H, W),
            fresh_targets=torch.randn(B, G * G, D),
            next_hidden=torch.randn(B, G * G, D),
            next_local_prev=None,
            survival_prob=0.5,
            hidden_tokens=torch.randn(1, G * G, D),
            local_tokens=None,
        )
        assert new_state.images.shape == (B, C, H, W)
        assert new_state.targets.shape == (B, G * G, D)
        assert new_state.hidden is not None
        assert new_state.hidden.shape == (B, G * G, D)
        assert new_state.local_prev is None

    def test_survival_zero_resets_all(self) -> None:
        B, C, H, W, D, G = 4, 3, 64, 64, 128, 16
        old_images = torch.randn(B, C, H, W)
        old_targets = torch.randn(B, G * G, D)
        state = TrainState.init(old_images, old_targets)

        fresh_images = torch.randn(B, C, H, W)
        fresh_targets = torch.randn(B, G * G, D)
        next_hidden = torch.randn(B, G * G, D)
        hidden_tokens = torch.randn(1, G * G, D)

        new_state = state.step(
            fresh_images, fresh_targets, next_hidden,
            next_local_prev=None,
            survival_prob=0.0,
            hidden_tokens=hidden_tokens,
            local_tokens=None,
        )
        # All items reset to fresh
        assert torch.equal(new_state.images, fresh_images)
        assert torch.equal(new_state.targets, fresh_targets)
        # Hidden resets to hidden_tokens
        assert new_state.hidden is not None
        assert torch.equal(new_state.hidden, hidden_tokens.expand(B, -1, -1))
        assert new_state.local_prev is None

    def test_survival_one_keeps_all(self) -> None:
        B, C, H, W, D, G = 4, 3, 64, 64, 128, 16
        old_images = torch.randn(B, C, H, W)
        old_targets = torch.randn(B, G * G, D)
        old_hidden = torch.randn(B, G * G, D)
        state = TrainState(images=old_images, targets=old_targets, hidden=old_hidden, local_prev=None)

        fresh_images = torch.randn(B, C, H, W)
        fresh_targets = torch.randn(B, G * G, D)
        next_hidden = torch.randn(B, G * G, D)
        hidden_tokens = torch.randn(1, G * G, D)

        new_state = state.step(
            fresh_images, fresh_targets, next_hidden,
            next_local_prev=None,
            survival_prob=1.0,
            hidden_tokens=hidden_tokens,
            local_tokens=None,
        )
        # All items kept
        assert torch.equal(new_state.images, old_images)
        assert torch.equal(new_state.targets, old_targets)
        # Hidden continues from next_hidden (detached)
        assert new_state.hidden is not None
        assert torch.equal(new_state.hidden, next_hidden)
        assert new_state.local_prev is None

    def test_hidden_detached(self) -> None:
        """Surviving hidden states are detached to cut BPTT across optimizer steps."""
        B, C, H, W, D, G = 4, 3, 64, 64, 128, 16
        state = TrainState.init(
            torch.randn(B, C, H, W),
            torch.randn(B, G * G, D),
        )
        next_hidden = torch.randn(B, G * G, D, requires_grad=True)
        new_state = state.step(
            torch.randn(B, C, H, W),
            torch.randn(B, G * G, D),
            next_hidden,
            next_local_prev=None,
            survival_prob=1.0,
            hidden_tokens=torch.randn(1, G * G, D),
            local_tokens=None,
        )
        # Surviving items should be detached
        assert new_state.hidden is not None
        assert not new_state.hidden.requires_grad

    def test_local_prev_survival_and_detach(self) -> None:
        """local_prev is properly handled when use_local_temporal=True."""
        B, C, H, W, D, G, N = 4, 3, 64, 64, 128, 16, 10
        state = TrainState.init(
            torch.randn(B, C, H, W),
            torch.randn(B, G * G, D),
        )

        next_hidden = torch.randn(B, G * G, D)
        next_local_prev = torch.randn(B, N, D, requires_grad=True)
        local_tokens = torch.randn(1, N, D)

        # survival_prob=1.0: all survive, use detached next_local_prev
        new_state = state.step(
            torch.randn(B, C, H, W),
            torch.randn(B, G * G, D),
            next_hidden,
            next_local_prev=next_local_prev,
            survival_prob=1.0,
            hidden_tokens=torch.randn(1, G * G, D),
            local_tokens=local_tokens,
        )
        assert new_state.local_prev is not None
        assert new_state.local_prev.shape == (B, N, D)
        assert not new_state.local_prev.requires_grad  # detached
        assert torch.equal(new_state.local_prev, next_local_prev.detach())

    def test_local_prev_reset_on_non_survival(self) -> None:
        """Non-survivors reset local_prev to local_tokens."""
        B, C, H, W, D, G, N = 4, 3, 64, 64, 128, 16, 10
        state = TrainState.init(
            torch.randn(B, C, H, W),
            torch.randn(B, G * G, D),
        )

        next_hidden = torch.randn(B, G * G, D)
        next_local_prev = torch.randn(B, N, D)
        local_tokens = torch.randn(1, N, D)

        # survival_prob=0.0: all reset
        new_state = state.step(
            torch.randn(B, C, H, W),
            torch.randn(B, G * G, D),
            next_hidden,
            next_local_prev=next_local_prev,
            survival_prob=0.0,
            hidden_tokens=torch.randn(1, G * G, D),
            local_tokens=local_tokens,
        )
        assert new_state.local_prev is not None
        assert torch.equal(new_state.local_prev, local_tokens.expand(B, -1, -1))
