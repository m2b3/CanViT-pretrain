"""Tests for Bernoulli survival state management."""

import torch

from avp_vit.train.state import TrainState


class TestTrainState:
    def test_init(self) -> None:
        B, C, H, W, D, G = 4, 3, 64, 64, 128, 16
        images = torch.randn(B, C, H, W)
        targets = torch.randn(B, G * G, D)
        state = TrainState.init(images, targets)
        assert state.scene is None
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
            next_scene=torch.randn(B, G * G, D),
            survival_prob=0.5,
            scene_tokens=torch.randn(1, G * G, D),
        )
        assert new_state.images.shape == (B, C, H, W)
        assert new_state.targets.shape == (B, G * G, D)
        assert new_state.scene is not None
        assert new_state.scene.shape == (B, G * G, D)

    def test_survival_zero_resets_all(self) -> None:
        B, C, H, W, D, G = 4, 3, 64, 64, 128, 16
        old_images = torch.randn(B, C, H, W)
        old_targets = torch.randn(B, G * G, D)
        state = TrainState.init(old_images, old_targets)

        fresh_images = torch.randn(B, C, H, W)
        fresh_targets = torch.randn(B, G * G, D)
        next_scene = torch.randn(B, G * G, D)
        scene_tokens = torch.randn(1, G * G, D)

        new_state = state.step(
            fresh_images, fresh_targets, next_scene,
            survival_prob=0.0, scene_tokens=scene_tokens,
        )
        # All items reset to fresh
        assert torch.equal(new_state.images, fresh_images)
        assert torch.equal(new_state.targets, fresh_targets)
        # Scene resets to scene_tokens
        assert new_state.scene is not None
        assert torch.equal(new_state.scene, scene_tokens.expand(B, -1, -1))

    def test_survival_one_keeps_all(self) -> None:
        B, C, H, W, D, G = 4, 3, 64, 64, 128, 16
        old_images = torch.randn(B, C, H, W)
        old_targets = torch.randn(B, G * G, D)
        old_scene = torch.randn(B, G * G, D)
        state = TrainState(images=old_images, targets=old_targets, scene=old_scene)

        fresh_images = torch.randn(B, C, H, W)
        fresh_targets = torch.randn(B, G * G, D)
        next_scene = torch.randn(B, G * G, D)
        scene_tokens = torch.randn(1, G * G, D)

        new_state = state.step(
            fresh_images, fresh_targets, next_scene,
            survival_prob=1.0, scene_tokens=scene_tokens,
        )
        # All items kept
        assert torch.equal(new_state.images, old_images)
        assert torch.equal(new_state.targets, old_targets)
        # Scene continues from next_scene (detached)
        assert new_state.scene is not None
        assert torch.equal(new_state.scene, next_scene)

    def test_scene_detached(self) -> None:
        B, C, H, W, D, G = 4, 3, 64, 64, 128, 16
        state = TrainState.init(
            torch.randn(B, C, H, W),
            torch.randn(B, G * G, D),
        )
        next_scene = torch.randn(B, G * G, D, requires_grad=True)
        new_state = state.step(
            torch.randn(B, C, H, W),
            torch.randn(B, G * G, D),
            next_scene,
            survival_prob=1.0,
            scene_tokens=torch.randn(1, G * G, D),
        )
        # Surviving items should be detached
        assert new_state.scene is not None
        assert not new_state.scene.requires_grad
