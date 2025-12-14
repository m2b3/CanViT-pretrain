"""Training state management with Bernoulli survival."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class TrainState:
    """Persistent state for Bernoulli survival training.

    Bernoulli survival enables learning to generalize across arbitrary inference
    lengths despite BPTT being restricted to short horizons. Each batch item has
    probability `survival_prob` of being "carried over" between optimizer steps:
    - Surviving items: same image/target, scene continues from previous step
    - Non-surviving items: fresh image/target, scene resets to learned scene_tokens

    This creates a geometric distribution of effective sequence lengths.
    """

    images: Tensor  # [B, C, H, W]
    targets: Tensor  # [B, G*G, D]
    scene: Tensor | None  # [B, G*G, D] or None (first step)

    @staticmethod
    def init(images: Tensor, targets: Tensor) -> "TrainState":
        """Initialize state with fresh batch, no scene history."""
        return TrainState(images=images, targets=targets, scene=None)

    def step(
        self,
        fresh_images: Tensor,
        fresh_targets: Tensor,
        next_scene: Tensor,
        survival_prob: float,
        scene_tokens: Tensor,
    ) -> "TrainState":
        """Update state with Bernoulli survival.

        Args:
            fresh_images: New images from dataloader [B, C, H, W]
            fresh_targets: Teacher patches for fresh images [B, G*G, D]
            next_scene: Scene output from forward step [B, G*G, D]
            survival_prob: Probability of keeping current item
            scene_tokens: Model's scene_tokens parameter [1, G*G, D]

        Returns:
            Updated TrainState for next iteration.
        """
        B = self.images.shape[0]
        device = self.images.device

        survive = torch.rand(B, device=device) < survival_prob
        s_img = survive.view(B, 1, 1, 1)
        s_feat = survive.view(B, 1, 1)

        images = torch.where(s_img, self.images, fresh_images)
        targets = torch.where(s_feat, self.targets, fresh_targets)

        # Survivors: detach to cut BPTT across optimizer steps
        # Non-survivors: reset to scene_tokens (gradients flow to scene_tokens param)
        scene_init = scene_tokens.expand(B, -1, -1)
        scene = torch.where(s_feat, next_scene.detach(), scene_init)

        return TrainState(images=images, targets=targets, scene=scene)
