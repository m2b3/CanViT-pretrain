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
    - Surviving items: same image/target, hidden state continues from previous step
    - Non-surviving items: fresh image/target, hidden resets to learned hidden_tokens

    This creates a geometric distribution of effective sequence lengths.

    IMPORTANT - Naming convention (consistent with AVPViT):
    - hidden: Internal state for CONTINUATION between timesteps
    - scene: Projected output for loss/viz (not stored here, computed on demand)
    """

    images: Tensor  # [B, C, H, W]
    targets: Tensor  # [B, G*G, D]
    hidden: Tensor | None  # [B, G*G, D] or None (first step) - for CONTINUATION

    @staticmethod
    def init(images: Tensor, targets: Tensor) -> "TrainState":
        """Initialize state with fresh batch, no hidden history."""
        return TrainState(images=images, targets=targets, hidden=None)

    def step(
        self,
        fresh_images: Tensor,
        fresh_targets: Tensor,
        next_hidden: Tensor,
        survival_prob: float,
        hidden_tokens: Tensor,
    ) -> "TrainState":
        """Update state with Bernoulli survival.

        Args:
            fresh_images: New images from dataloader [B, C, H, W]
            fresh_targets: Teacher patches for fresh images [B, G*G, D]
            next_hidden: Hidden state from forward step [B, G*G, D] (for CONTINUATION)
            survival_prob: Probability of keeping current item
            hidden_tokens: Model's hidden_tokens parameter [1, G*G, D]

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
        # Non-survivors: reset to hidden_tokens (gradients flow to hidden_tokens param)
        hidden_init = hidden_tokens.expand(B, -1, -1)
        hidden = torch.where(s_feat, next_hidden.detach(), hidden_init)

        return TrainState(images=images, targets=targets, hidden=hidden)
