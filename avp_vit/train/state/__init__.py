"""Training state management with fresh ratio survival."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class TrainState:
    """Persistent state for fresh-ratio survival training.

    Each step, a fixed ratio of the batch is replaced with fresh samples.
    Random permutation ensures stochastic lifetimes (geometric distribution)
    while allowing us to load/compute teacher features for only fresh_count images.

    IMPORTANT - Naming convention (consistent with AVPViT):
    - hidden: Internal state for CONTINUATION between timesteps
    - local_prev: Local state for CONTINUATION when use_local_temporal enabled
    - scene: Projected output for loss/viz (not stored here, computed on demand)
    """

    images: Tensor  # [B, C, H, W]
    targets: Tensor  # [B, G*G, D]
    hidden: Tensor | None  # [B, n_tokens, D] or None (first step)
    local_prev: Tensor | None  # [B, N, D] or None

    @staticmethod
    def init(
        images: Tensor,
        targets: Tensor,
        hidden_init: Tensor,
        local_init: Tensor | None,
    ) -> "TrainState":
        """Initialize state with first full batch."""
        return TrainState(images=images, targets=targets, hidden=hidden_init, local_prev=local_init)

    def step(
        self,
        fresh_images: Tensor,
        fresh_targets: Tensor,
        next_hidden: Tensor,
        next_local_prev: Tensor | None,
        hidden_init: Tensor,
        local_init: Tensor | None,
    ) -> "TrainState":
        """Update state: permute, replace first K with fresh.

        Args:
            fresh_images: New images [K, C, H, W] where K = fresh_count
            fresh_targets: Teacher patches for fresh images [K, G*G, D]
            next_hidden: Hidden state from forward step [B, n_tokens, D]
            next_local_prev: Local state from forward step [B, N, D] or None
            hidden_init: Initialized hidden for fresh samples [K, n_tokens, D]
            local_init: Initialized local for fresh samples [K, N, D] or None

        Returns:
            Updated TrainState for next iteration.
        """
        B = self.images.shape[0]
        K = fresh_images.shape[0]
        device = self.images.device

        # Random permutation: which indices "die" is stochastic
        perm = torch.randperm(B, device=device)

        # Permute current state
        images = self.images[perm]
        targets = self.targets[perm]
        hidden = next_hidden[perm].detach()
        local_prev = next_local_prev[perm].detach() if next_local_prev is not None else None

        # Replace first K with fresh
        images[:K] = fresh_images
        targets[:K] = fresh_targets
        hidden[:K] = hidden_init
        if local_prev is not None:
            assert local_init is not None
            local_prev[:K] = local_init

        return TrainState(images=images, targets=targets, hidden=hidden, local_prev=local_prev)
