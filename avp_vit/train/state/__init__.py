"""Batch state for fresh-ratio survival across optimizer steps."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class SurvivalBatch:
    """Batch state for fresh-ratio survival.

    Replaces a fraction of batch each step; surviving samples keep canvas for continuation.
    Fresh and survivor samples are randomly distributed (no deterministic positions).
    """

    images: Tensor  # [B, C, H, W]
    targets: Tensor  # [B, G*G, D] - patch tokens for scene/local loss
    cls_targets: Tensor  # [B, D] - CLS tokens for CLS loss
    canvas: Tensor  # [B, n_tokens, D]

    @staticmethod
    def init(
        images: Tensor, targets: Tensor, cls_targets: Tensor, canvas: Tensor
    ) -> "SurvivalBatch":
        """Initialize with random permutation to avoid first-step position bias."""
        B = images.shape[0]
        perm = torch.randperm(B, device=images.device)
        return SurvivalBatch(
            images=images[perm],
            targets=targets[perm],
            cls_targets=cls_targets[perm],
            canvas=canvas[perm],
        )

    def step(
        self,
        *,
        fresh_images: Tensor,
        fresh_targets: Tensor,
        fresh_cls_targets: Tensor,
        next_canvas: Tensor,
        canvas_init: Tensor,
    ) -> "SurvivalBatch":
        """Replace K samples with fresh, permute to randomize positions.

        Since batch is already randomly ordered from previous step's permutation,
        taking indices [K:] is equivalent to randomly selecting B-K survivors.
        Final permutation ensures fresh samples aren't at deterministic positions.
        """
        B = self.images.shape[0]
        K = fresh_images.shape[0]

        # Cat fresh + survivors (survivors = indices K: due to prior random order)
        images = torch.cat([fresh_images, self.images[K:]], dim=0)
        targets = torch.cat([fresh_targets, self.targets[K:]], dim=0)
        cls_targets = torch.cat([fresh_cls_targets, self.cls_targets[K:]], dim=0)
        # canvas_init NOT detached (learnable), next_canvas IS detached (no BPTT)
        canvas = torch.cat([canvas_init, next_canvas[K:].detach()], dim=0)

        # Permute to randomize final positions
        perm = torch.randperm(B, device=self.images.device)
        return SurvivalBatch(
            images=images[perm],
            targets=targets[perm],
            cls_targets=cls_targets[perm],
            canvas=canvas[perm],
        )
