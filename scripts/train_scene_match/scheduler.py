"""LR scheduling for curriculum training with warmup phase."""

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from .config import Config


def create_curriculum_scheduler(optimizer: Optimizer, cfg: Config) -> LambdaLR:
    """Create LR scheduler for two-phase curriculum training.

    Phase 1 (Warmup): Cycles through all grid sizes (largest first).
        Each sub-phase: linear warmup (first half) → cosine decay (second half)

    Phase 2 (Main): Normal curriculum training (smallest to largest).
        Single cosine decay from peak to 0.

    All phases use the same peak LR. The schedule is fully determined by cfg.
    """
    warmup_steps = cfg.warmup_steps
    warmup_per_size = cfg.warmup_steps_per_size
    main_steps = cfg.main_training_steps
    n_sizes = len(cfg.grid_sizes)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Warmup phase: which sub-phase are we in?
            sub_phase_idx = step // warmup_per_size
            sub_phase_idx = min(sub_phase_idx, n_sizes - 1)  # Clamp to last
            step_in_sub = step - sub_phase_idx * warmup_per_size

            # Linear warmup for first half, cosine decay for second half
            half = warmup_per_size // 2
            if step_in_sub < half:
                # Linear warmup: 0 → 1
                return step_in_sub / max(1, half)
            else:
                # Cosine decay: 1 → 0
                progress = (step_in_sub - half) / max(1, warmup_per_size - half)
                return 0.5 * (1 + math.cos(math.pi * progress))
        else:
            # Main training: cosine decay from 1 → 0
            step_in_main = step - warmup_steps
            progress = step_in_main / max(1, main_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)
