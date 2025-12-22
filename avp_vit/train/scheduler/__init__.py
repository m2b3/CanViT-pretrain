"""Learning rate scheduler utilities."""

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
)


def warmup_cosine_scheduler(
    optimizer: Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> LRScheduler:
    """Create warmup + cosine decay scheduler.

    Warmup: linear from 1/warmup_steps to 1.0 over warmup_steps.
    Decay: cosine from 1.0 to 0.0 over remaining steps.

    Args:
        optimizer: The optimizer to schedule.
        total_steps: Total training steps.
        warmup_steps: Number of warmup steps (0 = no warmup, pure cosine).

    Returns:
        LRScheduler (SequentialLR if warmup, else CosineAnnealingLR).
    """
    assert warmup_steps >= 0, "warmup_steps must be non-negative"
    assert warmup_steps < total_steps, "warmup_steps must be less than total_steps"

    if warmup_steps == 0:
        return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0.0)

    start_factor = 1.0 / warmup_steps
    warmup = LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=0.0,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )
