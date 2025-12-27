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
    peak_lr: float,
    start_lr: float | None = None,
    end_lr: float | None = None,
) -> LRScheduler:
    """Create warmup + cosine decay scheduler.

    Warmup: linear from start_lr to peak_lr over warmup_steps.
    Decay: cosine from peak_lr to end_lr over remaining steps.

    Args:
        optimizer: The optimizer to schedule.
        total_steps: Total training steps.
        warmup_steps: Number of warmup steps (0 = no warmup, pure cosine).
        peak_lr: Learning rate at end of warmup.
        start_lr: Learning rate at step 0. None = peak_lr / warmup_steps.
        end_lr: Learning rate at final step. None = 0.

    Returns:
        LRScheduler (SequentialLR if warmup, else CosineAnnealingLR).
    """
    assert warmup_steps >= 0, "warmup_steps must be non-negative"
    assert warmup_steps < total_steps, "warmup_steps must be less than total_steps"

    effective_end_lr = end_lr if end_lr is not None else 0.0

    if warmup_steps == 0:
        return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=effective_end_lr)

    effective_start_lr = start_lr if start_lr is not None else peak_lr / warmup_steps
    start_factor = effective_start_lr / peak_lr

    warmup = LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=effective_end_lr,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )
