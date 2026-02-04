"""Learning rate scheduler utilities."""

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    ConstantLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
)


def warmup_constant_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    peak_lr: float,
    start_lr: float | None = None,
) -> LRScheduler:
    """Create warmup + constant LR scheduler.

    Warmup: linear from start_lr to peak_lr over warmup_steps.
    Then: constant at peak_lr forever.

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Number of warmup steps (0 = constant from start).
        peak_lr: Learning rate after warmup.
        start_lr: Learning rate at step 0. None = peak_lr / warmup_steps.

    Returns:
        LRScheduler.
    """
    assert warmup_steps >= 0, "warmup_steps must be non-negative"

    if warmup_steps == 0:
        return ConstantLR(optimizer, factor=1.0, total_iters=0)

    effective_start_lr = start_lr if start_lr is not None else peak_lr / warmup_steps
    start_factor = effective_start_lr / peak_lr

    warmup = LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    constant = ConstantLR(optimizer, factor=1.0, total_iters=0)
    return SequentialLR(
        optimizer,
        schedulers=[warmup, constant],
        milestones=[warmup_steps],
    )
