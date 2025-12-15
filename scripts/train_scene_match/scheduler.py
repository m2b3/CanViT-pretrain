"""LR scheduling for curriculum training using PyTorch built-ins."""

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .config import Config


def create_curriculum_scheduler(optimizer: Optimizer, cfg: Config) -> SequentialLR:
    """Chain LinearLR (ramp) + CosineAnnealingLR (decay) for each schedule entry."""
    schedulers = []
    milestones = []
    step = 0

    for entry in cfg.get_schedule():
        assert entry.lr_ramp_steps > 0 or entry.lr_decay_steps > 0
        if entry.lr_ramp_steps > 0:
            schedulers.append(LinearLR(optimizer, 1 / entry.lr_ramp_steps, 1.0, entry.lr_ramp_steps))
            step += entry.lr_ramp_steps
            milestones.append(step)
        if entry.lr_decay_steps > 0:
            schedulers.append(CosineAnnealingLR(optimizer, entry.lr_decay_steps, 0))
            step += entry.lr_decay_steps
            milestones.append(step)

    assert step == cfg.n_steps, f"Schedule covers {step} steps, expected {cfg.n_steps}"
    return SequentialLR(optimizer, schedulers, milestones[:-1])
