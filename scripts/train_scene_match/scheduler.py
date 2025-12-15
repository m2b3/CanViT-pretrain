"""LR scheduling: linear warmup + cosine decay."""

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .config import Config


def create_scheduler(optimizer: Optimizer, cfg: Config) -> SequentialLR:
    """Linear warmup + cosine decay to zero."""
    warmup = LinearLR(
        optimizer,
        start_factor=1 / cfg.warmup_steps,
        end_factor=1.0,
        total_iters=cfg.warmup_steps,
    )
    decay = CosineAnnealingLR(
        optimizer,
        T_max=cfg.n_steps - cfg.warmup_steps,
        eta_min=0,
    )
    return SequentialLR(optimizer, [warmup, decay], milestones=[cfg.warmup_steps])
