"""Training state for probes."""

from dataclasses import dataclass

from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR

from canvit_eval.ade20k.probe import ProbeHead


@dataclass
class ProbeState:
    """Training state for one probe."""

    name: str
    head: ProbeHead
    optimizer: AdamW
    scheduler: SequentialLR
    best_mean_miou: float = 0.0
    _loss_sum: Tensor | None = None
    _grad_norm_sum: Tensor | None = None
    _count: int = 0

    def accumulate(self, loss: Tensor, grad_norm: Tensor) -> None:
        """Accumulate loss/grad_norm. NO GPU sync."""
        if self._loss_sum is None:
            self._loss_sum = loss.detach().clone()
            self._grad_norm_sum = grad_norm.detach().clone()
        else:
            self._loss_sum += loss.detach()
            assert self._grad_norm_sum is not None
            self._grad_norm_sum += grad_norm.detach()
        self._count += 1

    def get_and_reset(self) -> tuple[float, float]:
        """Get averaged stats and reset. SYNCS here."""
        assert self._loss_sum is not None and self._grad_norm_sum is not None
        avg_loss = (self._loss_sum / self._count).item()
        avg_grad = (self._grad_norm_sum / self._count).item()
        self._loss_sum = self._grad_norm_sum = None
        self._count = 0
        return avg_loss, avg_grad
