"""Position-aware running normalization for spatial token sequences."""

import torch
from torch import Tensor, nn


class PositionAwareNorm(nn.Module):
    """Running normalization with per-position stats for [B, N, D] sequences.

    Stats shape: [N, D] - one mean/var per token position per dimension.
    Stateful: first batch initializes stats directly (no warmup with wrong stats).
    All ops on GPU buffers without sync.
    """

    mean: Tensor
    var: Tensor
    _initialized: Tensor  # bool tensor, persisted in state_dict

    def __init__(
        self,
        n_tokens: int,
        embed_dim: int,
        grid_size: int,
        momentum: float = 0.1,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("mean", torch.zeros(n_tokens, embed_dim))
        self.register_buffer("var", torch.ones(n_tokens, embed_dim))
        self.register_buffer("_initialized", torch.tensor(False))

    @property
    def initialized(self) -> bool:
        """Whether running stats have been initialized from data."""
        return self._initialized.item()  # type: ignore[return-value]

    def forward(self, x: Tensor) -> Tensor:
        """Normalize x: [B, N, D] -> [B, N, D]. Updates stats only in train mode.

        Uses Chan's parallel variance algorithm adapted for EWMA. This handles any
        batch size seamlessly - for B=1 it reduces to Welford's online algorithm,
        for B>1 it combines within-batch and between-batch variance correctly.

        Reference: Chan, Golub, LeVeque (1979) "Updating Formulae and a Pairwise
        Algorithm for Computing Sample Variances"
        """
        if self.training:
            B = x.shape[0]
            with torch.no_grad():
                batch_mean = x.mean(dim=0)
                delta = batch_mean - self.mean

                # Within-batch variance. unbiased=False gives 0 for B=1 (not NaN).
                batch_var = x.var(dim=0, unbiased=False) if B > 1 else x.new_zeros(self.var.shape)

                # Effective momentum: B sequential updates with rate α equals
                # one batch update with rate m = 1 - (1-α)^B
                m = 1 - (1 - self.momentum) ** B

                if not self.initialized:
                    self.mean.copy_(batch_mean)
                    if B > 1:
                        self.var.copy_(batch_var)
                    # else: keep var=1 until we have enough samples
                    self._initialized.fill_(True)
                else:
                    self.mean.lerp_(batch_mean, m)
                    # Chan's formula: var = within-batch + between-batch variance
                    #   var_new = (1-m)*var_old + m*batch_var + (1-m)*m*δ²
                    #
                    # The (1-m)*m*δ² term is the "between-group variance" - it captures
                    # distribution shift. For B=1, batch_var=0 and this term provides
                    # the entire variance signal, reducing to Welford's formula:
                    #   var_new = (1-α)*(var_old + α*δ²)
                    self.var.copy_(
                        (1 - m) * self.var + m * batch_var + (1 - m) * m * delta**2
                    )

        return (x - self.mean) / (self.var + self.eps).sqrt()

    def denormalize(self, x: Tensor) -> Tensor:
        """Invert normalization: x * sqrt(var + eps) + mean."""
        return x * (self.var + self.eps).sqrt() + self.mean
