"""Hidden stream state management."""

import math
from typing import final

import torch
from torch import Tensor, nn


@final
class HiddenStreamParams(nn.Module):
    """Init params + recurrence normalizers for one hidden stream.

    Hidden layout: [cls | registers | spatial] where cls is always 1 token.
    """

    dim: int
    n_registers: int
    cls_init: nn.Parameter
    spatial_init: nn.Parameter
    registers: nn.Parameter | None
    cls_ln: nn.Module
    reg_ln: nn.Module
    spatial_ln: nn.Module

    def __init__(
        self,
        dim: int,
        n_registers: int,
        use_recurrence_ln: bool,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_registers = n_registers
        scale = 1.0 / math.sqrt(dim)

        # Init params (1/sqrt(D) for unit L2 norm)
        self.cls_init = nn.Parameter(torch.randn(1, 1, dim) * scale)
        self.spatial_init = nn.Parameter(torch.randn(1, 1, dim) * scale)
        self.registers = (
            nn.Parameter(torch.randn(1, n_registers, dim) * scale)
            if n_registers > 0
            else None
        )

        # Recurrence normalizers
        if use_recurrence_ln:
            self.cls_ln = nn.LayerNorm(dim)
            self.reg_ln = nn.LayerNorm(dim)
            self.spatial_ln = nn.LayerNorm(dim)
            nn.init.constant_(self.cls_ln.weight, scale)
            nn.init.constant_(self.reg_ln.weight, scale)
            nn.init.constant_(self.spatial_ln.weight, scale)
        else:
            self.cls_ln = nn.Identity()
            self.reg_ln = nn.Identity()
            self.spatial_ln = nn.Identity()

    @property
    def n_prefix(self) -> int:
        """Number of prefix tokens (cls + registers)."""
        return 1 + self.n_registers

    def init_hidden(self, batch_size: int, n_spatial: int) -> Tensor:
        """Create initial hidden state [B, 1 + n_reg + n_spatial, D]."""
        B = batch_size
        cls = self.cls_init.expand(B, -1, -1)
        spatial = self.spatial_init.expand(B, n_spatial, -1)

        if self.registers is not None:
            regs = self.registers.expand(B, -1, -1)
            return torch.cat([cls, regs, spatial], dim=1)
        return torch.cat([cls, spatial], dim=1)

    def normalize(self, hidden: Tensor) -> Tensor:
        """Normalize at recurrence boundary: [cls | registers | spatial]."""
        n_reg = self.n_registers
        cls = self.cls_ln(hidden[:, :1])
        reg = self.reg_ln(hidden[:, 1 : 1 + n_reg])
        spatial = self.spatial_ln(hidden[:, 1 + n_reg :])
        return torch.cat([cls, reg, spatial], dim=1)
