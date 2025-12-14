"""Tests for scheduler utilities."""

import torch
from torch import nn

from avp_vit.train.scheduler import warmup_cosine_scheduler


class TestWarmupCosineScheduler:
    def test_warmup_starts_near_zero(self) -> None:
        model = nn.Linear(10, 10)
        opt = torch.optim.AdamW(model.parameters(), lr=1.0)
        sched = warmup_cosine_scheduler(opt, total_steps=100, warmup_steps=10)

        # At step 0, LR should be start_factor * base_lr = (1/10) * 1.0 = 0.1
        assert abs(sched.get_last_lr()[0] - 0.1) < 1e-6

    def test_warmup_reaches_peak(self) -> None:
        model = nn.Linear(10, 10)
        opt = torch.optim.AdamW(model.parameters(), lr=1.0)
        sched = warmup_cosine_scheduler(opt, total_steps=100, warmup_steps=10)

        # Step through warmup
        for _ in range(10):
            sched.step()

        # At end of warmup, LR should be peak (1.0)
        assert abs(sched.get_last_lr()[0] - 1.0) < 1e-6

    def test_cosine_decays_to_zero(self) -> None:
        model = nn.Linear(10, 10)
        opt = torch.optim.AdamW(model.parameters(), lr=1.0)
        sched = warmup_cosine_scheduler(opt, total_steps=100, warmup_steps=10)

        # Step through all steps
        for _ in range(100):
            sched.step()

        # At end, LR should be ~0
        assert sched.get_last_lr()[0] < 1e-6

    def test_monotonic_warmup(self) -> None:
        model = nn.Linear(10, 10)
        opt = torch.optim.AdamW(model.parameters(), lr=1.0)
        sched = warmup_cosine_scheduler(opt, total_steps=100, warmup_steps=10)

        lrs = [sched.get_last_lr()[0]]
        for _ in range(10):
            sched.step()
            lrs.append(sched.get_last_lr()[0])

        # Warmup should be strictly increasing
        for i in range(len(lrs) - 1):
            assert lrs[i] < lrs[i + 1]

    def test_assertions_on_invalid_args(self) -> None:
        model = nn.Linear(10, 10)
        opt = torch.optim.AdamW(model.parameters(), lr=1.0)

        # warmup_steps must be positive
        try:
            warmup_cosine_scheduler(opt, total_steps=100, warmup_steps=0)
            assert False, "Should have raised"
        except AssertionError:
            pass

        # warmup_steps must be less than total_steps
        try:
            warmup_cosine_scheduler(opt, total_steps=100, warmup_steps=100)
            assert False, "Should have raised"
        except AssertionError:
            pass
