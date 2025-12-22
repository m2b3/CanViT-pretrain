"""Tests for probe utilities."""

import torch

from avp_vit.train.probe import compute_in1k_top1


def test_compute_in1k_top1_perfect() -> None:
    logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    labels = torch.tensor([0, 1, 2])
    assert compute_in1k_top1(logits, labels) == 100.0


def test_compute_in1k_top1_none() -> None:
    # All predict class 0, but none should be correct
    logits = torch.tensor([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    labels = torch.tensor([1, 2, 1])  # none are class 0
    assert compute_in1k_top1(logits, labels) == 0.0


def test_compute_in1k_top1_partial() -> None:
    # preds: [0, 1, 0, 1], labels: [0, 1, 1, 0] -> 2/4 = 50%
    logits = torch.tensor([[10.0, 0.0], [0.0, 10.0], [10.0, 0.0], [0.0, 10.0]])
    labels = torch.tensor([0, 1, 1, 0])
    assert compute_in1k_top1(logits, labels) == 50.0
