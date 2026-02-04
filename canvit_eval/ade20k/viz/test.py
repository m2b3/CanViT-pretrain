"""Tests for visualization."""

import numpy as np

from canvit_eval.ade20k.dataset import IGNORE_LABEL

from . import colorize_mask, correctness_map


def test_colorize_mask():
    """Colorize mask produces RGB array."""
    mask = np.array([[0, 1, 2], [IGNORE_LABEL, 50, 100]])
    out = colorize_mask(mask)
    assert out.shape == (2, 3, 3)
    assert out.dtype == np.uint8


def test_correctness_map():
    """Correctness map produces green/red/gray."""
    pred = np.array([[0, 1, 2], [0, 1, 2]])
    gt = np.array([[0, 0, 2], [IGNORE_LABEL, 1, 1]])
    out = correctness_map(pred, gt)
    assert out.shape == (2, 3, 3)
    # (0,0) correct -> green
    assert np.array_equal(out[0, 0], [0, 200, 0])
    # (0,1) wrong -> red
    assert np.array_equal(out[0, 1], [200, 0, 0])
    # (1,0) ignore -> gray
    assert np.array_equal(out[1, 0], [128, 128, 128])
