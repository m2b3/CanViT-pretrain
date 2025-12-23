"""Comet ML logging utilities for visualization."""

import gc
import io
import logging

import comet_ml
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

log = logging.getLogger(__name__)

# Comet curve budget - enforced at logging point
_curve_count = 0
_CURVE_BUDGET = 900


def log_curve(exp: comet_ml.Experiment, name: str, **kwargs) -> None:
    """Log curve with budget enforcement. Skips silently once exhausted."""
    global _curve_count
    if _curve_count >= _CURVE_BUDGET:
        if _curve_count == _CURVE_BUDGET:
            log.warning(f"Curve budget exhausted ({_CURVE_BUDGET}), skipping further curves")
            _curve_count += 1  # only warn once
        return
    exp.log_curve(name, **kwargs)
    _curve_count += 1


def log_figure(exp: comet_ml.Experiment, fig: Figure, name: str, step: int) -> None:
    """Log matplotlib figure to Comet. Aggressively cleans up to prevent memory leaks."""
    with io.BytesIO() as buf:
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        exp.log_image(buf, name=name, step=step)
    for ax in fig.axes:
        ax.clear()
    fig.clf()
    plt.close(fig)
    gc.collect()
