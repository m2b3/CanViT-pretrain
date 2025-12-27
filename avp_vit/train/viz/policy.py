"""Policy prediction visualization."""

import matplotlib.pyplot as plt
import numpy as np
from canvit.policy import PolicyOutput
from matplotlib.figure import Figure
from scipy.stats import gaussian_kde
from torch import Tensor


def plot_policy_predictions(
    starts_full: Tensor,
    starts_random: Tensor,
    preds_full: PolicyOutput,
    preds_random: PolicyOutput,
    min_scale: float = 0.05,
) -> Figure:
    """Plot policy predictions: positions (lines) and scales (KDE + scatter).

    Args:
        starts_full: (B, 2) centers from full scene viewpoints (all zeros)
        starts_random: (B, 2) centers from random viewpoints
        preds_full: Policy predictions given full scene context
        preds_random: Policy predictions given random context
        min_scale: Minimum scale for x-axis limits
    """
    fig, (ax_pos, ax_scale) = plt.subplots(1, 2, figsize=(10, 5))

    # Left: position scatter with lines from start to end
    for start, pred, color, label in [
        (starts_full, preds_full.position, "tab:blue", "full→policy"),
        (starts_random, preds_random.position, "tab:orange", "random→policy"),
    ]:
        start_np = start.cpu().float().numpy()
        pred_np = pred.cpu().float().numpy()
        for i in range(len(start_np)):
            ax_pos.plot([start_np[i, 0], pred_np[i, 0]], [start_np[i, 1], pred_np[i, 1]],
                       color=color, alpha=0.3, lw=1)
        ax_pos.scatter(start_np[:, 0], start_np[:, 1], c="white", edgecolors=color, s=20, alpha=0.8, zorder=5)
        ax_pos.scatter(pred_np[:, 0], pred_np[:, 1], c=color, s=20, alpha=0.8, label=label, zorder=6)

    ax_pos.set_xlim(-1.1, 1.1)
    ax_pos.set_ylim(-1.1, 1.1)
    ax_pos.set_aspect("equal")
    ax_pos.axhline(0, color="gray", lw=0.5, alpha=0.5)
    ax_pos.axvline(0, color="gray", lw=0.5, alpha=0.5)
    ax_pos.set_xlabel("x")
    ax_pos.set_ylabel("y")
    ax_pos.legend(loc="upper right", fontsize=8)
    ax_pos.set_title("Policy positions")

    # Right: scale distribution (KDE + scatter)
    x_kde = np.linspace(min_scale, 1.0, 200)
    for scales, color, label, y_offset in [
        (preds_full.scale, "tab:blue", "full→policy", 0.02),
        (preds_random.scale, "tab:orange", "random→policy", -0.02),
    ]:
        s_np = scales.cpu().float().numpy()
        kde = gaussian_kde(s_np, bw_method=0.1)
        ax_scale.plot(x_kde, kde(x_kde), color=color, lw=2, label=label)
        ax_scale.scatter(s_np, np.full_like(s_np, y_offset), c=color, s=15, alpha=0.5)
    ax_scale.axhline(0, color="gray", lw=0.5, alpha=0.3)
    ax_scale.set_xlim(min_scale, 1.0)
    ax_scale.set_xlabel("Scale")
    ax_scale.set_ylabel("Density")
    ax_scale.legend(loc="upper right", fontsize=8)
    ax_scale.set_title("Policy scales")

    plt.tight_layout()
    return fig
