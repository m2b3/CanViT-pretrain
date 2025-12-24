"""Policy prediction visualization."""

import matplotlib.pyplot as plt
from canvit.policy import PolicyOutput
from matplotlib.figure import Figure
from torch import Tensor


def plot_policy_predictions(
    starts_full: Tensor,
    starts_random: Tensor,
    preds_full: PolicyOutput,
    preds_random: PolicyOutput,
) -> Figure:
    """Plot policy predictions: positions (scatter+arrows) and scales (histogram).

    Args:
        starts_full: (B, 2) centers from full scene viewpoints (all zeros)
        starts_random: (B, 2) centers from random viewpoints
        preds_full: Policy predictions given full scene context
        preds_random: Policy predictions given random context
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

    # Right: scale histogram
    bins = 20
    ax_scale.hist(
        preds_full.scale.cpu().float().numpy(),
        bins=bins,
        alpha=0.6,
        label="full→policy",
        color="tab:blue",
    )
    ax_scale.hist(
        preds_random.scale.cpu().float().numpy(),
        bins=bins,
        alpha=0.6,
        label="random→policy",
        color="tab:orange",
    )
    ax_scale.set_xlabel("Scale")
    ax_scale.set_ylabel("Count")
    ax_scale.legend(loc="upper right", fontsize=8)
    ax_scale.set_title("Policy scales")

    plt.tight_layout()
    return fig
