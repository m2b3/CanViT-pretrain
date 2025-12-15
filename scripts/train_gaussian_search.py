"""Train AVP policy to find target-colored gaussian among distractors.

Visual search task demonstrating end-to-end differentiable policy learning.

Task:
- Canvas with N colored gaussian blobs (different colors)
- A target color is given as query
- Policy must center viewpoint on the blob matching target color

Key elements:
- Policy network: (hidden state, target color) → viewpoint
- Same image + different query → different expected behavior
- Reparameterization trick for differentiable sampling
- Reward: -distance(viewpoint_center, target_blob_center)
"""

import copy
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path

import comet_ml
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from dinov3.hub.backbones import dinov3_vits16
from matplotlib.figure import Figure
from torch import Tensor
from tqdm import tqdm
from ymc.lr import get_linear_scaled_lr
from ytch.device import get_sensible_device
from ytch.model import count_parameters

from avp_vit import AVPConfig, AVPViT
from avp_vit.backbone.dinov3 import DINOv3Backbone
from avp_vit.glimpse import Viewpoint
from avp_vit.train import warmup_cosine_scheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


# ============================================================================
# Synthetic data: multiple colored gaussian blobs on a canvas
# ============================================================================


def make_gaussian_grid(
    size: int, centers: Tensor, sigmas: Tensor, device: torch.device
) -> Tensor:
    """Create 2D gaussian intensity maps.

    Args:
        size: Grid size (pixels)
        centers: [N, 2] normalized centers in [-1, 1], (y, x) order
        sigmas: [N] normalized sigma (relative to size)
        device: Target device

    Returns:
        [N, size, size] intensity maps in [0, 1]
    """
    N = centers.shape[0]
    lin = torch.linspace(-1, 1, size, device=device)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")
    yy = yy.unsqueeze(0).expand(N, -1, -1)
    xx = xx.unsqueeze(0).expand(N, -1, -1)

    cy = centers[:, 0].view(N, 1, 1)
    cx = centers[:, 1].view(N, 1, 1)
    sig = sigmas.view(N, 1, 1)

    dist_sq = (yy - cy) ** 2 + (xx - cx) ** 2
    gaussian = torch.exp(-dist_sq / (2 * sig**2 + 1e-8))
    return gaussian


def hsv_to_rgb(h: Tensor, s: Tensor, v: Tensor) -> Tensor:
    """Vectorized HSV to RGB conversion.

    Args:
        h, s, v: [N] tensors in [0, 1]

    Returns:
        [N, 3] RGB tensor in [0, 1]
    """
    c = s * v
    h6 = h * 6.0
    x = c * (1 - (h6 % 2 - 1).abs())
    m = v - c

    hi = (h6.long() % 6).unsqueeze(-1)  # [N, 1]

    # Build RGB based on which sextant
    rgb = torch.zeros(h.shape[0], 3, device=h.device)
    rgb[:, 0] = torch.where(
        (hi.squeeze() == 0) | (hi.squeeze() == 5),
        c,
        torch.where((hi.squeeze() == 1) | (hi.squeeze() == 4), x, torch.zeros_like(c)),
    )
    rgb[:, 1] = torch.where(
        (hi.squeeze() == 1) | (hi.squeeze() == 2),
        c,
        torch.where((hi.squeeze() == 0) | (hi.squeeze() == 3), x, torch.zeros_like(c)),
    )
    rgb[:, 2] = torch.where(
        (hi.squeeze() == 3) | (hi.squeeze() == 4),
        c,
        torch.where((hi.squeeze() == 2) | (hi.squeeze() == 5), x, torch.zeros_like(c)),
    )
    rgb += m.unsqueeze(-1)
    return rgb


def generate_multi_blob_batch(
    B: int,
    size: int,
    n_blobs: int,
    device: torch.device,
    margin: float = 0.3,
    sigma_range: tuple[float, float] = (0.08, 0.12),
    marker_size: int = 6,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Generate batch of canvases with gray gaussian blobs + tiny colored markers.

    The colored markers at blob centers are small enough to be indistinguishable
    at full resolution (~0.4 patches) but visible when zoomed to min_scale (~1.7 patches).
    This forces active vision: model must zoom into blobs to identify colors.

    Uses grid-based placement with jitter for guaranteed separation.

    Args:
        B: Batch size
        size: Canvas size (pixels)
        n_blobs: Number of blobs per image
        device: Target device
        margin: Keep blob centers away from edges
        sigma_range: (min, max) sigma
        marker_size: Size of colored marker in pixels (default 6)

    Returns:
        images: [B, 3, size, size] in ImageNet-normalized space
        target_colors: [B, 3] target color (query)
        target_centers: [B, 2] true target centers in [-1, 1], (y, x)
        all_centers: [B, n_blobs, 2] all blob centers
    """
    # Generate distinct colors: evenly spaced hues
    hues = torch.linspace(0, 1, n_blobs + 1, device=device)[:-1]
    saturation = torch.ones(n_blobs, device=device) * 0.9
    value = torch.ones(n_blobs, device=device) * 0.9
    colors = hsv_to_rgb(hues, saturation, value)  # [n_blobs, 3]

    # Grid-based placement: arrange blobs on a grid with jitter
    grid_size = int((n_blobs**0.5) + 0.999)  # ceil
    valid_range = 1 - margin
    cell_size = 2 * valid_range / grid_size

    # Base grid positions
    base_positions = []
    for gy in range(grid_size):
        for gx in range(grid_size):
            if len(base_positions) >= n_blobs:
                break
            cy = -valid_range + cell_size * (gy + 0.5)
            cx = -valid_range + cell_size * (gx + 0.5)
            base_positions.append([cy, cx])
        if len(base_positions) >= n_blobs:
            break
    base_pos = torch.tensor(base_positions[:n_blobs], device=device)  # [n_blobs, 2]

    # Add random jitter within cell
    jitter_range = cell_size * 0.3
    jitter = (torch.rand(B, n_blobs, 2, device=device) * 2 - 1) * jitter_range
    all_centers = base_pos.unsqueeze(0) + jitter  # [B, n_blobs, 2]
    all_centers = all_centers.clamp(-valid_range, valid_range)

    # Random sigmas: [B, n_blobs]
    sigmas = (
        torch.rand(B, n_blobs, device=device) * (sigma_range[1] - sigma_range[0])
        + sigma_range[0]
    )

    # Create coordinate grids
    lin = torch.linspace(-1, 1, size, device=device)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")  # [size, size]

    # Vectorized gaussian computation: [B, n_blobs, size, size]
    cy = all_centers[:, :, 0].view(B, n_blobs, 1, 1)
    cx = all_centers[:, :, 1].view(B, n_blobs, 1, 1)
    sig = sigmas.view(B, n_blobs, 1, 1)

    dist_sq = (yy.unsqueeze(0).unsqueeze(0) - cy) ** 2 + (
        xx.unsqueeze(0).unsqueeze(0) - cx
    ) ** 2
    gaussians = torch.exp(-dist_sq / (2 * sig**2 + 1e-8))  # [B, n_blobs, size, size]

    # Sum gaussians to grayscale, then expand to RGB
    gray = gaussians.sum(dim=1, keepdim=True).clamp(0, 1)  # [B, 1, H, W]
    images = gray.expand(-1, 3, -1, -1).clone()  # [B, 3, H, W]

    # Shuffle color-to-position mapping per batch sample
    color_perm = torch.stack([torch.randperm(n_blobs, device=device) for _ in range(B)])
    colors_shuffled = colors[color_perm]  # [B, n_blobs, 3]

    # Paint tiny colored cross at blob centers BEFORE noise (so markers get noised too)
    # Cross is 1px thick, marker_size//2 px arms
    cross_arm = marker_size // 2
    for b in range(B):
        for i in range(n_blobs):
            py = int((all_centers[b, i, 0].item() + 1) / 2 * size)
            px = int((all_centers[b, i, 1].item() + 1) / 2 * size)
            color = colors_shuffled[b, i]  # [3]
            # Vertical arm (1px wide)
            y0, y1 = max(0, py - cross_arm), min(size, py + cross_arm + 1)
            if 0 <= px < size:
                for yy in range(y0, y1):
                    images[b, :, yy, px] = color
            # Horizontal arm (1px tall)
            x0, x1 = max(0, px - cross_arm), min(size, px + cross_arm + 1)
            if 0 <= py < size:
                for xx in range(x0, x1):
                    images[b, :, py, xx] = color

    # Add gaussian color noise AFTER markers - makes them blend in at full res
    noise_std = 0.15
    color_noise = torch.randn(B, 3, size, size, device=device) * noise_std
    images = (images + color_noise).clamp(0, 1)

    # Random target selection: [B]
    target_idx = torch.randint(n_blobs, (B,), device=device)

    # Gather target colors and centers
    target_colors = colors_shuffled[torch.arange(B, device=device), target_idx]  # [B, 3]
    target_centers = all_centers[torch.arange(B, device=device), target_idx]  # [B, 2]

    # Apply ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    images = (images - mean) / std

    return images, target_colors, target_centers, all_centers


# ============================================================================
# Policy network: (hidden state, target color) → viewpoint
# ============================================================================


class TransformerBlock(nn.Module):
    """Simple transformer block with pre-norm."""

    def __init__(self, dim: int, n_heads: int = 4, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViewpointPolicy(nn.Module):
    """Policy that predicts viewpoint from scene hidden state and target query.

    Architecture:
    1. Avg pool scene from HxW to pool_size x pool_size (preserves spatial structure)
    2. Project to policy hidden dim
    3. Append query token (from color embedding)
    4. Run through N transformer blocks
    5. Read from query token to predict viewpoint

    Outputs tanh-bounded center + sigmoid-bounded scale.
    Fixed gaussian noise for exploration (no learned std).
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        n_transformer_blocks: int = 2,
        n_heads: int = 4,
        pool_size: int = 4,
        min_scale: float = 0.25,
        max_scale: float = 1.0,
        noise_std: float = 0.1,
        center_head_init_scale: float = 0.1,
        scale_head_init_scale: float = 0.01,
    ) -> None:
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.noise_std = noise_std
        self.pool_size = pool_size

        # Project scene features to policy dim
        self.scene_proj = nn.Linear(embed_dim, hidden_dim)

        # Color embedding → query token
        self.color_embed = nn.Linear(3, hidden_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads) for _ in range(n_transformer_blocks)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)

        # Output MLP: norm → 2-layer SiLU MLP → heads
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.center_head = nn.Linear(hidden_dim, 2)
        self.scale_head = nn.Linear(hidden_dim, 1)

        # Init
        self._init_weights(center_head_init_scale, scale_head_init_scale)

    def _init_weights(self, center_head_init_scale: float, scale_head_init_scale: float) -> None:
        # Orthogonal init for projections
        nn.init.orthogonal_(self.scene_proj.weight, gain=1.0)
        nn.init.zeros_(self.scene_proj.bias)
        nn.init.orthogonal_(self.color_embed.weight, gain=1.0)
        nn.init.zeros_(self.color_embed.bias)

        # Orthogonal init for output MLP (gain=sqrt(2) for SiLU)
        for m in self.output_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=2**0.5)
                nn.init.zeros_(m.bias)

        # Small uniform init on final heads
        nn.init.uniform_(self.center_head.weight, -center_head_init_scale, center_head_init_scale)
        nn.init.zeros_(self.center_head.bias)
        nn.init.uniform_(self.scale_head.weight, -scale_head_init_scale, scale_head_init_scale)
        nn.init.zeros_(self.scale_head.bias)

    def forward(
        self, hidden: Tensor, target_color: Tensor, deterministic: bool = False
    ) -> tuple[Viewpoint, dict[str, Tensor]]:
        """Predict viewpoint from hidden state and target query.

        Args:
            hidden: Scene hidden state [B, H*W, D] (will be reshaped to spatial)
            target_color: Target color query [B, 3]
            deterministic: If True, no noise

        Returns:
            viewpoint: Viewpoint with centers and scales
            stats: Dict of statistics for logging
        """
        B, N, D = hidden.shape
        H = W = int(N**0.5)
        assert H * W == N, f"Expected square spatial dims, got {N} tokens"

        # Reshape to spatial and avg pool to pool_size x pool_size
        spatial = hidden.view(B, H, W, D).permute(0, 3, 1, 2)  # [B, D, H, W]
        pooled = nn.functional.adaptive_avg_pool2d(spatial, self.pool_size)  # [B, D, pool_size, pool_size]
        pooled = pooled.permute(0, 2, 3, 1).reshape(B, self.pool_size * self.pool_size, D)  # [B, pool_size^2, D]

        # Project to policy dim
        scene_tokens = self.scene_proj(pooled)  # [B, pool_size^2, hidden_dim]

        # Create query token from color
        query = self.color_embed(target_color).unsqueeze(1)  # [B, 1, hidden_dim]

        # Concatenate: [scene_tokens, query]
        tokens = torch.cat([scene_tokens, query], dim=1)  # [B, pool_size^2 + 1, hidden_dim]

        # Transformer blocks
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.final_norm(tokens)

        # Read from query token (last position) and pass through output MLP
        query_out = tokens[:, -1]  # [B, hidden_dim]
        query_out = self.output_mlp(query_out)  # [B, hidden_dim]

        # Pre-activation logits
        center_logits = self.center_head(query_out)  # [B, 2]
        scale_logit = self.scale_head(query_out).squeeze(-1)  # [B]

        # Add noise to logits (reparameterization), then bound with tanh/sigmoid
        if deterministic:
            noisy_center = center_logits
            noisy_scale_logit = scale_logit
        else:
            noisy_center = center_logits + torch.randn_like(center_logits) * self.noise_std
            noisy_scale_logit = scale_logit + torch.randn_like(scale_logit) * self.noise_std

        # Sigmoid bounds scale to [min_scale, max_scale]
        scale = torch.sigmoid(noisy_scale_logit) * (self.max_scale - self.min_scale) + self.min_scale
        # tanh bounds center, scaled by valid offset given scale
        max_offset = 1 - scale  # Valid center range: [-max_offset, max_offset]
        centers = torch.tanh(noisy_center) * max_offset.unsqueeze(-1)

        stats = {
            "center_logits_y": center_logits[:, 0],
            "center_logits_x": center_logits[:, 1],
            "scale_logit": scale_logit,
            "center_y": centers[:, 0],
            "center_x": centers[:, 1],
            "scale": scale,
        }

        viewpoint = Viewpoint(name="policy", centers=centers, scales=scale)
        return viewpoint, stats


# ============================================================================
# Visualization utilities
# ============================================================================


def log_figure(exp: comet_ml.Experiment, fig: Figure, name: str, step: int) -> None:
    """Log matplotlib figure to Comet."""
    with io.BytesIO() as buf:
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        exp.log_image(buf, name=name, step=step)
    plt.close(fig)


def plot_policy_scatter(
    pred_centers: Tensor,
    target_centers: Tensor,
    target_colors: Tensor,
    title: str,
    n_samples: int = 200,
) -> Figure:
    """Scatter plot of predicted vs target viewpoint centers.

    Args:
        pred_centers: [B, 2] predicted centers (y, x)
        target_centers: [B, 2] target centers (y, x)
        target_colors: [B, 3] RGB colors for each point
        title: Plot title
        n_samples: Max samples to plot

    Returns:
        Figure with scatter plot
    """
    n = min(n_samples, pred_centers.shape[0])

    pred = pred_centers[:n].cpu().numpy()
    tgt = target_centers[:n].cpu().numpy()
    colors = target_colors[:n].cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot targets as circles
    ax.scatter(
        tgt[:, 1],
        tgt[:, 0],
        c=colors,
        s=100,
        marker="o",
        edgecolors="black",
        linewidths=1,
        label="Target",
        alpha=0.7,
    )

    # Plot predictions as X marks
    ax.scatter(
        pred[:, 1],
        pred[:, 0],
        c=colors,
        s=60,
        marker="x",
        linewidths=2,
        label="Predicted",
        alpha=0.9,
    )

    # Draw lines connecting each pred-target pair
    for i in range(n):
        ax.plot(
            [tgt[i, 1], pred[i, 1]],
            [tgt[i, 0], pred[i, 0]],
            c=colors[i],
            alpha=0.3,
            linewidth=1,
        )

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    return fig


def plot_sample_images(
    images: Tensor,
    target_colors: Tensor,
    pred_centers: Tensor,
    target_centers: Tensor,
    n_samples: int = 8,
) -> Figure:
    """Plot sample images with predicted and target viewpoints.

    Args:
        images: [B, 3, H, W] ImageNet-normalized images
        target_colors: [B, 3] RGB target colors
        pred_centers: [B, 2] predicted centers (y, x)
        target_centers: [B, 2] target centers (y, x)
        n_samples: Number of samples to show
    """
    n = min(n_samples, images.shape[0])
    H, W = images.shape[2], images.shape[3]

    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    imgs = (images[:n] * std + mean).clamp(0, 1).cpu()

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i in range(n):
        ax = axes[i]
        img = imgs[i].permute(1, 2, 0).numpy()
        ax.imshow(img)

        # Convert normalized coords to pixels
        py_tgt = (target_centers[i, 0].item() + 1) / 2 * H
        px_tgt = (target_centers[i, 1].item() + 1) / 2 * W
        py_pred = (pred_centers[i, 0].item() + 1) / 2 * H
        px_pred = (pred_centers[i, 1].item() + 1) / 2 * W

        # Target as circle
        ax.scatter(px_tgt, py_tgt, c="white", s=200, marker="o", edgecolors="black", linewidths=2)
        # Prediction as X
        ax.scatter(px_pred, py_pred, c="lime", s=150, marker="x", linewidths=3)

        # Line connecting them
        ax.plot([px_tgt, px_pred], [py_tgt, py_pred], "w--", linewidth=2, alpha=0.7)

        # Show target color in corner
        color = target_colors[i].cpu().numpy()
        ax.add_patch(mpatches.Rectangle((5, 5), 30, 30, facecolor=color, edgecolor="white", linewidth=2))

        ax.set_title(f"dist={torch.norm(pred_centers[i] - target_centers[i]).item():.3f}")
        ax.axis("off")

    plt.tight_layout()
    return fig


def plot_trajectory_with_glimpses(
    images: Tensor,
    target_colors: Tensor,
    target_centers: Tensor,
    viewpoints: list[Viewpoint],
    glimpses: list[Tensor],
    sample_idx: int = 0,
) -> Figure:
    """Plot trajectory showing image, viewpoint boxes over time, and glimpses.

    Args:
        images: [B, 3, H, W] ImageNet-normalized images
        target_colors: [B, 3] RGB target colors
        target_centers: [B, 2] target centers (y, x) in [-1, 1]
        viewpoints: List of viewpoints at each timestep
        glimpses: List of [B, 3, G, G] glimpse tensors at each timestep
        sample_idx: Which batch sample to visualize
    """
    n_steps = len(viewpoints)
    H, W = images.shape[2], images.shape[3]

    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    img = (images[sample_idx : sample_idx + 1] * std + mean).clamp(0, 1)[0]
    img_np = img.permute(1, 2, 0).cpu().numpy()

    # Denormalize glimpses
    glimpse_imgs = []
    for g in glimpses:
        g_denorm = (g[sample_idx : sample_idx + 1] * std + mean).clamp(0, 1)[0]
        glimpse_imgs.append(g_denorm.permute(1, 2, 0).cpu().numpy())

    # Get boxes
    boxes = [vp.to_pixel_box(sample_idx, H, W) for vp in viewpoints]

    # Colors for each timestep
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, n_steps - 1)) for i in range(n_steps)]

    # Layout: main image + n_steps glimpses
    fig, axes = plt.subplots(1, n_steps + 1, figsize=(4 * (n_steps + 1), 4))

    # Main image with trajectory
    ax = axes[0]
    ax.imshow(img_np)

    # Draw target marker
    ty = (target_centers[sample_idx, 0].item() + 1) / 2 * H
    tx = (target_centers[sample_idx, 1].item() + 1) / 2 * W
    ax.scatter(tx, ty, c="white", s=200, marker="*", edgecolors="black", linewidths=2, zorder=10, label="Target")

    # Draw connecting lines
    if n_steps > 1:
        xs = [b.center_x for b in boxes]
        ys = [b.center_y for b in boxes]
        ax.plot(xs, ys, "-", color="white", linewidth=1.5, alpha=0.7, zorder=1)

    # Draw boxes
    for i, box in enumerate(boxes):
        rect = mpatches.Rectangle(
            (box.left, box.top),
            box.width,
            box.height,
            linewidth=2,
            edgecolor=colors[i],
            facecolor="none",
            label=f"t={i} (s={box.width / W:.2f})",
        )
        ax.add_patch(rect)
        ax.plot(box.center_x, box.center_y, "o", color=colors[i], markersize=6, zorder=2)

    # Target color patch
    tgt_color = target_colors[sample_idx].cpu().numpy()
    ax.add_patch(mpatches.Rectangle((5, 5), 30, 30, facecolor=tgt_color, edgecolor="white", linewidth=2))

    ax.set_title("Trajectory")
    ax.legend(loc="upper right", fontsize=7)
    ax.axis("off")

    # Glimpses
    for i, g_img in enumerate(glimpse_imgs):
        ax = axes[i + 1]
        ax.imshow(g_img)
        ax.set_title(f"Glimpse t={i}")
        ax.axis("off")

    plt.tight_layout()
    return fig


def plot_scale_distribution(
    scales_det: Tensor,
    scales_noised: Tensor,
    title: str = "Scale Distribution",
) -> Figure:
    """Plot histogram of scales (deterministic vs with noise).

    Args:
        scales_det: [B] scales from deterministic policy
        scales_noised: [B] scales from stochastic policy
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    s_det = scales_det.cpu().numpy()
    s_noised = scales_noised.cpu().numpy()

    ax.hist(s_det, bins=30, alpha=0.7, label=f"Deterministic (μ={s_det.mean():.3f})", density=True)
    ax.hist(s_noised, bins=30, alpha=0.7, label=f"With noise (μ={s_noised.mean():.3f})", density=True)

    ax.set_xlabel("Scale")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def compute_policy_grad_norms(policy: ViewpointPolicy) -> dict[str, float]:
    """Compute gradient norms for policy components."""
    def module_grad_norm(module: nn.Module) -> float:
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        if not grads:
            return 0.0
        return torch.stack([g.norm(2) for g in grads]).norm(2).item()

    return {
        "grad_policy_total": module_grad_norm(policy),
        "grad_scene_proj": module_grad_norm(policy.scene_proj),
        "grad_color_embed": module_grad_norm(policy.color_embed),
        "grad_transformer": sum(module_grad_norm(b) for b in policy.blocks),
        "grad_output_mlp": module_grad_norm(policy.output_mlp),
        "grad_center_head": module_grad_norm(policy.center_head),
        "grad_scale_head": module_grad_norm(policy.scale_head),
    }


def log_grad_breakdown(policy: ViewpointPolicy, step: int) -> None:
    """Log detailed grad breakdown to stdout (for debugging at step 0)."""
    norms = compute_policy_grad_norms(policy)
    log.info(f"Step {step} grad breakdown:")
    for name, val in norms.items():
        log.info(f"  {name}: {val:.6f}")


def summarize_policy_stats(stats: dict[str, Tensor]) -> dict[str, float]:
    """Summarize policy stats into key metrics.

    batch_spread_logits: How varied are pre-tanh outputs across the batch?
        Low = policy outputs similar values for all inputs (not learning to differentiate)
        High = policy outputs different values per input (learning)
    """
    return {
        "batch_spread_logits": (stats["center_logits_y"].std() + stats["center_logits_x"].std()).item() / 2,
        "batch_spread_centers": (stats["center_y"].std() + stats["center_x"].std()).item() / 2,
        "scale_mean": stats["scale"].mean().item(),
        "scale_std": stats["scale"].std().item(),
    }


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class Config:
    # Paths
    teacher_ckpt: Path = Path("dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
    # Model
    # scene_size = scene_grid_size * patch_size (14), glimpse similarly
    # min_scale = glimpse_grid_size / scene_grid_size
    avp: AVPConfig = field(
        default_factory=lambda: AVPConfig(
            scene_grid_size=16,  # 16*14=224px scene
            glimpse_grid_size=4,  # 4*14=56px glimpse, min_scale=0.25
            gate_init=1e-5,
            use_output_proj=True,
            use_scene_registers=False,
            gradient_checkpointing=False,
        )
    )
    policy_hidden_dim: int = 256
    policy_n_transformer_blocks: int = 2
    policy_n_heads: int = 4
    policy_pool_size: int = 4
    policy_noise_std: float = 0.1
    policy_center_head_init_scale: float = 0.1
    policy_scale_head_init_scale: float = 0.01  # 10x smaller - sigmoid more sensitive
    # Training
    n_steps_per_episode: int = 4
    n_steps: int = 10000
    batch_size: int = 256
    ref_lr: float = 1e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    # Task
    n_blobs: int = 4
    blob_margin: float = 0.3
    blob_sigma_min: float = 0.08
    blob_sigma_max: float = 0.12
    # Logging
    log_every: int = 20
    val_every: int = 100
    # Compilation
    compile: bool = False
    # Runtime
    device: torch.device = field(default_factory=get_sensible_device)

    @property
    def min_viewpoint_scale(self) -> float:
        return self.avp.glimpse_grid_size / self.avp.scene_grid_size

    @property
    def max_viewpoint_scale(self) -> float:
        return 1.0


# ============================================================================
# Model loading
# ============================================================================


def load_backbone(cfg: Config) -> DINOv3Backbone:
    model = dinov3_vits16(weights=str(cfg.teacher_ckpt), pretrained=True)
    backbone = DINOv3Backbone(model.eval().to(cfg.device))
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone


def create_avp(backbone: DINOv3Backbone, cfg: Config) -> AVPViT:
    backbone_copy = copy.deepcopy(backbone)
    for p in backbone_copy.parameters():
        p.requires_grad = False
    return AVPViT(backbone_copy, cfg.avp).to(cfg.device)


def compile_model(avp: AVPViT) -> None:
    assert isinstance(avp.backbone, DINOv3Backbone)
    blocks = avp.backbone._backbone.blocks
    for i in range(avp.backbone.n_blocks):
        blocks[i] = torch.compile(blocks[i])  # type: ignore[assignment]
    for i in range(avp.backbone.n_blocks):
        avp.read_attn[i] = torch.compile(avp.read_attn[i])  # type: ignore[assignment]
        avp.write_attn[i] = torch.compile(avp.write_attn[i])  # type: ignore[assignment]


# ============================================================================
# Training
# ============================================================================


def compute_distance_loss(viewpoint: Viewpoint, target_centers: Tensor) -> Tensor:
    """Mean L2 distance between viewpoint centers and target centers."""
    return torch.norm(viewpoint.centers - target_centers, dim=-1).mean()


def evaluate_policy(
    exp: comet_ml.Experiment,
    policy: ViewpointPolicy,
    avp: AVPViT,
    cfg: Config,
    step: int,
    prefix: str = "eval",
) -> float:
    """Evaluate policy with deterministic inference and log visualizations.

    Returns final distance.
    """
    from avp_vit.glimpse import extract_glimpse

    image_size = avp.scene_size
    glimpse_size = avp.cfg.glimpse_grid_size * avp.backbone.patch_size

    with torch.inference_mode():
        images, target_colors, target_centers, _ = generate_multi_blob_batch(
            cfg.batch_size,
            image_size,
            cfg.n_blobs,
            cfg.device,
            margin=cfg.blob_margin,
            sigma_range=(cfg.blob_sigma_min, cfg.blob_sigma_max),
        )

        B = images.shape[0]
        hidden = avp._init_hidden(B, None)

        # Run episode with deterministic policy - collect viewpoints and glimpses
        viewpoints: list[Viewpoint] = []
        glimpses: list[Tensor] = []
        scales_det: list[Tensor] = []
        dists_t: list[Tensor] = []

        for t in range(cfg.n_steps_per_episode):
            vp, stats = policy(hidden, target_colors, deterministic=True)
            viewpoints.append(vp)
            scales_det.append(stats["scale"])
            dists_t.append(torch.norm(vp.centers - target_centers, dim=-1).mean())

            # Extract glimpse for viz
            glimpse = extract_glimpse(images, vp, glimpse_size)
            glimpses.append(glimpse)

            out = avp.forward_step(images, vp, hidden, None)
            hidden = out.hidden

        final_vp = viewpoints[-1]
        final_dist = dists_t[-1].item()

        # Also run stochastic to compare scales
        hidden_fresh = avp._init_hidden(B, None)
        vp_noised, stats_noised = policy(hidden_fresh, target_colors, deterministic=False)
        scales_noised = stats_noised["scale"]

        # Log per-timestep distance
        for t, d in enumerate(dists_t):
            exp.log_metric(f"{prefix}/dist_t{t}", d.item(), step=step)
        exp.log_metric(f"{prefix}/dist", final_dist, step=step)

        # Scatter plot of final predictions (batch overview)
        fig = plot_policy_scatter(
            final_vp.centers, target_centers, target_colors,
            f"{prefix} - Step {step}"
        )
        log_figure(exp, fig, f"{prefix}/scatter", step)

        # Trajectory with glimpses (detailed view of one sample)
        fig = plot_trajectory_with_glimpses(
            images, target_colors, target_centers, viewpoints, glimpses, sample_idx=0
        )
        log_figure(exp, fig, f"{prefix}/trajectory", step)

        # Scale distribution
        fig = plot_scale_distribution(
            torch.cat(scales_det),  # All timesteps deterministic
            scales_noised,  # Single step with noise
            title=f"Scale Distribution - Step {step}",
        )
        log_figure(exp, fig, f"{prefix}/scales", step)

        return final_dist


def train(cfg: Config) -> None:
    """Train policy to find target-colored gaussian blobs."""
    log.info(f"Device: {cfg.device}")
    log.info(f"Config: {cfg}")

    exp = comet_ml.Experiment(
        project_name="avp-gaussian-search", auto_metric_logging=False
    )
    exp.log_parameters(
        {
            k: str(v) if isinstance(v, (torch.device, Path)) else v
            for k, v in cfg.__dict__.items()
        }
    )

    log.info("Loading backbone...")
    backbone = load_backbone(cfg)
    log.info(f"Backbone params: {count_parameters(backbone):,}")

    log.info("Creating AVP model...")
    avp = create_avp(backbone, cfg)
    if cfg.compile:
        compile_model(avp)

    # Freeze AVP - only train policy
    for p in avp.parameters():
        p.requires_grad = False

    log.info("Creating policy...")
    policy = ViewpointPolicy(
        embed_dim=backbone.embed_dim,
        hidden_dim=cfg.policy_hidden_dim,
        n_transformer_blocks=cfg.policy_n_transformer_blocks,
        n_heads=cfg.policy_n_heads,
        pool_size=cfg.policy_pool_size,
        min_scale=cfg.min_viewpoint_scale,
        max_scale=cfg.max_viewpoint_scale,
        noise_std=cfg.policy_noise_std,
        center_head_init_scale=cfg.policy_center_head_init_scale,
        scale_head_init_scale=cfg.policy_scale_head_init_scale,
    ).to(cfg.device)
    log.info(f"Policy params: {count_parameters(policy):,}")

    peak_lr = get_linear_scaled_lr(cfg.ref_lr, cfg.batch_size)
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=peak_lr, weight_decay=cfg.weight_decay
    )
    warmup_steps = int(cfg.n_steps * cfg.warmup_ratio)
    scheduler = warmup_cosine_scheduler(optimizer, cfg.n_steps, warmup_steps)
    log.info(f"Optimizer: peak_lr={peak_lr:.2e}, warmup={warmup_steps}")

    exp.log_parameters({"policy_params": count_parameters(policy), "peak_lr": peak_lr})

    # =========== INITIAL EVAL (step 0, before training) ===========
    log.info("Initial evaluation...")
    init_dist = evaluate_policy(exp, policy, avp, cfg, step=0, prefix="eval")
    log.info(f"Init dist: {init_dist:.4f}")

    from avp_vit.glimpse import extract_glimpse

    ema_loss = torch.tensor(0.0, device=cfg.device)
    alpha = 2 / (cfg.log_every + 1)
    image_size = avp.scene_size
    glimpse_size = avp.cfg.glimpse_grid_size * avp.backbone.patch_size

    log.info("Starting training...")
    pbar = tqdm(range(cfg.n_steps), desc="Training", unit="step")

    for step in pbar:
        # Generate batch
        images, target_colors, target_centers, _ = generate_multi_blob_batch(
            cfg.batch_size,
            image_size,
            cfg.n_blobs,
            cfg.device,
            margin=cfg.blob_margin,
            sigma_range=(cfg.blob_sigma_min, cfg.blob_sigma_max),
        )

        # Run episode (stochastic policy for exploration)
        # Loss is based on LAST timestep only (policy must navigate to target)
        # Collect viewpoints/glimpses for trajectory viz at val_every
        B = images.shape[0]
        hidden = avp._init_hidden(B, None)
        first_stats = None
        train_viewpoints: list[Viewpoint] = []
        train_glimpses: list[Tensor] = []

        for t in range(cfg.n_steps_per_episode):
            vp, stats = policy(hidden, target_colors, deterministic=False)
            if t == 0:
                first_stats = stats
            train_viewpoints.append(vp)
            train_glimpses.append(extract_glimpse(images, vp, glimpse_size))
            out = avp.forward_step(images, vp, hidden, None)
            hidden = out.hidden

        final_vp = train_viewpoints[-1]
        loss = compute_distance_loss(final_vp, target_centers)

        optimizer.zero_grad()
        loss.backward()

        # Compute grad norms BEFORE clipping (only at log steps to avoid overhead)
        grad_norms = None
        if step % cfg.log_every == 0:
            grad_norms = compute_policy_grad_norms(policy)

        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        # EMA (keep as tensor, no .item() in hot loop)
        ema_loss = loss.detach() if step == 0 else alpha * loss.detach() + (1 - alpha) * ema_loss

        if step % cfg.log_every == 0:
            assert grad_norms is not None
            assert first_stats is not None
            policy_stats = summarize_policy_stats(first_stats)

            # Log detailed grad breakdown on first step
            if step == 0:
                log_grad_breakdown(policy, step)

            exp.log_metrics({
                "loss": ema_loss.item(),
                "grad_norm": grad_norm.item(),
                **grad_norms,
                "lr": scheduler.get_last_lr()[0],
                "batch_spread_logits": policy_stats["batch_spread_logits"],
                "scale_mean": policy_stats["scale_mean"],
                "scale_std": policy_stats["scale_std"],
            }, step=step)

            pbar.set_postfix_str(f"loss={ema_loss.item():.3f}")

        # Training trajectory viz (stochastic, shows noise effect)
        if step > 0 and step % cfg.val_every == 0:
            with torch.inference_mode():
                fig = plot_trajectory_with_glimpses(
                    images, target_colors, target_centers,
                    train_viewpoints, train_glimpses, sample_idx=0
                )
                log_figure(exp, fig, "train/trajectory", step)

            # Eval with visualization (deterministic)
            evaluate_policy(exp, policy, avp, cfg, step=step, prefix="eval")

    # Final eval
    final_dist = evaluate_policy(exp, policy, avp, cfg, step=cfg.n_steps, prefix="eval")
    log.info(f"Training complete. Final dist: {final_dist:.4f}")
    exp.end()


def main() -> None:
    import tyro

    torch.set_float32_matmul_precision("high")
    cfg = tyro.cli(Config)
    train(cfg)


if __name__ == "__main__":
    main()
