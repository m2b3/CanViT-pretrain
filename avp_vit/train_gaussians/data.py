"""Synthetic gaussian blob data generation for AVP training."""

import torch
from torch import Tensor

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def imagenet_denormalize(t: Tensor) -> Tensor:
    """Denormalize ImageNet-normalized tensor to [0, 1]."""
    mean = torch.tensor(IMAGENET_MEAN, device=t.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=t.device).view(1, 3, 1, 1)
    return (t * std + mean).clamp(0, 1)


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

    hi = (h6.long() % 6).unsqueeze(-1)

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

    # Independent random positions per blob per sample (overlaps allowed)
    valid_range = 1 - margin
    all_centers = (torch.rand(B, n_blobs, 2, device=device) * 2 - 1) * valid_range

    # Random sigmas: [B, n_blobs]
    sigmas = (
        torch.rand(B, n_blobs, device=device) * (sigma_range[1] - sigma_range[0])
        + sigma_range[0]
    )

    # Create coordinate grids
    lin = torch.linspace(-1, 1, size, device=device)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")

    # Vectorized gaussian computation: [B, n_blobs, size, size]
    cy = all_centers[:, :, 0].view(B, n_blobs, 1, 1)
    cx = all_centers[:, :, 1].view(B, n_blobs, 1, 1)
    sig = sigmas.view(B, n_blobs, 1, 1)

    dist_sq = (yy.unsqueeze(0).unsqueeze(0) - cy) ** 2 + (
        xx.unsqueeze(0).unsqueeze(0) - cx
    ) ** 2
    gaussians = torch.exp(-dist_sq / (2 * sig**2 + 1e-8))

    # Sum gaussians to grayscale, then expand to RGB
    gray = gaussians.sum(dim=1, keepdim=True).clamp(0, 1)
    images = gray.expand(-1, 3, -1, -1).clone()

    # Shuffle color-to-position mapping per batch sample
    color_perm = torch.stack([torch.randperm(n_blobs, device=device) for _ in range(B)])
    colors_shuffled = colors[color_perm]

    # Paint tiny colored cross at blob centers BEFORE noise (so markers get noised too)
    cross_arm = marker_size // 2
    for b in range(B):
        for i in range(n_blobs):
            py = int((all_centers[b, i, 0].item() + 1) / 2 * size)
            px = int((all_centers[b, i, 1].item() + 1) / 2 * size)
            color = colors_shuffled[b, i]
            # Vertical arm
            y0, y1 = max(0, py - cross_arm), min(size, py + cross_arm + 1)
            if 0 <= px < size:
                for yy_idx in range(y0, y1):
                    images[b, :, yy_idx, px] = color
            # Horizontal arm
            x0, x1 = max(0, px - cross_arm), min(size, px + cross_arm + 1)
            if 0 <= py < size:
                for xx_idx in range(x0, x1):
                    images[b, :, py, xx_idx] = color

    # Add gaussian color noise AFTER markers - makes them blend in at full res
    noise_std = 0.15
    color_noise = torch.randn(B, 3, size, size, device=device) * noise_std
    images = (images + color_noise).clamp(0, 1)

    # Random target selection: [B]
    target_idx = torch.randint(n_blobs, (B,), device=device)

    # Gather target colors and centers
    target_colors = colors_shuffled[torch.arange(B, device=device), target_idx]
    target_centers = all_centers[torch.arange(B, device=device), target_idx]

    # Apply ImageNet normalization
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    images = (images - mean) / std

    return images, target_colors, target_centers, all_centers
