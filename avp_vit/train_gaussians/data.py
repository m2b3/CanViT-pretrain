"""Synthetic gaussian blob data generation for AVP training."""

import torch
from torch import Tensor

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def perlin_noise_2d(
    shape: tuple[int, int, int],
    scale: float,
    octaves: int,
    persistence: float,
    device: torch.device,
) -> Tensor:
    """Generate 2D Perlin noise on GPU.

    Args:
        shape: (B, H, W) output shape
        scale: Base frequency (larger = more detail)
        octaves: Number of noise layers to combine
        persistence: Amplitude multiplier per octave (typically 0.5)
        device: Target device

    Returns:
        [B, H, W] noise in [0, 1]
    """
    B, H, W = shape

    def fade(t: Tensor) -> Tensor:
        return t * t * t * (t * (t * 6 - 15) + 10)

    def lerp(a: Tensor, b: Tensor, t: Tensor) -> Tensor:
        return a + t * (b - a)

    noise = torch.zeros(B, H, W, device=device)
    amplitude = 1.0
    max_amplitude = 0.0

    for _ in range(octaves):
        freq = scale
        # Grid size for this octave
        grid_h = max(2, int(H / freq))
        grid_w = max(2, int(W / freq))

        # Random gradients at grid points: [B, grid_h+1, grid_w+1, 2]
        angles = torch.rand(B, grid_h + 1, grid_w + 1, device=device) * 2 * 3.14159
        gradients = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)

        # Pixel coordinates in grid space
        y_coords = torch.linspace(0, grid_h - 1e-5, H, device=device)
        x_coords = torch.linspace(0, grid_w - 1e-5, W, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

        # Integer and fractional parts
        y0 = yy.long()
        x0 = xx.long()
        y1 = y0 + 1
        x1 = x0 + 1
        fy = yy - y0.float()
        fx = xx - x0.float()

        # Fade curves
        u = fade(fx)
        v = fade(fy)

        # Gather gradients for 4 corners: [B, H, W, 2]
        # Expand indices for batch dimension
        b_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, H, W)
        y0_exp = y0.unsqueeze(0).expand(B, H, W)
        y1_exp = y1.unsqueeze(0).expand(B, H, W)
        x0_exp = x0.unsqueeze(0).expand(B, H, W)
        x1_exp = x1.unsqueeze(0).expand(B, H, W)

        g00 = gradients[b_idx, y0_exp, x0_exp]
        g01 = gradients[b_idx, y0_exp, x1_exp]
        g10 = gradients[b_idx, y1_exp, x0_exp]
        g11 = gradients[b_idx, y1_exp, x1_exp]

        # Distance vectors from corners to point
        fy_exp = fy.unsqueeze(0).expand(B, H, W)
        fx_exp = fx.unsqueeze(0).expand(B, H, W)

        d00 = torch.stack([fy_exp, fx_exp], dim=-1)
        d01 = torch.stack([fy_exp, fx_exp - 1], dim=-1)
        d10 = torch.stack([fy_exp - 1, fx_exp], dim=-1)
        d11 = torch.stack([fy_exp - 1, fx_exp - 1], dim=-1)

        # Dot products
        n00 = (g00 * d00).sum(dim=-1)
        n01 = (g01 * d01).sum(dim=-1)
        n10 = (g10 * d10).sum(dim=-1)
        n11 = (g11 * d11).sum(dim=-1)

        # Bilinear interpolation
        u_exp = u.unsqueeze(0).expand(B, H, W)
        v_exp = v.unsqueeze(0).expand(B, H, W)
        nx0 = lerp(n00, n01, u_exp)
        nx1 = lerp(n10, n11, u_exp)
        octave_noise = lerp(nx0, nx1, v_exp)

        noise = noise + octave_noise * amplitude
        max_amplitude += amplitude
        amplitude *= persistence
        scale *= 2

    # Normalize to [0, 1]
    noise = (noise / max_amplitude + 1) / 2
    return noise.clamp(0, 1)


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
    marker_size: int = 12,
    use_perlin_bg: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Generate batch of canvases with gaussian blobs + colored markers on perlin background.

    The colored markers at blob centers are visible when zoomed but harder to see at full res.
    Perlin noise background adds multiscale structure for more interesting reconstruction.

    Args:
        B: Batch size
        size: Canvas size (pixels)
        n_blobs: Number of blobs per image
        device: Target device
        margin: Keep blob centers away from edges
        sigma_range: (min, max) sigma
        marker_size: Size of colored marker in pixels
        use_perlin_bg: Use perlin noise background (vs flat gray)

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

    # Background: perlin noise or flat gray
    if use_perlin_bg:
        # Multi-octave perlin for multiscale structure
        bg = perlin_noise_2d((B, size, size), scale=32, octaves=4, persistence=0.5, device=device)
        bg = bg * 0.3 + 0.35  # Range ~[0.35, 0.65] - muted background
        bg = bg.unsqueeze(1).expand(-1, 3, -1, -1)
    else:
        bg = torch.full((B, 3, size, size), 0.5, device=device)

    # Add gaussians on top of background
    blob_intensity = gaussians.sum(dim=1, keepdim=True).clamp(0, 1)  # [B, 1, H, W]
    images = bg + blob_intensity * 0.5  # Blobs add brightness
    images = images.clamp(0, 1)

    # Shuffle color-to-position mapping per batch sample
    color_perm = torch.stack([torch.randperm(n_blobs, device=device) for _ in range(B)])
    colors_shuffled = colors[color_perm]

    # Paint colored cross at blob centers
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

    # Light noise on top
    noise_std = 0.08
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
