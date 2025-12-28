"""Visualization helpers. Pure numpy/PIL, no torch."""

import numpy as np
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA

from avp_vit.train.viz import fit_pca, pca_rgb, timestep_colors

from inference_app.types import Viewpoint

COL_W = 300


def upscale(arr: np.ndarray, size: int) -> Image.Image:
    """Upscale array to image."""
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).resize(
        (size, size), Image.Resampling.NEAREST
    )


def pca_img(features: np.ndarray, grid: int, pca: PCA, pc_off: int, *, norm: bool) -> Image.Image:
    """Create PCA visualization image."""
    return upscale(pca_rgb(pca=pca, features=features, H=grid, W=grid, normalize=norm, pc_offset=pc_off), COL_W)


def cos_sim_heatmap(features: np.ndarray, indices: list[int]) -> np.ndarray:
    """Cosine similarity vs anchor tokens. Returns [N] in [-1, 1]."""
    q = np.mean([features[i] for i in indices], axis=0)
    q_n = q / (np.linalg.norm(q) + 1e-8)
    f_n = features / (np.linalg.norm(features, axis=-1, keepdims=True) + 1e-8)
    return f_n @ q_n


def sim_to_img(sim: np.ndarray, grid: int, size: int, vmin: float, vmax: float, sharpness: float = 1.0) -> Image.Image:
    """Convert similarity [N] to heatmap. Blue=vmin, red=vmax."""
    s = sim.reshape(grid, grid)
    s = ((s - vmin) / (vmax - vmin + 1e-8)).clip(0, 1)
    s = np.power(s, sharpness)
    rgb = np.zeros((grid, grid, 3), dtype=np.uint8)
    rgb[..., 0] = (s * 255).astype(np.uint8)
    rgb[..., 2] = ((1 - s) * 255).astype(np.uint8)
    return Image.fromarray(rgb).resize((size, size), Image.Resampling.NEAREST)


def pos_to_token(nx: float, ny: float, grid: int) -> tuple[int, int, int]:
    """Normalized [0,1] position to (row, col, flat_idx)."""
    col = min(int(nx * grid), grid - 1)
    row = min(int(ny * grid), grid - 1)
    return row, col, row * grid + col


def draw_token_box(img: Image.Image, row: int, col: int, grid: int, color: tuple = (255, 255, 0)) -> Image.Image:
    """Draw rectangle around token position."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    cell = img.width / grid
    x0, y0 = col * cell, row * cell
    draw.rectangle([x0, y0, x0 + cell, y0 + cell], outline=color, width=2)
    return out


def draw_viewpoint_boxes(img: Image.Image, viewpoints: list[Viewpoint], H: int, W: int,
                         policy: tuple[float, float, float] | None = None) -> Image.Image:
    """Draw viewpoint boxes on image. Policy is (cy, cx, scale) if present."""
    out = img.copy()
    draw = ImageDraw.Draw(out, "RGBA")
    colors = timestep_colors(len(viewpoints)) if viewpoints else []

    for i, vp in enumerate(viewpoints):
        # Convert normalized coords to pixel coords
        cy_px = (vp.cy + 1) / 2 * H
        cx_px = (vp.cx + 1) / 2 * W
        half_h = vp.scale * H / 2
        half_w = vp.scale * W / 2

        r, g, b = (int(c * 255) for c in colors[i][:3])
        draw.rectangle(
            [cx_px - half_w, cy_px - half_h, cx_px + half_w, cy_px + half_h],
            outline=(r, g, b, 255), width=2
        )
        draw.ellipse([cx_px - 8, cy_px - 8, cx_px + 8, cy_px + 8], fill=(r, g, b, 200))
        draw.text((int(cx_px), int(cy_px)), str(i), fill=(255, 255, 255), anchor="mm")

    if policy:
        cy, cx, scale = policy
        cy_px = (cy + 1) / 2 * H
        cx_px = (cx + 1) / 2 * W
        half_h = scale * H / 2
        half_w = scale * W / 2
        draw.rectangle(
            [cx_px - half_w, cy_px - half_h, cx_px + half_w, cy_px + half_h],
            outline=(255, 200, 0, 200), width=2
        )
        draw.text((int(cx_px), int(cy_px)), "→", fill=(255, 200, 0), anchor="mm")

    return out


def click_to_viewpoint(cx: float, cy: float, scale: float, H: int, W: int, name: str) -> Viewpoint:
    """Convert pixel click to Viewpoint. cx/cy are pixel coords."""
    # Convert to normalized [-1, 1]
    cx_n = float(np.clip((cx / W) * 2 - 1, -(1 - scale), 1 - scale))
    cy_n = float(np.clip((cy / H) * 2 - 1, -(1 - scale), 1 - scale))
    return Viewpoint(cy=cy_n, cx=cx_n, scale=scale, name=name)


def upsample_features(features: np.ndarray, src_grid: int, dst_grid: int) -> np.ndarray:
    """Upsample features from src_grid to dst_grid using bilinear interpolation."""
    import torch
    import torch.nn.functional as F

    D = features.shape[-1]
    t = torch.from_numpy(features).float().view(src_grid, src_grid, D).permute(2, 0, 1).unsqueeze(0)
    out = F.interpolate(t, (dst_grid, dst_grid), mode="bilinear", align_corners=False)
    return out[0].permute(1, 2, 0).reshape(-1, D).numpy()


def glimpse_to_pil(glimpse: np.ndarray, size: int = 256) -> Image.Image:
    """Convert glimpse array [H,W,3] float to PIL Image."""
    return Image.fromarray((np.clip(glimpse, 0, 1) * 255).astype(np.uint8)).resize(
        (size, size), Image.Resampling.LANCZOS
    )
