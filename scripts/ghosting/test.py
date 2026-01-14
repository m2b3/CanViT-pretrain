"""Verification tests for ghosting phenomenon analysis.

This module contains three verification functions to validate the run_trajectory
implementation:
1. Smoke test - basic functionality with dummy inputs
2. Value checks - verify delta maps are in expected ranges
3. Comparison with visualization - ensure consistency with plot.py logic
"""

from dataclasses import dataclass
from pathlib import Path
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
from torch import Tensor
import torchvision.transforms.functional as TF

from avp_vit import ActiveCanViT
from avp_vit.checkpoint import load_model
from avp_vit.train.viz.image import imagenet_denormalize
from avp_vit.train.viz.plot import plot_multistep_pca

from .define_phenom import run_trajectory


def test_smoke(model: ActiveCanViT, device: torch.device) -> None:
    """Test basic functionality with dummy inputs.

    Args:
        model: The active vision model to test
        device: Device to run on (e.g., torch.device('cuda'))

    Raises:
        AssertionError: If any validation check fails
    """
    B = 2  # Reduced from 4 to save GPU memory
    canvas_grid = 32
    n_viewpoints = 5

    # Create dummy input
    images = torch.randn(B, 3, 224, 224, device=device)

    # Run trajectory
    result = run_trajectory(model, images, canvas_grid, 224, n_viewpoints)

    # Verify shapes: [T, B, G, G]
    expected_T = n_viewpoints
    assert result.delta_scene.shape == (expected_T, B, canvas_grid, canvas_grid), (
        f"Expected scene delta shape ({expected_T}, {B}, {canvas_grid}, {canvas_grid}), "
        f"got {result.delta_scene.shape}"
    )
    assert result.delta_hidden.shape == (expected_T, B, canvas_grid, canvas_grid), (
        f"Expected hidden delta shape ({expected_T}, {B}, {canvas_grid}, {canvas_grid}), "
        f"got {result.delta_hidden.shape}"
    )

    print("✓ Smoke test passed")


def test_values(
    delta_scene: np.ndarray, delta_hidden: np.ndarray
) -> None:
    """Verify delta map values are in expected ranges.

    Args:
        delta_scene: Scene delta maps from run_trajectory [T, B, G, G]
        delta_hidden: Hidden delta maps from run_trajectory [T, B, G, G]

    Raises:
        AssertionError: If any value check fails
    """
    # Check shapes match
    assert delta_scene.shape == delta_hidden.shape, (
        f"Shape mismatch: scene {delta_scene.shape} vs hidden {delta_hidden.shape}"
    )

    # Verify 4D shape [T, B, G, G]
    assert len(delta_scene.shape) == 4, (
        f"Expected 4D array, got shape {delta_scene.shape}"
    )
    _T, _B, G, G2 = delta_scene.shape
    assert G == G2, f"Expected square grid, got {G}x{G2}"

    # All should be in [0, 2] (cosine dissimilarity range)
    assert np.all((delta_scene >= 0) & (delta_scene <= 2)), (
        f"Scene delta out of range [0,2]: min={np.min(delta_scene)}, max={np.max(delta_scene)}"
    )
    assert np.all((delta_hidden >= 0) & (delta_hidden <= 2)), (
        f"Hidden delta out of range [0,2]: min={np.min(delta_hidden)}, max={np.max(delta_hidden)}"
    )

    # No NaN or Inf
    assert not np.any(np.isnan(delta_scene)), "NaN in scene delta"
    assert not np.any(np.isnan(delta_hidden)), "NaN in hidden delta"
    assert not np.any(np.isinf(delta_scene)), "Inf in scene delta"
    assert not np.any(np.isinf(delta_hidden)), "Inf in hidden delta"

    print("✓ Value checks passed")


def test_compare_with_viz(
    model: ActiveCanViT,
    images: Tensor,
    canvas_grid: int,
    glimpse_size_px: int,
    n_viewpoints: int,
) -> None:
    """Generate visualization using plot_multistep_pca.

    Uses the same visualization logic as the main training validation code
    to render a comprehensive multi-row diagnostic figure.

    Args:
        model: The active vision model
        images: Input images [B, C, H, W] - index 0 will be replaced with real image
        canvas_grid: Canvas grid size
        glimpse_size_px: Glimpse size in pixels
        n_viewpoints: Number of viewpoints to sample
    """
    # Load a real ImageNet image for sample 0
    img_path = Path("/home/sabrina/Documents/datasets/imagenet/val/n07717410/ILSVRC2012_val_00042721.JPEG")
    img_pil = Image.open(img_path).convert("RGB")

    # Resize to match expected image size and normalize with ImageNet stats
    img_size = images.shape[-1]
    img_pil = img_pil.resize((img_size, img_size))
    img_tensor = TF.to_tensor(img_pil)  # [3, H, W] in [0, 1]
    img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    images[0] = img_tensor.to(images.device)

    # Run trajectory to get all viz data
    result = run_trajectory(model, images, canvas_grid, glimpse_size_px, n_viewpoints)

    # Verify delta shapes
    T = result.delta_scene.shape[0]
    assert T == n_viewpoints, f"Expected {n_viewpoints} timesteps, got {T}"

    # Prepare full_img from the input images (denormalized to [0,1] RGB)
    full_img = imagenet_denormalize(images[0].cpu()).numpy()

    # Use final scene prediction as reference (avoids loading teacher backbone)
    teacher = result.scenes[-1]

    # Convert viewpoints to boxes and names
    H, W = images.shape[-2:]
    boxes = [vp.to_pixel_box(0, H, W) for vp in result.viewpoints]
    names = [vp.name for vp in result.viewpoints]

    # Compute glimpse grid size
    glimpse_grid = glimpse_size_px // model.backbone.patch_size_px

    # Generate comprehensive visualization
    fig = plot_multistep_pca(
        full_img=full_img,
        teacher=teacher,
        scenes=result.scenes,
        glimpses=result.glimpses,
        boxes=boxes,
        names=names,
        scene_grid_size=canvas_grid,
        glimpse_grid_size=glimpse_grid,
        initial_scene=result.initial_scene,
        hidden_spatials=result.hidden_spatials,
        initial_hidden_spatial=result.initial_hidden_spatial,
    )

    # Save figure
    output_path = Path("ghosting_delta_maps_mine.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Also plot delta_scene and delta_hidden from run_trajectory for comparison
    # Extract sample 0: [T, G, G]
    delta_scene_s0 = result.delta_scene[:, 0, :, :]
    delta_hidden_s0 = result.delta_hidden[:, 0, :, :]

    fig_deltas, axes = plt.subplots(2, T, figsize=(4 * T, 8))
    for t in range(T):
        # Top row: delta_scene
        im_s = axes[0, t].imshow(delta_scene_s0[t], cmap="viridis")
        axes[0, t].set_title(f"Δ Scene t={t}\n{names[t]}")
        axes[0, t].axis("off")
        fig_deltas.colorbar(im_s, ax=axes[0, t], fraction=0.046, pad=0.04)

        # Bottom row: delta_hidden
        im_h = axes[1, t].imshow(delta_hidden_s0[t], cmap="viridis")
        axes[1, t].set_title(f"Δ Hidden t={t}")
        axes[1, t].axis("off")
        fig_deltas.colorbar(im_h, ax=axes[1, t], fraction=0.046, pad=0.04)

    axes[0, 0].set_ylabel("Scene Deltas", fontsize=12)
    axes[1, 0].set_ylabel("Hidden Deltas", fontsize=12)
    fig_deltas.suptitle("Deltas from run_trajectory (for comparison with plot_multistep_pca)", fontsize=14)
    plt.tight_layout()

    delta_output_path = Path("ghosting_deltas_original.png")
    fig_deltas.savefig(delta_output_path, dpi=150, bbox_inches="tight")
    plt.close(fig_deltas)

    print("✓ Visualization saved to", output_path)
    print("✓ Delta comparison saved to", delta_output_path)


@dataclass
class Config:
    checkpoint: Path
    device: str = "mps"
    canvas_grid: int = 32
    glimpse_grid: int = 8
    n_viewpoints: int = 5


def main() -> None:
    """Run verification tests for ghosting phenomenon analysis."""
    cfg = tyro.cli(Config)
    device = torch.device(cfg.device)

    print(f"Loading model from {cfg.checkpoint}")
    model = load_model(cfg.checkpoint, device)

    patch_size = model.backbone.patch_size_px
    glimpse_size_px = cfg.glimpse_grid * patch_size
    img_size = cfg.canvas_grid * patch_size

    print(f"Device: {device}")
    print(f"Grid: {cfg.canvas_grid}, glimpse: {cfg.glimpse_grid}, image: {img_size}px")
    print(f"Viewpoints: {cfg.n_viewpoints}")
    print()

    # Run smoke test
    print("Running smoke test...")
    test_smoke(model, device)
    print()

    # Run trajectory with sample images to get deltas for value checks
    print("Running trajectory to generate deltas for value checks...")
    B = 2  # Reduced from 4 to save GPU memory
    images = torch.randn(B, 3, img_size, img_size, device=device)

    result = run_trajectory(
        model, images, cfg.canvas_grid, glimpse_size_px, cfg.n_viewpoints
    )

    print("Running value checks...")
    test_values(result.delta_scene, result.delta_hidden)
    print()

    # Run comparison test
    print("Running comparison test...")
    test_compare_with_viz(model, images, cfg.canvas_grid, glimpse_size_px, cfg.n_viewpoints)
    print()

    print("All tests passed! ✓")


if __name__ == "__main__":
    main()
