"""Ghosting phenomenon: accumulate delta maps across trajectory.

This module provides utilities to track how scene and hidden state representations
evolve across a multi-glimpse trajectory, computing cosine dissimilarity deltas
between consecutive timesteps.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from torch import Tensor

from canvit import GlimpseOutput
from canvit.viewpoint import Viewpoint as CanvitViewpoint

from avp_vit import ActiveCanViT
from avp_vit.train.viewpoint import Viewpoint, make_eval_viewpoints
from avp_vit.train.viz.image import imagenet_denormalize
from avp_vit.train.viz.metrics import cosine_dissimilarity


@dataclass
class Tableau:
    """Snapshot of batch state at a single timestep.

    Shape annotations:
        B = batch size
        G = canvas grid size (e.g., 32)
        g = glimpse grid size (e.g., 8)
        D = teacher feature dim (e.g., 768)
        C = canvas hidden dim (e.g., 384)
    """

    scenes: NDArray[np.floating]  # [B, G², D] predicted scenes in teacher space
    canvas_spatials: NDArray[np.floating]  # [B, G², C] raw canvas hidden state
    glimpse: NDArray[np.floating] | None  # [g, g, 3] denormalized glimpse for sample 0


@dataclass
class TrajectoryResult:
    """Result from run_trajectory with viz data for sample 0."""

    delta_scene: NDArray[np.floating]  # [T, B, G, G] scene deltas
    delta_hidden: NDArray[np.floating]  # [T, B, G, G] hidden deltas
    viewpoints: list["Viewpoint"]
    # Viz data for sample 0:
    scenes: list[NDArray[np.floating]]  # [T] of [G², D] after each glimpse
    initial_scene: NDArray[np.floating]  # [G², D] before first glimpse
    hidden_spatials: list[NDArray[np.floating]]  # [T] of [G², C]
    initial_hidden_spatial: NDArray[np.floating]  # [G², C]
    glimpses: list[NDArray[np.floating]]  # [T] of [g, g, 3] RGB

def run_trajectory(
    model: ActiveCanViT,
    images: Tensor,
    canvas_grid: int,
    glimpse_size_px: int,
    n_viewpoints: int,
) -> TrajectoryResult:
    """Run trajectory and return delta maps plus visualization data.

    This function executes a multi-glimpse trajectory on the model, extracting
    both predicted scene features (in teacher space) and canvas spatial hidden
    states at each timestep. It computes cosine dissimilarity between
    consecutive timesteps to produce "delta maps" showing where and how much
    the representations changed.

    Additionally, extracts visualization data for sample 0 (scenes, hidden
    states, glimpses) for use with plot_multistep_pca.

    Args:
        model: The active vision model
        images: Input images [B, C, H, W]
        canvas_grid: Spatial resolution of canvas (e.g., 32 for 32x32)
        glimpse_size_px: Size of each glimpse in pixels
        n_viewpoints: Number of glimpses to take

    Returns:
        TrajectoryResult containing delta maps and viz data for sample 0.
    """
    B = images.shape[0]
    viewpoints = make_eval_viewpoints(B, images.device, n_viewpoints=n_viewpoints)

    def init_fn(canvas: Tensor, _cls: Tensor) -> list[Tableau]:
        """Extract initial state before any glimpses."""
        # Extract canvas spatial tokens [B, G², C]
        canvas_spatial = model.get_spatial(canvas)
        canvas_spatial_np = canvas_spatial.detach().cpu().float().numpy()

        # Predict initial scene [B, G², D]
        predicted_scene_np = model.predict_teacher_scene(canvas).detach().cpu().float().numpy()

        # Create initial tableau (no glimpse at init)
        initial_tableau = Tableau(
            scenes=predicted_scene_np,
            canvas_spatials=canvas_spatial_np,
            glimpse=None,
        )
        return [initial_tableau]

    def step_fn(
        acc: list[Tableau], out: GlimpseOutput, _vp: CanvitViewpoint
    ) -> list[Tableau]:
        """Extract state at current timestep and append to accumulator."""
        # Extract canvas spatial tokens [B, G², C]
        canvas_spatial = model.get_spatial(out.canvas)
        canvas_spatial_np = canvas_spatial.detach().cpu().float().numpy()

        # Predict scene [B, G², D]
        predicted_scene = model.predict_teacher_scene(out.canvas)
        predicted_scene_np = predicted_scene.detach().cpu().float().numpy()

        # Extract glimpse for sample 0 (denormalized to [0,1] RGB)
        glimpse_np = imagenet_denormalize(out.glimpse[0].cpu()).numpy()

        # Create tableau for this timestep
        tableau = Tableau(
            scenes=predicted_scene_np,
            canvas_spatials=canvas_spatial_np,
            glimpse=glimpse_np,
        )
        acc.append(tableau)
        return acc

    # Run trajectory
    tableaux, _, _ = model.forward_reduce(
        image=images,
        viewpoints=viewpoints,  # pyright: ignore[reportArgumentType]
        canvas_grid_size=canvas_grid,
        glimpse_size_px=glimpse_size_px,
        init_fn=init_fn,
        step_fn=step_fn,
    )

    # Compute delta maps using batched cosine_dissimilarity
    # Stack all tableaux: [T+1, B, G², *]
    all_scenes = np.stack([t.scenes for t in tableaux], axis=0)
    all_canvas_spatials = np.stack([t.canvas_spatials for t in tableaux], axis=0)

    # Compute deltas between consecutive timesteps: [T, B, G²]
    scene_deltas = cosine_dissimilarity(all_scenes[1:], all_scenes[:-1])
    hidden_deltas = cosine_dissimilarity(all_canvas_spatials[1:], all_canvas_spatials[:-1])

    # Reshape to [T, B, G, G]
    T = len(tableaux) - 1
    delta_scene_maps = scene_deltas.reshape(T, B, canvas_grid, canvas_grid)
    delta_hidden_maps = hidden_deltas.reshape(T, B, canvas_grid, canvas_grid)

    # Extract viz data for sample 0
    initial_tableau = tableaux[0]
    step_tableaux = tableaux[1:]

    # Glimpses are only in step tableaux (not initial)
    glimpses = [t.glimpse for t in step_tableaux]
    assert all(g is not None for g in glimpses), "All step tableaux should have glimpses"

    return TrajectoryResult(
        delta_scene=delta_scene_maps,
        delta_hidden=delta_hidden_maps,
        viewpoints=viewpoints,
        scenes=[t.scenes[0] for t in step_tableaux],  # sample 0
        initial_scene=initial_tableau.scenes[0],
        hidden_spatials=[t.canvas_spatials[0] for t in step_tableaux],
        initial_hidden_spatial=initial_tableau.canvas_spatials[0],
        glimpses=glimpses,  # type: ignore[arg-type]
    )
