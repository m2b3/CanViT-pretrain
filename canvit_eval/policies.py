"""Evaluation policies for CanViT.

Wraps canvit_utils.policies with an interactive API that supports
uncertainty-guided viewpoint ordering.

Convention: centers are (y, x) in [-1, 1]. Scales are in (0, 1].
y grows downward (row index), x grows rightward (column index).
This matches the convention in canvit.viewpoint.
"""

import logging
from collections.abc import Callable
from typing import Literal, Protocol

PolicyName = Literal[
    "coarse_to_fine",
    "fine_to_coarse",
    "random",
    "full_then_random",
    "entropy_coarse_to_fine",
]

import torch
from canvit import RecurrentState, Viewpoint
from canvit_utils.policies import coarse_to_fine_viewpoints, random_viewpoints
from torch import Tensor

# Type for the function that extracts spatial features from the canvas.
# Signature: (canvas: Tensor[B, N, D]) -> Tensor[B, N, D_spatial]
GetSpatialFn = Callable[[Tensor], Tensor]

log = logging.getLogger(__name__)


class EvalPolicy(Protocol):
    """Protocol for evaluation policies."""

    @property
    def name(self) -> str: ...

    def step(self, t: int, state: RecurrentState | None) -> Viewpoint:
        """Return viewpoint for timestep t, given current canvas state."""
        ...


# ── Static policies (wrap existing functions) ──────────────────────


class StaticPolicy:
    """Pre-generated viewpoints, ignoring canvas state."""

    def __init__(self, policy_name: str, viewpoints: list[Viewpoint]) -> None:
        self._name = policy_name
        self._viewpoints = viewpoints

    @property
    def name(self) -> str:
        return self._name

    def step(self, t: int, state: RecurrentState | None) -> Viewpoint:
        return self._viewpoints[t]


# ── Fine-to-coarse ─────────────────────────────────────────────────


def fine_to_coarse_viewpoints(
    batch_size: int,
    device: torch.device,
    n_viewpoints: int,
) -> list[Viewpoint]:
    """Generate fine-to-coarse quadtree viewpoints (reversed C2F).

    Visits the finest scale first, then coarser scales.
    Same quadtree structure as C2F but levels are traversed in reverse.
    Within each level, order is shuffled per batch item.
    """
    # Build levels finest-first
    levels: list[list[tuple[float, float, float]]] = []
    level = 0
    total = 0
    while total < n_viewpoints:
        lvl_vps = _level_viewpoints(level)
        levels.append(lvl_vps)
        total += len(lvl_vps)
        level += 1
    levels.reverse()  # finest first

    result: list[Viewpoint] = []
    for level_vps in levels:
        level_t = torch.tensor(level_vps, device=device, dtype=torch.float32)
        n = len(level_vps)
        if n == 1:
            perms = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        else:
            perms = torch.stack([torch.randperm(n, device=device) for _ in range(batch_size)])
        for i in range(n):
            if len(result) >= n_viewpoints:
                return result
            idx = perms[:, i]
            result.append(Viewpoint(centers=level_t[idx, :2], scales=level_t[idx, 2]))
    return result[:n_viewpoints]


# ── Entropy-guided C2F ─────────────────────────────────────────────


def _level_viewpoints(level: int) -> list[tuple[float, float, float]]:
    """Compute (y, x, scale) for all crops at a given C2F level.

    Level 0: 1 crop (full scene, scale=1.0)
    Level 1: 4 crops (2×2, scale=0.5)
    Level 2: 16 crops (4×4, scale=0.25)
    """
    n = 2**level
    scale = 1.0 / n
    result: list[tuple[float, float, float]] = []
    for row in range(n):
        for col in range(n):
            cy = (2 * row + 1) * scale - 1.0
            cx = (2 * col + 1) * scale - 1.0
            result.append((cy, cx, scale))
    return result


def _build_tile_masks(
    crop_centers: list[tuple[float, float, float]],
    canvas_grid: int,
    device: torch.device,
) -> Tensor:
    """Precompute boolean tile masks. Returns [n_tiles, G, G].

    Only correct for power-of-2 canvas grids (8, 16, 32, 64, ...).
    """
    G = canvas_grid
    assert G > 0 and (G & (G - 1)) == 0, f"canvas_grid must be a power of 2, got {G}"
    coords = torch.linspace(-1 + 1 / G, 1 - 1 / G, G, device=device)
    crops_t = torch.tensor(crop_centers, device=device)  # [n_tiles, 3] = (y, x, s)
    cy = crops_t[:, 0]  # [n_tiles]
    cx = crops_t[:, 1]
    s = crops_t[:, 2]
    # [n_tiles, G]: does each canvas row/col fall within each tile?
    row_in = (coords.unsqueeze(0) - cy.unsqueeze(1)).abs() <= s.unsqueeze(1)
    col_in = (coords.unsqueeze(0) - cx.unsqueeze(1)).abs() <= s.unsqueeze(1)
    # [n_tiles, G, G]
    return row_in.unsqueeze(2) & col_in.unsqueeze(1)


def _tile_mean_uncertainty(
    uncertainty_map: Tensor,
    tile_masks: Tensor,
) -> Tensor:
    """Mean uncertainty per tile, averaged across the tile's spatial cells.

    Fully vectorized, no Python loops or GPU syncs.

    Args:
        uncertainty_map: [B, G, G] — per-canvas-cell uncertainty (higher = more uncertain)
        tile_masks: [n_tiles, G, G] — precomputed boolean masks (which cells belong to each tile)

    Returns:
        [B, n_tiles] — for each image and each tile, the mean uncertainty
        across the canvas cells covered by that tile
    """
    # [B, 1, G, G] * [1, n_tiles, G, G] → sum over (G, G) → [B, n_tiles]
    n_cells = tile_masks.sum(dim=(1, 2)).clamp(min=1).float()  # [n_tiles]
    masked = uncertainty_map.unsqueeze(1) * tile_masks.unsqueeze(0).float()  # [B, n_tiles, G, G]
    return masked.sum(dim=(2, 3)) / n_cells.unsqueeze(0)


class EntropyGuidedC2F:
    """Coarse-to-fine with within-level ordering by probe prediction entropy.

    At each C2F level, visits crops in order of decreasing entropy
    (most uncertain first). Entropy is the categorical entropy of the
    segmentation probe's softmax predictions at each spatial position.
    The probe is already run at every timestep for mIoU computation,
    so this adds zero overhead.

    Within level 0 (single full-scene crop): deterministic.
    Within levels 1-2: greedy ordering — re-computes entropy after each
    step and picks the highest-entropy unvisited crop.
    """

    def __init__(
        self,
        batch_size: int,
        device: torch.device,
        canvas_grid: int,
        *,
        probe: torch.nn.Module,
        get_spatial_fn: GetSpatialFn,
    ) -> None:
        self._batch_size = batch_size
        self._device = device
        self._canvas_grid = canvas_grid
        self._probe = probe
        self._get_spatial_fn = get_spatial_fn

        self._levels = [_level_viewpoints(lvl) for lvl in range(3)]
        self._level_starts: list[int] = []
        t = 0
        for lvl in self._levels:
            self._level_starts.append(t)
            t += len(lvl)
        self._total = t
        assert self._total == 21, f"Expected 21 viewpoints (1+4+16), got {self._total}"

        # Precompute tile masks for levels 1 and 2
        self._tile_masks: list[Tensor | None] = [None]  # level 0 has no tiles
        for lvl in range(1, 3):
            self._tile_masks.append(_build_tile_masks(self._levels[lvl], canvas_grid, device))

        # Per-image visited mask: [B, n_tiles] per level
        self._visited: list[Tensor | None] = [None for _ in self._levels]

    @property
    def name(self) -> str:
        return "entropy_coarse_to_fine"

    def _compute_entropy(self, state: RecurrentState) -> Tensor:
        """[B, G, G] categorical entropy from probe predictions."""
        spatial = self._get_spatial_fn(state.canvas)  # [B, N, D]
        B = spatial.shape[0]
        G = self._canvas_grid
        features = spatial.view(B, G, G, -1)
        logits = self._probe(features.float())  # [B, C, G, G]
        log_probs = torch.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=1)  # [B, G, G]
        return entropy

    def _pick_per_image(self, entropy: Tensor, level_idx: int) -> Tensor:
        """Per-image best unvisited crop. Returns [B] crop indices."""
        masks = self._tile_masks[level_idx]
        assert masks is not None
        scores = _tile_mean_uncertainty(entropy, masks)  # [B, n_tiles]
        visited = self._visited[level_idx]
        assert visited is not None
        # -inf for visited tiles so they're never picked
        scores = scores.masked_fill(visited, float("-inf"))
        chosen = scores.argmax(dim=1)  # [B]
        # Mark as visited
        visited.scatter_(1, chosen.unsqueeze(1), True)
        return chosen

    def step(self, t: int, state: RecurrentState | None) -> Viewpoint:
        level_idx = 0
        for i in range(len(self._level_starts) - 1):
            if t >= self._level_starts[i + 1]:
                level_idx = i + 1
        pos_in_level = t - self._level_starts[level_idx]

        crops = self._levels[level_idx]
        B = self._batch_size

        if level_idx == 0 or state is None:
            # Level 0: full scene, deterministic
            cy, cx, s = crops[0]
            centers = torch.tensor([[cy, cx]], device=self._device).expand(B, -1)
            scales = torch.full((B,), s, device=self._device)
            return Viewpoint(centers=centers, scales=scales)

        # Reset visited mask at start of each level
        if pos_in_level == 0:
            self._visited[level_idx] = torch.zeros(B, len(crops), dtype=torch.bool, device=self._device)

        entropy = self._compute_entropy(state)
        chosen = self._pick_per_image(entropy, level_idx)  # [B]

        # Build per-image viewpoints
        all_crops = torch.tensor(crops, device=self._device)  # [n_tiles, 3]
        selected = all_crops[chosen]  # [B, 3]
        return Viewpoint(centers=selected[:, :2], scales=selected[:, 2])


# ── Factory ────────────────────────────────────────────────────────


def make_eval_policy(
    policy_name: PolicyName | str,
    batch_size: int,
    device: torch.device,
    n_viewpoints: int,
    *,
    canvas_grid: int = 32,
    min_scale: float = 0.05,
    max_scale: float = 1.0,
    start_with_full_scene: bool = True,
    probe: torch.nn.Module | None = None,
    get_spatial_fn: GetSpatialFn | None = None,
) -> EvalPolicy:
    """Create an evaluation policy by name.

    Supported: coarse_to_fine/c2f, random/iid,
    full_then_random/fullrand, entropy_coarse_to_fine.

    For entropy_coarse_to_fine, pass probe= and get_spatial_fn=.
    """
    ALIASES = {
        "c2f": "coarse_to_fine",
        "f2c": "fine_to_coarse",
        "fullrand": "full_then_random",
        "iid": "random",
    }
    resolved = ALIASES.get(policy_name, policy_name)

    if resolved == "entropy_coarse_to_fine":
        assert probe is not None, "entropy_coarse_to_fine requires probe="
        assert get_spatial_fn is not None, "entropy_coarse_to_fine requires get_spatial_fn="
        # FIXME: hardcoded to exactly 3 levels (1+4+16=21). Should support arbitrary n_levels.
        assert n_viewpoints == 21, f"entropy_coarse_to_fine requires n_viewpoints=21, got {n_viewpoints}"
        return EntropyGuidedC2F(
            batch_size, device, canvas_grid,
            probe=probe, get_spatial_fn=get_spatial_fn,
        )

    if resolved == "coarse_to_fine":
        return StaticPolicy(resolved, coarse_to_fine_viewpoints(batch_size, device, n_viewpoints))

    if resolved == "fine_to_coarse":
        return StaticPolicy(resolved, fine_to_coarse_viewpoints(batch_size, device, n_viewpoints))

    if resolved in ("random", "full_then_random"):
        return StaticPolicy(resolved, random_viewpoints(
            batch_size, device, n_viewpoints,
            min_scale=min_scale, max_scale=max_scale,
            start_with_full_scene=(resolved == "full_then_random" or start_with_full_scene),
        ))

    raise ValueError(f"Unknown policy: {policy_name!r}")
