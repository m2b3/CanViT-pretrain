"""Evaluation policies for CanViT.

Wraps canvit_utils.policies with an interactive API that supports
uncertainty-guided viewpoint ordering.

Convention: centers are (y, x) in [-1, 1]. Scales are in (0, 1].
y grows downward (row index), x grows rightward (column index).
This matches the convention in canvit.viewpoint.
"""

import logging
from typing import Protocol

import torch
from canvit import RecurrentState, Viewpoint
from canvit_utils.policies import coarse_to_fine_viewpoints, random_viewpoints
from torch import Tensor

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


def _tile_mean_uncertainty(
    uncertainty_map: Tensor,
    crop_centers: list[tuple[float, float, float]],
    canvas_grid: int,
) -> Tensor:
    """Compute mean uncertainty per tile.

    Args:
        uncertainty_map: [B, G, G] — higher = more uncertain
        crop_centers: list of (y, x, scale) tuples
        canvas_grid: G

    Returns:
        [B, n_tiles] mean uncertainty per tile
    """
    B, G = uncertainty_map.shape[0], canvas_grid
    device = uncertainty_map.device

    # Canvas cell coordinates in [-1, 1]
    coords = torch.linspace(-1 + 1 / G, 1 - 1 / G, G, device=device)

    scores = torch.zeros(B, len(crop_centers), device=device)
    for i, (cy, cx, s) in enumerate(crop_centers):
        # Boolean mask: which canvas cells fall within this crop
        row_in = (coords - cy).abs() <= s
        col_in = (coords - cx).abs() <= s
        mask = row_in[:, None] & col_in[None, :]  # [G, G]
        n_cells = mask.sum().clamp(min=1)
        scores[:, i] = (uncertainty_map * mask.unsqueeze(0)).sum(dim=(1, 2)) / n_cells

    return scores


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
        get_spatial_fn: object,
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

        self._orderings: list[list[int]] = [[] for _ in self._levels]

    @property
    def name(self) -> str:
        return "entropy_c2f"

    def _compute_entropy(self, state: RecurrentState) -> Tensor:
        """[B, G, G] categorical entropy from probe predictions."""
        spatial = self._get_spatial_fn(state.canvas)  # [B, N, D]
        B = spatial.shape[0]
        G = self._canvas_grid
        features = spatial.view(B, G, G, -1)
        with torch.no_grad():
            logits = self._probe(features.float())  # [B, C, G, G]
        log_probs = torch.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=1)  # [B, G, G]
        return entropy

    def _pick_best_unvisited(self, entropy: Tensor, level_idx: int) -> int:
        """Pick the unvisited crop with highest mean entropy (majority vote across batch)."""
        crops = self._levels[level_idx]
        scores = _tile_mean_uncertainty(entropy, crops, self._canvas_grid)
        visited = set(self._orderings[level_idx])
        remaining = [i for i in range(len(crops)) if i not in visited]
        assert remaining, f"No remaining crops at level {level_idx}"
        mask = torch.full((len(crops),), float("-inf"), device=self._device)
        for i in remaining:
            mask[i] = 0.0
        # Majority vote: average scores across batch, then argmax
        return int((scores.mean(dim=0) + mask).argmax().item())

    def step(self, t: int, state: RecurrentState | None) -> Viewpoint:
        # Determine level
        level_idx = 0
        for i in range(len(self._level_starts) - 1):
            if t >= self._level_starts[i + 1]:
                level_idx = i + 1
        pos_in_level = t - self._level_starts[level_idx]

        if level_idx == 0 or state is None:
            crop_idx = 0
        else:
            if pos_in_level == 0:
                self._orderings[level_idx] = []
            entropy = self._compute_entropy(state)
            crop_idx = self._pick_best_unvisited(entropy, level_idx)
            self._orderings[level_idx].append(crop_idx)

        cy, cx, s = self._levels[level_idx][crop_idx]
        centers = torch.tensor([[cy, cx]], device=self._device).expand(self._batch_size, -1)
        scales = torch.full((self._batch_size,), s, device=self._device)
        return Viewpoint(centers=centers, scales=scales)


# ── Factory ────────────────────────────────────────────────────────


def make_eval_policy(
    policy_name: str,
    batch_size: int,
    device: torch.device,
    n_viewpoints: int,
    *,
    canvas_grid: int = 32,
    min_scale: float = 0.05,
    max_scale: float = 1.0,
    start_with_full_scene: bool = True,
    probe: torch.nn.Module | None = None,
    get_spatial_fn: object | None = None,
) -> EvalPolicy:
    """Create an evaluation policy by name.

    Supported: coarse_to_fine/c2f, random/iid,
    full_then_random/fullrand, entropy_c2f.

    For entropy_c2f, pass probe= and get_spatial_fn=.
    """
    ALIASES = {"c2f": "coarse_to_fine", "fullrand": "full_then_random", "iid": "random"}
    resolved = ALIASES.get(policy_name, policy_name)

    if resolved == "entropy_c2f":
        assert probe is not None, "entropy_c2f requires probe="
        assert get_spatial_fn is not None, "entropy_c2f requires get_spatial_fn="
        assert n_viewpoints == 21, f"entropy_c2f requires n_viewpoints=21, got {n_viewpoints}"
        return EntropyGuidedC2F(
            batch_size, device, canvas_grid,
            probe=probe, get_spatial_fn=get_spatial_fn,
        )

    if resolved == "coarse_to_fine":
        return StaticPolicy(resolved, coarse_to_fine_viewpoints(batch_size, device, n_viewpoints))

    if resolved in ("random", "full_then_random"):
        return StaticPolicy(resolved, random_viewpoints(
            batch_size, device, n_viewpoints,
            min_scale=min_scale, max_scale=max_scale,
            start_with_full_scene=(resolved == "full_then_random" or start_with_full_scene),
        ))

    raise ValueError(f"Unknown policy: {policy_name!r}")
