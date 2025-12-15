"""Curriculum learning utilities for scene size progression.

Curriculum training starts at small grid sizes and progressively increases:
G=16 (256 tokens) → G=32 (1024 tokens) → G=64 (4096 tokens)

Key formulas:
- batch_size ∝ 1/G² (quadratic scaling - tokens ∝ G²)
- n_eval_viewpoints ∝ G (linear - glimpses reuse hidden state)
- fresh_ratio = n_viewpoints_per_step / n_eval
"""

from dataclasses import dataclass


def batch_size_for_grid(grid_size: int, max_grid_size: int, max_batch_size: int) -> int:
    """Compute batch size for a grid size.

    Tokens ∝ G², so batch ∝ 1/G² to maintain constant VRAM.
    """
    ratio = max_grid_size // grid_size
    return max(1, max_batch_size * ratio * ratio)


def n_eval_viewpoints(grid_size: int, glimpse_grid_size: int) -> int:
    """Expected number of viewpoints for evaluation.

    Linear in G (not quadratic) since glimpses reuse hidden state.
    Reference: G=16 → 5 viewpoints.
    """
    _ = glimpse_grid_size  # Unused but kept for API consistency
    return max(2, round(5 * grid_size / 16))


def fresh_ratio_for_grid(
    grid_size: int, glimpse_grid_size: int, n_viewpoints_per_step: int
) -> float:
    """Fresh ratio to achieve expected viewpoint count.

    E[glimpses] = n_viewpoints_per_step / fresh_ratio = n_eval
    """
    return n_viewpoints_per_step / n_eval_viewpoints(grid_size, glimpse_grid_size)


@dataclass
class CurriculumStage:
    """Configuration for one curriculum stage. Computed properties avoid redundancy."""

    scene_grid_size: int
    glimpse_grid_size: int
    patch_size: int
    batch_size: int
    fresh_count: int
    n_viewpoints_per_step: int

    @property
    def scene_size_px(self) -> int:
        return self.scene_grid_size * self.patch_size

    @property
    def n_scene_tokens(self) -> int:
        return self.scene_grid_size ** 2

    @property
    def n_eval(self) -> int:
        return n_eval_viewpoints(self.scene_grid_size, self.glimpse_grid_size)

    @property
    def fresh_ratio_intended(self) -> float:
        return self.n_viewpoints_per_step / self.n_eval

    @property
    def fresh_ratio_actual(self) -> float:
        return self.fresh_count / self.batch_size

    @property
    def expected_glimpses_intended(self) -> float:
        return self.n_viewpoints_per_step / self.fresh_ratio_intended

    @property
    def expected_glimpses_actual(self) -> float:
        return self.n_viewpoints_per_step / self.fresh_ratio_actual

    @property
    def min_viewpoint_scale(self) -> float:
        return self.glimpse_grid_size / self.scene_grid_size


def create_curriculum_stage(
    scene_grid_size: int,
    glimpse_grid_size: int,
    patch_size: int,
    max_grid_size: int,
    max_batch_size: int,
    n_viewpoints_per_step: int,
) -> CurriculumStage:
    """Create a curriculum stage with computed batch size and fresh count."""
    batch_size = batch_size_for_grid(scene_grid_size, max_grid_size, max_batch_size)
    fresh_ratio = fresh_ratio_for_grid(
        scene_grid_size, glimpse_grid_size, n_viewpoints_per_step
    )
    fresh_count = max(1, round(fresh_ratio * batch_size))

    return CurriculumStage(
        scene_grid_size=scene_grid_size,
        glimpse_grid_size=glimpse_grid_size,
        patch_size=patch_size,
        batch_size=batch_size,
        fresh_count=fresh_count,
        n_viewpoints_per_step=n_viewpoints_per_step,
    )


def log_curriculum_stage(stage: CurriculumStage, logger: "logging.Logger") -> None:
    """Log curriculum stage configuration with all computed values."""
    import logging

    assert isinstance(logger, logging.Logger)
    logger.info(
        f"Stage G={stage.scene_grid_size} ({stage.n_scene_tokens} scene tokens):"
    )
    logger.info(
        f"  scene_size_px={stage.scene_size_px}, "
        f"batch_size={stage.batch_size} "
        f"({stage.batch_size // (stage.scene_grid_size // 16) ** 2 if stage.scene_grid_size >= 16 else stage.batch_size}x max)"
    )
    logger.info(
        f"  fresh_count={stage.fresh_count}, "
        f"fresh_ratio={stage.fresh_ratio_actual:.3f} "
        f"(intended={stage.fresh_ratio_intended:.3f})"
    )
    logger.info(
        f"  E[glimpses]={stage.expected_glimpses_actual:.1f} "
        f"(intended={stage.expected_glimpses_intended:.1f}) "
        f"- min_scale={stage.min_viewpoint_scale:.4f}"
    )
