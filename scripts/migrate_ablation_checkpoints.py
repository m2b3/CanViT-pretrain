"""Migrate pre-unification ablation checkpoints to current format.

BACKGROUND:
Before commit 21e9bbc (2026-02-23), the training loop stored standardizer
stats as top-level keys `scene_norm_state` and `cls_norm_state`, separate
from `model.state_dict()`. After unification, standardizer stats live inside
the model state dict as `scene_standardizers.<G>.{mean,var,_initialized}`
and `cls_standardizers.<G>.{mean,var,_initialized}`.

The 12 ablation checkpoints (canvit-train commit 27ac70a, 2026-02-27) were
trained with the pre-unification code. When loaded by current code via
`load_model()` / `load_state_dict_flexible()`, the standardizer stats
silently end up uninitialized (mean=0, var=1, _initialized=False) because:
  1. The model's state_dict has standardizer keys (from model.__init__)
  2. The checkpoint's state_dict has the SAME keys but with zeros
  3. The REAL stats are in the separate top-level keys
  4. load_state_dict_flexible loads the zeros, ignoring the top-level keys

This script migrates the stats from legacy keys into the state_dict,
removes the legacy keys, and saves the migrated checkpoint.

USAGE:
    uv run python scripts/migrate_ablation_checkpoints.py --ckpt-dir /path/to/checkpoints/

    Reads all *.pt files in the directory, migrates in-place (overwrites).
    Use --dry-run to preview without writing.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


@dataclass
class Args:
    ckpt_dir: Path
    dry_run: bool = False


def migrate_checkpoint(path: Path, dry_run: bool) -> bool:
    """Migrate one checkpoint. Returns True if modified."""
    raw = torch.load(path, map_location="cpu", weights_only=False)

    scene_legacy = raw.get("scene_norm_state")
    cls_legacy = raw.get("cls_norm_state")

    if scene_legacy is None or cls_legacy is None:
        log.info("  %s: no legacy keys, skipping", path.name)
        return False

    # Verify legacy stats are actually initialized
    assert scene_legacy["_initialized"].item(), f"{path.name}: legacy scene stats not initialized"
    assert cls_legacy["_initialized"].item(), f"{path.name}: legacy cls stats not initialized"

    # Find the grid size from canvas_patch_grid_sizes
    grids = raw["canvas_patch_grid_sizes"]
    assert len(grids) == 1, f"Expected 1 grid size, got {grids}"
    G = str(grids[0])

    sd = raw["state_dict"]

    # Verify current state_dict has uninitialized standardizers
    scene_key = f"scene_standardizers.{G}._initialized"
    cls_key = f"cls_standardizers.{G}._initialized"
    assert scene_key in sd, f"Missing {scene_key} in state_dict"
    assert not sd[scene_key].item(), f"{path.name}: scene standardizer already initialized in state_dict"

    # Migrate: copy legacy stats into state_dict
    for prefix, legacy in [("scene_standardizers", scene_legacy), ("cls_standardizers", cls_legacy)]:
        for stat_name in ["mean", "var", "_initialized"]:
            key = f"{prefix}.{G}.{stat_name}"
            sd[key] = legacy[stat_name]

    # Verify migration
    assert sd[scene_key].item(), "Migration failed: scene still not initialized"
    assert sd[cls_key].item(), "Migration failed: cls still not initialized"

    # Remove legacy keys
    del raw["scene_norm_state"]
    del raw["cls_norm_state"]

    log.info("  %s: migrated (grid=%s, scene_mean_norm=%.2f, cls_mean_norm=%.2f)",
             path.name, G,
             sd[f"scene_standardizers.{G}.mean"].abs().sum().item(),
             sd[f"cls_standardizers.{G}.mean"].abs().sum().item())

    if not dry_run:
        torch.save(raw, path)
        log.info("  %s: saved", path.name)

    return True


def main(args: Args) -> None:
    assert args.ckpt_dir.is_dir(), f"Not a directory: {args.ckpt_dir}"

    files = sorted(args.ckpt_dir.glob("*.pt"))
    assert len(files) > 0, f"No .pt files in {args.ckpt_dir}"

    log.info("Migrating %d checkpoints in %s%s",
             len(files), args.ckpt_dir, " (DRY RUN)" if args.dry_run else "")

    n_migrated = 0
    for f in files:
        if migrate_checkpoint(f, args.dry_run):
            n_migrated += 1

    log.info("Done: %d/%d migrated%s", n_migrated, len(files),
             " (dry run, nothing written)" if args.dry_run else "")


if __name__ == "__main__":
    main(tyro.cli(Args))
