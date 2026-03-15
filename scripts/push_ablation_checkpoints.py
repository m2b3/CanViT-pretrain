"""Push ablation checkpoints to private HuggingFace Hub repos.

Handles legacy standardizer migration in-memory — works on both raw
(un-migrated) and migrated checkpoints. No separate migration step needed.

Naming: {owner}/canvitb16-abl-{slug}-{YYYYMMDD}
  Date from checkpoint timestamp. Slug from registry.

Usage:
    uv run python scripts/push_ablation_checkpoints.py \
        --ckpt-dir ~/projects/canvit-eval-workspace/ablation_checkpoints/ \
        --dry-run
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import tyro
from torch import Tensor

from canvit.model.pretraining.hub import upload_to_hf
from canvit.model.pretraining.impl import CanViTForPretraining

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ckpt_dir stem → short slug for HF repo name.
_SLUG: dict[str, str] = {
    "abl-baseline-200k": "baseline",
    "abl-qkvo-dcan256-200k": "qkvo-dcan256",
    "abl-qkvo-dcan384-200k": "qkvo-dcan384",
    "abl-dcan256-200k": "dcan256",
    "abl-no-dense-200k": "no-dense",
    "abl-no-fiid-200k": "no-fiid-1riid",
    "abl-2riid-no-fiid-200k": "no-fiid-2riid",
    "abl-no-bptt-200k": "no-bptt",
    "abl-no-reads-200k": "no-reads",
    "abl-no-vpe-200k": "no-vpe",
    "abl-rw-stride6-200k": "rw-stride6",
    "abl-vit-s-200k": "vit-s",
}

# Keys that are large/non-serializable — excluded from HF metadata.
_SKIP_METADATA = {"state_dict", "optimizer_state", "scheduler_state"}


def _migrate_standardizers_in_place(raw: dict) -> None:
    """Migrate legacy standardizer keys into state_dict if needed. Mutates raw."""
    scene_legacy = raw.get("scene_norm_state")
    cls_legacy = raw.get("cls_norm_state")
    if scene_legacy is None:
        return  # Already migrated or current format

    assert cls_legacy is not None, "scene_norm_state present but cls_norm_state missing"
    assert scene_legacy["_initialized"].item(), "Legacy scene stats not initialized"
    assert cls_legacy["_initialized"].item(), "Legacy cls stats not initialized"

    grids = raw["canvas_patch_grid_sizes"]
    assert len(grids) == 1, f"Expected 1 grid size, got {grids}"
    G = str(grids[0])
    sd = raw["state_dict"]

    for prefix, legacy in [("scene_standardizers", scene_legacy), ("cls_standardizers", cls_legacy)]:
        for stat_name in ["mean", "var", "_initialized"]:
            sd[f"{prefix}.{G}.{stat_name}"] = legacy[stat_name]

    del raw["scene_norm_state"]
    del raw["cls_norm_state"]
    log.info("    migrated standardizers in-memory (grid=%s)", G)


def _verify_standardizers(model: CanViTForPretraining) -> None:
    """Assert all standardizers are initialized."""
    for G in model.canvas_patch_grid_sizes:
        _, scene_std = model.standardizers(G)
        assert scene_std.initialized, (
            f"Standardizer not initialized for grid {G} after loading. "
            "Checkpoint may be corrupt."
        )


@dataclass
class Args:
    ckpt_dir: Path
    owner: str = "canvit"
    dry_run: bool = False


def main(args: Args) -> None:
    assert args.ckpt_dir.is_dir(), f"Not a directory: {args.ckpt_dir}"
    files = sorted(args.ckpt_dir.glob("*.pt"))
    assert len(files) > 0, f"No .pt files in {args.ckpt_dir}"

    log.info("%s %d checkpoints from %s",
             "DRY RUN:" if args.dry_run else "Pushing", len(files), args.ckpt_dir)

    for f in files:
        stem = f.stem
        slug = _SLUG.get(stem)
        assert slug is not None, (
            f"Unknown checkpoint '{stem}' — not in _SLUG. Known: {sorted(_SLUG)}"
        )

        raw = torch.load(f, map_location="cpu", weights_only=False)
        _migrate_standardizers_in_place(raw)

        step = raw["step"]
        ts = datetime.fromisoformat(raw["timestamp"])
        date_str = ts.strftime("%Y%m%d")
        repo_id = f"{args.owner}/canvitb16-abl-{slug}-{date_str}"

        log.info("  %s → %s (step=%d, %s)", stem, repo_id, step, ts.date())

        if args.dry_run:
            continue

        # Reconstruct model from (possibly migrated) raw checkpoint
        from canvit.backbone import create_backbone
        from canvit.model.pretraining.impl import CanViTForPretrainingConfig
        import dacite

        cfg = dacite.from_dict(CanViTForPretrainingConfig, raw["model_config"])
        model = CanViTForPretraining(
            backbone=create_backbone(raw["backbone_name"]),
            cfg=cfg,
            backbone_name=raw["backbone_name"],
            canvas_patch_grid_sizes=raw["canvas_patch_grid_sizes"],
        )
        model.load_state_dict(raw["state_dict"])
        _verify_standardizers(model)

        meta = {k: v for k, v in raw.items() if k not in _SKIP_METADATA}
        upload_to_hf(model, repo_id, private=True, extra_metadata=meta)

        del model, raw
        torch.cuda.empty_cache()

    log.info("Done.")


if __name__ == "__main__":
    main(tyro.cli(Args))
