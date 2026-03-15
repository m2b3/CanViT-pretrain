"""Push migrated ablation checkpoints to private HuggingFace Hub repos.

Loads each checkpoint, extracts all metadata, uploads model weights + rich
model card with full provenance. Requires migrated checkpoints (standardizers
in state_dict).

Naming: {owner}/canvitb16-abl-{slug}-{YYYYMMDD}
  Date from checkpoint timestamp. All ablations share the same date (same
  training campaign). Slug from registry.

Usage:
    # Dry run (print repo IDs + metadata without pushing):
    uv run python scripts/push_ablation_checkpoints.py \
        --ckpt-dir ~/projects/canvit-eval-workspace/ablation_checkpoints/ \
        --dry-run

    # Push:
    uv run python scripts/push_ablation_checkpoints.py \
        --ckpt-dir ~/projects/canvit-eval-workspace/ablation_checkpoints/
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import tyro

from canvit.model.pretraining.hub import upload_to_hf
from canvit.model.pretraining.impl import CanViTForPretraining

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ckpt_dir stem → short slug for HF repo name.
# Source of truth: analysis/ablations/__init__.py in CanViT-Toward-AVFMs.
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


def _extract_metadata(ckpt: dict, slug: str) -> dict:
    """Extract all checkpoint metadata (everything except weights/optimizer)."""
    skip = {"state_dict", "optimizer_state", "scheduler_state"}
    meta = {k: v for k, v in ckpt.items() if k not in skip}
    meta["ablation_slug"] = slug
    return meta


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
            f"Unknown checkpoint stem '{stem}' — not in _SLUG. "
            f"Known: {sorted(_SLUG)}"
        )

        ckpt = torch.load(f, map_location="cpu", weights_only=False)
        step = ckpt["step"]
        ts = datetime.fromisoformat(ckpt["timestamp"])
        date_str = ts.strftime("%Y%m%d")

        repo_id = f"{args.owner}/canvitb16-abl-{slug}-{date_str}"

        log.info("  %s → %s (step=%d, %s)", stem, repo_id, step, ts.date())

        if args.dry_run:
            continue

        model = CanViTForPretraining.from_checkpoint(f)
        meta = _extract_metadata(ckpt, slug)
        upload_to_hf(model, repo_id, private=True, extra_metadata=meta)
        del model
        torch.cuda.empty_cache()

    log.info("Done.")


if __name__ == "__main__":
    main(tyro.cli(Args))
