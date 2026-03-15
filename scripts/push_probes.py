"""Push trained segmentation probes to HuggingFace Hub.

Loads probe .pt checkpoints (canvit_eval format), converts to
SegmentationProbe (canvit_utils), and uploads with full metadata.

Usage:
    uv run python scripts/push_probes.py \
        --probe ~/projects/canvit-eval-workspace/probes/probe_512_32.pt \
        --repo-id canvit/probe-ade20k-s512-c32-40k \
        --dry-run
"""

import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from huggingface_hub import HfApi
from safetensors.torch import save_file

from canvit_utils.probes import SegmentationProbe

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

ADE20K_NUM_CLASSES = 150


@dataclass
class Args:
    probe: Path
    repo_id: str
    dry_run: bool = False


def main(args: Args) -> None:
    assert args.probe.exists(), f"Not found: {args.probe}"

    raw = torch.load(args.probe, map_location="cpu", weights_only=False)

    # Extract probe state_dict (handle both formats)
    if "feat_type" in raw:
        state_dict = raw["probe_state_dict"]
        feat_type = raw["feat_type"]
    elif "probe_state_dicts" in raw:
        state_dict = raw["probe_state_dicts"]["canvas_hidden"]
        feat_type = "canvas_hidden"
    else:
        assert False, f"Unknown probe checkpoint format. Keys: {sorted(raw.keys())}"

    config = raw.get("config", {})
    best_mious = raw.get("best_mious_per_t", raw.get("best_mious", []))

    # Infer probe params from state_dict
    embed_dim = state_dict["conv.weight"].shape[1]
    num_classes = state_dict["conv.weight"].shape[0]
    assert num_classes == ADE20K_NUM_CLASSES, f"Expected {ADE20K_NUM_CLASSES} classes, got {num_classes}"
    use_ln = "ln.weight" in state_dict
    dropout = config.get("dropout", 0.1)

    log.info("Probe: embed_dim=%d, num_classes=%d, use_ln=%s, dropout=%s, feat_type=%s",
             embed_dim, num_classes, use_ln, dropout, feat_type)
    log.info("Best mIoUs: %s", [f"{m:.4f}" for m in best_mious] if best_mious else "not available")

    # Construct SegmentationProbe and load weights
    probe = SegmentationProbe(
        embed_dim=embed_dim,
        num_classes=num_classes,
        dropout=dropout,
        use_ln=use_ln,
    )
    result = probe.load_state_dict(state_dict, strict=True)
    assert not result.missing_keys and not result.unexpected_keys, f"State dict mismatch: {result}"

    log.info("  → %s%s", args.repo_id, " (DRY RUN)" if args.dry_run else "")

    if args.dry_run:
        return

    # Build HF repo contents
    hf_config = {
        "embed_dim": embed_dim,
        "num_classes": num_classes,
        "dropout": dropout,
        "use_ln": use_ln,
        "metadata": {
            "dataset": "ade20k",
            "feature_type": feat_type,
            "best_mious_per_t": best_mious,
            "training_config": config,
            "source_file": args.probe.name,
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        (tmppath / "config.json").write_text(json.dumps(hf_config, indent=2, default=str))
        save_file(probe.state_dict(), tmppath / "model.safetensors")

        api = HfApi()
        api.create_repo(args.repo_id, private=True, exist_ok=True)
        api.upload_folder(folder_path=tmpdir, repo_id=args.repo_id)

    log.info("Pushed to https://huggingface.co/%s", args.repo_id)


if __name__ == "__main__":
    main(tyro.cli(Args))
