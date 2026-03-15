"""Push trained segmentation probes to HuggingFace Hub.

Handles both canvas probe and DINOv3 probe checkpoint formats.
Infers all params from the checkpoint — nothing hardcoded.

Usage:
    uv run python scripts/push_probes.py \
        --probe path/to/best.pt \
        --repo-id canvit/probe-ade20k-dinov3-vitb16-512px-40k \
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


@dataclass
class Args:
    probe: Path
    repo_id: str
    dry_run: bool = False


def _extract_state_dict_and_metadata(raw: dict) -> tuple[dict, dict]:
    """Extract probe state_dict and all metadata from any checkpoint format.

    Returns (state_dict, metadata_dict). metadata_dict contains EVERYTHING
    from the checkpoint except the state_dict itself.
    """
    # Canvas probe (new format): has feat_type key
    if "feat_type" in raw:
        sd = raw["probe_state_dict"]
        meta = {k: v for k, v in raw.items() if k != "probe_state_dict"}
        return sd, meta

    # Canvas probe (legacy format): has probe_state_dicts dict
    if "probe_state_dicts" in raw:
        # Take canvas_hidden (the primary feature type)
        sd = raw["probe_state_dicts"]["canvas_hidden"]
        meta = {k: v for k, v in raw.items() if k != "probe_state_dicts"}
        meta["feat_type"] = "canvas_hidden"
        meta["_legacy_format"] = True
        return sd, meta

    # DINOv3 probe: has probe_state_dict but no feat_type
    if "probe_state_dict" in raw:
        sd = raw["probe_state_dict"]
        meta = {k: v for k, v in raw.items() if k != "probe_state_dict"}
        meta["feat_type"] = "dinov3_spatial"
        return sd, meta

    assert False, f"Unknown probe checkpoint format. Keys: {sorted(raw.keys())}"


def main(args: Args) -> None:
    assert args.probe.exists(), f"Not found: {args.probe}"

    raw = torch.load(args.probe, map_location="cpu", weights_only=False)
    state_dict, meta = _extract_state_dict_and_metadata(raw)

    # Infer probe architecture from state_dict (never hardcode)
    embed_dim = state_dict["conv.weight"].shape[1]
    num_classes = state_dict["conv.weight"].shape[0]
    use_ln = "ln.weight" in state_dict
    dropout = meta.get("config", {}).get("dropout", None)
    assert dropout is not None, "dropout not found in checkpoint config"

    log.info("Probe: embed_dim=%d, num_classes=%d, use_ln=%s, dropout=%s",
             embed_dim, num_classes, use_ln, dropout)
    log.info("  feat_type: %s", meta.get("feat_type"))
    log.info("  model: %s", meta.get("model", meta.get("config", {}).get("model_repo", "?")))
    log.info("  step: %s", meta.get("step"))
    log.info("  source: %s", args.probe.name)

    # Construct and load
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

    # HF config: architecture params + ALL checkpoint metadata
    hf_config: dict = {
        "embed_dim": embed_dim,
        "num_classes": num_classes,
        "dropout": dropout,
        "use_ln": use_ln,
        "metadata": meta,
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
