"""Push all 40k-step ADE20K probes to HuggingFace Hub.

Reads probe checkpoints from Nibi (or local), derives repo ID from
checkpoint metadata, and pushes via SegmentationProbe + safetensors.

Naming convention (derived from checkpoint metadata, never hardcoded):
  Canvas probes:  canvit/probe-ade20k-{steps}k-s{scene}-c{grid}-{model_slug}
  DINOv3 probes:  canvit/probe-ade20k-{steps}k-{model_slug}-{resolution}px

Usage:
    uv run python scripts/push_all_probes.py --probe-dir /path/to/probes --dry-run
"""

import json
import logging
import re
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
    probe_dir: Path
    owner: str = "canvit"
    dry_run: bool = False


def _find_best_pt(d: Path) -> Path:
    """Find the best checkpoint in a probe directory."""
    # Canvas probes: canvas_hidden_best_t*_miou*_step*.pt
    # DINOv3 probes: best_miou*_step*.pt
    candidates = list(d.glob("*best*miou*.pt"))
    assert len(candidates) >= 1, f"No best checkpoint in {d}: {list(d.glob('*.pt'))}"
    # Take the one with highest mIoU in the filename
    def _extract_miou(p: Path) -> float:
        m = re.search(r"miou([\d.]+)", p.name)
        return float(m.group(1)) if m else 0.0
    return max(candidates, key=_extract_miou)


def _make_repo_id(owner: str, config: dict, is_dinov3: bool) -> str:
    """Derive HF repo ID from checkpoint metadata. No hardcoded names."""
    steps_k = config["max_steps"] // 1000

    if is_dinov3:
        # DINOv3: model is e.g. "facebook/dinov3-vitb16-pretrain-lvd1689m"
        model_raw = config["model"]
        # Extract short name: dinov3-vitb16
        slug = model_raw.split("/")[-1].replace("-pretrain-lvd1689m", "")
        resolution = config["resolution"]
        return f"{owner}/probe-ade20k-{steps_k}k-{slug}-{resolution}px"
    else:
        # Canvas: model_repo is e.g. "canvit/canvitb16-add-vpe-pretrain-..."
        model_slug = config["model_repo"].split("/")[-1]
        scene = config.get("image_size", config.get("scene_size"))
        assert scene is not None, f"No scene size in config: {sorted(config.keys())}"
        grid = scene // 16  # patch_size
        return f"{owner}/probe-ade20k-{steps_k}k-s{scene}-c{grid}-{model_slug}"


def main(args: Args) -> None:
    assert args.probe_dir.is_dir(), f"Not a directory: {args.probe_dir}"

    # Find all probe directories (each contains best*.pt)
    probe_dirs = sorted(d for d in args.probe_dir.iterdir()
                        if d.is_dir() and (d.name.startswith("dinov3-") or d.name.startswith("canvit")))

    log.info("%s %d probe directories in %s",
             "DRY RUN:" if args.dry_run else "Pushing", len(probe_dirs), args.probe_dir)

    for d in probe_dirs:
        best_pt = _find_best_pt(d)
        raw = torch.load(best_pt, map_location="cpu", weights_only=False)
        config = raw.get("config", {})
        is_dinov3 = "model" in config and "model_repo" not in config

        max_steps = config.get("max_steps")
        if max_steps != 40000:
            log.info("  SKIP %s (max_steps=%s, want 40000)", d.name, max_steps)
            continue

        repo_id = _make_repo_id(args.owner, config, is_dinov3)

        # Infer architecture from state_dict
        sd = raw["probe_state_dict"]
        embed_dim = sd["conv.weight"].shape[1]
        num_classes = sd["conv.weight"].shape[0]
        use_ln = "ln.weight" in sd
        dropout = config.get("dropout")
        assert dropout is not None

        log.info("  %s → %s (embed=%d, ln=%s, file=%s)",
                 d.name, repo_id, embed_dim, use_ln, best_pt.name)

        if args.dry_run:
            continue

        probe = SegmentationProbe(embed_dim=embed_dim, num_classes=num_classes,
                                  dropout=dropout, use_ln=use_ln)
        result = probe.load_state_dict(sd, strict=True)
        assert not result.missing_keys and not result.unexpected_keys

        # Forward ALL metadata
        meta = {k: v for k, v in raw.items() if k != "probe_state_dict"}
        hf_config: dict = {
            "embed_dim": embed_dim, "num_classes": num_classes,
            "dropout": dropout, "use_ln": use_ln,
            "metadata": meta,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "config.json").write_text(json.dumps(hf_config, indent=2, default=str))
            save_file(probe.state_dict(), tmppath / "model.safetensors")
            api = HfApi()
            api.create_repo(repo_id, private=True, exist_ok=True)
            api.upload_folder(folder_path=tmpdir, repo_id=repo_id)

        log.info("    pushed")
        del probe

    log.info("Done.")


if __name__ == "__main__":
    main(tyro.cli(Args))
