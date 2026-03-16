"""Unified evaluation script: reproduce ALL paper results from HuggingFace checkpoints.

Subcommands:
    probes      ADE20K probe eval (DINOv3 + CanViT canvas) → beating-teacher table
    policies    ADE20K policy eval (6 policies × 2 resolutions × N runs) → mIoU figure/table
    in1k        ImageNet-1K classification (4 policies × N runs) → IN1K figure/table
    ablations   Held-out reconstruction quality (12 ablations × N runs) → ablation tables

All models/probes loaded from HuggingFace Hub. No local checkpoints needed.
All outputs are .pt files consumed by CanViT-Toward-AVFMs export scripts.

Environment variables:
    ADE20K_ROOT     Path to ADE20K dataset (required for probes, policies, ablations)
    IN1K_VAL_IMAGE_DIR  Path to ImageNet-1K val/ (required for in1k, defaults to /datashare/imagenet/ILSVRC2012/val)

Usage:
    ADE20K_ROOT=/datasets/ADE20k/ADEChallengeData2016 uv run python scripts/eval.py probes
    ADE20K_ROOT=... uv run python scripts/eval.py policies --n-runs 5
    uv run python scripts/eval.py in1k --n-runs 5
    ADE20K_ROOT=... uv run python scripts/eval.py ablations --teacher-cache teacher_cache.pt
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import tyro

log = logging.getLogger(__name__)

# ── Shared constants ──────────────────────────────────────────────────────────

CANVIT_MODEL = "canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02"

# ── Helpers ───────────────────────────────────────────────────────────────────


def _ade20k_root() -> Path:
    root = Path(os.environ["ADE20K_ROOT"])
    assert root.is_dir(), f"ADE20K_ROOT not a directory: {root}"
    return root


def _skip(out: Path, tag: str) -> bool:
    if out.exists():
        log.info("SKIP %s", tag)
        return True
    return False


# ── Subcommands ───────────────────────────────────────────────────────────────


@dataclass
class ProbesCmd:
    """ADE20K probe eval: DINOv3 baselines + CanViT canvas probes."""
    out_dir: Path = Path("results_probe_eval")
    batch_size: int = 32

    def run(self) -> None:
        from canvit_eval.ade20k.eval_dinov3_probe import DINOv3ProbeEvalConfig, evaluate as eval_dv3
        from canvit_eval.ade20k.evaluate import EvalConfig, evaluate as eval_cv

        self.out_dir.mkdir(parents=True, exist_ok=True)
        root = _ade20k_root()

        for variant, resolutions in [("dv3b", [128,144,160,192,256,384,512]),
                                      ("dv3s", [128,144,160,192,256,384,512])]:
            for res in resolutions:
                out = self.out_dir / f"{variant}_{res}px.pt"
                if _skip(out, f"{variant}_{res}px"): continue
                log.info("DINOv3 %s %dpx", variant, res)
                eval_dv3(DINOv3ProbeEvalConfig(
                    probe_repo=f"canvit/probe-ade20k-40k-{variant}-{res}px",
                    ade20k_root=root, output=out, scene_size=512, batch_size=self.batch_size,
                ))

        for slug, scene, grid in [("s512-c8-in21k",512,8), ("s512-c16-in21k",512,16),
                                    ("s512-c32-in21k",512,32), ("s1024-c64-in21k",1024,64)]:
            out = self.out_dir / f"canvit_{slug}.pt"
            if _skip(out, slug): continue
            log.info("CanViT %s (scene=%d, grid=%d)", slug, scene, grid)
            eval_cv(EvalConfig(
                probe_repo=f"canvit/probe-ade20k-40k-{slug}", ade20k_root=root, output=out,
                model_repo=CANVIT_MODEL, policy="coarse_to_fine", n_timesteps=1,
                scene_size=scene, canvas_grid=grid, batch_size=self.batch_size,
            ))


@dataclass
class PoliciesCmd:
    """ADE20K policy eval: 6 policies × 2 resolutions × N runs."""
    out_dir: Path = Path("results_policy_eval")
    n_runs: int = 5
    n_timesteps: int = 21

    _CONFIGS = [
        ("canvit/probe-ade20k-40k-s512-c32-in21k", 512, 32, 32),
        ("canvit/probe-ade20k-40k-s1024-c64-in21k", 1024, 64, 8),
    ]
    _POLICIES = ["coarse_to_fine", "fine_to_coarse", "full_then_random",
                 "random", "entropy_coarse_to_fine", "constant_full_scene"]
    _DETERMINISTIC = {"constant_full_scene"}

    def run(self) -> None:
        from canvit_eval.ade20k.evaluate import EvalConfig, evaluate as eval_cv

        self.out_dir.mkdir(parents=True, exist_ok=True)
        root = _ade20k_root()

        for run in range(self.n_runs):
            for probe_repo, scene, grid, bs in self._CONFIGS:
                for policy in self._POLICIES:
                    if policy in self._DETERMINISTIC and run > 0: continue
                    tag = f"{policy}_s{scene}_c{grid}_run{run}"
                    out = self.out_dir / f"{tag}.pt"
                    if _skip(out, tag): continue
                    log.info("RUN %s", tag)
                    eval_cv(EvalConfig(
                        probe_repo=probe_repo, ade20k_root=root, output=out,
                        model_repo=CANVIT_MODEL, policy=policy, n_timesteps=self.n_timesteps,
                        scene_size=scene, canvas_grid=grid, batch_size=bs,
                    ))


@dataclass
class In1kCmd:
    """ImageNet-1K classification: 4 policies × N runs."""
    out_dir: Path = Path("results_in1k")
    n_runs: int = 5
    n_viewpoints: int = 21

    _POLICIES = ["coarse_to_fine", "fine_to_coarse", "full_then_random", "random"]

    def run(self) -> None:
        from canvit_eval.in1k.evaluate import Config as IN1KConfig, evaluate as eval_in1k

        self.out_dir.mkdir(parents=True, exist_ok=True)

        for policy in self._POLICIES:
            for run in range(self.n_runs):
                tag = f"in1k_{policy}_run{run}"
                out = self.out_dir / f"{tag}.pt"
                if _skip(out, tag): continue
                log.info("RUN %s", tag)
                eval_in1k(IN1KConfig(output=out, policy=policy, n_viewpoints=self.n_viewpoints))


@dataclass
class AblationsCmd:
    """Ablation reconstruction quality: 12 checkpoints × N runs."""
    out_dir: Path = Path("results_ablations_riid")
    n_runs: int = 5
    n_timesteps: int = 10
    teacher_cache: Path | None = None

    # HF repos — matches analysis/ablations/__init__.py registry.
    _REPOS = [
        "canvit/canvitb16-abl-baseline-2026-03-15",
        "canvit/canvitb16-abl-qkvo-dcan256-2026-03-15",
        "canvit/canvitb16-abl-qkvo-dcan384-2026-03-15",
        "canvit/canvitb16-abl-dcan256-2026-03-15",
        "canvit/canvitb16-abl-no-dense-2026-03-15",
        "canvit/canvitb16-abl-no-fiid-1riid-2026-03-15",
        "canvit/canvitb16-abl-no-fiid-2riid-2026-03-15",
        "canvit/canvitb16-abl-no-bptt-2026-03-15",
        "canvit/canvitb16-abl-no-reads-2026-03-15",
        "canvit/canvitb16-abl-no-vpe-2026-03-15",
        "canvit/canvitb16-abl-rw-stride6-2026-03-15",
        "canvit/canvitb16-abl-vit-s-2026-03-15",
    ]

    @staticmethod
    def _repo_to_name(repo: str) -> str:
        """canvit/canvitb16-abl-baseline-2026-03-15 → abl-baseline-200k"""
        parts = repo.split("/")[1].replace("canvitb16-", "").rsplit("-", 3)
        return f"{parts[0]}-200k"

    def run(self) -> None:
        from canvit_eval.reconstruction import ReconstructionEvalConfig, evaluate as eval_recon

        self.out_dir.mkdir(parents=True, exist_ok=True)
        image_dir = _ade20k_root() / "images" / "validation"

        for repo in self._REPOS:
            name = self._repo_to_name(repo)
            for run in range(self.n_runs):
                out = self.out_dir / f"{name}_run{run}.pt"
                if _skip(out, f"{name} run{run}"): continue
                log.info("RUN %s run%d", name, run)
                eval_recon(ReconstructionEvalConfig(
                    model_repo=repo, image_dir=image_dir, output=out,
                    n_timesteps=self.n_timesteps, batch_size=16, teacher_cache=self.teacher_cache,
                ))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cmd = tyro.cli(
        Annotated[ProbesCmd, tyro.conf.subcommand("probes")]
        | Annotated[PoliciesCmd, tyro.conf.subcommand("policies")]
        | Annotated[In1kCmd, tyro.conf.subcommand("in1k")]
        | Annotated[AblationsCmd, tyro.conf.subcommand("ablations")]
    )
    cmd.run()
    log.info("DONE")


if __name__ == "__main__":
    main()
