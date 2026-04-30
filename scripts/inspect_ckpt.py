"""Inspect a training checkpoint.

Usage:
    uv run python scripts/inspect_ckpt.py <checkpoint.pt>
    uv run python scripts/inspect_ckpt.py <checkpoint.pt> --smoke-test
"""

import sys
from pathlib import Path

import torch


def print_info(path: Path) -> None:
    resolved = path.resolve() if path.is_symlink() else path
    ckpt = torch.load(resolved, weights_only=False, map_location="cpu")

    sched = ckpt.get("scheduler_state")
    step = sched["last_epoch"] if sched else ckpt.get("step")
    n_params = sum(p.numel() for p in ckpt["state_dict"].values()) / 1e6
    git = ckpt.get("git_commit")
    git_str = f"{git[:8]}{'*' if ckpt.get('git_dirty') else ''}" if git else "n/a"

    print(f"path:       {path}" + (f" -> {resolved.name}" if path.is_symlink() else ""))
    print(f"size:       {resolved.stat().st_size / 1024**2:.0f} MB")
    print(f"step:       {step}")
    print(f"train_loss: {ckpt.get('train_loss', 'n/a')}")
    print(f"params:     {n_params:.1f}M")
    print(f"has_optim:  {'yes' if ckpt.get('optimizer_state') is not None else 'no'}")
    print(f"has_sched:  {'yes' if sched else 'no'}")
    print()
    print(f"backbone:   {ckpt['backbone_name']}")
    print(f"teacher:    {ckpt['teacher_name']}")
    print(f"dataset:    {ckpt['dataset']}")
    print(f"scene_res:  {ckpt['scene_resolution']}px")
    print(f"glimpse:    {ckpt['glimpse_grid_size']}")
    print(f"grids:      {ckpt['canvas_patch_grid_sizes']}")
    print()
    print(f"last_save:  {ckpt.get('timestamp', 'n/a')}")
    print(f"comet:      {ckpt.get('comet_id', 'n/a')}")

    prov_history = ckpt.get("provenance_history")
    if prov_history:
        print(f"\nprovenance history: {len(prov_history)} entries")
        prev = None
        for ts in sorted(prov_history):
            p = prov_history[ts]
            git = p.get("git_commit")
            git_s = f"{git[:8]}{'*' if p.get('git_dirty') else ''}" if git else "n/a"
            host = p.get("hostname", "n/a")
            task = p.get("slurm_array_task_id", "n/a")
            changed = prev is not None and (
                p.get("git_commit") != prev.get("git_commit")
                or p.get("hostname") != prev.get("hostname")
            )
            marker = "  CHANGED" if changed else ""
            print(f"  {ts}  git={git_s}  host={host}  task={task}{marker}")
            prev = p
    else:
        # No provenance_history — show last-save fields.
        print(f"hostname:   {ckpt.get('hostname', 'n/a')}")
        print(f"git:        {git_str}")
        print(f"slurm_job:  {ckpt.get('slurm_job_id', 'n/a')}")
        print(f"slurm_task: {ckpt.get('slurm_array_task_id', 'n/a')}")

    config_history = ckpt.get("training_config_history")
    if config_history:
        timestamps = sorted(config_history)
        print(f"\nconfig history: {len(timestamps)} entries")
        prev = None
        for ts in timestamps:
            cfg = config_history[ts]
            if prev is None:
                print(f"  {ts}  (initial)")
            else:
                changed = {k for k in cfg.keys() | prev.keys() if cfg.get(k) != prev.get(k)}
                if changed:
                    print(f"  {ts}  CHANGED:")
                    for k in sorted(changed):
                        print(f"    {k}: {prev.get(k)!r} -> {cfg.get(k)!r}")
                else:
                    print(f"  {ts}  (no changes)")
            prev = cfg


def smoke_test(path: Path) -> None:
    """Load model from checkpoint, run one forward pass with dummy input."""
    from canvit_pytorch.viewpoint import Viewpoint

    from canvit_pretrain.checkpoint import load_model

    print("\n--- smoke test ---")
    print("loading model...", end=" ", flush=True)
    model, ckpt = load_model(path)
    print("ok")

    glimpse_px = ckpt["glimpse_grid_size"] * model.backbone.patch_size_px
    canvas_grid = model.canvas_patch_grid_sizes[0]

    print(f"forward (glimpse={glimpse_px}px, canvas_grid={canvas_grid})...", end=" ", flush=True)
    with torch.no_grad():
        state = model.init_state(batch_size=1, canvas_grid_size=canvas_grid)
        vp = Viewpoint(centers=torch.zeros(1, 2), scales=torch.ones(1, 1))
        dummy = torch.randn(1, 3, glimpse_px, glimpse_px)
        out = model.forward(glimpse=dummy, state=state, viewpoint=vp, canvas_grid_size=canvas_grid)
    print("ok")
    print(f"output canvas: {out.state.canvas.shape}")


def main() -> None:
    assert len(sys.argv) > 1, __doc__
    path = Path(sys.argv[1])
    resolved = path.resolve() if path.is_symlink() else path

    print_info(path)

    if "--smoke-test" in sys.argv[2:]:
        smoke_test(resolved)


if __name__ == "__main__":
    main()
