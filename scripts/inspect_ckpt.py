"""Inspect a training checkpoint. Usage: uv run python scripts/inspect_ckpt.py <path>"""

import sys
from pathlib import Path

import torch


def main() -> None:
    assert len(sys.argv) > 1, "Usage: uv run python scripts/inspect_ckpt.py <checkpoint.pt>"
    path = Path(sys.argv[1])
    resolved = path.resolve() if path.is_symlink() else path
    ckpt = torch.load(resolved, weights_only=False, map_location="cpu")

    sched = ckpt.get("scheduler_state")
    step = sched["last_epoch"] if sched else ckpt.get("step")
    n_params = sum(p.numel() for p in ckpt["state_dict"].values()) / 1e6
    has_optim = ckpt.get("optimizer_state") is not None
    git = ckpt.get("git_commit")
    git_str = f"{git[:8]}{'*' if ckpt.get('git_dirty') else ''}" if git else "n/a"

    print(f"path:       {path}" + (f" -> {resolved.name}" if path.is_symlink() else ""))
    print(f"size:       {resolved.stat().st_size / 1024**2:.0f} MB")
    print(f"step:       {step}")
    print(f"train_loss: {ckpt.get('train_loss', 'n/a')}")
    print(f"params:     {n_params:.1f}M")
    print(f"resumable:  {'yes' if has_optim and sched else 'no (missing optimizer/scheduler)'}")
    print()
    print(f"backbone:   {ckpt['backbone_name']}")
    print(f"teacher:    {ckpt['teacher_name']}")
    print(f"dataset:    {ckpt['dataset']}")
    print(f"scene_res:  {ckpt['scene_resolution']}px")
    print(f"glimpse:    {ckpt['glimpse_grid_size']}")
    print(f"grids:      {ckpt['canvas_patch_grid_sizes']}")
    print()
    print(f"timestamp:  {ckpt.get('timestamp', 'n/a')}")
    print(f"hostname:   {ckpt.get('hostname', 'n/a')}")
    print(f"git:        {git_str}")
    print(f"comet:      {ckpt.get('comet_id', 'n/a')}")
    print(f"slurm_job:  {ckpt.get('slurm_job_id', 'n/a')}")
    print(f"slurm_task: {ckpt.get('slurm_array_task_id', 'n/a')}")

    history = ckpt.get("training_config_history")
    if history:
        print(f"\nconfig history: {len(history)} entries")
        for ts in sorted(history):
            print(f"  {ts}")


if __name__ == "__main__":
    main()
