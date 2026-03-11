#!/bin/bash
# Interactive GPU session on Nibi
# Usage (from repo root): bash slurm/interactive.sh [time] [mem]

TIME="${1:-0:20:00}"
MEM="${2:-32G}"
REPO_DIR="$(pwd)"

exec srun --account=rrg-skrishna_gpu --gres=gpu:1 --mem="$MEM" --cpus-per-task=16 --time="$TIME" --pty bash -c "
cd '$REPO_DIR'
source slurm/env.sh
uv sync
exec bash
"
