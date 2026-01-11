#!/bin/bash
# Interactive GPU session on Nibi
# Usage: bash ~/scratch/avp-vit/slurm/interactive.sh [time]
#   bash ~/scratch/avp-vit/slurm/interactive.sh         # 1h
#   bash ~/scratch/avp-vit/slurm/interactive.sh 2:00:00 # 2h

TIME="${1:-1:00:00}"

exec srun --account=rrg-skrishna_gpu --gres=gpu:h100:1 --mem=64G --cpus-per-task=16 --time="$TIME" --pty bash -c "
cd ~/scratch/avp-vit
source slurm/env.sh
uv sync
exec bash
"
