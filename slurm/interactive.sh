#!/bin/bash
# Interactive GPU session on Nibi
# Usage: bash slurm/interactive.sh [time] [--bench]
#   bash slurm/interactive.sh           # 1h, drops to shell
#   bash slurm/interactive.sh 2:00:00   # 2h
#   bash slurm/interactive.sh --bench   # 1h, opens ipython with bench preload

TIME="${1:-1:00:00}"
BENCH=""

if [[ "$1" == "--bench" ]]; then
    TIME="1:00:00"
    BENCH=1
elif [[ "$2" == "--bench" ]]; then
    BENCH=1
fi

exec srun --account=rrg-skrishna_gpu --gres=gpu:h100:1 --mem=32G --cpus-per-task=8 --time="$TIME" --pty bash -c "
cd ~/scratch/avp-vit
source slurm/env.sh
uv sync
echo ''
if [[ -n '$BENCH' ]]; then
    echo '=== Starting ipython with bench preload ==='
    exec uv run ipython -i scripts/bench_interactive.py
else
    echo '=== Ready. Try: uv run ipython -i scripts/bench_interactive.py ==='
    exec bash
fi
"
