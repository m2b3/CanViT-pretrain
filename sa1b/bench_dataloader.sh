#!/bin/bash
#SBATCH --job-name=bench-dataloader
#SBATCH --account=def-skrishna
#SBATCH --time=0:15:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=logs/bench_dataloader_%j.out
#SBATCH --error=logs/bench_dataloader_%j.err

# CPU-only benchmark of SA-1B dataloader.
# Uses the real AllShardsDataset code path.

set -euo pipefail
source slurm/env.sh
mkdir -p logs

SHARD_DIR="$SA1B_FEATURES_DIR/sa1b/dinov3_vitb16/1024/shards"

echo "========================================"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "CPUs:         $SLURM_CPUS_PER_TASK"
echo "Date:         $(date -Iseconds)"
echo "========================================"

time uv run python sa1b/bench_dataloader.py \
    --shard-dir "$SHARD_DIR" \
    --tar-dir "$SA1B_TAR_DIR" \
    --image-sizes 1024 1500 \
    --workers 0 1 2 4 \
    --n-serial 200 \
    --n-batches 30
