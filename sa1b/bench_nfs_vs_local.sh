#!/bin/bash
#SBATCH --job-name=bench-nfs-local
#SBATCH --account=def-skrishna
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=logs/bench_nfs_vs_local_%j.out
#SBATCH --error=logs/bench_nfs_vs_local_%j.err

# NFS vs local NVMe benchmark for shard access.
# Tests: sequential read, mmap patterns, local staging, DataLoader throughput.
# Needs more mem than bench_dataloader (copies 66GB shard to $SLURM_TMPDIR).

set -euo pipefail
source slurm/env.sh
mkdir -p logs

SHARD_DIR="$SA1B_FEATURES_DIR/sa1b/dinov3_vitb16/1024/shards"
SHARD_PATH="$SHARD_DIR/sa_000020.pt"
TAR_PATH="$SA1B_TAR_DIR/sa_000020.tar"

echo "========================================"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "CPUs:         $SLURM_CPUS_PER_TASK"
echo "Mem:          $SLURM_MEM_PER_NODE MB"
echo "TMPDIR:       $SLURM_TMPDIR"
echo "Date:         $(date -Iseconds)"
echo "========================================"

time uv run python sa1b/bench_nfs_vs_local.py \
    --shard-path "$SHARD_PATH" \
    --tar-path "$TAR_PATH" \
    --tmpdir "$SLURM_TMPDIR" \
    --workers 0 1 2 4 8 \
    --n-serial 500 \
    --n-batches 40
