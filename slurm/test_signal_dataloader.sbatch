#!/bin/bash
#SBATCH --job-name=signal-dl-test
#SBATCH --time=0:05:00
#SBATCH --signal=USR1@60
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=signal_dl_test_%j.out

# 5-minute job with DataLoaders, SIGUSR1 60 seconds before walltime
# Tests signal handling with actual DataLoader workers

set -uo pipefail

echo "=== Job $SLURM_JOB_ID starting at $(date) ==="
echo "Working directory: $(pwd)"

# Adjust these paths for your cluster
DATA_DIR="${DATA_DIR:-/datasets/ILSVRC/Data/CLS-LOC/train}"
INDEX_DIR="${INDEX_DIR:-$SCRATCH/imagenet_index}"

srun python3 slurm/test_signal_dataloader.py \
    --data-dir "$DATA_DIR" \
    --index-dir "$INDEX_DIR" \
    --num-workers 4 \
    --batch-size 32 \
    --target-steps 10000

EXIT_CODE=$?
echo "Python exited with code $EXIT_CODE"
echo "=== Job finished at $(date) ==="
