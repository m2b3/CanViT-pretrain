#!/bin/bash
#SBATCH --job-name=resume-test
#SBATCH --array=0-2%1
#SBATCH --time=00:10:00
#SBATCH --output=logs/resume-test_%A_%a.out
#SBATCH --error=logs/resume-test_%A_%a.err
# TODO: Add your partition, GPU, etc settings

set -euo pipefail

# === CONFIGURE THESE ===
FEATURE_BASE_DIR="/path/to/your/features"
FEATURE_IMAGE_ROOT="/path/to/your/images"
# =======================

export COMET_API_KEY=$(cat ~/comet_api_key.txt)

echo "=== Job Info ==="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "================"

# Ultra-minimal settings for fast testing:
# - 10 steps per job = ~10 seconds of actual training
# - checkpoint every 5 steps = 2 checkpoints per job
# - validate every 5 steps = 2 validations per job
# - warmup 2 steps = essentially instant
# - 3 array tasks = tests 2 resume cycles

python -m avp_vit.train \
    --run-name "resume-test-$(date +%Y%m%d-%H%M%S)" \
    --steps-per-job 10 \
    --ckpt-every 5 \
    --val-every 5 \
    --log-every 2 \
    --warmup-steps 2 \
    --batch-size 4 \
    --feature-base-dir "$FEATURE_BASE_DIR" \
    --feature-image-root "$FEATURE_IMAGE_ROOT" \
    --compile false \
    --n-trials 1

echo "=== Job $SLURM_ARRAY_TASK_ID completed ==="
