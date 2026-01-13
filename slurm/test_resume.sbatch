#!/bin/bash
#SBATCH --job-name=resume-test
#SBATCH --account=rrg-skrishna_gpu
#SBATCH --array=0-2%1
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/resume-test_%A_%a.out
#SBATCH --error=logs/resume-test_%A_%a.err

# Minimal resume test: 3 jobs, ~10 steps each, tests full resume cycle
# Usage: sbatch slurm/test_resume.sbatch

set -euo pipefail

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== Resume Test ==="
log "Job: $SLURM_JOB_ID, Array: $SLURM_ARRAY_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
log "Host: $(hostname)"

cd ~/scratch/avp-vit
source slurm/env.sh

log "Installing dependencies..."
uv sync

mkdir -p logs

# Use fixed run name so all array tasks share the same checkpoint dir
RUN_NAME="resume-test-$$"
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    # First task generates the name and saves it
    echo "$RUN_NAME" > /tmp/resume_test_run_name_$SLURM_ARRAY_JOB_ID
else
    # Subsequent tasks read the name (wait for file to exist)
    for i in {1..30}; do
        [ -f /tmp/resume_test_run_name_$SLURM_ARRAY_JOB_ID ] && break
        sleep 1
    done
    RUN_NAME=$(cat /tmp/resume_test_run_name_$SLURM_ARRAY_JOB_ID)
fi

log "Run name: $RUN_NAME"

# Ultra-minimal settings:
# - 10 steps/job, checkpoint every 5 = tests checkpoint + resume
# - 3 array tasks = tests 2 resume cycles
log "Starting training..."
exec uv run python -m avp_vit.train \
    --run-name "$RUN_NAME" \
    --steps-per-job 10 \
    --ckpt-every 5 \
    --val-every 5 \
    --log-every 2 \
    --warmup-steps 2 \
    --batch-size 4 \
    --feature-base-dir "$FEATURES_DIR/in21k" \
    --feature-image-root "$IN21K_DIR" \
    --val-dir "$IN1K_VAL_DIR" \
    --train-index-dir "$INDEX_DIR" \
    --ckpt-dir "$CHECKPOINTS_DIR" \
    --teacher-ckpt "$DINOV3_VITB16_CKPT" \
    --compile false \
    --n-trials 1
