#!/bin/bash
#SBATCH --job-name=avp-train
#SBATCH --account=rrg-skrishna_gpu
#SBATCH --time=00:30:00
#SBATCH --array=0-400%1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/train-%A_%a.out
#SBATCH --error=logs/train-%A_%a.err
#SBATCH --mail-user=me@yberreby.com
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE

# AVP-ViT training: 401 jobs × 4992 steps ≈ 2M steps (~8 days)
#
# Default (just run it):
#   sbatch slurm/train.sbatch
#
# Continue existing run:
#   sbatch --array=0-99%1 slurm/train.sbatch --run-name train-XXXXXXX
#
# Quick test:
#   sbatch --array=0-2%1 --time=00:10:00 slurm/train.sbatch --steps-per-job 10

set -eux

log() { echo "[$(date '+%H:%M:%S')] $*"; }
timed() { local start=$SECONDS; "$@"; log "  ↳ $((SECONDS - start))s"; }

log "=== AVP-ViT Training ==="
if [ -n "${SLURM_ARRAY_JOB_ID:-}" ]; then
    log "Array Job: $SLURM_ARRAY_JOB_ID, Task: $SLURM_ARRAY_TASK_ID, Host: $(hostname)"
    RUN_NAME="train-${SLURM_ARRAY_JOB_ID}"
else
    log "Job: $SLURM_JOB_ID on $(hostname)"
    RUN_NAME=""
fi
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
log "Args: $*"

cd ~/scratch/avp-vit
source slurm/env.sh

# Dependencies
log "Installing dependencies..."
timed uv sync

# Ensure logs dir exists
mkdir -p logs

# Training with precomputed features
log "Starting training..."
RUN_NAME_ARG=()
[ -n "$RUN_NAME" ] && RUN_NAME_ARG=(--run-name "$RUN_NAME")
exec uv run python -m avp_vit.train \
    "${RUN_NAME_ARG[@]}" \
    --feature-base-dir "$FEATURES_DIR/in21k" \
    --feature-image-root "$IN21K_DIR" \
    --val-dir "$IN1K_VAL_DIR" \
    --train-index-dir "$INDEX_DIR" \
    --ckpt-dir "$CHECKPOINTS_DIR" \
    --teacher-ckpt "$DINOV3_VITB16_CKPT" \
    --n-trials 1 \
    --val-every 1000 \
    --no-use-checkpointing \
    "$@"
