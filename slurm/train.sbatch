#!/bin/bash
#SBATCH --job-name=avp-train
#SBATCH --account=rrg-skrishna_gpu
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err

# AVP-ViT training on IN21k
# Usage: sbatch slurm/train.sbatch [tyro overrides...]
# Examples:
#   sbatch slurm/train.sbatch --batch-size 128
#   sbatch slurm/train.sbatch --n-trials 1

set -eu

log() { echo "[$(date '+%H:%M:%S')] $*"; }
timed() { local start=$SECONDS; "$@"; log "  ↳ $((SECONDS - start))s"; }

log "=== AVP-ViT Training ==="
log "Job: $SLURM_JOB_ID on $(hostname)"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
log "Args: $*"

cd ~/scratch/avp-vit
source slurm/env.sh

# Dependencies
log "Installing dependencies..."
timed uv sync

# Training
log "Starting training..."
exec uv run python -m avp_vit.train \
    --train-dir "$IN21K_DIR" \
    --val-dir "$IN1K_VAL_DIR" \
    --train-index-dir "$INDEX_DIR" \
    --ckpt-dir "$CHECKPOINTS_DIR" \
    --teacher-ckpt "$DINOV3_VITB16_CKPT" \
    "$@"
