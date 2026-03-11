#!/bin/bash
#SBATCH --job-name=canvit-in1k
#SBATCH --account=rrg-skrishna_gpu
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/eval-in1k-%j.out
#SBATCH --error=logs/eval-in1k-%j.err

# CanViT ImageNet-1k evaluation
# Usage: sbatch slurm/eval_in1k.sbatch

set -eux

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== CanViT IN1k Evaluation ==="
log "Job: $SLURM_JOB_ID on $(hostname)"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
log "Args: $*"

source slurm/env.sh

mkdir -p logs outputs

log "Running evaluation..."
exec uv run python -m canvit_eval.in1k "$@"
