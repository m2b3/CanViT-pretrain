#!/bin/bash
#SBATCH --job-name=canvit-ade20k
#SBATCH --account=rrg-skrishna_gpu
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/eval-ade20k-%j.out
#SBATCH --error=logs/eval-ade20k-%j.err

# CanViT ADE20K probe training
# Usage: sbatch slurm/eval_ade20k.sbatch

set -eux

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== CanViT ADE20K Evaluation ==="
log "Job: $SLURM_JOB_ID on $(hostname)"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
log "Args: $*"

cd ~/scratch/CanViT-train
source slurm/env.sh

mkdir -p logs

# Extract ADE20K to SLURM_TMPDIR if needed
ADE20K_ROOT="$SLURM_TMPDIR/ADEChallengeData2016"
if [ ! -d "$ADE20K_ROOT" ]; then
    log "Extracting ADE20K dataset..."
    unzip -q "$ADE20K_ZIP" -d "$SLURM_TMPDIR"
fi

log "Running probe training..."
exec uv run python -m canvit_eval.ade20k \
    --ade20k-root "$ADE20K_ROOT" \
    "$@"
