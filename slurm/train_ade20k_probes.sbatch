#!/bin/bash
#SBATCH --job-name=ade20k-train
#SBATCH --account=rrg-skrishna_gpu
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/ade20k-train-%j.out
#SBATCH --error=logs/ade20k-train-%j.err

# ADE20K probe training
# Usage: sbatch slurm/train_ade20k_probes.sbatch [--max-steps 5000] [...]

set -eux

cd ~/scratch/CanViT-train
source slurm/env.sh
mkdir -p logs

echo "=== ADE20K Probe Training ==="
echo "Job: $SLURM_JOB_ID on $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo N/A)"

# Extract ADE20K
export ADE20K_ROOT="$SLURM_TMPDIR/ADEChallengeData2016"
[ ! -d "$ADE20K_ROOT" ] && unzip -q "$ADE20K_ZIP" -d "$SLURM_TMPDIR"

exec uv run python -m canvit_eval.ade20k train "$@"
