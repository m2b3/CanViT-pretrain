#!/bin/bash
#SBATCH --job-name=ade20k-eval
#SBATCH --account=rrg-skrishna_gpu
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/ade20k-eval-%j.out
#SBATCH --error=logs/ade20k-eval-%j.err

# ADE20K probe evaluation (single run)
# Usage: sbatch slurm/eval_ade20k.sbatch --probe-ckpt /path/to/best.pt [--policy random] [...]

set -eux

cd ~/scratch/CanViT-train
source slurm/env.sh
mkdir -p logs

echo "=== ADE20K Probe Evaluation ==="
echo "Job: $SLURM_JOB_ID on $(hostname)"

# Extract ADE20K
export ADE20K_ROOT="$SLURM_TMPDIR/ADEChallengeData2016"
[ ! -d "$ADE20K_ROOT" ] && unzip -q "$ADE20K_ZIP" -d "$SLURM_TMPDIR"

exec uv run python -m canvit_eval.ade20k evaluate "$@"
