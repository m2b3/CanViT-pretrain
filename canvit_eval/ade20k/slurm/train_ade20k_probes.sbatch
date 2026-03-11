#!/bin/bash
#SBATCH --job-name=ade20k-train
#SBATCH --account=rrg-skrishna_gpu
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/ade20k-train-%j.out
#SBATCH --error=logs/ade20k-train-%j.err

# Usage: sbatch canvit_eval/ade20k/slurm/train_ade20k_probes.sbatch [--max-steps 5000] [...]

set -eux
source canvit_eval/ade20k/slurm/setup.sh
mkdir -p logs

echo "=== ADE20K Canvas Probe Training ==="
echo "Job: $SLURM_JOB_ID on $(hostname)"

exec uv run python -m canvit_eval.ade20k train "$@"
