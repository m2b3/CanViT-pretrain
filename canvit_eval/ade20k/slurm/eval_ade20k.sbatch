#!/bin/bash
#SBATCH --job-name=ade20k-eval
#SBATCH --account=rrg-skrishna_gpu
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/ade20k-eval-%j.out
#SBATCH --error=logs/ade20k-eval-%j.err

# Usage: sbatch canvit_eval/ade20k/slurm/eval_ade20k.sbatch --probe-ckpt /path/to/best.pt [...]

set -eux
source canvit_eval/ade20k/slurm/setup.sh
mkdir -p logs

echo "=== ADE20K Canvas Probe Evaluation ==="
echo "Job: $SLURM_JOB_ID on $(hostname)"

exec uv run python -m canvit_eval.ade20k evaluate "$@"
