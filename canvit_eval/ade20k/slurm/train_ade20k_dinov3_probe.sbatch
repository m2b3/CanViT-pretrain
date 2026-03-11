#!/bin/bash
#SBATCH --job-name=ade20k-dinov3-train
#SBATCH --account=rrg-skrishna_gpu
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/ade20k-dinov3-train-%j.out
#SBATCH --error=logs/ade20k-dinov3-train-%j.err

# DINOv3 baseline probe training at a given resolution
# Usage:
#   sbatch canvit_eval/ade20k/slurm/train_ade20k_dinov3_probe.sbatch 128
#   sbatch canvit_eval/ade20k/slurm/train_ade20k_dinov3_probe.sbatch 256
#   sbatch canvit_eval/ade20k/slurm/train_ade20k_dinov3_probe.sbatch 128 --model facebook/dinov3-vits16-pretrain

set -eux

RESOLUTION="${1:?Usage: sbatch train_ade20k_dinov3_probe.sbatch <resolution> [extra args...]}"
shift

source canvit_eval/ade20k/slurm/setup.sh
mkdir -p logs

echo "=== DINOv3 Probe Training ==="
echo "Resolution: ${RESOLUTION}px"
echo "Job: $SLURM_JOB_ID on $(hostname)"

exec uv run python -m canvit_eval.ade20k train-dinov3-probe --resolution "$RESOLUTION" "$@"
