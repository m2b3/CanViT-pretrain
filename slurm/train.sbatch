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
#   sbatch slurm/train.sbatch --batch-size 128 --n-trials 1

set -e

echo "=== AVP-ViT Training ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Overrides: $@"
echo ""

# Environment
export PATH=$HOME/.local/bin:$PATH
module load java/17.0.6
export UV_CACHE_DIR="$SLURM_TMPDIR/.uv-cache"
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv"
export HF_HOME="$SCRATCH/.huggingface"
export TORCH_HOME="$SCRATCH/.torch"
export TORCH_COMPILE_CACHE_DIR="$SCRATCH/.torch_compile_cache"

# Comet
if [ -f ~/comet_api_key.txt ]; then
    export COMET_API_KEY=$(cat ~/comet_api_key.txt)
fi

cd ~/scratch/avp-vit

echo "=== Installing dependencies ==="
uv sync

echo ""
echo "=== Starting training ==="
uv run python -m avp_vit.train \
    --train-dir /datashare/imagenet/winter21_whole \
    --val-dir /datashare/imagenet/ILSVRC2012/val \
    --index-dir "$SCRATCH/in21k_index" \
    --ckpt-dir "$SCRATCH/avp_checkpoints" \
    --teacher-ckpt ~/projects/def-skrishna/checkpoints/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
    "$@"

echo ""
echo "=== Done ==="
