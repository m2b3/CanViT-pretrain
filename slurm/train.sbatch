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
#   sbatch slurm/train.sbatch --batch-size 32
#   sbatch slurm/train.sbatch --grid-sizes 16 32 64

set -e

echo "=== AVP-ViT Training ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Overrides: $@"
echo ""

# Environment setup
module load java/17.0.6
export PATH=$HOME/.local/bin:$PATH
export UV_CACHE_DIR="$SLURM_TMPDIR/.uv-cache"
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv"

# Comet
if [ -f ~/comet_api_key.txt ]; then
    export COMET_API_KEY=$(cat ~/comet_api_key.txt)
fi

# torch.compile cache
export TORCH_COMPILE_CACHE_DIR="$SCRATCH/.torch_compile_cache"

cd ~/scratch/avp-vit

echo "=== Installing dependencies ==="
uv sync

echo ""
echo "=== Starting training ==="
uv run python -m scripts.train_scene_match \
    --train-dir /datashare/imagenet/winter21_whole \
    --val-dir /datashare/imagenet/ILSVRC2012/val \
    --index-dir "$SCRATCH/avp_indices" \
    "$@"

echo ""
echo "=== Done ==="
