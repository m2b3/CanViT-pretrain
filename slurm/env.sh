# Nibi cluster environment
# Source this: source slurm/env.sh

echo "[env] Setting up environment..."

export PATH=$HOME/.local/bin:$PATH
module load java/17.0.6 2>/dev/null && echo "[env] Loaded java/17.0.6" || true

# ==============================================================================
# UV / TORCH CACHES
# ==============================================================================

if [ -n "$SLURM_TMPDIR" ]; then
    export UV_CACHE_DIR="$SLURM_TMPDIR/.uv-cache"
    export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv"
    echo "[env] Using SLURM_TMPDIR for uv cache/venv"
else
    echo "[env] No SLURM_TMPDIR (interactive session)"
fi

export HF_HOME="$SCRATCH/.huggingface"
export TORCH_HOME="$SCRATCH/.torch"
export TORCH_COMPILE_CACHE_DIR="$SCRATCH/.torch_compile_cache"

# ==============================================================================
# DATASETS
# ==============================================================================

export IN21K_DIR=/datashare/imagenet/winter21_whole
export IN1K_VAL_DIR=/datashare/imagenet/ILSVRC2012/val

# ==============================================================================
# STORAGE
# ==============================================================================

export INDEX_DIR="$SCRATCH/dataset_indexes"
export CHECKPOINTS_DIR=~/projects/def-skrishna/checkpoints
export FEATURES_DIR=~/projects/def-skrishna/dinov3_dense_features

# ==============================================================================
# KNOWN CHECKPOINTS
# ==============================================================================

export DINOV3_VITB16_CKPT="$CHECKPOINTS_DIR/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
# export DINOV3_VITL16_CKPT="$CHECKPOINTS_DIR/dinov3/dinov3_vitl16_pretrain_....pth"

# ==============================================================================
# COMET
# ==============================================================================

if [ -f ~/comet_api_key.txt ]; then
    export COMET_API_KEY=$(cat ~/comet_api_key.txt)
    echo "[env] Loaded COMET_API_KEY"
else
    echo "[env] No comet_api_key.txt (optional)"
fi

echo "[env] Done"
