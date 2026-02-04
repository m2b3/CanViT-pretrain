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
export ADE20K_ZIP=~/projects/def-skrishna/datasets/ade20k/ADEChallengeData2016.zip

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
export CANVIT_FLAGSHIP_CKPT="$CHECKPOINTS_DIR/CanViT-flagship.pt"

# ==============================================================================
# COMET
# ==============================================================================

if [ -f ~/comet_api_key.txt ]; then
    export COMET_API_KEY=$(cat ~/comet_api_key.txt)
    echo "[env] Loaded COMET_API_KEY"
else
    echo "[env] No comet_api_key.txt (optional)"
fi

# ==============================================================================
# HUGGINGFACE (required for private model repos)
# ==============================================================================

if [ -f ~/hf_token.txt ]; then
    export HF_TOKEN=$(cat ~/hf_token.txt)
    echo "[env] Loaded HF_TOKEN"
elif [ -f "$HF_HOME/token" ]; then
    export HF_TOKEN=$(cat "$HF_HOME/token")
    echo "[env] Loaded HF_TOKEN from $HF_HOME/token"
else
    echo "[env] No HF token found (may fail on private repos)"
fi

# Sync and activate venv (only in SLURM jobs where we control venv location)
uv sync
if [ -n "$UV_PROJECT_ENVIRONMENT" ]; then
    source "$UV_PROJECT_ENVIRONMENT/bin/activate"
else
    echo "[env] No venv activation (use 'uv run' directly)"
fi

echo "[env] Done"
