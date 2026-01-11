# AVP-ViT environment setup for Nibi
# Source this: source slurm/env.sh

echo "[env] Setting up environment..."

export PATH=$HOME/.local/bin:$PATH
module load java/17.0.6 2>/dev/null && echo "[env] Loaded java/17.0.6" || true

# Fast local storage if in SLURM job
if [ -n "$SLURM_TMPDIR" ]; then
    export UV_CACHE_DIR="$SLURM_TMPDIR/.uv-cache"
    export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv"
    echo "[env] Using SLURM_TMPDIR for uv cache/venv"
else
    echo "[env] No SLURM_TMPDIR (not in job?)"
fi

# Caches on scratch (persistent)
export HF_HOME="$SCRATCH/.huggingface"
export TORCH_HOME="$SCRATCH/.torch"
export TORCH_COMPILE_CACHE_DIR="$SCRATCH/.torch_compile_cache"
echo "[env] HF_HOME=$HF_HOME"
echo "[env] TORCH_HOME=$TORCH_HOME"

# Comet
if [ -f ~/comet_api_key.txt ]; then
    export COMET_API_KEY=$(cat ~/comet_api_key.txt)
    echo "[env] Loaded COMET_API_KEY"
else
    echo "[env] No comet_api_key.txt (optional)"
fi

echo "[env] Done"
