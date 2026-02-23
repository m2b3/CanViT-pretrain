#!/bin/bash
#SBATCH --job-name=canvit-sa1b
#SBATCH --account=rrg-skrishna_gpu
#SBATCH --time=01:00:00
#SBATCH --array=0-99%1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=logs/sa1b-train-%A_%a.out
#SBATCH --error=logs/sa1b-train-%A_%a.err
#SBATCH --mail-user=me@yberreby.com
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE

# SA-1B continual pretraining: tar extraction → training.
# Each task extracts tars to SLURM_TMPDIR, then trains for steps_per_job steps.
#
# First run (seed from HF Hub):
#   sbatch sa1b/train.sh
#
# Continue existing run:
#   sbatch --array=0-99%1 sa1b/train.sh --run-name sa1b-train-XXXXXXX
#
# Quick test (1 shard = 174 steps):
#   sbatch --array=0-0%1 --time=00:20:00 sa1b/train.sh --steps-per-job 174

set -eux

log() { echo "[$(date '+%H:%M:%S')] $*"; }
timed() { local start=$SECONDS; "$@"; log "  ↳ $((SECONDS - start))s"; }

# === CONFIG ===
BATCH_SIZE=64
SHARDS_PER_JOB=28       # 28 shards × 174 batches/shard = 4872 steps
STEPS_PER_JOB=4872      # Must equal SHARDS_PER_JOB × batches_per_shard
HF_SEED="canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02"

log "=== SA-1B Continual Pretraining ==="
if [ -n "${SLURM_ARRAY_JOB_ID:-}" ]; then
    log "Array Job: $SLURM_ARRAY_JOB_ID, Task: $SLURM_ARRAY_TASK_ID, Host: $(hostname)"
    RUN_NAME="sa1b-train-${SLURM_ARRAY_JOB_ID}"
else
    log "Job: $SLURM_JOB_ID on $(hostname)"
    RUN_NAME="sa1b-manual-$(date +%Y%m%d-%H%M)"
fi

# Override RUN_NAME if user passed --run-name in "$@"
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
    if [ "${ARGS[i]}" = "--run-name" ]; then
        RUN_NAME="${ARGS[i+1]}"
        break
    fi
done

log "Run name: $RUN_NAME"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
log "Args: $*"

source slurm/env.sh

mkdir -p logs

# === PATHS ===
SHARDS_DIR="$SA1B_FEATURES_DIR/sa1b/dinov3_vitb16/1024/shards"
IMAGE_DIR="$SLURM_TMPDIR/sa1b_images"
RUN_DIR="$CHECKPOINTS_DIR/$RUN_NAME"
mkdir -p "$IMAGE_DIR"

# === PLAN: determine which tars to extract ===
log "Planning tar extraction (run_dir=$RUN_DIR, shards_dir=$SHARDS_DIR)..."
TAR_INDICES=$(uv run python sa1b/plan_job.py "$RUN_DIR" "$SHARDS_DIR" "$BATCH_SIZE" "$SHARDS_PER_JOB")
N_TARS=$(echo "$TAR_INDICES" | wc -l | tr -d ' ')
log "Need $N_TARS tars: $(echo $TAR_INDICES | tr '\n' ' ')"

# === EXTRACT tars to SLURM_TMPDIR ===
log "Extracting $N_TARS tars to $IMAGE_DIR..."
for tar_idx in $TAR_INDICES; do
    TAR_PATH="$SA1B_TAR_DIR/sa_${tar_idx}.tar"
    if [ ! -f "$TAR_PATH" ]; then
        log "FATAL: Tar not found: $TAR_PATH"
        exit 1
    fi
    log "  Extracting $TAR_PATH..."
    timed tar xf "$TAR_PATH" --strip-components=1 --wildcards -C "$IMAGE_DIR" '*.jpg'
done
log "Extraction done."

# === TRAINING ===
# --hf-seed-ckpt is safe to always pass: ignored on RESUME (latest.pt takes priority).
# Normalizer auto-detects uninitialized standardizers — no need for --reset-normalizer.
log "Starting training..."
exec uv run python -m canvit_pretrain.train \
    --run-name "$RUN_NAME" \
    --hf-seed-ckpt "$HF_SEED" \
    --feature-base-dir "$SA1B_FEATURES_DIR/sa1b" \
    --feature-image-root "$IMAGE_DIR" \
    --val-dir "$IN1K_VAL_IMAGE_DIR" \
    --ckpt-dir "$CHECKPOINTS_DIR" \
    --canvas-patch-grid-size 64 \
    --scene-resolution 1024 \
    --steps-per-job "$STEPS_PER_JOB" \
    --batch-size "$BATCH_SIZE" \
    --dataset sa1b \
    "$@"
