#!/bin/bash
#SBATCH --account=rrg-skrishna_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --output=logs/export_sa1b_%j.out
#SBATCH --error=logs/export_sa1b_%j.err

# ==============================================================================
# Export DINOv3 features for SA-1B images at 1024px.
#
# Single job: extracts ALL tars in $SA1B_TAR_DIR to SLURM_TMPDIR,
# builds one parquet, runs export_features.py (untouched).
# Features land in one flat shards/ dir — compatible with training loader.
#
# For 3 tars (~33k images, ~33GB extracted): fits on SLURM_TMPDIR.
# For larger scale, will need a different approach.
#
# USAGE:
#   sbatch scripts/sa1b/export_features.sh
# ==============================================================================

set -euo pipefail

source slurm/env.sh
mkdir -p logs

# ==============================================================================
# CONFIG
# ==============================================================================

IMAGE_SIZE=1024
TEACHER_MODEL=dinov3_vitb16
TEACHER_CKPT="$DINOV3_VITB16_CKPT"
SHARD_SIZE=4096
BATCH_SIZE=32

LOCAL_IMAGE_DIR="$SLURM_TMPDIR/sa1b_images"
LOCAL_PARQUET="$SLURM_TMPDIR/sa1b_index.parquet"
OUT_DIR="$SA1B_FEATURES_DIR/sa1b/${TEACHER_MODEL}/${IMAGE_SIZE}"

# ==============================================================================
# LOGGING
# ==============================================================================

echo "========================================"
echo "SLURM_JOB_ID:       $SLURM_JOB_ID"
echo "Node:               $(hostname)"
echo "Date:               $(date -Iseconds)"
echo "Git commit:         $(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
echo "========================================"
echo "GPU:                $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "========================================"
echo "SA1B_TAR_DIR:       $SA1B_TAR_DIR"
echo "IMAGE_SIZE:         $IMAGE_SIZE"
echo "OUT_DIR:            $OUT_DIR"
echo "BATCH_SIZE:         $BATCH_SIZE"
echo "========================================"

# ==============================================================================
# SANITY CHECKS
# ==============================================================================

[[ -d "$SA1B_TAR_DIR" ]] || { echo "FATAL: Tar dir not found: $SA1B_TAR_DIR" >&2; exit 1; }
[[ -f "$TEACHER_CKPT" ]] || { echo "FATAL: Teacher ckpt not found: $TEACHER_CKPT" >&2; exit 1; }

TARS=("$SA1B_TAR_DIR"/*.tar)
[[ ${#TARS[@]} -gt 0 ]] || { echo "FATAL: No .tar files in $SA1B_TAR_DIR" >&2; exit 1; }
echo "Found ${#TARS[@]} tars"

# ==============================================================================
# STEP 1: Extract JPEGs from ALL tars to local SSD
# ==============================================================================

mkdir -p "$LOCAL_IMAGE_DIR"
START_TIME=$(date +%s)

for tar in "${TARS[@]}"; do
    echo "Extracting $(basename "$tar") ..."
    tar xzf "$tar" --wildcards '*/sa_*.jpg' --strip-components=1 -C "$LOCAL_IMAGE_DIR"
done

N_IMAGES=$(find "$LOCAL_IMAGE_DIR" -name '*.jpg' | wc -l)
ELAPSED=$(($(date +%s) - START_TIME))
echo "Extracted $N_IMAGES images from ${#TARS[@]} tars in ${ELAPSED}s"

# ==============================================================================
# STEP 2: Build parquet index
# ==============================================================================

echo "Building parquet index..."
uv run python scripts/sa1b/build_parquet.py \
    --image-dir "$LOCAL_IMAGE_DIR" \
    --output "$LOCAL_PARQUET"

# ==============================================================================
# STEP 3: Export features
# ==============================================================================

echo "Exporting features..."
START_TIME=$(date +%s)

uv run python scripts/export_features.py \
    --parquet "$LOCAL_PARQUET" \
    --image-root "$LOCAL_IMAGE_DIR" \
    --out-dir "$OUT_DIR" \
    --teacher-ckpt "$TEACHER_CKPT" \
    --teacher-model "$TEACHER_MODEL" \
    --image-size "$IMAGE_SIZE" \
    --shard-size "$SHARD_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --start-shard 0 \
    --end-shard 1000

EXIT_CODE=$?
ELAPSED=$(($(date +%s) - START_TIME))

echo "========================================"
echo "Exit code:          $EXIT_CODE"
echo "Elapsed (export):   ${ELAPSED}s"
echo "End time:           $(date -Iseconds)"
echo "========================================"

exit $EXIT_CODE
