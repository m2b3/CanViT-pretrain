#!/bin/bash
#SBATCH --account=rrg-skrishna_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=logs/export_sa1b_%A_%a.out
#SBATCH --error=logs/export_sa1b_%A_%a.err

# ==============================================================================
# Export DINOv3 features for SA-1B images at 1024px.
#
# Each array task handles one tar: extracts JPEGs to SLURM_TMPDIR,
# builds a parquet index, runs export_features.py (untouched).
#
# USAGE:
#   sbatch --array=0-2 slurm/export_sa1b_features.sh     # first 3 tars
#   sbatch --array=0 slurm/export_sa1b_features.sh        # just tar 0
#
# Array task ID maps to line (ID+2) in sa1b_links.tsv (line 1 = header).
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

TASK_ID=${SLURM_ARRAY_TASK_ID:?Must run as array job}
TAR_LINE=$((TASK_ID + 2))
TAR_NAME=$(sed -n "${TAR_LINE}p" "$SA1B_LINKS" | cut -f1)
TAR_PATH="$SA1B_TAR_DIR/$TAR_NAME"

LOCAL_IMAGE_DIR="$SLURM_TMPDIR/sa1b_images"
LOCAL_PARQUET="$SLURM_TMPDIR/sa1b_index.parquet"
# Each tar gets its own output dir to avoid shard ID collisions across tasks.
TAR_STEM="${TAR_NAME%.tar}"
OUT_DIR="$SA1B_FEATURES_DIR/sa1b/${TEACHER_MODEL}/${IMAGE_SIZE}/${TAR_STEM}"

# ==============================================================================
# LOGGING
# ==============================================================================

echo "========================================"
echo "SLURM_JOB_ID:       $SLURM_JOB_ID"
echo "SLURM_ARRAY_TASK_ID: $TASK_ID"
echo "Node:               $(hostname)"
echo "Date:               $(date -Iseconds)"
echo "Git commit:         $(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
echo "========================================"
echo "GPU:                $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "========================================"
echo "TAR_NAME:           $TAR_NAME"
echo "TAR_PATH:           $TAR_PATH"
echo "IMAGE_SIZE:         $IMAGE_SIZE"
echo "OUT_DIR:            $OUT_DIR"
echo "BATCH_SIZE:         $BATCH_SIZE"
echo "========================================"

# ==============================================================================
# SANITY CHECKS
# ==============================================================================

[[ -f "$TAR_PATH" ]] || { echo "FATAL: Tar not found: $TAR_PATH" >&2; exit 1; }
[[ -f "$TEACHER_CKPT" ]] || { echo "FATAL: Teacher ckpt not found: $TEACHER_CKPT" >&2; exit 1; }

# ==============================================================================
# STEP 1: Extract JPEGs from tar to local SSD
# ==============================================================================

echo "Extracting JPEGs from $TAR_NAME to $LOCAL_IMAGE_DIR ..."
mkdir -p "$LOCAL_IMAGE_DIR"
START_TIME=$(date +%s)

tar xzf "$TAR_PATH" --wildcards '*/sa_*.jpg' --strip-components=1 -C "$LOCAL_IMAGE_DIR"

N_IMAGES=$(ls "$LOCAL_IMAGE_DIR"/*.jpg 2>/dev/null | wc -l)
ELAPSED=$(($(date +%s) - START_TIME))
echo "Extracted $N_IMAGES images in ${ELAPSED}s"

# ==============================================================================
# STEP 2: Build parquet index
# ==============================================================================

echo "Building parquet index..."
uv run python scripts/build_parquet.py \
    --image-dir "$LOCAL_IMAGE_DIR" \
    --output "$LOCAL_PARQUET"

# ==============================================================================
# STEP 3: Export features (single shard range for this tar)
# ==============================================================================

# One tar ≈ 11k images. At shard_size=4096, that's ~3 shards.
# We export all shards for this tar (start=0, end=large number — script handles overflow).
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
    --end-shard 100

EXIT_CODE=$?
ELAPSED=$(($(date +%s) - START_TIME))

echo "========================================"
echo "Exit code:          $EXIT_CODE"
echo "Elapsed (export):   ${ELAPSED}s"
echo "End time:           $(date -Iseconds)"
echo "========================================"

exit $EXIT_CODE
