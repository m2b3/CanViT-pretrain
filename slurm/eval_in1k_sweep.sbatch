#!/bin/bash
#SBATCH --job-name=canvit-in1k-sweep
#SBATCH --account=rrg-skrishna_gpu
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/eval-in1k-sweep-%A_%a.out
#SBATCH --error=logs/eval-in1k-sweep-%A_%a.err

# IN1k evaluation sweep: N rollouts of a given policy
#
# Usage:
#   sbatch --array=0-1 slurm/eval_in1k_sweep.sbatch c2f
#   sbatch --array=0-1 slurm/eval_in1k_sweep.sbatch fullrand
#   sbatch --array=0-1 slurm/eval_in1k_sweep.sbatch iid

set -eux

NAME="${1:?Usage: sbatch --array=0-N eval_in1k_sweep.sbatch <c2f|fullrand|iid>}"

cd ~/scratch/CanViT-train
source slurm/env.sh

case "$NAME" in
    c2f)
        POLICY="coarse_to_fine"
        EXTRA_ARGS=""
        ;;
    fullrand)
        POLICY="random"
        EXTRA_ARGS="--random-start-full"
        ;;
    iid)
        POLICY="random"
        EXTRA_ARGS="--no-random-start-full"
        ;;
    *)
        echo "Unknown policy name: $NAME (expected c2f, fullrand, iid)"
        exit 1
        ;;
esac

# Build output path: outputs/in1k_eval/{policy}_{timestamp}_{git}_{jobid}_{taskid}.pt
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
GIT_SHORT=$(git rev-parse --short HEAD)
OUTPUT_DIR="outputs/in1k_eval"
OUTPUT="${OUTPUT_DIR}/in1k_${NAME}_${TIMESTAMP}_${GIT_SHORT}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.pt"

mkdir -p logs "$OUTPUT_DIR"

echo "Policy: $NAME ($POLICY)"
echo "Output: $OUTPUT"
exec uv run python -m canvit_eval.in1k \
    --policy "$POLICY" \
    --output "$OUTPUT" \
    $EXTRA_ARGS
