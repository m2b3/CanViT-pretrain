#!/bin/bash
#SBATCH --job-name=eval-ade20k-policy
#SBATCH --account=rrg-skrishna_gpu
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/eval-ade20k-policy-%A_%a.out
#SBATCH --error=logs/eval-ade20k-policy-%A_%a.err

# ADE20K probe evaluation: run a policy with array jobs for variance
#
# Usage:
#   sbatch --array=0-4 canvit_eval/ade20k/slurm/eval_ade20k_policy.sbatch c2f /path/to/probe.pt
#   sbatch --array=0-4 canvit_eval/ade20k/slurm/eval_ade20k_policy.sbatch fullrand /path/to/probe.pt
#   sbatch --array=0-4 canvit_eval/ade20k/slurm/eval_ade20k_policy.sbatch iid /path/to/probe.pt

set -eux

NAME="${1:?Usage: sbatch --array=0-N eval_ade20k_policy.sbatch <c2f|fullrand|iid> <probe_ckpt>}"
PROBE_CKPT="${2:?Missing probe checkpoint path}"

source canvit_eval/ade20k/slurm/setup.sh

case "$NAME" in
    c2f)
        POLICY="coarse_to_fine"
        EXTRA_ARGS=""
        ;;
    fullrand)
        POLICY="random"
        EXTRA_ARGS="--start-full"
        ;;
    iid)
        POLICY="random"
        EXTRA_ARGS="--no-start-full"
        ;;
    *)
        echo "Unknown policy name: $NAME (expected c2f, fullrand, iid)"
        exit 1
        ;;
esac

# Output path
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
GIT_SHORT=$(git rev-parse --short HEAD)
OUTPUT_DIR="outputs/ade20k_eval"
OUTPUT="${OUTPUT_DIR}/ade20k_${NAME}_${TIMESTAMP}_${GIT_SHORT}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.pt"

mkdir -p logs "$OUTPUT_DIR"

echo "Policy: $NAME ($POLICY)"
echo "Probe: $PROBE_CKPT"
echo "Output: $OUTPUT"
exec uv run python -m canvit_eval.ade20k evaluate \
    --probe-ckpt "$PROBE_CKPT" \
    --policy "$POLICY" \
    --output "$OUTPUT" \
    $EXTRA_ARGS
