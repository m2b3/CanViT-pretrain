#!/bin/bash
#SBATCH --job-name=canvit-in1k-sweep
#SBATCH --account=rrg-skrishna_gpu
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/eval-in1k-sweep-%A_%a.out
#SBATCH --error=logs/eval-in1k-sweep-%A_%a.err
#SBATCH --array=0-5

# IN1k evaluation sweep: 3 policies × 2 rollouts = 6 jobs
# Usage: sbatch slurm/eval_in1k_sweep.sbatch

set -eux

cd ~/scratch/CanViT-train
source slurm/env.sh
mkdir -p logs outputs

# Decode array task ID into policy config
# 0-1: coarse_to_fine
# 2-3: random with start_full=True (full-then-random)
# 4-5: random with start_full=False (IID)

TASK=$SLURM_ARRAY_TASK_ID
ROLLOUT=$((TASK % 2))

if [ $TASK -lt 2 ]; then
    POLICY="coarse_to_fine"
    NAME="c2f"
    EXTRA_ARGS=""
elif [ $TASK -lt 4 ]; then
    POLICY="random"
    NAME="fullrand"
    EXTRA_ARGS="--random-start-full"
else
    POLICY="random"
    NAME="iid"
    EXTRA_ARGS="--no-random-start-full"
fi

OUTPUT="outputs/in1k_${NAME}_r${ROLLOUT}.pt"

echo "Task $TASK: policy=$POLICY, rollout=$ROLLOUT, output=$OUTPUT"
exec uv run python -m canvit_eval.in1k \
    --policy "$POLICY" \
    --output "$OUTPUT" \
    $EXTRA_ARGS
