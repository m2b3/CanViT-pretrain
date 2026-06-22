#!/bin/bash
#SBATCH --job-name=ww-sweep
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/ww-%A_%a.out
#SBATCH --error=logs/ww-%A_%a.err
#SBATCH --mail-type=NONE

# CPU array: WeightWatcher per checkpoint. Array index selects a line from a
# manifest of checkpoint paths.
#   sbatch --account=def-skrishna --array=0-N%K slurm/ww_sweep.sbatch <manifest> <out_dir>
# --time is intentionally small (job is ~1-2 min) so it backfills fast; size it
# from one timed job first.

set -eu
source slurm/env.sh
mkdir -p logs

MANIFEST="$1"
OUT="$2"
CKPT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
echo "[$(date '+%H:%M:%S')] task ${SLURM_ARRAY_TASK_ID} -> ${CKPT}"

exec uv run python scripts/ww_sweep.py --ckpt "$CKPT" --out-dir "$OUT"
