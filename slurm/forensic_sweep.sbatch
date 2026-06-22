#!/bin/bash
#SBATCH --job-name=forensic-sweep
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/forensic-%A_%a.out
#SBATCH --error=logs/forensic-%A_%a.err
#SBATCH --mail-type=NONE

# CPU array: run <pyscript> --ckpt <line> --out-dir <out> per checkpoint.
#   sbatch --array=0-N%K slurm/forensic_sweep.sbatch <pyscript> <manifest> <out_dir>
# Supersedes ww_sweep.sbatch (reproduce WW via: forensic_sweep.sbatch scripts/ww_sweep.py ...).
set -eu
source slurm/env.sh
mkdir -p logs
read -ra SCRIPT <<< "$1"   # "$1" may be "script.py --flag" -> split into argv
MANIFEST="$2"; OUT="$3"
CKPT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
echo "[$(date '+%H:%M:%S')] task ${SLURM_ARRAY_TASK_ID} -> ${CKPT}"
exec uv run python "${SCRIPT[@]}" --ckpt "$CKPT" --out-dir "$OUT"
