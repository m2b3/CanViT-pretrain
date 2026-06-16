#!/bin/bash
#SBATCH --job-name=canvit-postmortem-cpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/postmortem-%j.out
#SBATCH --error=logs/postmortem-%j.err
#SBATCH --mail-type=NONE

# CPU-only forensic helper: runs `uv run python <args>` inside the project env.
# Use for read-only postmortem scans (checkpoint health, data shard scans) that
# are too heavy for the login node. GPU work uses train.sbatch / a GPU job.
#
#   sbatch --account=def-skrishna slurm/postmortem_cpu.sbatch \
#       scripts/postmortem_ckpt_health.py <run_dir> --last 8

set -eu

log() { echo "[$(date '+%H:%M:%S')] $*"; }
log "=== CanViT Postmortem (CPU) ==="
log "Job: ${SLURM_JOB_ID:-none} on $(hostname)"
log "Args: $*"

source slurm/env.sh
mkdir -p logs

log "Running: uv run python $*"
exec uv run python "$@"
