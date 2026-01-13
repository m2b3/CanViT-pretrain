#!/bin/bash
#SBATCH --job-name=signal-test
#SBATCH --time=0:03:00
#SBATCH --signal=USR1@60
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=signal_test_%j.out

# 3-minute job, SIGUSR1 60 seconds before walltime
#
# SLURM signal delivery quirk:
# - Default: signals go to "job steps" (srun), NOT batch shell
# - B: prefix sends to batch shell only
# - Without srun, there are no job steps to signal!
#
# Solution: use srun so Python becomes a job step and receives signal directly

set -uo pipefail

echo "=== Job $SLURM_JOB_ID starting at $(date) ==="
echo "Working directory: $(pwd)"

# srun makes Python a "job step" that receives SLURM signals
srun python3 slurm/test_signal.py
EXIT_CODE=$?
echo "Python exited with code $EXIT_CODE"

# Check if done
if [ -f "signal_test_DONE" ]; then
    echo "=== Training complete! ==="
    exit 0
fi

# Resubmit if exited cleanly
if [ $EXIT_CODE -eq 0 ]; then
    echo "=== Resubmitting... ==="
    sbatch slurm/test_signal.sbatch
else
    echo "=== Job failed, not resubmitting ==="
    exit $EXIT_CODE
fi
