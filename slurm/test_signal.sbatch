#!/bin/bash
#SBATCH --job-name=signal-test
#SBATCH --time=0:01:00
#SBATCH --signal=B:TERM@10
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=signal_test_%j.out

# 1-minute job, SIGTERM 10 seconds before walltime
# At 0.1s/step, we do ~10 steps/sec = ~500 steps in 50s before SIGTERM
# Target is 1000 steps, so we'll need 2 jobs to complete

set -euo pipefail

echo "=== Job $SLURM_JOB_ID starting at $(date) ==="
echo "Working directory: $(pwd)"

# Run the test
python3 slurm/test_signal.py

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
