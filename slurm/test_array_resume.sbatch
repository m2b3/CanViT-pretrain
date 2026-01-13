#!/bin/bash
#SBATCH --job-name=array-resume-test
#SBATCH --array=1-5%1
#SBATCH --time=0:01:00
#SBATCH --signal=USR1@30
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=array_test_%A_%a.out

# Test: job array + signal-based checkpointing
#
# Config:
#   - 60 second jobs
#   - Signal 30s before timeout (so ~30s of work per job)
#   - Python does 1 step/second, target 90 steps
#   - Expected: ~3 jobs to complete
#
# What we're testing:
#   1. Array %1 runs jobs sequentially
#   2. Signal delivered to Python via srun
#   3. Checkpoint saved on signal
#   4. Next job resumes from checkpoint

echo "=== Job $SLURM_ARRAY_JOB_ID / Task $SLURM_ARRAY_TASK_ID ==="
echo "Started: $(date)"
echo "Working dir: $(pwd)"

# Early exit if done
if [ -f "array_test_DONE" ]; then
    echo "Training complete (DONE file exists)"
    exit 0
fi

# Run via srun so Python receives signals
srun python3 slurm/test_array_resume.py
EXIT_CODE=$?

echo "Python exit code: $EXIT_CODE"
echo "Finished: $(date)"
