# Seamless Resume System for Continuous Training

**Date:** 2026-01-13
**Status:** Strategy revised, testing job arrays + signals

---

## The Problem

Training requires millions of steps (weeks/months). On shared SLURM clusters:
- **Long jobs** (24h): Need large contiguous window. Queue for days.
- **Short jobs** (30min): Fit in scheduler gaps. Start fast.

**Goal:** Use many short jobs that checkpoint and resume automatically.

---

## Two Approaches to Job Chaining

### Option A: Self-Resubmission
```bash
# At end of job script:
if work_should_continue; then
    sbatch ${BASH_SOURCE[0]}
fi
```
- Dynamic: runs exactly as long as needed
- Complex: bash logic to check and resubmit
- Risk: infinite loops if continuation test is wrong

### Option B: Job Arrays
```bash
#SBATCH --array=1-1000%1   # 1000 jobs, one at a time
```
- Simple: all jobs queued upfront
- Predictable: fixed number of jobs
- If training finishes early, remaining jobs exit immediately (cheap)
- If under-estimated, submit another array

**Decision: Job Arrays.** Simpler, less can go wrong.

---

## Two Approaches to Checkpointing

### Option A: Periodic Only
- Checkpoint every N steps (e.g., every 10 min of work)
- Job times out at walltime
- Next job resumes from last checkpoint
- **Lost work per job:** up to 1 checkpoint interval

### Option B: Signal-Based
- SLURM sends signal before timeout (`--signal=USR1@60`)
- Python catches signal, checkpoints immediately, exits
- Next job resumes with zero lost work
- Requires `srun` to make Python a "job step" that receives signals

**Math on lost work (periodic only):**
- 30-min jobs, 10-min checkpoint interval
- Each job loses ~10 min when killed
- Over 500 jobs = ~80 hours of wasted GPU time
- That's ~30% overhead!

**Math on lost work (signal-based):**
- Checkpoint 60s before timeout
- Lost work per job: ~0
- Overhead: ~3% (checkpoint time only)

**Decision: Signal-Based.** The efficiency gain justifies the complexity.

---

## Combined Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│  Job Array + Signal-Based Checkpointing                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  #SBATCH --array=1-1000%1                                       │
│  #SBATCH --time=0:30:00                                         │
│  #SBATCH --signal=USR1@60                                       │
│                                                                 │
│  Task 1: [start] ─────── signal ─────► [checkpoint & exit]      │
│          step 0           t=29m         step 8500               │
│                                              │                  │
│  Task 2: [resume] ────── signal ─────► [checkpoint & exit]      │
│          step 8500        t=29m         step 17000              │
│                                              │                  │
│  Task 3: [resume] ───────────────────► [DONE, exit]             │
│          step 17000                     step 20000              │
│                                                                 │
│  Tasks 4-1000: [check DONE file] ────► [exit immediately]       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key mechanics:**
1. Array ensures jobs run sequentially (`%1`)
2. Signal ensures clean checkpoint before timeout
3. DONE file stops remaining jobs from doing unnecessary work
4. Periodic checkpoint is fallback if signal fails

---

## SLURM Signal Delivery

**Critical insight from docs:**
> "By default all job steps will be signaled, but not the batch shell itself."

This means:
- Without `srun`: no "job steps" exist → signal goes nowhere
- With `srun`: Python becomes a job step → receives signal directly

```bash
# WRONG - signal not delivered:
python3 train.py

# CORRECT - Python receives signal:
srun python3 train.py
```

---

## Test Script Design

**Goal:** Validate in ~3 minutes that:
1. Job array runs jobs sequentially
2. Signal is delivered via srun
3. Checkpoint saved on signal
4. Next job resumes from checkpoint

**Config:**
- 60-second jobs
- Signal 30s before timeout
- 1 step/second, target 90 steps
- Expected: ~3 jobs to complete

**Files:**
- `slurm/test_array_resume.py` - mock training with checkpoint
- `slurm/test_array_resume.sbatch` - job array script

**To run:**
```bash
# Clean up any previous test
rm -f array_test_*.json array_test_DONE array_test_*.out

# Submit
sbatch slurm/test_array_resume.sbatch

# Watch
watch -n 5 'cat array_test_*.out 2>/dev/null | tail -50'
```

**Expected output:**
```
=== Job 12345 / Task 1 ===
[task=1] Starting fresh (no checkpoint)
[task=1] Step 5/90
...
[task=1] Step 25/90
[task=1] >>> SIGUSR1 received, will checkpoint <<<
[task=1] Checkpoint saved: step 28
[task=1] Exiting cleanly after signal

=== Job 12345 / Task 2 ===
[task=2] Resumed from step 28
[task=2] Step 30/90
...
[task=2] >>> SIGUSR1 received, will checkpoint <<<
[task=2] Checkpoint saved: step 58

=== Job 12345 / Task 3 ===
[task=3] Resumed from step 58
[task=3] Step 60/90
...
[task=3] COMPLETE! Reached 90 steps

=== Job 12345 / Task 4 ===
Training complete (DONE file exists)
```

---

## Implementation for Real Training

### Changes needed in `loop.py`:

1. **Signal handler for SIGUSR1** (already exists for debug, repurpose)
2. **Check shutdown flag in training loop**
3. **Auto-resume from latest checkpoint in ckpt_dir**

### Changes needed in `config.py`:

1. Remove `auto_resume` flag - just always auto-resume if checkpoint exists
2. Keep `reset_opt_and_sched=False` as default

### Changes needed in `checkpoint/__init__.py`:

1. `find_latest()` - already added
2. Atomic writes - already added

### SLURM script for production:

```bash
#!/bin/bash
#SBATCH --job-name=avp-train
#SBATCH --array=1-2000%1
#SBATCH --time=0:30:00
#SBATCH --signal=USR1@60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/train_%A_%a.out

CKPT_DIR="${SCRATCH}/avp-checkpoints/run1"
mkdir -p "$CKPT_DIR" logs

# Early exit if done
if [ -f "${CKPT_DIR}/DONE" ]; then
    echo "Training complete"
    exit 0
fi

srun python -m avp_vit.train \
    --ckpt-dir "$CKPT_DIR" \
    --n-steps 5000000
```

---

## Fallback: No Signals

If signal handling proves unreliable on this cluster:

1. Remove `--signal` directive
2. Set `ckpt_every` to checkpoint multiple times per job (e.g., every 5 min)
3. Accept ~15-20% overhead from lost work

This is the "simple approach" - works, just less efficient.

---

## Open Questions

1. **Does your cluster support `srun` in batch scripts?** Some clusters restrict srun usage.
2. **What's the minimum job time?** Very short jobs (<1 min) may have timing issues.
3. **Signal timing jitter:** Docs say signals may arrive "up to 60 seconds early" - need to test.

---

## Next Steps

1. **Test the array+signal approach** with `test_array_resume.sbatch`
2. If signals work: implement in real training loop
3. If signals don't work: fall back to periodic-only checkpointing
