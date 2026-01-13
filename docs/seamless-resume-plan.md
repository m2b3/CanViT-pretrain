# Seamless Resume System for Continuous Training

**Date:** 2026-01-13
**Status:** Planning complete, ready for implementation

---

# Part 1: Architecture & Reasoning

This section explains the problem, solution, and design decisions. Readable without code context.

---

## The Core Problem

Training requires millions of optimization steps. On shared clusters, SLURM schedules jobs by finding contiguous time windows.

- **Long jobs** (e.g., 24h): Need 24h contiguous window. Rare. Queue for days.
- **Short jobs** (e.g., 30min): Fit in gaps between other jobs. Abundant. Start fast.

**Insight:** Instead of one long job, use many short jobs that checkpoint and resume automatically.

---

## The Solution Model

```
┌─────────────────────────────────────────────────────────────┐
│  Continuous Training via Short Jobs                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Job 1: [start] ──────────────────────────► [checkpoint]    │
│         step 0                               step 8000      │
│                                                     │       │
│                            ┌────────────────────────┘       │
│                            ▼                                │
│  Job 2: [load checkpoint] ────────────────► [checkpoint]    │
│         step 8000                            step 16000     │
│                                                     │       │
│                            ┌────────────────────────┘       │
│                            ▼                                │
│  Job 3: [load] ───────────────────────────► [checkpoint]    │
│         ...                                  ...            │
│                                                             │
│  Job N: [load] ───────────────────────────► [DONE]          │
│         step 990000                          step 1000000   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

Each job:
1. Finds latest checkpoint (or starts fresh)
2. Loads full state (model, optimizer, step number)
3. Continues training from where it left off
4. Saves checkpoint when walltime approaches (SIGTERM)
5. Resubmits itself (unless training complete)

**Result:** Training progresses whenever cluster has capacity. No babysitting.

---

## Key Design Decisions

### 1. Learning Rate Schedule

**Cosine decay is incompatible with this workflow.**

Cosine decay requires knowing total steps upfront (T_max). But we're training "until good enough" — duration unknown. And even if known, resuming mid-cosine after scheduler state loss creates discontinuities.

**Decision:** Linear warmup → constant LR. Can train indefinitely. Resume anywhere without schedule issues.

### 2. Checkpoint Triggers

Two ways a checkpoint happens:
1. **Natural:** Every N steps (e.g., 10,000)
2. **Forced:** SIGTERM received (walltime approaching)

At ~5 steps/sec, 10k steps = 33 min. For 30-min jobs, natural checkpoints may not occur.

**Decision:** SIGTERM handler is mandatory. It's the safety net that guarantees we save before death.

### 3. Experiment Tracking Continuity

Metrics should show one continuous training curve, not 100 fragmented experiments.

**Decision:** When resuming, continue the same Comet experiment (using saved experiment ID).

### 4. Simplicity Over Flexibility

Could have many flags: "continue Comet?", "reset optimizer?", "reset scheduler?", "reset step counter?"

**Decision:** One mode. When resuming, continue everything. Want different behavior? Use different checkpoint directory. No flag matrix to remember.

### 5. Single GPU

Multi-GPU (DDP) adds: distributed state saving, rank coordination, gradient sync edge cases, debugging complexity.

**Decision:** Single GPU for now. H100 throughput is sufficient. Revisit if bottlenecked.

---

## SLURM Signal Mechanics

When your job approaches walltime, SLURM sends signals:

```
--time=0:30:00 --signal=B:TERM@60

Timeline:
├─ 0:00:00   Job starts
│
├─ 0:29:00   SIGTERM sent (60s before walltime)
│            → Python catches signal
│            → Sets "please checkpoint" flag
│            → Current step completes
│            → Checkpoint saves (~10-30s)
│            → Job exits cleanly
│
├─ 0:30:00   Walltime reached
│            → SIGKILL sent (if still running)
│            → Instant death, no cleanup
```

**The 60-second grace period is critical.** Must be enough time to finish current step + save checkpoint.

---

## Failure Modes & Mitigations

| Failure | What Happens | Mitigation |
|---------|--------------|------------|
| SIGKILL during checkpoint write | File half-written, corrupt | Atomic write: save to `.tmp`, then rename |
| Job crashes immediately, resubmits | Infinite loop of failing jobs | Check step > previous step before resubmit |
| Grace period too short | Checkpoint doesn't complete | Use 60-120s grace; checkpoint is fast |
| Comet API down | Can't create/continue experiment | Catch exception, warn, continue without tracking |
| Two jobs run simultaneously | Race on checkpoint file | Include SLURM job ID in filename |

---

## Clock Time Budget (30-min job)

| Phase | Duration | Notes |
|-------|----------|-------|
| Startup (Python, imports) | ~30s | Fast with cached deps |
| Model loading | ~30s | Teacher + student |
| `torch.compile` | ~60-90s | Only first forward pass |
| **Training** | **~26-27 min** | Actual useful work |
| Checkpoint save | ~10-30s | Model + optimizer state |

**Effective utilization:** ~87-90%. Acceptable.

---

## What This Enables

1. **No queue anxiety:** Short jobs start fast
2. **No babysitting:** Jobs resubmit themselves
3. **No lost work:** SIGTERM saves progress
4. **Clean metrics:** One continuous experiment
5. **Flexible duration:** Train until satisfied, not until T_max
6. **Robust to preemption:** Same mechanism handles both walltime and preemption

---

# Part 2: Implementation Details

This section covers specific code locations, changes needed, and technical uncertainties.

---

## Current Code Issues

### Issue 1: Loop starts at step 0

**Location:** `avp_vit/train/loop.py:344`
```python
pbar = tqdm(range(cfg.n_steps + 1), desc="Training", unit="step")
```

**Problem:** Always iterates 0 to n_steps, regardless of loaded checkpoint step.

**Fix:** `range(start_step, cfg.n_steps + 1)`

---

### Issue 2: New Comet experiment each run

**Location:** `avp_vit/train/loop.py:115`
```python
exp = comet_ml.Experiment(project_name="avp-vit-scene-match", ...)
```

**Problem:** Creates new experiment. Checkpoint has `comet_id` but it's never used.

**Fix:**
```python
if ckpt_data and ckpt_data.get("comet_id"):
    exp = comet_ml.ExistingExperiment(experiment_key=ckpt_data["comet_id"], ...)
else:
    exp = comet_ml.Experiment(...)
```

**Requires:** Loading checkpoint BEFORE this line (currently happens at line ~197).

---

### Issue 3: No SIGTERM handler

**Location:** `avp_vit/train/loop.py:111`
```python
signal.signal(signal.SIGUSR1, _handle_sigusr1)
```

**Problem:** Only SIGUSR1 handled. SIGTERM (SLURM's "you're about to die" signal) is ignored.

**Fix:** Add identical handler for SIGTERM:
```python
signal.signal(signal.SIGTERM, _handle_sigusr1)  # Same handler works
```

---

### Issue 4: No automatic checkpoint discovery

**Location:** `avp_vit/train/config.py:71`
```python
resume_ckpt: Path | None = None
```

**Problem:** Must manually specify path. Can't auto-resume.

**Fix:** Add `auto_resume: bool = False`. When True, search `ckpt_dir` for latest `.pt` file.

---

### Issue 5: Default loses optimizer state

**Location:** `avp_vit/train/config.py:73-74`
```python
reset_opt_and_sched: bool = True
```

**Problem:** Default resets optimizer momentum on resume.

**Fix:** Change default to `False`.

---

### Issue 6: Non-atomic checkpoint writes

**Location:** `avp_vit/checkpoint/__init__.py:114`
```python
torch.save(data, path)
```

**Problem:** If SIGKILL during write, file is corrupt.

**Fix:**
```python
tmp = path.with_suffix(".pt.tmp")
torch.save(data, tmp)
tmp.rename(path)  # Atomic on POSIX
```

---

## Files to Modify

| File | Changes | Risk |
|------|---------|------|
| `avp_vit/train/config.py` | Add `auto_resume`, flip `reset_opt_and_sched` default | Low |
| `avp_vit/train/loop.py` | SIGTERM handler, reorder checkpoint load, ExistingExperiment, fix loop range | Medium - most complex |
| `avp_vit/checkpoint/__init__.py` | Add `find_latest()`, atomic writes | Low |
| New: `slurm/train_continuous.sbatch` | SLURM script with self-resubmit | Low |

---

## Code Flow Change in `loop.py`

**Current order:**
```
109: def train(cfg, trial)
111:   signal.signal(SIGUSR1, ...)
115:   exp = comet_ml.Experiment(...)      ← Creates NEW experiment
128:   exp.log_parameters(...)
133:   teacher = load_teacher(...)
...
174:   optimizer = AdamW(...)
175:   scheduler = warmup_constant_scheduler(...)
...
197:   if cfg.resume_ckpt:                 ← Too late! Comet already created
        ckpt_data = load_checkpoint(...)
...
344:   for step in range(cfg.n_steps + 1): ← Always starts at 0
```

**Required order:**
```
def train(cfg, trial)
  signal handlers (SIGUSR1 + SIGTERM)

  # Checkpoint load FIRST
  ckpt_data = None
  start_step = 0
  if cfg.auto_resume:
      ckpt_path = find_latest(cfg.ckpt_dir)
      if ckpt_path:
          ckpt_data = load_checkpoint(ckpt_path)
          start_step = ckpt_data.get("step") or 0
  elif cfg.resume_ckpt:
      ckpt_data = load_checkpoint(cfg.resume_ckpt)
      start_step = ckpt_data.get("step") or 0

  # Comet init (can now use comet_id from checkpoint)
  if ckpt_data and ckpt_data.get("comet_id"):
      exp = ExistingExperiment(experiment_key=ckpt_data["comet_id"], ...)
  else:
      exp = Experiment(...)

  # Rest of setup...

  for step in range(start_step, cfg.n_steps + 1):  ← Fixed
```

---

## New Functions Needed

### `checkpoint/__init__.py`

```python
def find_latest(ckpt_dir: Path) -> Path | None:
    """Find most recent .pt file in directory by modification time."""
    if not ckpt_dir.exists():
        return None
    candidates = list(ckpt_dir.glob("*.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)
```

---

## Uncertainties & Questions

### 1. Comet ExistingExperiment API

**Uncertainty:** Exact API for `ExistingExperiment`. Does it need `previous_experiment` param? Does it auto-continue metrics?

**Resolution:** Check Comet docs or test empirically.

### 2. Checkpoint file naming with job ID

**Current:** `{comet_experiment_id}.pt`

**Question:** Should we include SLURM job ID to prevent races?

**Tentative answer:** No. One experiment = one checkpoint file. Race conditions shouldn't occur with proper SLURM setup (no duplicate running jobs). If paranoid, use file locking.

### 3. What if checkpoint is from different config?

**Current:** `training_config_history` tracks config changes but doesn't validate.

**Question:** Should we warn/error if resuming with incompatible config (e.g., different model size)?

**Tentative answer:** Model loading already handles shape mismatches gracefully. Add a warning log if config differs significantly. Don't block.

### 4. Data loader state

**Question:** Is random sampler state saved? Does it matter?

**Answer:** Not saved. For IN21k-scale with random sampling, statistically doesn't matter. Each job sees slightly different data distribution. Actually good for regularization.

### 5. EMA tracker state

**Question:** `EMATracker` for smoothed metrics isn't checkpointed. Problem?

**Answer:** Minor. On resume, EMA restarts. Small discontinuity in smoothed metrics for ~100 steps. Not worth checkpoint bloat.

### 6. `torch.compile` on resume

**Question:** Does `torch.compile` re-trigger on each job?

**Answer:** Yes. ~60-90s overhead per job. Unavoidable without persistent compilation cache (complex). Accept it.

---

## Testing Plan

### Phase 1: CPU Mock (no GPU queue wait)

```python
# test_slurm_resume.py
# - Fake "training" that increments a counter
# - Handles SIGTERM
# - Saves/loads state to file
# - Prints progress

# Submit as 1-min CPU job with --signal=B:TERM@10
# Verify: SIGTERM fires, state saves, next job continues
```

### Phase 2: Short GPU Test

- 10-min job, 1000 steps
- Verify checkpoint save/load with real model
- Verify Comet continuation
- Verify optimizer state preserved (check momentum buffers)

### Phase 3: Production

- 30-min jobs
- Full training run
- Monitor for issues

---

## SLURM Script Template

```bash
#!/bin/bash
#SBATCH --job-name=avp-train
#SBATCH --time=0:30:00
#SBATCH --signal=B:TERM@60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/train_%j.out

set -euo pipefail

CKPT_DIR="${SCRATCH}/avp-checkpoints"
TARGET_STEPS=5000000
DONE_FILE="${CKPT_DIR}/DONE"

mkdir -p "$CKPT_DIR"
mkdir -p logs

echo "Job $SLURM_JOB_ID starting at $(date)"

# Run training
python -m avp_vit.train \
    --auto-resume \
    --ckpt-dir "$CKPT_DIR" \
    --n-steps "$TARGET_STEPS"

EXIT_CODE=$?

# Check completion
if [ -f "$DONE_FILE" ]; then
    echo "Training complete!"
    exit 0
fi

# Check if we made progress (prevent infinite fail loop)
# This requires the Python code to write current step somewhere
# Or we just check exit code

if [ $EXIT_CODE -eq 0 ]; then
    echo "Resubmitting..."
    sbatch "$0"
else
    echo "Job failed with exit code $EXIT_CODE, not resubmitting"
    exit $EXIT_CODE
fi
```

---

## Implementation Order

1. **`checkpoint/__init__.py`**: `find_latest()`, atomic `save()`
2. **`config.py`**: Add `auto_resume`, change `reset_opt_and_sched` default
3. **`loop.py`**: Reorder (checkpoint before Comet), SIGTERM handler, ExistingExperiment, loop range
4. **CPU test script**: Validate SLURM mechanics without GPU
5. **GPU test**: Short real training test
6. **SLURM script**: Production self-resubmitting script

---

## Rollback Plan

All changes are additive or default flips. If something breaks:
- `auto_resume=False` + explicit `--resume-ckpt` = old behavior
- `reset_opt_and_sched=True` = old behavior
- SIGTERM handler is purely additive (no downside)

Low risk. Easy rollback.
