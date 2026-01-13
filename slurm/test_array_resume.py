#!/usr/bin/env python3
"""Minimal test: job array + signal-based checkpointing.

Tests:
1. Job array runs jobs sequentially (%1)
2. Signal delivered via srun before timeout
3. Checkpoint saved on signal
4. Next array job resumes from checkpoint

Target: 90 steps at 1s each = 90s total work
Job time: 60s with signal at 30s = ~30 steps per job
Expected: ~3 array jobs to complete

Run: sbatch slurm/test_array_resume.sbatch
"""

import json
import os
import signal
import sys
import time
from pathlib import Path

CKPT = Path("array_test_ckpt.json")
DONE = Path("array_test_DONE")
TARGET = 90
STEP_TIME = 1.0

shutdown = False


def log(msg: str) -> None:
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "?")
    print(f"[task={task_id}] {msg}", flush=True)


def on_signal(signum: int, frame: object) -> None:
    global shutdown
    shutdown = True
    name = "SIGUSR1" if signum == signal.SIGUSR1 else f"SIG{signum}"
    log(f">>> {name} received, will checkpoint <<<")


def load() -> int:
    if CKPT.exists():
        step = json.loads(CKPT.read_text())["step"]
        log(f"Resumed from step {step}")
        return step
    log("Starting fresh (no checkpoint)")
    return 0


def save(step: int) -> None:
    tmp = CKPT.with_suffix(".tmp")
    tmp.write_text(json.dumps({"step": step, "time": time.time()}))
    tmp.rename(CKPT)
    log(f"Checkpoint saved: step {step}")


def main() -> None:
    signal.signal(signal.SIGUSR1, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    if DONE.exists():
        log("DONE file exists, nothing to do")
        return

    step = load()

    while step < TARGET:
        if shutdown:
            save(step)
            log("Exiting cleanly after signal")
            return

        time.sleep(STEP_TIME)
        step += 1

        if step % 5 == 0:
            log(f"Step {step}/{TARGET}")

    save(step)
    DONE.touch()
    log(f"COMPLETE! Reached {TARGET} steps")


if __name__ == "__main__":
    main()
