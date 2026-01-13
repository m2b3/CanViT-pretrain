#!/usr/bin/env python3
"""Test SLURM signal handling. No GPU, no deps beyond stdlib.

Simulates training: increments counter, handles SIGTERM, saves/loads state.
Submit with: sbatch test_signal.sbatch
"""

import json
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

STATE_FILE = Path("signal_test_state.json")
LOG_FILE = Path("signal_test_log.txt")

# Global state
current_step = 0
checkpoint_requested = False


def log(msg: str) -> None:
    """Append to log file with timestamp."""
    line = f"[{datetime.now().isoformat()}] {msg}"
    print(line, flush=True)
    with LOG_FILE.open("a") as f:
        f.write(line + "\n")


def handle_sigterm(signum: int, frame: object) -> None:
    """Handle SIGTERM from SLURM."""
    global checkpoint_requested
    checkpoint_requested = True
    log(f"SIGTERM received! (signal {signum}) Will checkpoint after current step.")


def save_state() -> None:
    """Save current step to file."""
    tmp = STATE_FILE.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump({"step": current_step, "timestamp": datetime.now().isoformat()}, f)
    tmp.rename(STATE_FILE)  # atomic
    log(f"Checkpoint saved: step={current_step}")


def load_state() -> int:
    """Load step from file, or return 0 if no state."""
    if not STATE_FILE.exists():
        return 0
    with STATE_FILE.open() as f:
        data = json.load(f)
    return data.get("step", 0)


def main() -> None:
    global current_step, checkpoint_requested

    # Register signal handler
    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGUSR1, handle_sigterm)  # For manual testing
    log("Signal handlers registered (SIGTERM, SIGUSR1)")

    # Load state
    current_step = load_state()
    log(f"Starting from step {current_step}")

    target_step = 1000  # Arbitrary target

    # "Training" loop
    while current_step < target_step:
        # Simulate one training step (~0.1s)
        time.sleep(0.1)
        current_step += 1

        if current_step % 50 == 0:
            log(f"Step {current_step}/{target_step}")

        # Check for checkpoint request
        if checkpoint_requested:
            log("Checkpoint requested, saving and exiting...")
            save_state()
            log("Exiting cleanly.")
            sys.exit(0)

    # Training complete
    log(f"Training complete! Reached step {current_step}")
    save_state()

    # Write DONE marker
    Path("signal_test_DONE").touch()
    log("DONE marker written.")


if __name__ == "__main__":
    main()
