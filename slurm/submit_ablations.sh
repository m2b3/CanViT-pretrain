#!/bin/bash
# ╔═══════════════════════════════════════════════════════════════════╗
# ║  ABLATION BATCH SUBMISSION — 2026-02-27                         ║
# ║                                                                 ║
# ║  THIS SCRIPT SUBMITS 8 ARRAY JOBS × 41 TASKS = 328 JOBS.       ║
# ║  Each job uses 1× H100 for ~30 min.                            ║
# ║  Total: ~164 GPU-hours.                                         ║
# ║                                                                 ║
# ║  DO NOT RUN THIS LIGHTLY.                                       ║
# ╚═══════════════════════════════════════════════════════════════════╝
#
# Ablation grid (200k steps, 20k warmup, additive baseline w/ VPE):
#
#   1. Baseline           — asymmetric, D_can=1024
#   2. +QKVO, D_can=256   — full projections, FLOP-matched (n_h=2)
#   3. +QKVO, D_can=384   — full projections, slightly above (n_h=3)
#   4. D_can=512           — asymmetric, halved canvas (n_h=4)
#   5. D_can=256           — asymmetric, quarter canvas (n_h=2)
#   6. No F-IID rollout    — R-IID only
#   7. No canvas reads     — write-only canvas
#   8. No VPE              — no viewpoint position encoding

set -euo pipefail

ARRAY="0-40%1"          # 41 jobs × 4992 steps/job ≈ 200k steps
WARMUP="20000"
SBATCH="slurm/train.sbatch"

confirm() {
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  ABOUT TO SUBMIT 8 ARRAY JOBS (328 total SLURM tasks)"
    echo "  ~164 GPU-hours on H100s"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    read -p "Type YES to confirm: " answer
    if [ "$answer" != "YES" ]; then
        echo "Aborted."
        exit 1
    fi
}

submit() {
    local name="$1"; shift
    echo "Submitting: $name"
    sbatch --array="$ARRAY" -J "$name" "$SBATCH" --warmup-steps "$WARMUP" --run-name "$name" "$@"
}

confirm

submit abl-baseline-200k

submit abl-qkvo-dcan256-200k \
    --model.canvas-proj-mode full --model.canvas-num-heads 2

submit abl-qkvo-dcan384-200k \
    --model.canvas-proj-mode full --model.canvas-num-heads 3

submit abl-dcan512-200k \
    --model.canvas-num-heads 4

submit abl-dcan256-200k \
    --model.canvas-num-heads 2

submit abl-no-fiid-200k \
    --n-full-start-branches 0

submit abl-no-reads-200k \
    --model.no-enable-reads

submit abl-no-vpe-200k \
    --model.no-enable-vpe

echo ""
echo "All 8 ablations submitted. Use 'squeue -u \$USER' to monitor."
