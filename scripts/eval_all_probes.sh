#!/bin/bash
# Evaluate ALL ADE20K probes (DINOv3 + CanViT) from HuggingFace Hub.
# Produces fresh mIoU results for the beating-teacher table and DINOv3 curves.
#
# Usage:
#   ADE20K_ROOT=/datasets/ADE20k/ADEChallengeData2016 bash scripts/eval_all_probes.sh
#
# Outputs to: results_probe_eval/*.pt (one per probe)

set -euo pipefail

: "${ADE20K_ROOT:?Set ADE20K_ROOT}"
OUT_DIR="${1:-results_probe_eval}"
mkdir -p "$OUT_DIR"

# DINOv3 ViT-B probes (7 resolutions)
for res in 128 144 160 192 256 384 512; do
  out="$OUT_DIR/dv3b_${res}px.pt"
  if [ -f "$out" ]; then echo "SKIP $out"; continue; fi
  echo "$(date +%H:%M:%S) DINOv3 ViT-B ${res}px"
  uv run python -m canvit_eval.ade20k eval-dinov3-probe \
    --probe-repo "canvit/probe-ade20k-40k-dv3b-${res}px" \
    --scene-size 512 --output "$out" --batch-size 32 \
    2>&1 | tee "$OUT_DIR/dv3b_${res}px.log"
done

# DINOv3 ViT-S probes (7 resolutions)
for res in 128 144 160 192 256 384 512; do
  out="$OUT_DIR/dv3s_${res}px.pt"
  if [ -f "$out" ]; then echo "SKIP $out"; continue; fi
  echo "$(date +%H:%M:%S) DINOv3 ViT-S ${res}px"
  uv run python -m canvit_eval.ade20k eval-dinov3-probe \
    --probe-repo "canvit/probe-ade20k-40k-dv3s-${res}px" \
    --scene-size 512 --output "$out" --batch-size 32 \
    2>&1 | tee "$OUT_DIR/dv3s_${res}px.log"
done

# CanViT canvas probes (t=0 only = 1 timestep, C2F policy)
# NOTE: model_repo is the TRAINING checkpoint, not the eval resolution.
# The IN21K model is always s512px (trained at 512); it's EVALUATED at different scene sizes.
CANVIT_IN21K="canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02"
# SA1B deprioritized for v0 — not included in eval.

for spec in "s512-c8-in21k:512:8:$CANVIT_IN21K" "s512-c16-in21k:512:16:$CANVIT_IN21K" "s512-c32-in21k:512:32:$CANVIT_IN21K" "s1024-c64-in21k:1024:64:$CANVIT_IN21K"; do
  IFS=: read -r slug scene grid model_repo <<< "$spec"
  out="$OUT_DIR/canvit_${slug}.pt"
  if [ -f "$out" ]; then echo "SKIP $out"; continue; fi
  echo "$(date +%H:%M:%S) CanViT $slug (scene=${scene}, grid=${grid})"
  uv run python -m canvit_eval.ade20k evaluate \
    --probe-repo "canvit/probe-ade20k-40k-${slug}" \
    --model-repo "$model_repo" \
    --policy coarse_to_fine --n-timesteps 1 \
    --scene-size "$scene" --canvas-grid "$grid" \
    --output "$out" --batch-size 32 \
    2>&1 | tee "$OUT_DIR/canvit_${slug}.log"
done

echo "$(date +%H:%M:%S) ALL DONE"
echo "Results in $OUT_DIR/"
ls -lh "$OUT_DIR/"*.pt
