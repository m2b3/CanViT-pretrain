#!/bin/bash
# D_can=256 (n_h=2, d_h=128), asymmetric. Quarter canvas width, no QKVO.
set -euo pipefail
source "$(dirname "$0")/common.sh"
submit abl-dcan256-200k --model.canvas-num-heads 2
