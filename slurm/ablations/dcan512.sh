#!/bin/bash
# D_can=512 (n_h=4, d_h=128), asymmetric. Halved canvas width, no QKVO.
set -euo pipefail
source "$(dirname "$0")/common.sh"
submit abl-dcan512-200k --model.canvas-num-heads 4
