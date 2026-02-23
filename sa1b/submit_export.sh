#!/bin/bash
# Submit export jobs for all tars that don't have a corresponding shard yet.
# Usage: bash sa1b/submit_export.sh [--dry-run]
set -euo pipefail

source slurm/env.sh

OUT_DIR="$SA1B_FEATURES_DIR/sa1b/dinov3_vitb16/1024/shards"

# Find tar indices that need exporting
pending=()
for tar in "$SA1B_TAR_DIR"/sa_*.tar; do
    idx=$(basename "$tar" .tar | sed 's/sa_//')
    shard="$OUT_DIR/sa_${idx}.pt"
    [[ -f "$shard" ]] || pending+=("$((10#$idx))")
done

if [[ ${#pending[@]} -eq 0 ]]; then
    echo "Nothing to do: all tars have shards."
    exit 0
fi

# Build comma-separated array spec
IFS=','; array_spec="${pending[*]}"; unset IFS
echo "Pending: ${#pending[@]} tars"
echo "Array:   $array_spec"

if [[ "${1:-}" == "--dry-run" ]]; then
    echo "(dry run, not submitting)"
    exit 0
fi

sbatch --array="$array_spec" sa1b/export_features.sh
