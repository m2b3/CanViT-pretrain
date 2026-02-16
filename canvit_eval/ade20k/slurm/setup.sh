#!/bin/bash
# ADE20K environment setup. Source this before running any ADE20K job.
# Works for both sbatch scripts and interactive sessions.
#
# Usage:
#   source canvit_eval/ade20k/slurm/setup.sh

source slurm/env.sh

export ADE20K_ROOT="${SLURM_TMPDIR:?SLURM_TMPDIR not set — are you in a SLURM job?}/ADEChallengeData2016"

if [ ! -d "$ADE20K_ROOT" ]; then
    echo "Extracting ADE20K to $ADE20K_ROOT ..."
    unzip -q "${ADE20K_ZIP:?ADE20K_ZIP not set — check slurm/env.sh}" -d "$SLURM_TMPDIR"
    echo "Done."
fi
