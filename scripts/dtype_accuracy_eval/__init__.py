"""Measure IN1k val accuracy at different CLS feature storage dtypes.

Single teacher inference pass, then compare probe accuracy when CLS
features are cast to different dtypes (simulating storage precision loss).

Dtypes tested: fp32, fp16 (our shard storage), bf16, fp8.

Usage (on cluster, after `source slurm/env.sh`):
    uv run -m scripts.dtype_accuracy_eval \
        --val-dir $IN1K_VAL_DIR \
        --teacher-ckpt $DINOV3_VITB16_CKPT
"""
