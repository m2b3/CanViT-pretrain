"""Shared configuration defaults for gaussian blob training scripts."""

from pathlib import Path

from avp_vit import AVPConfig

# Paths
DEFAULT_TEACHER_CKPT = Path("dinov3_vits16_pretrain_lvd1689m-08c60483.pth")

# AVP defaults (shared between scripts)
DEFAULT_AVP_CONFIG = AVPConfig(
    scene_grid_size=16,  # 16*14=224px scene
    glimpse_grid_size=4,  # 4*14=56px glimpse, min_scale=0.25
    gate_init=1e-4,
    use_output_proj=True,
    use_scene_registers=True,
    gradient_checkpointing=False,
)

# Training defaults
DEFAULT_N_STEPS = 10000
DEFAULT_BATCH_SIZE = 64
DEFAULT_REF_LR = 1e-5  # per-sample LR; peak_lr = ref_lr * batch_size
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_GRAD_CLIP = 1.0

# Task defaults
DEFAULT_N_BLOBS = 2
DEFAULT_BLOB_MARGIN = 0.3
DEFAULT_BLOB_SIGMA_MIN = 0.08
DEFAULT_BLOB_SIGMA_MAX = 0.12

# Logging defaults
DEFAULT_LOG_EVERY = 20
DEFAULT_VAL_EVERY = 100
