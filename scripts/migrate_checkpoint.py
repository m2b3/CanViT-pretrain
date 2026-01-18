"""Migrate old checkpoint format to current CanViT architecture.

See REFERENCE_CKPT_MIGRATION.md for details on what changed.
"""

import argparse
import logging
from pathlib import Path

import torch
from torch import Tensor

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

N_CANVAS_REGISTERS = 16
LAYER_SCALE_INIT = 1.0


def migrate_ewa(old_state: dict[str, Tensor], old_prefix: str, new_prefix: str) -> dict[str, Tensor]:
    """Migrate ElementwiseAffine → SplitElementwiseAffine.

    OLD: {prefix}.scale, {prefix}.bias
    NEW: {prefix}.delta_rest_scale, delta_rest_bias, delta_prefix_scale, delta_prefix_bias
         + init_* buffers (handled by model init)
    """
    result = {}

    old_scale = old_state[f"{old_prefix}.scale"]
    old_bias = old_state[f"{old_prefix}.bias"]

    # Rest (spatial tokens): delta = old - init (init is 1 for scale, 0 for bias)
    result[f"{new_prefix}.delta_rest_scale"] = old_scale - 1.0
    result[f"{new_prefix}.delta_rest_bias"] = old_bias

    # Prefix (registers): broadcast old shared params to per-register
    result[f"{new_prefix}.delta_prefix_scale"] = (old_scale - 1.0).unsqueeze(0).expand(N_CANVAS_REGISTERS, -1).clone()
    result[f"{new_prefix}.delta_prefix_bias"] = old_bias.unsqueeze(0).expand(N_CANVAS_REGISTERS, -1).clone()

    return result


def migrate_layer_scale(old_state: dict[str, Tensor], old_prefix: str, new_prefix: str) -> dict[str, Tensor]:
    """Migrate LayerScale reparameterization.

    OLD: {prefix}.scale (learned, init to LAYER_SCALE_INIT)
    NEW: {prefix}.delta_scale (param) + init_scale (buffer)
    """
    old_scale = old_state[f"{old_prefix}.scale"]
    return {f"{new_prefix}.delta_scale": old_scale - LAYER_SCALE_INIT}


def migrate_checkpoint(old_ckpt: dict) -> dict:
    """Migrate old checkpoint to new format."""
    old_state = old_ckpt["state_dict"]
    new_state = {}
    migrated_keys = set()

    # === 1. Simple renames ===
    renames = {
        "register_init": "canvas_register_init",
        "spatial_init": "canvas_spatial_init",
        "cls_init": "recurrent_cls_init",
    }
    for old_key, new_key in renames.items():
        if old_key in old_state:
            new_state[new_key] = old_state[old_key]
            migrated_keys.add(old_key)
            log.info(f"RENAME: {old_key} → {new_key}")

    # === 2. Direct copies (compatible keys) ===
    compatible_prefixes = [
        "backbone.",
        "canvas_rope_periods",
        "vpe_encoder.",
        "glimpse_cls_head.",
        "glimpse_patches_head.",
        "scene_patches_head.",
        # Attention components that don't need migration
        "read_attn.0.attn.q_transform.",
        "read_attn.0.attn.out_transform.",
        "read_attn.0.attn.pre_q_ln.",
        "read_attn.0.attn.pre_kv_ln.",
        "read_attn.1.attn.q_transform.",
        "read_attn.1.attn.out_transform.",
        "read_attn.1.attn.pre_q_ln.",
        "read_attn.1.attn.pre_kv_ln.",
        "read_attn.2.attn.q_transform.",
        "read_attn.2.attn.out_transform.",
        "read_attn.2.attn.pre_q_ln.",
        "read_attn.2.attn.pre_kv_ln.",
        "write_attn.0.attn.k_transform.",
        "write_attn.0.attn.v_transform.",
        "write_attn.0.attn.pre_q_ln.",
        "write_attn.0.attn.pre_kv_ln.",
        "write_attn.1.attn.k_transform.",
        "write_attn.1.attn.v_transform.",
        "write_attn.1.attn.pre_q_ln.",
        "write_attn.1.attn.pre_kv_ln.",
        "write_attn.2.attn.k_transform.",
        "write_attn.2.attn.v_transform.",
        "write_attn.2.attn.pre_q_ln.",
        "write_attn.2.attn.pre_kv_ln.",
    ]

    for key, value in old_state.items():
        if key in migrated_keys:
            continue
        for prefix in compatible_prefixes:
            if key.startswith(prefix):
                new_state[key] = value
                migrated_keys.add(key)
                break

    # === 3. EWA migrations (read attention k/v transforms) ===
    for i in range(3):  # 3 read attention layers
        for transform in ["k_transform", "v_transform"]:
            old_prefix = f"read_attn.{i}.attn.{transform}"
            if f"{old_prefix}.scale" in old_state:
                ewa_state = migrate_ewa(old_state, old_prefix, old_prefix)
                new_state.update(ewa_state)
                migrated_keys.add(f"{old_prefix}.scale")
                migrated_keys.add(f"{old_prefix}.bias")
                log.info(f"MIGRATE EWA: {old_prefix}")

    # === 4. EWA migrations (write attention q/out transforms) ===
    for i in range(3):  # 3 write attention layers
        for transform in ["q_transform", "out_transform"]:
            old_prefix = f"write_attn.{i}.attn.{transform}"
            if f"{old_prefix}.scale" in old_state:
                ewa_state = migrate_ewa(old_state, old_prefix, old_prefix)
                new_state.update(ewa_state)
                migrated_keys.add(f"{old_prefix}.scale")
                migrated_keys.add(f"{old_prefix}.bias")
                log.info(f"MIGRATE EWA: {old_prefix}")

    # === 5. LayerScale migrations ===
    for i in range(3):
        old_prefix = f"read_attn.{i}.scale"
        if f"{old_prefix}.scale" in old_state:
            ls_state = migrate_layer_scale(old_state, old_prefix, old_prefix)
            new_state.update(ls_state)
            migrated_keys.add(f"{old_prefix}.scale")
            log.info(f"MIGRATE LayerScale: {old_prefix}")

    # === 6. Keys to drop ===
    drop_keys = [
        "local_cls_init",
        "scene_global_cls_ln.weight",
        "scene_global_cls_ln.bias",
        "scene_canvas_ln.weight",
        "scene_canvas_ln.bias",
        "scene_cls_proj.weight",
        "scene_cls_proj.bias",
    ]
    for key in drop_keys:
        if key in old_state:
            migrated_keys.add(key)
            log.info(f"DROP: {key}")

    # === Check for unmigrated keys ===
    unmigrated = set(old_state.keys()) - migrated_keys
    if unmigrated:
        log.warning(f"UNMIGRATED KEYS: {sorted(unmigrated)}")

    # === Build new checkpoint ===
    new_ckpt = {
        "state_dict": new_state,
        "model_config": old_ckpt["model_config"],
        "step": old_ckpt["step"],
        "backbone": old_ckpt["backbone"],
        "teacher_dim": old_ckpt["teacher_dim"],
        "comet_id": old_ckpt["comet_id"],
        "git_commit": old_ckpt["git_commit"],
        "timestamp": old_ckpt["timestamp"],
        "train_loss": old_ckpt.get("train_loss"),
        # Normalizer states
        "scene_norm_state": old_ckpt.get("scene_norm_state"),
        "cls_norm_state": old_ckpt.get("cls_norm_state"),
        "glimpse_patches_norm_state": old_ckpt.get("glimpse_patches_norm_state"),
        "glimpse_cls_norm_state": old_ckpt.get("glimpse_cls_norm_state"),
        # Mark as migrated
        "migrated_from_commit": old_ckpt["git_commit"],
        "migration_notes": "scene_cls_head reinitialized (architecture changed)",
    }

    return new_ckpt


def main():
    parser = argparse.ArgumentParser(description="Migrate old checkpoint to new format")
    parser.add_argument("input", type=Path, help="Input checkpoint path")
    parser.add_argument("output", type=Path, help="Output checkpoint path")
    args = parser.parse_args()

    log.info(f"Loading {args.input}")
    old_ckpt = torch.load(args.input, map_location="cpu", weights_only=False)

    log.info(f"Old checkpoint: step={old_ckpt['step']}, commit={old_ckpt['git_commit'][:8]}")

    new_ckpt = migrate_checkpoint(old_ckpt)

    log.info(f"Saving to {args.output}")
    torch.save(new_ckpt, args.output)

    # Summary
    old_keys = len(old_ckpt["state_dict"])
    new_keys = len(new_ckpt["state_dict"])
    log.info(f"Done: {old_keys} old keys → {new_keys} new keys")
    log.info("NOTE: scene_cls_head must be reinitialized (not in migrated state_dict)")


if __name__ == "__main__":
    main()
