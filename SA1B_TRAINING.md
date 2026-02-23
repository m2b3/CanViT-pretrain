# SA-1B Continual Pretraining

**Goal**: Continually pretrain CanViT-B flagship checkpoint (2M steps on IN21k @ 512px) at **1024px on SA-1B**.
First run: get loss on Comet, verify the model works at higher resolution.

**Branch**: `sa1b` (worktree: `~/code/CanViT-train-SA1B`)

---

## Architecture Decisions

### HF Seed Checkpoint
- **Source**: `canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02` on HF Hub
- **Format**: `config.json` + `model.safetensors` (239 keys) — NOT CheckpointData `.pt`
- **Config**: `additive` update mode, `enable_vpe=True`, `gate_bias_init=None`, `teacher_dim=768`
- **Solution**: `hf_seed_ckpt` config option. Downloads from HF, extracts state_dict + config, overrides `cfg.model`, proceeds with seed mode (step=0, fresh optimizer).
- **Grid size change**: 32 → 64. Only 6 standardizer keys mismatch (3 missing for "64", 3 unexpected for "32"). All 233 core weights load perfectly. Handled by existing `strict=False` + regex filter.
- **IMPORTANT**: `cfg.model` defaults are `convex` + `gate_bias_init=-2.0`. The HF model is `additive` + `gate_bias_init=None`. Using defaults crashes. The `hf_seed_ckpt` path MUST override `cfg.model` with the HF config.

### Storage Model
| What | Where | Persistence |
|---|---|---|
| SA-1B tars (~10.5 GB each) | `~/projects/def-akrish/datasets/sa1b/tars/` (NFS) | Permanent |
| SA-1B feature shards (~65 GB each) | `$SA1B_FEATURES_DIR/sa1b/dinov3_vitb16/1024/shards/` (NFS) | Permanent |
| Extracted JPEGs (~10.3 GB/tar) | `$SLURM_TMPDIR/sa1b_images/` | Per-job |
| Checkpoints | `$CHECKPOINTS_DIR/{run_name}/` (NFS) | Permanent |

### Shard Loading Strategy
- Feature `.pt` shards on NFS, read with `mmap=True` (cheap, no copy needed).
- Images extracted from tars to SLURM_TMPDIR before training starts.
- `--feature-base-dir "$SA1B_FEATURES_DIR/sa1b"` → shards path auto-constructed: `{base}/dinov3_vitb16/1024/shards/`.
- `--feature-image-root "$SLURM_TMPDIR/sa1b_images"` → extracted JPEGs.
- Shard loader iterates all shards on NFS, starts at `start_shard`, processes `steps_per_job` steps.
- Only images from shards within [start_shard, start_shard + shards_per_job] are accessed.
- **No symlinks needed.** Just extract the right tars.

### Image Resolution
- **Teacher export**: 1024px input → features at 1024px (baked at export time).
- **Training images**: loaded at `scene_resolution=1024px`, same transform as export (`val_transform`).
- **Student glimpses**: 128px² crops from the loaded scene image. Higher scene resolution = more fine-grained info when student looks "up close".
- The dataloader image resolution affects glimpse pixel richness and CPU/GPU memory, NOT alignment with the teacher features (those are already fixed from export).
- No assertion currently validates `scene_resolution == G * patch_size`. For SA-1B: 1024 == 64 × 16 = 1024.

### Tar Structure
- SA-1B tars from Meta have `./` prefix: entries like `./sa_226692.jpg` (flat, no subdirectory).
- **train.sh** must use `--wildcards` flag with `'*.jpg'` — without it, GNU tar's default glob doesn't match across `/` and extracts nothing. Also `--strip-components=1` to strip `./`.
- **export_features.py** lists members with `tar tf` then extracts by exact path — unaffected by glob issues.

### Job Array Design
- `%1` concurrency (1 job at a time, like IN21k).
- `steps_per_job` = multiple of `batches_per_shard`. batches_per_shard = 11186 // 64 = **174**.
- 174 × 28 = **4872 steps/job = 28 shards = 28 tars** (~288 GB extracted).
- A helper script (`sa1b/plan_job.py`) reads the checkpoint to determine start_step, outputs tar indices.
- Robust to failures: if task N fails mid-job, restart reads same checkpoint, re-extracts same tars.

### Key Numbers
| Metric | Value | Source |
|---|---|---|
| Images per tar | ~11,186 | tar tvf on sa_000020 |
| JPEG size per tar | ~10.3 GB | tar tvf estimate |
| Feature shard size | ~65-70 GB | sa_000020.pt on def-areynaud |
| batch_size | 64 | config default |
| batches_per_shard | 174 | 11186 // 64 |
| shards_per_job | 7 | Reduced from 28 (1024px is ~4x more expensive/step) |
| steps_per_job | 1218 | 174 × 7 |
| SLURM_TMPDIR budget | ~375 GB | 3 TB / 8 GPUs |
| Extracted size/job | ~72 GB | 7 × 10.3 |
| Canvas grid size | 64 | 1024 / 16 |
| Scene resolution | 1024px | Target for SA-1B |
| VRAM estimate | <80 GB | 18.4 GB @ 512/32 × ~4x, plus constant factors |

---

## What's DONE

### Training Pipeline (commit `956f501`)
- `sa1b/train.sh` — SLURM array job. Extracts tars via `plan_job.py`, then trains.
- `sa1b/plan_job.py` — reads checkpoint → computes which shards/tars needed → outputs tar indices.
- `hf_seed_ckpt` config option in `loop.py` — downloads from HF, overrides cfg.model, seed mode.
- `strict=False` + regex filter for grid-size-change standardizer mismatches.
- Standardizers travel with model state_dict via `model.standardizers(G)`.

### Export Pipeline
- `sa1b/export_features.py` — 1 tar → 1 shard. Atomic save. Idempotent.
- `sa1b/export_features.sh` — SLURM array job. `gpu:h100:1`, 32G RAM, 16 CPUs, 10 min.
- `sa1b/submit_export.sh` — Auto-submits missing shards. `--dry-run` supported. **No lock on concurrent runs.**
- First shard exported: `sa_000020.pt` (70,394 MB, 11186 images).

### Download
- `sa1b/download.py` running on Nibi (tmux `sa1b-dl`). ~73 MB/s.
- Non-contiguous tar indices. submit_export.sh handles this.

### Env Setup
- `.envrc.nibi` committed. `SA1B_TAR_DIR`, `SA1B_FEATURES_DIR`, `CANVIT_FLAGSHIP_CKPT` defined.
- `sa1b/sa1b_links.tsv` committed (1000 image tars).

---

## What's PENDING

### 1. First training run
- Job 9081051 submitted (`--array=0-0%1 --time=00:20:00 --mem=32G --steps-per-job 174`).
- Runs 174 steps on 1 shard (sa_000020). Quick pipeline validation.
- Monitor: Comet for loss, VRAM usage, no crashes.
- Previous attempt (9079933) died silently — probably OOM with old 96G code + 70GB shard loaded into RAM.

### 2. Export jobs
- Job 9081089 (47 shards, 32G mem). Pending H100 availability.

### 3. Full training run
- Once first run validates, submit with `--array=0-N%1` for real training.

---

## Tensions / Open Questions

1. **Shuffling**: SA-1B images within a tar are geographically correlated (contiguous IDs = same region). No shuffle for now — verified visually that 4 sequential images look diverse enough. Revisit if loss curves show issues.
2. **Variable shard sizes**: `ShardedFeatureLoader` assumes uniform shard sizes (line 132). SA-1B shards are ~11,186 but not guaranteed identical. Should be fine for first run.
3. **Batch size at 1024px**: 64 @ 512px uses 18.4 GB. At 1024px/grid64, canvas is 4x larger. ~50-70 GB estimated. Should fit H100 80GB. May need to reduce if OOM.
4. **Training memory**: 96G requested but probably only needs 32-48G (shards are mmap'd, images are small). Export genuinely needs 96G for the 66 GB accumulation buffer.
5. **Export mmap optimization**: DONE (`8636703`). numpy.memmap on SLURM_TMPDIR, torch.from_numpy for save. `--mem` 96G → 32G.
6. **Warmup for continual pretraining**: Config has 100K warmup steps. For seed from pretrained model, this is very long (LR stays near 1e-7 for ~100K steps). May want shorter warmup or no warmup.
7. **No assertion for `scene_resolution == G * patch_size`**: Would silently break if these diverge.

---

## Bugs Found and Fixed

| Date | Commit | Bug | Impact |
|---|---|---|---|
| 2026-02-23 | `956f501` | End-of-job `save_checkpoint()` passed removed `scene_norm_state`/`cls_norm_state` kwargs | Would crash with TypeError after every training job — ALL training wasted |
| 2026-02-23 | `956f501` | `train.sh` tar extraction: `'*.jpg'` doesn't match `./sa_226692.jpg` without `--wildcards` | Zero images extracted, training crashes on first image load |
| 2026-02-23 | `956f501` | `ShardedFeatureLoader.__init__` loaded first shard without `mmap=True` | ~70 GB loaded into RAM just to call `len()`, risking OOM on 96G nodes |

---

## Changelog

| Date | Commit | What |
|---|---|---|
| 2026-02-23 | `a8813b6` | Reduce train.sh: workers 16→4, cpus 16→8, steps 4872→1218 |
| 2026-02-23 | `8636703` | Mmap export buffers, reduce --mem 96G→32G (export+train) |
| 2026-02-23 | `f3051a9` | Remove dead build_parquet.py |
| 2026-02-23 | `956f501` | Fix 3 bugs: checkpoint crash, tar extraction, shard OOM |
| 2026-02-23 | `2a2e316` | Add `hf_seed_ckpt` config + loop support, plan_job.py, train.sh |
| 2026-02-23 | `3539567` | Add --max-concurrent to submit_export.sh |
| 2026-02-22 | `21e9bbc` | Unify standardizers: model.standardizers(G) in loop |
| 2026-02-22 | various | Export pipeline (export_features.py/.sh, submit_export.sh) |
