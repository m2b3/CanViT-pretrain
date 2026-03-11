# SA-1B Continual Pretraining

**Goal**: Continually pretrain CanViT-B flagship checkpoint (2M steps on IN21k @ 512px) at **1024px on SA-1B**.
First run: get loss on Comet, verify the model works at higher resolution.

**Branch**: merged into `main` (previously `sa1b`, worktree `~/code/CanViT-train-SA1B`)

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
| Checkpoints | `$CHECKPOINTS_DIR/{run_name}/` (NFS) | Permanent |

### Data Loading Strategy
- Feature `.pt` shards on NFS, read with `mmap=True` (cheap, no copy).
- **Images read directly from mmap'd tar files** — no extraction step. `TarImageReader` reads images via mmap slicing.
- **Pre-built tar indexes** (`.tar.idx` files): created by `sa1b/build_tar_indexes.py`, loaded in main process before fork (~0.01s). Workers inherit via COW. Shards lacking a `.tar.idx` are **filtered out** at startup.
- All DataLoader workers share mmap pages via fork COW. No RAM duplication.
- `--feature-base-dir "$SA1B_FEATURES_DIR/sa1b"` → shards path auto-constructed: `{base}/dinov3_vitb16/1024/shards/`.
- `--tar-dir "$SA1B_TAR_DIR"` → images read directly from tar files.
- Shard→tar mapping: `sa_000020.pt` → `sa_000020.tar` (via `shard_path.stem + ".tar"`).
- Shard loader iterates all shards, starts at `start_shard`, processes `steps_per_job` steps.
- **Exact resume**: on restart, skips partial-shard batches (not just whole shards) to avoid replaying data.

### Image Resolution
- **Teacher export**: 1024px input → features at 1024px (baked at export time).
- **Training images**: loaded at `scene_resolution=1024px`, same transform as export (`preprocess` from `canvit_utils.transforms`).
- **Student glimpses**: 128px² crops from the loaded scene image. Higher scene resolution = more fine-grained info when student looks "up close".
- The dataloader image resolution affects glimpse pixel richness and CPU/GPU memory, NOT alignment with the teacher features (those are already fixed from export).
- No assertion currently validates `scene_resolution == G * patch_size`. For SA-1B: 1024 == 64 × 16 = 1024.

### Tar Structure
- SA-1B tars from Meta have `./` prefix: entries like `./sa_226692.jpg` (flat, no subdirectory).
- `scan_tar_headers()` strips the prefix: `member.name.split("/", 1)[-1]`.
- `sa1b/export_features.py` reads images from mmap'd tars (same as training).

### Job Array Design
- `%1` concurrency (1 job at a time, like IN21k).
- `steps_per_job` = multiple of `batches_per_shard`. batches_per_shard = 11186 // 64 = **174**.
- Default: 1218 steps/job (7 shards). Override via `--steps-per-job`.
- No extraction step — Python handles everything. Startup is near-instant.

### Key Numbers
| Metric | Value | Source |
|---|---|---|
| Images per tar | ~11,186 | tar tvf on sa_000020 |
| JPEG size per tar | ~10.3 GB | tar tvf estimate |
| Feature shard size | ~65-70 GB | sa_000020.pt on def-areynaud |
| batch_size | 64 | sa1b/train.sh |
| batches_per_shard | 174 | 11186 // 64 |
| shards_per_job | 7 | Default (1024px is ~4x more expensive/step) |
| steps_per_job | 1218 | 174 × 7 |
| Tar index load | ~0.01s | Pre-built `.tar.idx` file |
| Tar index build | ~56s | `scan_tar_headers()` on 70 GB tar (one-time) |
| Canvas grid size | 64 | 1024 / 16 |
| Scene resolution | 1024px | Target for SA-1B |
| SLURM --mem | 256G | sa1b/train.sh (tuned from 48G→64G→128G→192G→256G) |
| SLURM workers | 12 | sa1b/train.sh (tuned from 4→8→12) |
| SLURM cpus-per-task | 12 | sa1b/train.sh |

---

## What's DONE

### Training Pipeline (commit `e1ef147`)
- `sa1b/train.sh` — SLURM array job. No extraction — images read from mmap'd tars.
- `canvit_pretrain/train/data/tar_images.py` — `TarImageReader`: tarfile index + mmap reads.
- `canvit_pretrain/train/data/shards.py` — `AllShardsDataset` supports `tar_dir` (SA-1B) or `image_root` (IN21k).
- `hf_seed_ckpt` config option in `loop.py` — downloads from HF, overrides cfg.model, seed mode.
- `strict=False` + regex filter for grid-size-change standardizer mismatches.
- Standardizers travel with model state_dict via `model.standardizers(G)`.

### Export Pipeline
- `sa1b/export_features.py` — 1 tar → 1 shard. Atomic save. Idempotent.
- `sa1b/export_features.sh` — SLURM array job. `gpu:h100:1`, 48G RAM, 16 CPUs, 10 min.
- `sa1b/submit_export.sh` — Auto-submits missing shards. `--dry-run` supported. **No lock on concurrent runs.**

### Tar Index Infrastructure
- `sa1b/build_tar_indexes.py` — Pre-builds `.tar.idx` files (SHA256 + offset index). Parallel via `ProcessPoolExecutor`.
- `sa1b/build_tar_indexes.sh` — SLURM script for batch index building (CPU-only, no GPU).
- Training **requires** `.tar.idx` files — shards without them are filtered out at startup.

### Download
- `sa1b/download.py` running on Nibi (tmux `sa1b-dl`). ~73 MB/s.
- Non-contiguous tar indices. submit_export.sh handles this.

### Env Setup
- `.envrc.nibi` committed. `SA1B_TAR_DIR`, `SA1B_FEATURES_DIR`, `CANVIT_FLAGSHIP_CKPT` defined.
- `sa1b/sa1b_links.tsv` committed (1000 image tars).

---

## Status (needs cluster verification)

**TODO: verify on Nibi** — the following was true as of the last code commit (Feb 25).
Check Comet, logs, and cluster state to confirm current status.

- Export pipeline: code complete. Unknown how many shards were successfully exported.
- Tar-ordered re-export: code landed (commit `d1bf94b`). Unknown if all shards were re-exported.
- Tar indexes: `build_tar_indexes.py` exists. Unknown if indexes were built for all tars.
- Training smoketest: early attempts ran (9079933, 9081051, 9082081, 9084008, 9086760). Unknown final outcome.
- Full training run: **not confirmed started**. Requires human validation of smoketest results first.

---

## Tensions / Open Questions

1. **Shuffling**: SA-1B images within a tar are geographically correlated (contiguous IDs = same region). No shuffle for now — verified visually that 4 sequential images look diverse enough. Revisit if loss curves show issues.
2. **Variable shard sizes**: `ShardedFeatureLoader` assumes uniform shard sizes (line 132). SA-1B shards are ~11,186 but not guaranteed identical. Should be fine for first run.
3. **Batch size at 1024px**: 64 @ 512px uses 18.4 GB. At 1024px/grid64, canvas is 4x larger. ~50-70 GB estimated. Should fit H100 80GB. May need to reduce if OOM.
4. **Training memory**: 256G requested (tuned up from 48G through multiple OOM rounds). Shards and tars are mmap'd (shared via COW). Export needs 48G.
5. **Warmup for continual pretraining**: Set to 2000 steps in `sa1b/train.sh` (down from default 100K).
6. **Cosine scheduler**: Available via `--cosine-total-steps N`. Not currently used in `sa1b/train.sh` (warmup → constant).
7. **`scene_resolution` vs `G * patch_size`**: These are independent. `scene_resolution` = pixel size of loaded training images. `G * patch_size` = teacher input resolution (baked at export time). They don't have to match — student sees images at `scene_resolution`, teacher features are precomputed.

---

## Bugs Found and Fixed

| Date | Commit | Bug | Impact |
|---|---|---|---|
| 2026-02-23 | `d1bf94b` | Export shards saved paths in alphabetical order (not tar order) | Random I/O across 70GB tars during training. Causes data loading bottleneck. |
| 2026-02-23 | `956f501` | End-of-job `save_checkpoint()` passed removed `scene_norm_state`/`cls_norm_state` kwargs | Would crash with TypeError after every training job — ALL training wasted |
| 2026-02-23 | `e1ef147` | Tar extraction blocked training for 8+ min on 10-min GPU job | Only 9 seconds of actual training before TIME LIMIT kill. Replaced with mmap tar reading. |
| 2026-02-23 | `956f501` | `train.sh` tar extraction: `'*.jpg'` doesn't match `./sa_226692.jpg` without `--wildcards` | Zero images extracted, training crashes on first image load |
| 2026-02-23 | `956f501` | `ShardedFeatureLoader.__init__` loaded first shard without `mmap=True` | ~70 GB loaded into RAM just to call `len()`, risking OOM on 96G nodes |
| 2026-02-23 | `47b6c9c` | `.envrc` credential `&&` chains return exit 1 under `set -e` | ALL submitted jobs (train + 47 exports) died in <1s. Zero work done. |
| 2026-02-23 | `47b6c9c` | `init_normalizer_stats_from_shard` loaded shard to GPU without mmap | Would OOM trying to load 70GB shard to H100. Even CPU would need 141GB float32. |

---

## Changelog

| Date | Commit | What |
|---|---|---|
| 2026-02-25 | `5df4c3b` | Bump export memory 32G → 48G (OOM on shard save) |
| 2026-02-25 | `ff6cc9e` | Filter shards to only those with `.tar.idx` at startup |
| 2026-02-25 | `95b3001` | Make TarImageReader.close() idempotent, add `__del__` |
| 2026-02-25 | `4c2dbef` | Fix resume config bug, per-worker instrumentation, bump --mem to 256G |
| 2026-02-24 | `9c51c04` | Pre-build tar indexes: eliminate worker contention on shard transitions |
| 2026-02-24 | `1f1b2fc` | Add SLURM script for building all tar indexes (CPU-only) |
| 2026-02-24 | various | Memory tuning: 64G → 128G → 192G → 256G, workers 4 → 8 → 12 |
| 2026-02-24 | `e11aedf` | Per-op timing instrumentation in AllShardsDataset |
| 2026-02-24 | `14f0713` | Add NFS vs local NVMe benchmark |
| 2026-02-24 | various | Rewrite + fix dataloader benchmark |
| 2026-02-23 | `1c2c9c9` | Add optional cosine decay: `--cosine-total-steps N` |
| 2026-02-23 | `57a3d32` | Exact resume: skip partial-shard batches instead of replaying |
| 2026-02-23 | `d1bf94b` | Export shards in tar order (not alphabetical), add data/gpu timing |
| 2026-02-23 | `e1ef147` | Read SA-1B images directly from mmap'd tars (no extraction) |
| 2026-02-23 | `5b1e45a` | Make normalizer_max_samples a config parameter |
| 2026-02-23 | `47b6c9c` | Fix .envrc set-e crash + normalizer shard OOM |
| 2026-02-23 | `956f501` | Fix 3 bugs: checkpoint crash, tar extraction, shard OOM |
| 2026-02-23 | `2a2e316` | Add `hf_seed_ckpt` config + loop support, train.sh |
| 2026-02-22 | `21e9bbc` | Unify standardizers: model.standardizers(G) in loop |
| 2026-02-22 | various | Export pipeline (sa1b/export_features.py/.sh, submit_export.sh) |
