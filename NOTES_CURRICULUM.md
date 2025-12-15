# Scene Size Curriculum - Implementation Status

## COMPLETED (commits on multi-trainer branch)

- ✅ `n_scene_registers: int = 32` replaces `use_scene_registers: bool` (commit 3c19d21)
- ✅ `AVPViT.set_scene_grid_size(new_size)` method added
- ✅ `SurvivalBatch` replaces `TrainState` (commit 1226dcb)
- ✅ `avp_vit/train/norm/PositionAwareNorm` created (commit 67508bf)
- ✅ `avp_vit/train/curriculum/` created with formulas + CurriculumStage (commit da17552)
- ✅ `make_curriculum_eval_viewpoints()` added (commit 43ad0fd)

## REMAINING WORK

### Phase 2: Restructure training script

Current: `scripts/train_scene_match.py` (654 lines)
Target: `scripts/train_scene_match/` directory

```
scripts/train_scene_match/
├── __init__.py       # empty
├── __main__.py       # entry point, parse args, call train()
├── config.py         # Config dataclass
├── data.py           # create_loaders_for_curriculum()
├── model.py          # load_teacher, create_avp
├── viz.py            # viz_and_log, eval_and_log
└── train.py          # train() main loop with curriculum
```

**Key changes in train.py:**
1. Create `dict[int, DataLoader]` for each grid size at startup
2. Create `dict[int, PositionAwareNorm]` for each grid size
3. Use `avp.set_scene_grid_size(G)` at curriculum transitions
4. Use `torch.compile(..., dynamic=True)` for varying token counts
5. Log per-grid metrics: `grid{G}/train/loss`, `grid{G}/val/loss`

**Curriculum transition logic:**
```python
grid_sizes = [16, 32, 64]  # configurable
steps_per_stage = n_steps // len(grid_sizes)
current_stage_idx = step // steps_per_stage
if current_stage_idx changed:
    G = grid_sizes[current_stage_idx]
    avp.set_scene_grid_size(G)
    # Switch to loader/norm for this G
```

**Checkpoint contents:**
```python
checkpoint = {
    "avp": avp.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "step": step,
    "current_grid_size": G,
    "norm_states": {G: norm.state_dict() for G, norm in norms.items()},
}
```

### Phase 3: Verify

1. Run training with curriculum [16, 32, 64]
2. Check startup logs show correct computed values
3. Check transitions logged with context
4. Check Comet metrics show grid-specific values
5. Verify checkpoint save/load works

---

## Key Reference Values

With `bs_max=32, G_max=64, g=7, n_viewpoints_per_step=2`:

| G | tokens | batch | n_eval | fresh_ratio | fresh_count | E[glimpses] |
|---|--------|-------|--------|-------------|-------------|-------------|
| 16 | 256 | 512 | 5 | 0.400 | 205 | 5.0 |
| 32 | 1024 | 128 | 10 | 0.200 | 26 | 9.9 |
| 64 | 4096 | 32 | 20 | 0.100 | 3 | 21.3 |

---

## Files Modified/Created

**Modified:**
- `avp_vit/model/__init__.py` - n_scene_registers, set_scene_grid_size
- `avp_vit/train/__init__.py` - export SurvivalBatch
- `avp_vit/train/state/__init__.py` - SurvivalBatch class
- `scripts/train_scene_match.py` - will be restructured

**Created:**
- `avp_vit/train/norm/__init__.py` - PositionAwareNorm
- `avp_vit/train/norm/test.py`
- `avp_vit/train/curriculum/__init__.py` - formulas, CurriculumStage
- `avp_vit/train/curriculum/test.py`

---

## Important Design Decisions

1. **n_scene_registers is TOTAL count** (split 50/50 persistent/ephemeral)
2. **survival_batches NOT in checkpoint** - ephemeral, rebuilt on load
3. **n_viewpoints_per_step=2 always** - fresh_ratio controls trajectory length
4. **Skip G=8** - min_scale=0.875 is useless
5. **No preflight checks** - just use low warmup, exercise actual code paths
