# Scene Size Curriculum - Implementation Status

## COMPLETED (commits on multi-trainer branch)

- ✅ `n_scene_registers: int = 32` replaces `use_scene_registers: bool` (commit 3c19d21)
- ✅ `AVPViT.set_scene_grid_size(new_size)` method added
- ✅ `SurvivalBatch` replaces `TrainState` (commit 1226dcb)
- ✅ `avp_vit/train/norm/PositionAwareNorm` created (commit 67508bf)
- ✅ `avp_vit/train/curriculum/` created with formulas + CurriculumStage (commit da17552)

## REMAINING WORK

### Phase 1.4: Update viewpoint module (IN PROGRESS)

Add to `avp_vit/train/viewpoint/__init__.py`:

```python
def make_curriculum_eval_viewpoints(B: int, G: int, g: int, device) -> list[Viewpoint]:
    """Generate eval viewpoints for curriculum stage. Uses quadrant recursion."""
    from avp_vit.train.curriculum import n_eval_viewpoints
    n_eval = n_eval_viewpoints(G, g)
    all_vps = [Viewpoint.full_scene(B, device)]

    # Depth needed: sum of 4^i from i=0 to d = (4^(d+1)-1)/3 >= n_eval
    max_depth = max(1, math.ceil(math.log((3 * n_eval + 1), 4)) - 1)

    for depth in range(1, max_depth + 1):
        level_vps = _quadrants_at_depth(B, depth, device)
        random.shuffle(level_vps)
        all_vps.extend(level_vps)

    return all_vps[:n_eval]

def _quadrants_at_depth(B: int, depth: int, device) -> list[Viewpoint]:
    """Generate all 4^depth quadrants at given depth."""
    # depth=1: 4 quadrants (scale=0.5)
    # depth=2: 16 sub-quadrants (scale=0.25)
    scale = 0.5 ** depth
    n = 2 ** depth  # grid divisions per side
    vps = []
    for i in range(n):
        for j in range(n):
            # Center of cell (i,j) in [-1,1] coords
            cx = -1 + scale + 2 * scale * j
            cy = -1 + scale + 2 * scale * i
            centers = torch.full((B, 2), 0.0, device=device)
            centers[:, 0] = cx
            centers[:, 1] = cy
            scales = torch.full((B,), scale, device=device)
            vps.append(Viewpoint(f"depth{depth}_{i}_{j}", centers, scales))
    return vps
```

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
