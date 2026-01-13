# AVP-ViT Design Philosophy

## Vision

**Goal**: Active Vision Foundation Models. Scalable, general-purpose, task-agnostic active vision pretraining - done right. Fast training, fast inference, clean gradient signal, smart design.

**Three pillars:**
- **Seeing**: Single-step behavior. Given a glimpse, can you make sense of it?
- **Integration**: Spatiotemporal filling-in from partial views across time. Recurrence, not pooling - enables top-down feedback.
- **Deciding where to look**: Policy. Use what you've seen to inform what to look at next.

## Design Choices

**Sequential glimpses**: Each glimpse is a crop at (center, scale). Canvas and CLS persist across steps - recurrence enables top-down feedback, not just bottom-up aggregation.

**Fixed canvas size**: Bounds memory. Glimpse positions vary, canvas dimensions don't grow.

**Efficiency**: Small glimpse = cheap per step. Canvas cross-attention cheaper than full self-attention. See `scripts/flops.py`.

**Distillation**: Student reconstructs frozen teacher (DINOv3) features. Clean gradient signal.

## Training Philosophy

**Generality**: Model should work with ANY viewpoint sequence - random, fixed, learned, whatever. Training uses diverse viewpoint types to ensure this.

**Exploration via structure**: Multiple branches with different viewpoint types. Policy can be deterministic (no noise injection) because other branches provide diversity. Backbone trains on all branches, stays robust.

**Maximize teacher signal**: Teacher inference is expensive. Don't waste it. Full-view branches ensure we extract maximum value from each image.

**Difficulty gradient**: Partial views from the start (not just after seeing full image) forces real filling-in. The hard case, not the easy case.

**Policy is opportunistic**: Cheap to train alongside everything else - just another branch. Even if policy isn't good yet, it:
- Shapes representations for task-specific policy finetuning later
- Keeps VPE token policy-relevant during pretraining
- Gives something to monitor during iteration

**How to evaluate policy**: full→policy should beat full→random and full→full. "You've seen everything once, now where should you look to do even better?" If it can't beat random second look, policy isn't helping yet.

**Current state (Jan 2026)**: Policy learning is experimental. Might take long pretraining to start mattering. The branching structure may evolve.
