@README.md

## Project-Specific Rules

### CRITICAL: Metrics and Mask Handling
- **NEVER downsample GT masks for any reason**
- **ALWAYS upsample predictions (proba with bilinear, THEN threshold) to full mask resolution**
- Threshold value (0.5) is in `TBD` as `MASK_THRESHOLD` - use `binarize()`, never hardcode
- Use `upsample_proba_and_binarize()` for predictions, never upsample binary masks

### Model
- DINOv3 ViT-B/16 (NOT DINOv2!)
- Patch size: 16px (NOT 14px)
- Feature dim: 768

### Reference C values
C ∝ 1/patches. See `REFERENCE_C` in TBD.
