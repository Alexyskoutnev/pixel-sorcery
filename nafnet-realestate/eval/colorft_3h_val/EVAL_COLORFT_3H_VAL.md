# ColorFT (3h) — Validation Evaluation

This folder contains evaluation artifacts comparing:
- **Baseline**: `NAFNet_RealEstate_Fast` @ `net_g_12000.pth`
- **Fine-tune**: `NAFNet_RealEstate_ColorFT_3h` @ `net_g_latest.pth` (4500 iters)

The fine-tune adds a **color-aware pixel loss** (RGB L1 + masked YCbCr chroma constraint) and oversamples curated “color drift / repaint” failures via `meta_info_file`.

## How this eval was produced

Val inference (57 images) was run twice using the same runner:
- `create_dataset_triptychs.py --backend torch --splits val --no-save-pred`

Outputs (untracked) live under:
- `nafnet-realestate/dataset_triptychs/torch/torch/baseline_12000_val/`
- `nafnet-realestate/dataset_triptychs/torch/torch/colorft_3h_latest_val/`

This folder keeps **only** the small CSV/JSON summaries + a derived color-drift CSV:
- `baseline_12000_metrics.csv`
- `baseline_12000_summary.json`
- `colorft_3h_latest_metrics.csv`
- `colorft_3h_latest_summary.json`
- `color_drift_val.csv`

## Key results

### PSNR (whole-image)
Whole-image PSNR moved slightly (expected when adding a color-focused objective):
- Baseline mean PSNR: **21.704**
- ColorFT mean PSNR:  **21.609**

### Color drift proxy (high-saturation GT mask)
`color_drift_val.csv` is computed from stitched triptychs `(LQ | PRED | GT)` by:
- splitting the triptych into `pred` and `gt`
- computing HSV on both
- masking pixels where GT is “meaningfully colorful”: `S(gt) >= 0.25` and `0.10 <= V(gt) <= 0.95`
- reporting:
  - `sat_ratio = mean(S(pred)) / mean(S(gt))` on the mask (1.0 is ideal)
  - `hue_abs_deg = mean circular absolute hue error` on the mask (lower is better)

Aggregate saturation behavior improved:
- **All val images (n=57)**: sat_ratio mean improved from ~**0.720 → 0.766**
- **Stress subset (n=5)**: sat_ratio mean improved from ~**0.557 → 0.650**

The stress subset corresponds to the curated “repaint” failures (val IDs):
`0005.jpg, 0013.jpg, 0017.jpg, 0023.jpg, 0040.jpg`

## Notes / caveats

- This is a *proxy* metric intended to catch the “saturation collapse” failure mode you described; it’s not a full perceptual color metric (ΔE00, etc).
- The stitched triptychs are JPEG, so this underestimates true errors compared to lossless evaluation.
- The BasicSR training log reported `nondist_validation is not implemented`, so this evaluation run is the authoritative val check for the 3h run.

