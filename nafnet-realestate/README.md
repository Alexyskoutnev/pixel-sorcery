# NAFNet Real Estate Photo Enhancement (HDR / Editor-Style)

This project trains and deploys a **paired image-to-image enhancement** model for real estate photos:

**unedited input (LQ)** → **professionally edited target (GT)**

It is built for the hackathon constraints: **small dataset (577 pairs)**, **high-res images**, and a need for **fast, production-ready inference** on DGX Spark / GB10.

## TL;DR

- **Model:** NAFNet (width=32, ~29.2M params, ~111MB FP32)
- **Baseline training:** `NAFNet_RealEstate_Fast` (12k iters, ~5h)
- **Second-stage training:** `NAFNet_RealEstate_ColorFT_3h` (4.5k iters, ~3h) using a **color-aware loss** to reduce hue/saturation drift on large solid surfaces (walls, ceilings).
- **Deployment:**
  - Local inference API (warm GPU model, batching, SSE progress, tiers): `api_server.py`
  - Mobile export path (ONNX → Core ML FP16): `mobile_models/`

## Models

**Hugging Face:** https://huggingface.co/SebRincon/nafnet-realestate

| Format | File | Size | Use Case |
|--------|------|------|----------|
| PyTorch | `nafnet_realestate.pth` | ~117 MB | training/fine-tuning, server inference |
| ONNX | `nafnet_realestate.onnx` | ~117 MB | cross-platform / mobile export |
| Core ML | convert from ONNX | ~56 MB | iOS (FP16) |

## Architecture (What We Trained)

**NAFNet** is an image restoration architecture that works well for “editor-like” global enhancement when trained as a paired mapping.

We use the exact width-32 layout everywhere (training + inference):
- `width=32`
- `middle_blk_num=12`
- `enc_blk_nums=[2, 2, 4, 8]`
- `dec_blk_nums=[2, 2, 2, 2]`
- `scale=1` (enhancement, not upscaling)

Implementation references:
- Inference script: `inference.py:1`
- API server: `api_server.py:1`
- Training configs: `nafnet_realestate_pipeline/configs/nafnet_fast.yml:1`, `nafnet_realestate_pipeline/configs/nafnet_color_finetune_3h.yml:1`

## Loss Functions (Exact Formulas)

### Baseline model (L1 + perceptual)

Config: `nafnet_realestate_pipeline/configs/nafnet_fast.yml:1`

Let `ŷ = fθ(x)` be the model output for input `x`, and `y` be the ground truth target.

Baseline objective:

```
L_total = 1.0 * L1(ŷ, y) + 0.1 * L_perc(ŷ, y)
```

Perceptual loss uses VGG19 feature L1 distances with layer weights:
- `conv3_4: 0.5`
- `conv4_4: 0.5`
- `conv5_4: 1.0`

### Color fine-tune model (ColorAwareLoss + perceptual)

Config: `nafnet_realestate_pipeline/configs/nafnet_color_finetune_3h.yml:1`  
Loss implementation: `nafnet_realestate_pipeline/patches/basicsr/losses/color_aware_loss.py:1`

We keep the same perceptual term, but replace the pixel loss with a color-aware pixel loss:

```
L_total = 1.0 * ( 1.0 * L1_RGB(ŷ, y) + 0.25 * L_cbcr_masked(ŷ, y) ) + 0.1 * L_perc(ŷ, y)
```

Where `L_cbcr_masked` is an L1 constraint on chroma channels (Cb/Cr) in YCbCr space, averaged only over pixels where **GT chroma** exceeds a threshold:

- Convert RGB → YCbCr in `[0, 1]`
- Define chroma magnitude for GT:

```
m = sqrt( (Cb_gt - 0.5)^2 + (Cr_gt - 0.5)^2 + eps )
mask = 1[m > 0.06]        # hard mask (or sigmoid option)
```

- Compute masked CbCr L1:

```
L_cbcr_masked = sum( mask * mean(|Cb_pred-Cb_gt|, |Cr_pred-Cr_gt|) ) / (sum(mask) + eps)
```

Intuition: the baseline L1+perceptual can look “globally right” while still washing out saturated painted surfaces. This explicitly punishes chroma drift where GT is truly chromatic.

## Data + Augmentation (How We Got More Signal From 577 Pairs)

Dataset layout in this repo:
- `datasets/realestate/train/{lq,gt}` (520 pairs)
- `datasets/realestate/val/{lq,gt}` (57 pairs)

Baseline augmentation + sampling:
- Random paired crop to `gt_size=256`
- Paired flips/rotations:
  - `use_hflip: true`
  - `use_rot: true` (dihedral transforms)
- `dataset_enlarge_ratio: 50` (repeat sampling so 520 images behave like ~26k samples/epoch)

Color fine-tune sampling:
- Uses `meta_info_file` to oversample curated “color failure” examples:
  - `nafnet_realestate_pipeline/meta/train_color_focus.txt`
  - `nafnet_realestate_pipeline/meta/val_color_stress.txt`
- Meta generation script:
  - `nafnet_realestate_pipeline/scripts/09_make_color_focus_meta.py`

## Training (Baseline + Color Fine-Tune)

All training runs on CUDA (GB10). The configs are iteration-based (BasicSR).

### Baseline (fast) — ~4–5 hours

Config:
- `nafnet_realestate_pipeline/configs/nafnet_fast.yml:1`

Run:
```bash
conda activate nafnet
bash nafnet_realestate_pipeline/scripts/04_train.sh
```

Key hyperparameters:
- `total_iter: 12000`
- `batch_size_per_gpu: 20`
- Optimizer: AdamW (`lr=1e-3`, `weight_decay=1e-4`)
- Scheduler: CosineAnnealingRestartLR (`eta_min=1e-7`)
- EMA: `ema_decay=0.999`

### Color fine-tune — ~3 hours

Config:
- `nafnet_realestate_pipeline/configs/nafnet_color_finetune_3h.yml:1`

Run (applies BasicSR loss patch + generates meta files automatically):
```bash
conda activate nafnet
bash nafnet_realestate_pipeline/scripts/08_train_color_finetune_3h.sh
```

Key hyperparameters:
- `total_iter: 4500`
- AdamW `lr=2e-4`
- Same perceptual weight (`0.1`)
- Pixel loss replaced with `ColorAwareLoss`

## Evaluation / Debugging Tooling

### Triptychs (dataset-wide visual inspection + RAM/time logs)

Creates `(LQ | PRED | GT)` triptychs and logs per-image timing + RSS:
- `create_dataset_triptychs.py:1`

Example:
```bash
conda activate nafnet
python create_dataset_triptychs.py --backend torch --device cuda
```

Outputs:
- `dataset_triptychs/torch/<run_id>/metrics.csv`
- `dataset_triptychs/torch/<run_id>/summary.json`
- plus triptych images

### Benchmarks

See:
- `BENCHMARK_RESULTS.md:1`
- `benchmark_inference.py:1`
- `benchmark_sweep_torch.py:1` (resolution / fp32 vs fp16 sweeps)

### Optional: share results via S3 (private)

Generate pre-signed URL galleries for outputs stored in a private bucket:
- `generate_s3_gallery.py:1`

## Inference (CLI, API, Mobile)

### CLI inference

Single image:
```bash
python inference.py --input photo.jpg --output enhanced.jpg --device cuda
```

Folder:
```bash
python inference.py --input ./photos --output ./enhanced --device cuda
```

### API inference (FastAPI + SSE + batching)

Run:
```bash
conda activate nafnet
python api_server.py --host 0.0.0.0 --port 8000
```

Docs:
- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`
- Full spec + examples: [API.md](API.md)

Key features:
- Warm model(s) on startup (`WARMUP_ITERS`)
- 3 resolution tiers: `1024`, `2048`, `full`
- Batch upload: multiple files or a zip
- SSE progress stream: `/v1/jobs/{job_id}/events`
- Multi-model support: drop checkpoints into `~/models` and select with `model=...`

Smoke test:
```bash
conda activate nafnet
bash smoke_test_api.sh --tier 1024 --model default --image test_input/0000.jpg
```

### Mobile deployment (ONNX → Core ML FP16)

Mobile guide:
- `mobile_models/README.md:1`

Export to ONNX:
- `nafnet_realestate_pipeline/scripts/07_export_onnx.py:1`

Core ML conversion (macOS):
```bash
cd mobile_models
pip install coremltools
python convert_on_mac.py
```

## Performance

Primary reference: `BENCHMARK_RESULTS.md:1`

Measured on GB10 (avg ~7.25MP):
- ~**4.0s per image** at ~7MP
- ~**8.3GB VRAM peak**
- ~**581MB net RAM usage**

Scaling (estimates, see benchmark doc for details):
- 1080p (~2.1MP): ~1.2s
- 1440p (~3.7MP): ~2.0s
- 3K (~7.3MP): ~4.0s

Notes for stable latency:
- Keep the process warm (API server stays up; warmup runs on startup).
- Be careful with `CUDNN_BENCHMARK=1` if your requests are different sizes; it can cause large first-request spikes.

## Project Map

```
nafnet-realestate/
├── api_server.py                         # Inference API (warm GPU, batching, SSE, tiers, multi-model)
├── API.md                                # API spec + curl examples
├── inference.py                          # Simple CLI inference
├── create_dataset_triptychs.py           # (LQ | PRED | GT) dataset triptychs + RAM/time logs
├── benchmark_inference.py                # Single-run inference benchmark
├── BENCHMARK_RESULTS.md                  # Published benchmark numbers
├── nafnet_realestate_pipeline/
│   ├── configs/                          # Training configs (baseline + color fine-tune)
│   ├── scripts/                          # Train/export helpers
│   └── patches/                          # Patched BasicSR loss code for color fine-tuning
└── mobile_models/                        # ONNX + Core ML conversion + iOS integration notes
```

## References

- NAFNet paper: https://arxiv.org/abs/2204.04676
- Original NAFNet repo: https://github.com/megvii-research/NAFNet
