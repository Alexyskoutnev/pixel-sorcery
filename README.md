# Pixel Sorcery — Real Estate Photo Editing System (DGX Spark / GB10)

This repo contains an end-to-end real estate photo editing system built on a **DGX Spark (NVIDIA GB10)**:

- Ingest paired before/after data (S3)
- Train a fast image-to-image enhancement model (NAFNet)
- Evaluate with repeatable tooling
- Deploy for **local low-latency inference** (API) and **mobile** (Core ML)

The core project lives in `nafnet-realestate/`.

## Challenge (HDR / Real Estate Photo Editing)

We are given **577 paired images**:
- **Input (X / LQ):** unedited real estate photo
- **Output (Y / GT):** professional editor output

Goal: learn `fθ(X) → Y` as closely as possible (color, exposure, realism), while keeping inference efficient.

Dataset source:
- `s3://hackathon-dec12/autohdr-real-estate-577/`

## What We Built (Working System)

**1) Data ingestion**
- Download dataset from S3: `download-images.sh:1`
- Prepare paired dataset layout under `nafnet-realestate/datasets/realestate/...` (see pipeline).

**2) Training pipeline (two-stage)**
- **Baseline model:** NAFNet fine-tune optimized for the hackathon window.
- **Color-fidelity fine-tune:** a second run starting from the baseline, adding a color-aware loss focused on chroma stability (fix “wall repainting” failures).

**3) Evaluation + debugging tools**
- Triptych generator `(LQ | PRED | GT)` across train/val with timing + RAM logs: `nafnet-realestate/create_dataset_triptychs.py:1`
- Benchmark scripts for inference time + VRAM/RAM: `nafnet-realestate/benchmark_inference.py:1`, `nafnet-realestate/BENCHMARK_RESULTS.md:1`
- S3 gallery generator for visual inspection of outputs (private bucket): `nafnet-realestate/generate_s3_gallery.py:1`

**4) Deployment**
- **FastAPI inference server** (warm GPU model, batching, SSE progress, multi-model selection, 3 resolution tiers):
  - Code: `nafnet-realestate/api_server.py:1`
  - Spec: `nafnet-realestate/API.md:1`
- **Mobile path**:
  - Export to ONNX: `nafnet-realestate/nafnet_realestate_pipeline/scripts/07_export_onnx.py:1`
  - Convert ONNX → Core ML FP16 for iOS: `nafnet-realestate/mobile_models/README.md:1`

## Quick Demo (Judge Flow)

1) Start the API (local GPU inference):
```bash
conda activate nafnet
python nafnet-realestate/api_server.py --host 0.0.0.0 --port 8000
```

2) Run the smoke test (submits one image, streams SSE events, downloads output):
```bash
conda activate nafnet
bash nafnet-realestate/smoke_test_api.sh --tier 1024 --model default --image nafnet-realestate/test_input/0000.jpg
```

3) Open the API docs:
- `http://127.0.0.1:8000/docs`

## Performance (Why Spark)

From the benchmark run on **100 high-res images** (avg ~7.25MP) on **NVIDIA GB10**:
- ~**4.0s / image** at ~7MP
- ~**8.3GB VRAM peak**
- ~**581MB net RAM usage** (model + inference)

Details: `nafnet-realestate/BENCHMARK_RESULTS.md:1`

**Spark story (why DGX Spark):**
- The dataset is high-res; training + inference benefit heavily from **CUDA + cuDNN**.
- **128GB unified memory** allows larger batches and more aggressive buffering without OOM churn.
- Running inference **locally** gives low latency and avoids shipping private images to third parties.

## Judging Criteria Mapping (How To Talk About It)

**1) Technical Execution & Completeness**
- Full workflow: ingest → train → evaluate → deploy API/mobile.
- Non-trivial engineering: two-stage training, custom loss, tiered inference server (queues + warm models + SSE progress + batching).

**2) NVIDIA Ecosystem & Spark Utility**
- GPU-first training/inference with CUDA/cuDNN (PyTorch + BasicSR).
- Optional production path via ONNX export to enable **TensorRT** acceleration.
- Spark enables fast iteration on high-res data with limited dataset size.

**3) Value & Impact**
- Produces editor-like enhancements (HDR/exposure/color) from raw photos.
- Usable immediately via API (batch upload + download) or on-device (Core ML).

**4) Frontier Factor**
- Color-fidelity fine-tuning targets a real failure mode (solid surfaces drifting hue/saturation).
- Performance knobs: tiered resolution inference, warm models, parallel job queueing.

## HDR Challenge Submission Notes (Checklist)

What to provide:
- Sample outputs (zip)
- Written summary: technique, training time, inference time, VRAM usage, optimizations/tradeoffs

Where to look:
- Technique + training details + loss formulas: `nafnet-realestate/README.md:1`
- Performance numbers: `nafnet-realestate/BENCHMARK_RESULTS.md:1`
- How to run inference across datasets and generate zip outputs:
  - `nafnet-realestate/create_dataset_triptychs.py:1`
  - `nafnet-realestate/api_server.py:1` + `nafnet-realestate/API.md:1`

### Submission write-up (copy/paste)

**Model / technique**
- Architecture: NAFNet (width=32), trained as paired image-to-image enhancement (scale=1).
- Stage 1: baseline fine-tune with pixel L1 + perceptual VGG19 loss.
- Stage 2: color-fidelity fine-tune starting from the baseline, adding a chroma-aware loss (masked Cb/Cr constraint) to reduce hue/saturation drift on saturated walls/solid surfaces.

**Why this approach**
- NAFNet is a strong restoration backbone that trains quickly and runs fast at high resolution.
- Patch-based training (256px crops) lets us learn from 3K+ source images within the hackathon window.
- The color fine-tune targets a real failure mode (solid-color “repainting”) without inflating the model size.

**Training time**
- Baseline: ~5 hours on GB10 (12k iterations).
- Color fine-tune: ~3 hours on GB10 (4.5k iterations).

**Inference time / cost (GB10)**
- ~4s per ~7MP image (3K-class), ~8.3GB VRAM peak, ~581MB net RAM usage.
- We also expose tiers (1024 / 2048 / full) for speed-quality tradeoffs via the API.

**Optimizations / tradeoffs**
- Kept the model small (width=32) so it can run on phones after ONNX → Core ML FP16 conversion.
- “Warm” inference server: model loads once; optional warmup reduces first-request latency.
- Batching support: multi-file or zip upload, per-item streaming progress (SSE), zip download.
- Tradeoff: patch training reduces global context during training; full-resolution inference recovers global appearance in practice.

### Produce a ZIP for the final test set

When you get the final test set from the organizers:

```bash
conda activate nafnet
python nafnet-realestate/inference.py --input /path/to/test_set --output /tmp/outputs --device cuda --model nafnet-realestate/BasicSR/experiments/NAFNet_RealEstate_Fast/models/net_g_12000.pth
cd /tmp/outputs
zip -r outputs.zip .
```

Upload (example):
```bash
aws s3 cp outputs.zip s3://hackathon-dec12/<your-submission-prefix>/outputs.zip
```

## Docs / Entry Points

- DGX Spark setup: `docs/dgx-setup.md:1`
- Model + training + evaluation: `nafnet-realestate/README.md:1`
- API spec + examples: `nafnet-realestate/API.md:1`
