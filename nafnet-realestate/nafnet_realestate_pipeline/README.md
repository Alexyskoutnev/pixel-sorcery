# NAFNet Real Estate Photo Enhancement - 4-5 Hour Training Pipeline

## 🎯 What This Does
Trains a model to transform bad real estate photos into professionally enhanced ones (HDR, exposure correction, color grading) in **4-5 hours** on your GB10.

## 📋 Prerequisites
- NVIDIA GB10 with 128GB unified memory
- 577 paired before/after images at 3300×2200
- Ubuntu 22.04+ or similar Linux

---

## 🚀 QUICK START (5 Steps)

```bash
# Step 1: Setup environment (~10 minutes)
./scripts/01_setup_environment.sh

# Step 2: Prepare your dataset (~5 minutes)
python scripts/02_prepare_dataset.py \
    --before /path/to/your/before_photos \
    --after /path/to/your/after_photos \
    --output ./datasets/realestate

# Step 3: Verify everything is ready (~1 minute)
python scripts/03_verify_setup.py

# Step 4: Start training (~4-5 hours)
./scripts/04_train.sh

# Step 5: Test your model (~2 minutes)
python scripts/05_inference.py \
    --input /path/to/test_image.png \
    --output ./results/enhanced.png
```

---

## 📁 Folder Structure

```
nafnet_realestate_pipeline/
├── README.md                    # This file
├── configs/
│   └── nafnet_fast.yml          # Training config (4-5 hours)
├── scripts/
│   ├── 01_setup_environment.sh  # Install dependencies
│   ├── 02_prepare_dataset.py    # Organize your images
│   ├── 03_verify_setup.py       # Check everything works
│   ├── 04_train.sh              # Launch training
│   ├── 05_inference.py          # Run on new images
│   ├── 06_batch_inference.py    # Process many images
│   └── 07_export_onnx.py        # Export for production
├── datasets/                    # Your prepared data goes here
│   └── realestate/
│       ├── train/
│       │   ├── lq/              # Before images
│       │   └── gt/              # After images
│       └── val/
│           ├── lq/
│           └── gt/
├── experiments/                 # Training outputs appear here
└── results/                     # Enhanced images saved here
```

---

## ⏱️ Expected Timeline

| Step | Duration | What Happens |
|------|----------|--------------|
| 1. Setup | ~10 min | Install PyTorch, BasicSR, NAFNet |
| 2. Dataset prep | ~5 min | Organize 577 image pairs |
| 3. Verification | ~1 min | Confirm GPU, data, config ready |
| 4. Training | **4-5 hours** | Model learns your enhancement style |
| 5. Testing | ~2 min | Enhance a test image |

**Total: ~4.5-5.5 hours**

---

## 📊 Training Monitoring

Watch training progress:
```bash
# In a new terminal
tensorboard --logdir experiments/NAFNet_RealEstate_Fast/tb_logger --port 6006
```

Key metrics to watch:
- `l_pix` (pixel loss): Should decrease steadily
- `l_percep` (perceptual loss): May fluctuate but trend down
- `psnr_val`: Target **25+ dB** (higher = better)

---

## 🔧 Troubleshooting

### "CUDA out of memory"
Edit `configs/nafnet_fast.yml`:
```yaml
batch_size_per_gpu: 8  # Reduce from 12 to 8
```

### "No module named basicsr"
```bash
cd NAFNet && pip install -e . && cd ..
```

### Training seems stuck
Check GPU usage:
```bash
watch -n 1 nvidia-smi
```

---

## 📚 Resources

- NAFNet Paper: https://arxiv.org/abs/2204.04676
- NAFNet GitHub: https://github.com/megvii-research/NAFNet
- BasicSR Docs: https://basicsr.readthedocs.io
- Pretrained weights: https://drive.google.com/drive/folders/1lsByk21Xw-6aW7epCwOQxvm6HYCQZPHZ

---

## 🎉 After Training

Your trained model will be at:
```
experiments/NAFNet_RealEstate_Fast/models/net_g_30000.pth
```

Use it to enhance any real estate photo:
```bash
python scripts/05_inference.py \
    --input bad_photo.png \
    --output enhanced_photo.png \
    --model experiments/NAFNet_RealEstate_Fast/models/net_g_30000.pth
```
