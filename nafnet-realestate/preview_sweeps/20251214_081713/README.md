# Torch Sweep Preview (val x4)

Generated with:

```bash
conda activate nafnet
python nafnet-realestate/benchmark_sweep_torch.py \
  --device cuda --splits val --limit 4 \
  --long-sides 1024,1536,2048,full \
  --warmup 10 --optimize --channels-last \
  --save-fp16-checkpoint
```

Key file: `sweep_summary.json`

Quick takeaways (mean inference time):
- `1024`: FP32 ~0.34s, FP16 ~0.22s
- `2048`: FP32 ~1.50s, FP16 ~0.97s
- `full` (~7.26MP): FP32 ~3.93s, FP16 ~2.59s

Visuals: see each folder’s `quadtych/` for `(LQ | FP32 | FP16 | GT)`.

