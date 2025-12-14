# NAFNet Real Estate Enhancement API (FastAPI)

GPU-backed image enhancement API using the **FP32 PyTorch NAFNet** checkpoint.

## Run

```bash
conda activate nafnet

# Optional config
export TORCH_MODEL_PATH="nafnet-realestate/BasicSR/experiments/NAFNet_RealEstate_Fast/models/net_g_12000.pth"
export DEVICE=cuda                    # cuda|cpu
export GPU_SLOTS=1                    # number of warm model replicas (VRAM scales with this)
export WARMUP_ITERS=2                 # warmup forward passes per tier on startup
export CHANNELS_LAST=1                # 1/0
export CUDNN_BENCHMARK=0              # 1 can cause huge first-request spikes if shapes vary
export API_RUNS_DIR="nafnet-realestate/api_runs"

python nafnet-realestate/api_server.py --host 0.0.0.0 --port 8000
```

Interactive docs:
- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/healthz` | basic health + queue sizes + RSS |
| GET | `/v1/tiers` | supported tiers (`1024`, `2048`, `full`) |
| POST | `/v1/jobs` | submit 1+ images (multipart or zip) |
| GET | `/v1/jobs/{job_id}` | job status + per-item trace |
| GET | `/v1/jobs/{job_id}/events` | SSE event stream (progress) |
| GET | `/v1/items/{item_id}` | download a single output image (JPEG) |
| GET | `/v1/jobs/{job_id}/download` | download batch outputs as zip |

## Submit (single image)

```bash
curl -sS -X POST \
  -F tier=1024 \
  -F jpeg_quality=95 \
  -F images=@nafnet-realestate/test_input/0000.jpg \
  http://127.0.0.1:8000/v1/jobs | jq .
```

## Submit (batch: multiple files)

```bash
curl -sS -X POST \
  -F tier=2048 \
  -F jpeg_quality=95 \
  -F images=@nafnet-realestate/test_input/0000.jpg \
  -F images=@nafnet-realestate/test_input/0001.jpg \
  -F images=@nafnet-realestate/test_input/0002.jpg \
  http://127.0.0.1:8000/v1/jobs | jq .
```

## Submit (batch: zip upload)

```bash
tmpzip="$(mktemp --suffix=.zip)"
rm -f "$tmpzip"  # mktemp creates the file; zip expects to create it
zip -j "$tmpzip" nafnet-realestate/test_input/0000.jpg nafnet-realestate/test_input/0001.jpg >/dev/null

curl -sS -X POST \
  -F tier=full \
  -F jpeg_quality=95 \
  -F zip_file=@"$tmpzip" \
  http://127.0.0.1:8000/v1/jobs | jq .
```

## Stream progress (SSE)

```bash
JOB_ID="..."
curl -N http://127.0.0.1:8000/v1/jobs/$JOB_ID/events
```

Events include:
- `job_created`, `job_upload_saved`
- `item_enqueued`, `item_started`, `item_done`, `item_error`
- `job_done`, `job_error`, `end`

## Download outputs

```bash
# per-item
ITEM_ID="..."
curl -L "http://127.0.0.1:8000/v1/items/$ITEM_ID" -o out.jpg

# batch zip (ready when job status is "done")
JOB_ID="..."
curl -L "http://127.0.0.1:8000/v1/jobs/$JOB_ID/download" -o outputs.zip
```

## Tracing / performance fields

Each item has:
- `trace.decode_ms`, `trace.preprocess_ms`, `trace.infer_ms`, `trace.encode_ms`, `trace.total_worker_ms`
- `trace.rss_before_mb`, `trace.rss_after_mb`
- `trace.gpu_peak_alloc_mb` (best effort; most accurate when `GPU_SLOTS=1`)
- `queue_wait_ms` (time spent waiting in the tier queue)

## Cloudflare Tunnel (quick)

```bash
cloudflared tunnel --url http://127.0.0.1:8000
```

If you’re demoing “real-time” progress, prefer `/v1/jobs/{job_id}/events` (SSE) plus per-item downloads as each finishes.
