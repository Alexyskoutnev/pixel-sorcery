#!/usr/bin/env python3
"""
FastAPI inference server for NAFNet (FP32) with 3 resolution tiers + queues.

Goals:
  - Keep the model warm (load once, warmup once)
  - Accept single images or batch (zip or multi-file)
  - Process via tiered queues: 1024 / 2048 / full
  - Provide per-item tracing timestamps + RAM/GPU memory metrics
  - Provide SSE progress stream + zip download for batch results

Run (inside conda env):
  conda activate nafnet
  python api_server.py --host 127.0.0.1 --port 8000

Cloudflare tunnel (example):
  cloudflared tunnel --url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import queue
import re
import shutil
import sys
import threading
import time
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse


SCRIPT_DIR = Path(__file__).resolve().parent


def _now_ms() -> int:
    return int(time.time() * 1000)


def _perf_ms() -> float:
    return time.perf_counter() * 1000.0


def _safe_name(name: str) -> str:
    name = Path(name).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name[:180] if len(name) > 180 else name


def _get_rss_mb() -> float:
    import psutil  # type: ignore

    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def _torch_device(device: str) -> Any:
    import torch  # type: ignore

    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _resize_long_side_max(img_bgr: Any, long_side: int | None) -> Any:
    import cv2  # type: ignore

    if long_side is None:
        return img_bgr
    h, w = img_bgr.shape[:2]
    current = max(h, w)
    if current <= long_side:
        return img_bgr
    scale = long_side / float(current)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _pad_to_multiple_of_8(img_bgr: Any) -> tuple[Any, tuple[int, int], tuple[int, int]]:
    import cv2  # type: ignore

    h, w = img_bgr.shape[:2]
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h == 0 and pad_w == 0:
        return img_bgr, (0, 0), (h, w)
    padded = cv2.copyMakeBorder(img_bgr, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    return padded, (pad_h, pad_w), (h, w)


def _decode_image_bytes(data: bytes) -> Any:
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _encode_jpeg_bytes(img_bgr: Any, quality: int) -> bytes:
    import cv2  # type: ignore

    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def _load_model(checkpoint_path: Path, device: str, *, channels_last: bool, cudnn_benchmark: bool) -> Any:
    import torch  # type: ignore

    sys.path.insert(0, str(SCRIPT_DIR / "BasicSR"))
    from basicsr.archs.NAFNet_arch import NAFNet  # type: ignore

    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)

    model = NAFNet(
        img_channel=3,
        width=32,
        middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2],
    )

    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "params" in ckpt:
        model.load_state_dict(ckpt["params"])
    elif isinstance(ckpt, dict) and "params_ema" in ckpt:
        model.load_state_dict(ckpt["params_ema"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    model = model.to(_torch_device(device))
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    return model


@dataclass
class ItemTrace:
    # Epoch ms timestamps (server-side).
    t_request_received_ms: int | None = None
    t_saved_ms: int | None = None
    t_enqueued_ms: int | None = None
    t_worker_start_ms: int | None = None
    t_decode_done_ms: int | None = None
    t_preprocess_done_ms: int | None = None
    t_infer_start_ms: int | None = None
    t_infer_end_ms: int | None = None
    t_encode_done_ms: int | None = None
    t_done_ms: int | None = None

    # Durations (ms).
    decode_ms: float | None = None
    preprocess_ms: float | None = None
    infer_ms: float | None = None
    encode_ms: float | None = None
    total_worker_ms: float | None = None

    # Memory.
    rss_before_mb: float | None = None
    rss_after_mb: float | None = None
    gpu_alloc_before_mb: float | None = None
    gpu_alloc_after_mb: float | None = None
    gpu_peak_alloc_mb: float | None = None


ItemStatus = Literal["queued", "processing", "done", "error"]
Tier = Literal["1024", "2048", "full"]


@dataclass
class Item:
    item_id: str
    job_id: str
    filename: str
    tier: Tier
    jpeg_quality: int
    input_path: str
    output_path: str | None = None
    status: ItemStatus = "queued"
    error: str | None = None
    width: int | None = None
    height: int | None = None
    megapixels: float | None = None
    trace: ItemTrace = field(default_factory=ItemTrace)


JobStatus = Literal["queued", "processing", "done", "error"]


@dataclass
class Job:
    job_id: str
    created_ms: int
    tier: Tier
    jpeg_quality: int
    items: list[str] = field(default_factory=list)  # item_ids
    status: JobStatus = "queued"
    error: str | None = None
    out_dir: str = ""
    zip_path: str | None = None
    finalizing: bool = False
    events: list[dict[str, Any]] = field(default_factory=list)


class ServerState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.jobs: dict[str, Job] = {}
        self.items: dict[str, Item] = {}
        self.queues: dict[Tier, "queue.Queue[str]"] = {
            "1024": queue.Queue(),
            "2048": queue.Queue(),
            "full": queue.Queue(),
        }
        self.model_pool: "queue.Queue[Any]" = queue.Queue()
        self.device: str = "cuda"
        self.channels_last: bool = True
        self.cudnn_benchmark: bool = False
        self.gpu_slots: int = 1
        self.runs_dir: Path = SCRIPT_DIR / "api_runs"

    def push_event(self, job_id: str, event: dict[str, Any]) -> None:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return
            event = dict(event)
            event.setdefault("ts_ms", _now_ms())
            job.events.append(event)


STATE = ServerState()
APP = FastAPI(title="NAFNet Real Estate Enhancement API", version="0.1")


def _tier_long_side(tier: Tier) -> int | None:
    if tier == "full":
        return None
    return int(tier)


def _public_item(item: Item) -> dict[str, Any]:
    trace = asdict(item.trace)
    queue_wait_ms = None
    if item.trace.t_enqueued_ms is not None and item.trace.t_worker_start_ms is not None:
        queue_wait_ms = item.trace.t_worker_start_ms - item.trace.t_enqueued_ms
    return {
        "item_id": item.item_id,
        "job_id": item.job_id,
        "filename": item.filename,
        "tier": item.tier,
        "jpeg_quality": item.jpeg_quality,
        "status": item.status,
        "error": item.error,
        "width": item.width,
        "height": item.height,
        "megapixels": item.megapixels,
        "trace": trace,
        "queue_wait_ms": queue_wait_ms,
        "output_url": f"/v1/items/{item.item_id}" if item.status == "done" else None,
    }


def _public_job(job: Job, items: list[Item]) -> dict[str, Any]:
    done = sum(1 for i in items if i.status == "done")
    err = sum(1 for i in items if i.status == "error")
    return {
        "job_id": job.job_id,
        "tier": job.tier,
        "jpeg_quality": job.jpeg_quality,
        "status": job.status,
        "error": job.error,
        "items_total": len(items),
        "items_done": done,
        "items_error": err,
        "items": [_public_item(i) for i in items],
        "events_url": f"/v1/jobs/{job.job_id}/events",
        "download_url": f"/v1/jobs/{job.job_id}/download",
        "zip_url": f"/v1/jobs/{job.job_id}/download" if job.zip_path else None,
        "events_count": len(job.events),
    }


def _model_infer(model: Any, device: str, img_bgr: Any, *, channels_last: bool) -> Any:
    import numpy as np  # type: ignore
    import torch  # type: ignore

    padded, _pad, orig_hw = _pad_to_multiple_of_8(img_bgr)

    # BGR -> RGB, normalize [0,1], HWC->CHW->NCHW float32
    img_rgb = padded[:, :, ::-1].copy()
    img_rgb = img_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)
    if channels_last:
        tensor = tensor.contiguous(memory_format=torch.channels_last)
    tensor = tensor.to(_torch_device(device), dtype=torch.float32)

    if device == "cuda":
        torch.cuda.synchronize()
    with torch.inference_mode():
        out = model(tensor)
    if device == "cuda":
        torch.cuda.synchronize()

    out = out.squeeze(0).detach().cpu().clamp(0, 1).numpy()
    out = (np.transpose(out, (1, 2, 0)) * 255.0).astype(np.uint8)
    out_bgr = out[:, :, ::-1]
    h, w = orig_hw
    return out_bgr[:h, :w]


def _worker_loop(tier: Tier) -> None:
    import torch  # type: ignore

    q = STATE.queues[tier]
    while True:
        item_id = q.get()
        try:
            with STATE.lock:
                item = STATE.items.get(item_id)
                if not item:
                    continue
                job = STATE.jobs.get(item.job_id)
                if job and job.status == "queued":
                    job.status = "processing"
                item.status = "processing"
                item.trace.t_worker_start_ms = _now_ms()
            STATE.push_event(item.job_id, {"type": "item_started", "item_id": item_id, "tier": tier})

            t0 = _perf_ms()
            rss_before = _get_rss_mb()

            if STATE.device == "cuda":
                gpu_before = torch.cuda.memory_allocated() / (1024 * 1024)
                if STATE.gpu_slots == 1:
                    torch.cuda.reset_peak_memory_stats()
            else:
                gpu_before = 0.0

            data = Path(item.input_path).read_bytes()

            t_decode0 = _perf_ms()
            img = _decode_image_bytes(data)
            t_decode1 = _perf_ms()
            if img is None:
                raise RuntimeError("Could not decode image (cv2.imdecode returned None)")

            item.trace.decode_ms = t_decode1 - t_decode0
            item.trace.t_decode_done_ms = _now_ms()

            t_pre0 = _perf_ms()
            img = _resize_long_side_max(img, _tier_long_side(tier))
            item.height, item.width = img.shape[:2]
            item.megapixels = (item.height * item.width) / 1_000_000.0
            t_pre1 = _perf_ms()
            item.trace.preprocess_ms = t_pre1 - t_pre0
            item.trace.t_preprocess_done_ms = _now_ms()

            # Acquire a model slot for GPU work.
            item.trace.t_infer_start_ms = _now_ms()
            t_inf0 = _perf_ms()
            model = STATE.model_pool.get()
            try:
                out_bgr = _model_infer(model, STATE.device, img, channels_last=STATE.channels_last)
            finally:
                STATE.model_pool.put(model)
            t_inf1 = _perf_ms()
            item.trace.infer_ms = t_inf1 - t_inf0
            item.trace.t_infer_end_ms = _now_ms()

            t_enc0 = _perf_ms()
            out_bytes = _encode_jpeg_bytes(out_bgr, item.jpeg_quality)
            t_enc1 = _perf_ms()
            item.trace.encode_ms = t_enc1 - t_enc0
            item.trace.t_encode_done_ms = _now_ms()

            if not item.output_path:
                raise RuntimeError("Missing output_path on item")
            out_path = Path(item.output_path)
            out_path.write_bytes(out_bytes)

            rss_after = _get_rss_mb()
            if STATE.device == "cuda":
                gpu_after = torch.cuda.memory_allocated() / (1024 * 1024)
                gpu_peak = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if STATE.gpu_slots == 1 else None
            else:
                gpu_after = 0.0
                gpu_peak = None

            t1 = _perf_ms()

            with STATE.lock:
                item.output_path = str(out_path)
                item.status = "done"
                item.trace.rss_before_mb = rss_before
                item.trace.rss_after_mb = rss_after
                item.trace.gpu_alloc_before_mb = gpu_before
                item.trace.gpu_alloc_after_mb = gpu_after
                item.trace.gpu_peak_alloc_mb = gpu_peak
                item.trace.total_worker_ms = t1 - t0
                item.trace.t_done_ms = _now_ms()

            STATE.push_event(
                item.job_id,
                {
                    "type": "item_done",
                    "item_id": item_id,
                    "tier": tier,
                    "output_url": f"/v1/items/{item_id}",
                    "trace": asdict(item.trace),
                },
            )

            _maybe_finalize_job(item.job_id)

        except Exception as e:
            with STATE.lock:
                item = STATE.items.get(item_id)
                if item:
                    item.status = "error"
                    item.error = str(e)
            if item:
                STATE.push_event(item.job_id, {"type": "item_error", "item_id": item_id, "error": str(e)})
                _maybe_finalize_job(item.job_id)
        finally:
            q.task_done()


def _maybe_finalize_job(job_id: str) -> None:
    with STATE.lock:
        job = STATE.jobs.get(job_id)
        if not job or job.status in {"done", "error"}:
            return
        if job.finalizing:
            return
        item_ids = list(job.items)
        items = [STATE.items.get(i) for i in item_ids]
        if any(i is None for i in items):
            return
        if any(i.status in {"queued", "processing"} for i in items if i is not None):
            return
        if any(i.status == "error" for i in items if i is not None):
            job.status = "error"
            job.error = "One or more items failed"
            STATE.push_event(job_id, {"type": "job_error", "error": job.error})
            return
        # All items done; start finalization (zip) once.
        job.finalizing = True

    # Create zip outside lock (can be slow).
    try:
        job_dir = Path(job.out_dir)
        zip_path = job_dir / "outputs.zip"
        outputs_dir = job_dir / "outputs"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for p in sorted(outputs_dir.glob("*.jpg")):
                zf.write(p, arcname=p.name)
        with STATE.lock:
            job = STATE.jobs.get(job_id)
            if job:
                job.zip_path = str(zip_path)
                job.status = "done"
        STATE.push_event(job_id, {"type": "job_done", "zip_url": f"/v1/jobs/{job_id}/download"})
    except Exception as e:
        with STATE.lock:
            job = STATE.jobs.get(job_id)
            if job:
                job.status = "error"
                job.error = f"Failed to create zip: {e}"
        STATE.push_event(job_id, {"type": "job_error", "error": f"Failed to create zip: {e}"})


async def _save_upload_file(upload: UploadFile, dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _copy() -> int:
        with dest.open("wb") as f:
            shutil.copyfileobj(upload.file, f)
        return dest.stat().st_size

    return await asyncio.to_thread(_copy)


def _extract_zip(zip_path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            if not name:
                continue
            if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            safe = _safe_name(name)
            dest = out_dir / safe
            with zf.open(info) as src, dest.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(dest)
    return extracted


@APP.on_event("startup")
def _startup() -> None:
    if not os.environ.get("CONDA_PREFIX"):
        print("WARNING: CONDA_PREFIX not set; are you running inside the expected conda env?", file=sys.stderr)

    runs_dir = Path(os.environ.get("API_RUNS_DIR", str(SCRIPT_DIR / "api_runs"))).resolve()
    STATE.runs_dir = runs_dir
    STATE.runs_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(
        os.environ.get(
            "TORCH_MODEL_PATH",
            str(SCRIPT_DIR / "BasicSR" / "experiments" / "NAFNet_RealEstate_Fast" / "models" / "net_g_12000.pth"),
        )
    )
    STATE.device = os.environ.get("DEVICE", "cuda")
    STATE.channels_last = os.environ.get("CHANNELS_LAST", "1") not in {"0", "false", "False"}
    STATE.cudnn_benchmark = os.environ.get("CUDNN_BENCHMARK", "0") not in {"0", "false", "False"}
    STATE.gpu_slots = max(1, int(os.environ.get("GPU_SLOTS", "1")))
    warmup_iters = max(0, int(os.environ.get("WARMUP_ITERS", "2")))

    if not model_path.exists():
        raise RuntimeError(f"Model not found: {model_path}")

    print(f"[startup] device={STATE.device} gpu_slots={STATE.gpu_slots} channels_last={STATE.channels_last}", file=sys.stderr)
    print(f"[startup] runs_dir={STATE.runs_dir}", file=sys.stderr)
    print(f"[startup] loading model: {model_path}", file=sys.stderr)
    t0 = time.perf_counter()
    for _ in range(STATE.gpu_slots):
        m = _load_model(
            model_path,
            STATE.device,
            channels_last=STATE.channels_last,
            cudnn_benchmark=STATE.cudnn_benchmark,
        )
        STATE.model_pool.put(m)
    print(f"[startup] loaded {STATE.gpu_slots} model(s) in {time.perf_counter()-t0:.2f}s", file=sys.stderr)

    # Warmup (synthetic sizes) to reduce first-request latency.
    try:
        import numpy as np  # type: ignore

        warm_shapes = {
            "1024": (768, 1024),
            "2048": (1536, 2048),
            "full": (2200, 3300),
        }
        for tier, (h, w) in warm_shapes.items():
            if warmup_iters <= 0:
                continue
            img = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
            for _ in range(warmup_iters):
                # Use one model from pool for warmup, then return.
                m = STATE.model_pool.get()
                try:
                    _ = _model_infer(m, STATE.device, img, channels_last=STATE.channels_last)
                finally:
                    STATE.model_pool.put(m)
        print("[startup] warmup done", file=sys.stderr)
    except Exception as e:
        print(f"[startup] warmup skipped: {e}", file=sys.stderr)

    # Start worker threads.
    for tier in ("1024", "2048", "full"):
        t = threading.Thread(target=_worker_loop, args=(tier,), daemon=True)
        t.start()


@APP.get("/healthz")
def healthz() -> dict[str, Any]:
    with STATE.lock:
        job_count = len(STATE.jobs)
        item_count = len(STATE.items)
        q_sizes = {k: STATE.queues[k].qsize() for k in STATE.queues}
    return {
        "ok": True,
        "time_ms": _now_ms(),
        "device": STATE.device,
        "gpu_slots": STATE.gpu_slots,
        "channels_last": STATE.channels_last,
        "cudnn_benchmark": STATE.cudnn_benchmark,
        "rss_mb": _get_rss_mb(),
        "jobs": job_count,
        "items": item_count,
        "queue_sizes": q_sizes,
    }


@APP.get("/v1/tiers")
def tiers() -> dict[str, Any]:
    return {
        "tiers": [
            {"tier": "1024", "long_side_max": 1024},
            {"tier": "2048", "long_side_max": 2048},
            {"tier": "full", "long_side_max": None},
        ]
    }


@APP.post("/v1/jobs")
async def create_job(
    tier: Tier = Form("1024"),
    jpeg_quality: int = Form(95),
    images: list[UploadFile] | None = File(None),
    zip_file: UploadFile | None = File(None),
) -> JSONResponse:
    if images is None:
        images = []
    if not images and zip_file is None:
        raise HTTPException(status_code=400, detail="Provide `images` or `zip_file`.")
    if tier not in {"1024", "2048", "full"}:
        raise HTTPException(status_code=400, detail="Invalid tier.")
    jpeg_quality = int(jpeg_quality)
    if jpeg_quality < 50 or jpeg_quality > 100:
        raise HTTPException(status_code=400, detail="jpeg_quality must be 50..100")

    job_id = uuid.uuid4().hex
    job_dir = STATE.runs_dir / job_id
    inputs_dir = job_dir / "inputs"
    outputs_dir = job_dir / "outputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    created_ms = _now_ms()
    job = Job(
        job_id=job_id,
        created_ms=created_ms,
        tier=tier,
        jpeg_quality=jpeg_quality,
        out_dir=str(job_dir),
    )

    # Save uploads to disk.
    saved_paths: list[Path] = []
    t_req0 = _perf_ms()
    if zip_file is not None:
        zip_path = inputs_dir / "upload.zip"
        await _save_upload_file(zip_file, zip_path)
        saved_paths = _extract_zip(zip_path, inputs_dir)
    else:
        for up in images:
            dest = inputs_dir / _safe_name(up.filename or "image.jpg")
            await _save_upload_file(up, dest)
            saved_paths.append(dest)
    t_req1 = _perf_ms()

    if not saved_paths:
        raise HTTPException(status_code=400, detail="No images found in upload.")

    # Create items + enqueue.
    with STATE.lock:
        STATE.jobs[job_id] = job

    STATE.push_event(job_id, {"type": "job_created", "tier": tier, "count": len(saved_paths)})

    for p in saved_paths:
        item_id = uuid.uuid4().hex
        out_name = f"{item_id}_{Path(p.name).stem}.jpg"
        item = Item(
            item_id=item_id,
            job_id=job_id,
            filename=p.name,
            tier=tier,
            jpeg_quality=jpeg_quality,
            input_path=str(p),
            output_path=str(outputs_dir / out_name),
        )
        item.trace.t_request_received_ms = created_ms
        item.trace.t_saved_ms = _now_ms()
        item.trace.t_enqueued_ms = _now_ms()
        with STATE.lock:
            STATE.items[item_id] = item
            STATE.jobs[job_id].items.append(item_id)
        STATE.queues[tier].put(item_id)
        STATE.push_event(job_id, {"type": "item_enqueued", "item_id": item_id, "filename": p.name, "tier": tier})

    # Save request timing into job events.
    STATE.push_event(
        job_id,
        {
            "type": "job_upload_saved",
            "save_ms": (t_req1 - t_req0),
            "rss_mb": _get_rss_mb(),
        },
    )

    with STATE.lock:
        job = STATE.jobs[job_id]
        items = [STATE.items[i] for i in job.items if i in STATE.items]
        payload = _public_job(job, items)
    return JSONResponse(payload)


@APP.get("/v1/jobs/{job_id}")
def get_job(job_id: str) -> JSONResponse:
    with STATE.lock:
        job = STATE.jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        items = [STATE.items[i] for i in job.items if i in STATE.items]
        payload = _public_job(job, items)
    return JSONResponse(payload)


@APP.get("/v1/jobs/{job_id}/download")
def download_job(job_id: str) -> FileResponse:
    with STATE.lock:
        job = STATE.jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        if job.status != "done" or not job.zip_path:
            raise HTTPException(status_code=409, detail="job not ready")
        zip_path = job.zip_path
    return FileResponse(zip_path, media_type="application/zip", filename=f"{job_id}_{job.tier}.zip")


@APP.get("/v1/items/{item_id}")
def get_item(item_id: str) -> FileResponse:
    with STATE.lock:
        item = STATE.items.get(item_id)
        if not item:
            raise HTTPException(status_code=404, detail="item not found")
        if item.status != "done" or not item.output_path or not Path(item.output_path).exists():
            raise HTTPException(status_code=409, detail=f"item not ready (status={item.status})")
        path = item.output_path
        filename = f"{Path(item.filename).stem}.jpg"
    return FileResponse(path, media_type="image/jpeg", filename=filename)


@APP.get("/v1/jobs/{job_id}/events")
async def job_events(job_id: str, since: int = 0) -> StreamingResponse:
    # SSE stream of job events (one client is fine for demo). Use `since` as an index cursor.
    with STATE.lock:
        if job_id not in STATE.jobs:
            raise HTTPException(status_code=404, detail="job not found")

    async def _gen() -> Any:
        cursor = max(0, int(since))
        last_keepalive = time.time()
        while True:
            with STATE.lock:
                job = STATE.jobs.get(job_id)
                if not job:
                    break
                total = len(job.events)
                events = job.events[cursor:]
                job_done = job.status in {"done", "error"}
            for ev in events:
                cursor += 1
                ev_json = json.dumps(ev, separators=(",", ":"))
                ev_type = ev.get("type", "message")
                yield f"event: {ev_type}\n"
                yield f"data: {ev_json}\n\n"
            if job_done and cursor >= total:
                yield "event: end\ndata: {}\n\n"
                break
            await asyncio.sleep(0.25)
            if time.time() - last_keepalive > 15.0:
                yield ": keepalive\n\n"
                last_keepalive = time.time()

    return StreamingResponse(_gen(), media_type="text/event-stream")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(APP, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
