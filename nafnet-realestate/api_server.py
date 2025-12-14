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
import hashlib
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
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel


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
    model_id: str
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
    model_id: str
    tier: Tier
    jpeg_quality: int
    items: list[str] = field(default_factory=list)  # item_ids
    status: JobStatus = "queued"
    error: str | None = None
    out_dir: str = ""
    zip_path: str | None = None
    finalizing: bool = False
    events: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ModelEntry:
    model_id: str
    checkpoint_path: str
    filename: str
    loaded: bool = False
    loading: bool = False
    error: str | None = None
    load_time_s: float | None = None
    loaded_ms: int | None = None
    last_used_ms: int | None = None
    pool: "queue.Queue[Any]" = field(default_factory=queue.Queue, repr=False)


class ServerState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.model_load_cond = threading.Condition(self.lock)
        self.jobs: dict[str, Job] = {}
        self.items: dict[str, Item] = {}
        self.queues: dict[Tier, "queue.Queue[str]"] = {
            "1024": queue.Queue(),
            "2048": queue.Queue(),
            "full": queue.Queue(),
        }
        self.models: dict[str, ModelEntry] = {}
        self.model_aliases: dict[str, str] = {}
        self.default_model_id: str = ""
        self.models_dirs: list[Path] = []
        self.device: str = "cuda"
        self.channels_last: bool = True
        self.cudnn_benchmark: bool = False
        self.gpu_slots: int = 1
        self.warmup_iters: int = 2
        self.infer_semaphore = threading.Semaphore(1)
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


class ItemTraceModel(BaseModel):
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

    decode_ms: float | None = None
    preprocess_ms: float | None = None
    infer_ms: float | None = None
    encode_ms: float | None = None
    total_worker_ms: float | None = None

    rss_before_mb: float | None = None
    rss_after_mb: float | None = None
    gpu_alloc_before_mb: float | None = None
    gpu_alloc_after_mb: float | None = None
    gpu_peak_alloc_mb: float | None = None


class ItemPublicModel(BaseModel):
    item_id: str
    job_id: str
    model_id: str
    filename: str
    tier: Tier
    jpeg_quality: int
    status: ItemStatus
    error: str | None = None
    width: int | None = None
    height: int | None = None
    megapixels: float | None = None
    trace: ItemTraceModel
    queue_wait_ms: int | None = None
    output_url: str | None = None


class JobPublicModel(BaseModel):
    job_id: str
    model_id: str
    tier: Tier
    jpeg_quality: int
    status: JobStatus
    error: str | None = None
    items_total: int
    items_done: int
    items_error: int
    items: list[ItemPublicModel]
    events_url: str
    download_url: str
    zip_url: str | None = None
    events_count: int


class HealthzModel(BaseModel):
    ok: bool
    time_ms: int
    device: str
    gpu_slots: int
    channels_last: bool
    cudnn_benchmark: bool
    rss_mb: float
    jobs: int
    items: int
    queue_sizes: dict[str, int]


class TiersModel(BaseModel):
    tiers: list[dict[str, Any]]


class ModelPublicModel(BaseModel):
    model_id: str
    filename: str
    loaded: bool
    error: str | None = None
    load_time_s: float | None = None


class ModelsListModel(BaseModel):
    default_model: str
    aliases: dict[str, str]
    models: list[ModelPublicModel]


def _public_item(item: Item) -> dict[str, Any]:
    trace = asdict(item.trace)
    queue_wait_ms = None
    if item.trace.t_enqueued_ms is not None and item.trace.t_worker_start_ms is not None:
        queue_wait_ms = item.trace.t_worker_start_ms - item.trace.t_enqueued_ms
    return {
        "item_id": item.item_id,
        "job_id": item.job_id,
        "model_id": item.model_id,
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
        "model_id": job.model_id,
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


def _short_hash(text: str) -> str:
    return hashlib.blake2s(text.encode("utf-8"), digest_size=4).hexdigest()


def _models_dirs_from_env() -> list[Path]:
    raw = os.environ.get("MODELS_DIRS")
    if raw is None:
        raw = str(Path.home() / "models")
    dirs: list[Path] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        dirs.append(Path(part).expanduser())
    # Convenience: also check for a local `nafnet-realestate/models/` folder.
    local = (SCRIPT_DIR / "models").resolve()
    if local not in dirs:
        dirs.append(local)
    return dirs


def _discover_model_files(dirs: list[Path]) -> list[Path]:
    found: list[Path] = []
    for d in dirs:
        try:
            if not d.exists() or not d.is_dir():
                continue
        except Exception:
            continue
        for pat in ("*.pth", "*.pt"):
            found.extend(sorted([p for p in d.glob(pat) if p.is_file()]))
    # Deduplicate by resolved path.
    uniq: dict[str, Path] = {}
    for p in found:
        try:
            rp = str(p.resolve())
        except Exception:
            rp = str(p)
        uniq[rp] = p
    return sorted(uniq.values(), key=lambda p: (p.name, str(p)))


def _register_model_path(path: Path, *, preferred_id: str | None = None) -> str:
    path = path.expanduser()
    try:
        path = path.resolve()
    except Exception:
        pass
    if not path.exists():
        raise RuntimeError(f"Model not found: {path}")
    if not path.is_file():
        raise RuntimeError(f"Model path is not a file: {path}")

    model_id = preferred_id or _safe_name(path.stem)

    with STATE.lock:
        existing = STATE.models.get(model_id)
        if existing and Path(existing.checkpoint_path).resolve() != path:
            model_id = f"{model_id}_{_short_hash(str(path))}"
            existing = STATE.models.get(model_id)
        if existing:
            return existing.model_id
        entry = ModelEntry(model_id=model_id, checkpoint_path=str(path), filename=path.name)
        STATE.models[model_id] = entry
        return model_id


def _public_models_list() -> dict[str, Any]:
    with STATE.lock:
        models = [
            {
                "model_id": m.model_id,
                "filename": m.filename,
                "loaded": m.loaded,
                "error": m.error,
                "load_time_s": m.load_time_s,
            }
            for m in sorted(STATE.models.values(), key=lambda e: e.model_id)
        ]
        return {
            "default_model": STATE.default_model_id,
            "aliases": dict(STATE.model_aliases),
            "models": models,
        }


def _is_allowed_model_path(path: Path) -> bool:
    try:
        rp = path.resolve()
    except Exception:
        rp = path
    with STATE.lock:
        dirs = list(STATE.models_dirs)
        # Always allow the already-registered model paths.
        registered = {Path(e.checkpoint_path) for e in STATE.models.values()}
    for p in registered:
        try:
            if rp == p.resolve():
                return True
        except Exception:
            if str(rp) == str(p):
                return True
    for d in dirs:
        try:
            if rp.is_relative_to(d.resolve()):
                return True
        except Exception:
            continue
    return False


def _resolve_model_id(requested: str) -> str:
    key = (requested or "").strip()
    if not key:
        key = "default"

    # Allow direct file path within allowed dirs (e.g. ~/models/foo.pth).
    if "/" in key or key.startswith("~"):
        p = Path(key).expanduser()
        if not _is_allowed_model_path(p):
            raise HTTPException(status_code=400, detail="Model path is not in an allowed models dir.")
        model_id = _register_model_path(p)
        return model_id

    with STATE.lock:
        if key in STATE.model_aliases:
            key = STATE.model_aliases[key]
        if key in STATE.models:
            return key
        # Match by filename if unique.
        matches = [m.model_id for m in STATE.models.values() if m.filename == key]

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise HTTPException(status_code=400, detail=f"Ambiguous model name '{requested}'. Use model_id instead.")
    raise HTTPException(status_code=400, detail=f"Unknown model '{requested}'. See GET /v1/models.")


def _ensure_model_loaded(model_id: str) -> None:
    with STATE.lock:
        entry = STATE.models.get(model_id)
        if not entry:
            raise HTTPException(status_code=404, detail="model not found")
        if entry.loaded:
            return
        if entry.error:
            raise HTTPException(status_code=500, detail=f"model load failed: {entry.error}")
        if entry.loading:
            # Wait for the other loader to finish.
            while entry.loading:
                STATE.model_load_cond.wait(timeout=0.5)
            if entry.loaded:
                return
            raise HTTPException(status_code=500, detail=f"model load failed: {entry.error or 'unknown error'}")
        entry.loading = True
        checkpoint_path = Path(entry.checkpoint_path)
        channels_last = STATE.channels_last
        cudnn_benchmark = STATE.cudnn_benchmark
        device = STATE.device
        gpu_slots = STATE.gpu_slots
        warmup_iters = STATE.warmup_iters

    # Load outside the lock (can take seconds).
    models: list[Any] = []
    t0 = time.perf_counter()
    err: str | None = None
    try:
        for _ in range(gpu_slots):
            m = _load_model(checkpoint_path, device, channels_last=channels_last, cudnn_benchmark=cudnn_benchmark)
            models.append(m)
        # Publish models into the pool.
        with STATE.lock:
            entry = STATE.models.get(model_id)
            if not entry:
                raise RuntimeError("model entry disappeared")
            # If a previous failed load left items in the pool, clear it.
            while not entry.pool.empty():
                try:
                    entry.pool.get_nowait()
                except Exception:
                    break
            for m in models:
                entry.pool.put(m)

        # Warmup (best-effort) to reduce first-request latency.
        if warmup_iters > 0:
            try:
                import numpy as np  # type: ignore

                warm_shapes = {
                    "1024": (768, 1024),
                    "2048": (1536, 2048),
                    "full": (2200, 3300),
                }
                for _tier, (h, w) in warm_shapes.items():
                    img = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
                    for _ in range(warmup_iters):
                        if device == "cuda":
                            STATE.infer_semaphore.acquire()
                        try:
                            with STATE.lock:
                                entry = STATE.models.get(model_id)
                                if not entry:
                                    raise RuntimeError("model entry disappeared")
                                entry.last_used_ms = _now_ms()
                            m = entry.pool.get()
                            try:
                                _ = _model_infer(m, device, img, channels_last=channels_last)
                            finally:
                                entry.pool.put(m)
                        finally:
                            if device == "cuda":
                                STATE.infer_semaphore.release()
            except Exception:
                pass
    except Exception as e:
        err = str(e)
    load_time_s = time.perf_counter() - t0

    with STATE.lock:
        entry = STATE.models.get(model_id)
        if entry:
            entry.loading = False
            entry.load_time_s = load_time_s
            entry.loaded_ms = _now_ms()
            if err:
                entry.error = err
            else:
                entry.loaded = True
                entry.error = None
        STATE.model_load_cond.notify_all()

    if err:
        raise HTTPException(status_code=500, detail=f"Failed to load model '{model_id}': {err}")


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

            # Acquire a global infer slot + a model instance.
            item.trace.t_infer_start_ms = _now_ms()
            t_inf0 = _perf_ms()
            with STATE.lock:
                entry = STATE.models.get(item.model_id)
                if entry:
                    entry.last_used_ms = _now_ms()
            if not entry or not entry.loaded:
                raise RuntimeError(f"Model not loaded: {item.model_id}")

            if STATE.device == "cuda":
                STATE.infer_semaphore.acquire()
            try:
                model = entry.pool.get()
                try:
                    out_bgr = _model_infer(model, STATE.device, img, channels_last=STATE.channels_last)
                finally:
                    entry.pool.put(model)
            finally:
                if STATE.device == "cuda":
                    STATE.infer_semaphore.release()
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

    default_model_path = Path(
        os.environ.get(
            "TORCH_MODEL_PATH",
            str(SCRIPT_DIR / "BasicSR" / "experiments" / "NAFNet_RealEstate_Fast" / "models" / "net_g_12000.pth"),
        )
    )
    STATE.device = os.environ.get("DEVICE", "cuda")
    STATE.channels_last = os.environ.get("CHANNELS_LAST", "1") not in {"0", "false", "False"}
    STATE.cudnn_benchmark = os.environ.get("CUDNN_BENCHMARK", "0") not in {"0", "false", "False"}
    STATE.gpu_slots = max(1, int(os.environ.get("GPU_SLOTS", "1")))
    STATE.warmup_iters = max(0, int(os.environ.get("WARMUP_ITERS", "2")))
    STATE.infer_semaphore = threading.Semaphore(STATE.gpu_slots)

    if not default_model_path.exists():
        raise RuntimeError(f"Model not found: {default_model_path}")

    # Register models (default + any in MODELS_DIRS / local models folder).
    STATE.models_dirs = _models_dirs_from_env()
    discovered = _discover_model_files(STATE.models_dirs)

    default_id = _register_model_path(default_model_path, preferred_id=_safe_name(default_model_path.stem))
    with STATE.lock:
        STATE.default_model_id = default_id
        STATE.model_aliases.setdefault("default", default_id)

    for p in discovered:
        try:
            if p.resolve() == default_model_path.resolve():
                continue
        except Exception:
            if str(p) == str(default_model_path):
                continue
        try:
            _register_model_path(p)
        except Exception:
            continue

    print(
        f"[startup] device={STATE.device} gpu_slots={STATE.gpu_slots} channels_last={STATE.channels_last} cudnn_benchmark={STATE.cudnn_benchmark}",
        file=sys.stderr,
    )
    print(f"[startup] runs_dir={STATE.runs_dir}", file=sys.stderr)
    print(f"[startup] default_model={STATE.default_model_id} (from TORCH_MODEL_PATH)", file=sys.stderr)
    print(f"[startup] discovered_models={len(discovered)} dirs={','.join(str(d) for d in STATE.models_dirs)}", file=sys.stderr)
    print(f"[startup] loading default model '{STATE.default_model_id}' ...", file=sys.stderr)
    _ensure_model_loaded(STATE.default_model_id)
    print("[startup] default model ready", file=sys.stderr)

    # Start worker threads.
    for tier in ("1024", "2048", "full"):
        t = threading.Thread(target=_worker_loop, args=(tier,), daemon=True)
        t.start()


@APP.get("/healthz", response_model=HealthzModel)
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


@APP.get("/v1/tiers", response_model=TiersModel)
def tiers() -> dict[str, Any]:
    return {
        "tiers": [
            {"tier": "1024", "long_side_max": 1024},
            {"tier": "2048", "long_side_max": 2048},
            {"tier": "full", "long_side_max": None},
        ]
    }


@APP.get("/v1/models", response_model=ModelsListModel)
def list_models() -> dict[str, Any]:
    return _public_models_list()


@APP.post("/v1/models/rescan", response_model=ModelsListModel)
def rescan_models() -> dict[str, Any]:
    # Re-scan configured model dirs and register new model files (does not unload existing).
    with STATE.lock:
        dirs = list(STATE.models_dirs)
    for p in _discover_model_files(dirs):
        try:
            _register_model_path(p)
        except Exception:
            continue
    return _public_models_list()


@APP.post("/v1/models/{model}/load", response_model=ModelPublicModel)
def load_model(model: str) -> dict[str, Any]:
    model_id = _resolve_model_id(model)
    _ensure_model_loaded(model_id)
    with STATE.lock:
        entry = STATE.models.get(model_id)
        if not entry:
            raise HTTPException(status_code=404, detail="model not found")
        return {
            "model_id": entry.model_id,
            "filename": entry.filename,
            "loaded": entry.loaded,
            "error": entry.error,
            "load_time_s": entry.load_time_s,
        }


@APP.post("/v1/jobs", response_model=JobPublicModel)
async def create_job(
    model: str = Form("default"),
    tier: Tier = Form("1024"),
    jpeg_quality: int = Form(95),
    images: list[UploadFile] | None = File(None),
    zip_file: UploadFile | None = File(None),
) -> dict[str, Any]:
    if images is None:
        images = []
    if not images and zip_file is None:
        raise HTTPException(status_code=400, detail="Provide `images` or `zip_file`.")
    if tier not in {"1024", "2048", "full"}:
        raise HTTPException(status_code=400, detail="Invalid tier.")
    jpeg_quality = int(jpeg_quality)
    if jpeg_quality < 50 or jpeg_quality > 100:
        raise HTTPException(status_code=400, detail="jpeg_quality must be 50..100")

    model_id = _resolve_model_id(model)
    _ensure_model_loaded(model_id)

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
        model_id=model_id,
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

    STATE.push_event(job_id, {"type": "job_created", "tier": tier, "model_id": model_id, "count": len(saved_paths)})

    for p in saved_paths:
        item_id = uuid.uuid4().hex
        out_name = f"{item_id}_{Path(p.name).stem}.jpg"
        item = Item(
            item_id=item_id,
            job_id=job_id,
            model_id=model_id,
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
    return payload


@APP.get("/v1/jobs/{job_id}", response_model=JobPublicModel)
def get_job(job_id: str) -> dict[str, Any]:
    with STATE.lock:
        job = STATE.jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        items = [STATE.items[i] for i in job.items if i in STATE.items]
        payload = _public_job(job, items)
    return payload


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

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        # Helps with some reverse proxies that buffer streaming responses.
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(_gen(), media_type="text/event-stream", headers=headers)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(APP, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
