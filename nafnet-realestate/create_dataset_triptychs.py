#!/usr/bin/env python3
"""
Create dataset triptychs: (LQ input | model output | GT).

Supports:
  - ONNXRuntime inference (default) for validating exported ONNX ("Onyx") behavior
  - Optional PyTorch inference for comparing against native weights

Also logs per-image timing + RAM usage to help diagnose performance and memory spikes.

Example (PyTorch):
  conda activate <your-env>
  python create_dataset_triptychs.py --backend torch --device cuda

Example (ONNX):
  conda activate <your-env>
  python create_dataset_triptychs.py --backend onnx
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Event, Thread
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent


def _is_image_path(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png"}


def _now_compact() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _get_rss_mb() -> float:
    try:
        import psutil  # type: ignore
    except Exception:
        psutil = None

    if psutil is not None:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    # Fallback: ru_maxrss is max RSS, not current, but better than nothing.
    try:
        import resource

        ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KB, macOS reports bytes.
        if sys.platform == "darwin":
            return ru_maxrss / (1024 * 1024)
        return ru_maxrss / 1024
    except Exception:
        return float("nan")


def _require_conda(no_conda_check: bool, expected_env: str | None) -> None:
    if no_conda_check:
        return

    conda_prefix = os.environ.get("CONDA_PREFIX")
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if not conda_prefix:
        raise SystemExit(
            "ERROR: This script must be run inside your conda environment.\n"
            "  - Activate it first (example): `conda activate nafnet`\n"
            "  - Or bypass with: `--no-conda-check`"
        )
    if expected_env and conda_env and conda_env != expected_env:
        print(
            f"WARNING: Expected conda env '{expected_env}', but active env is '{conda_env}'.",
            file=sys.stderr,
        )


def _collect_pairs(dataset_root: Path, splits: Iterable[str]) -> list[tuple[str, Path, Path]]:
    pairs: list[tuple[str, Path, Path]] = []
    missing = 0

    for split in splits:
        lq_dir = dataset_root / split / "lq"
        gt_dir = dataset_root / split / "gt"
        if not lq_dir.exists() or not gt_dir.exists():
            raise SystemExit(f"ERROR: Missing dataset dirs for split '{split}': {lq_dir} / {gt_dir}")

        gt_by_stem = {p.stem: p for p in gt_dir.iterdir() if p.is_file() and _is_image_path(p)}
        lq_paths = sorted([p for p in lq_dir.iterdir() if p.is_file() and _is_image_path(p)])

        for lq_path in lq_paths:
            gt_path = gt_by_stem.get(lq_path.stem)
            if gt_path is None:
                missing += 1
                continue
            pairs.append((split, lq_path, gt_path))

    if missing:
        print(f"WARNING: {missing} LQ files had no matching GT (skipped).", file=sys.stderr)
    return pairs


def _pad_to_multiple_of_8(img_bgr: Any) -> tuple[Any, tuple[int, int], tuple[int, int]]:
    import cv2  # type: ignore

    h, w = img_bgr.shape[:2]
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h == 0 and pad_w == 0:
        return img_bgr, (pad_h, pad_w), (h, w)
    padded = cv2.copyMakeBorder(img_bgr, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    return padded, (pad_h, pad_w), (h, w)


def _bgr_uint8_to_onnx_input(img_bgr_uint8: Any) -> Any:
    import numpy as np  # type: ignore

    img_rgb = img_bgr_uint8[:, :, ::-1].copy()
    img_rgb = img_rgb.astype(np.float32) / 255.0
    chw = np.transpose(img_rgb, (2, 0, 1))
    nchw = chw[None, ...]
    return np.ascontiguousarray(nchw)


def _onnx_output_to_bgr_uint8(output_nchw: Any, orig_hw: tuple[int, int]) -> Any:
    import numpy as np  # type: ignore

    out = output_nchw
    if isinstance(out, list):
        out = out[0]
    out = np.asarray(out)
    if out.ndim == 4:
        out = out[0]
    out = np.clip(out, 0.0, 1.0)
    out = (np.transpose(out, (1, 2, 0)) * 255.0).astype(np.uint8)
    out_bgr = out[:, :, ::-1]
    h, w = orig_hw
    return out_bgr[:h, :w]


def _psnr_uint8(img1: Any, img2: Any) -> float:
    import numpy as np  # type: ignore

    a = img1.astype(np.float32) / 255.0
    b = img2.astype(np.float32) / 255.0
    mse = float(np.mean((a - b) ** 2))
    if mse == 0.0:
        return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def _mean_bgr(img: Any) -> tuple[float, float, float]:
    import numpy as np  # type: ignore

    mean = np.mean(img.astype(np.float32), axis=(0, 1))
    return float(mean[0]), float(mean[1]), float(mean[2])


def _stitch_triptych(
    img_lq: Any,
    img_pred: Any,
    img_gt: Any,
    *,
    pred_label: str,
    divider_px: int = 6,
    jpeg_quality: int = 90,
) -> Any:
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    if img_lq is None or img_pred is None or img_gt is None:
        raise ValueError("One or more images are None")

    base_h = img_lq.shape[0]

    def _resize_to_h(img: Any) -> Any:
        h, w = img.shape[:2]
        if h == base_h:
            return img
        scale = base_h / h
        new_w = max(1, int(round(w * scale)))
        return cv2.resize(img, (new_w, base_h), interpolation=cv2.INTER_AREA)

    img_pred = _resize_to_h(img_pred)
    img_gt = _resize_to_h(img_gt)

    divider = np.ones((base_h, divider_px, 3), dtype=np.uint8) * 128
    stitched = np.hstack([img_lq, divider, img_pred, divider, img_gt])

    # Labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0
    thickness = 4

    def _label_at(x_offset: int, text: str) -> None:
        x = int(x_offset + 30)
        y = 80
        cv2.putText(stitched, text, (x, y), font, font_scale, (0, 0, 0), thickness + 6)
        cv2.putText(stitched, text, (x, y), font, font_scale, (255, 255, 255), thickness)

    # Put labels on separate copies so we don't modify the originals on disk (but we do for stitched).
    # Label positions depend on each subimage offset.
    x0 = 0
    x1 = img_lq.shape[1] + divider_px
    x2 = x1 + img_pred.shape[1] + divider_px

    _label_at(x0, "INPUT (LQ)")
    _label_at(x1, pred_label)
    _label_at(x2, "GT")

    return stitched, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]


def _sample_rss_peak_mb(stop: Event, interval_sec: float, out_max: list[float]) -> None:
    peak = -1.0
    while not stop.is_set():
        rss = _get_rss_mb()
        if rss == rss:  # not NaN
            peak = max(peak, rss)
        time.sleep(interval_sec)
    out_max.append(peak)


def _run_with_rss_sampler(fn: Any, interval_sec: float) -> tuple[Any, float]:
    if interval_sec <= 0:
        return fn(), float("nan")
    stop = Event()
    peak_list: list[float] = []
    sampler = Thread(target=_sample_rss_peak_mb, args=(stop, interval_sec, peak_list), daemon=True)
    sampler.start()
    try:
        result = fn()
    finally:
        stop.set()
        sampler.join(timeout=1.0)
    peak = peak_list[0] if peak_list else float("nan")
    return result, peak


@dataclass(frozen=True)
class Row:
    backend: str
    split: str
    filename: str
    width: int
    height: int
    megapixels: float
    preprocess_s: float
    inference_s: float
    postprocess_s: float
    total_s: float
    rss_before_mb: float
    rss_after_mb: float
    rss_peak_during_infer_mb: float
    psnr_pred_gt: float
    mean_lq_b: float
    mean_lq_g: float
    mean_lq_r: float
    mean_pred_b: float
    mean_pred_g: float
    mean_pred_r: float
    mean_gt_b: float
    mean_gt_g: float
    mean_gt_r: float


def _load_onnx_session(model_path: Path, provider: str, num_threads: int | None) -> tuple[Any, str, str, list[str]]:
    try:
        import onnxruntime as ort  # type: ignore
    except Exception as e:
        raise SystemExit(
            "ERROR: onnxruntime is not available.\n"
            "  - Make sure you activated the correct conda env.\n"
            f"  - Import error: {e}"
        )

    available = ort.get_available_providers()

    sess_options = ort.SessionOptions()
    if num_threads is not None and num_threads > 0:
        sess_options.intra_op_num_threads = int(num_threads)
        sess_options.inter_op_num_threads = int(num_threads)

    if provider == "auto":
        provider = "cuda" if "CUDAExecutionProvider" in available else "cpu"

    providers: list[str]
    if provider == "cpu":
        providers = ["CPUExecutionProvider"]
    elif provider == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise SystemExit(
                "ERROR: You requested `--onnx-provider cuda` but this Python environment does not have "
                "CUDAExecutionProvider enabled for ONNX Runtime.\n"
                f"  Available providers: {available}\n"
                "  Notes:\n"
                "    - On Linux aarch64, `onnxruntime-gpu` wheels are often unavailable.\n"
                "    - Fastest workaround is usually PyTorch CUDA; otherwise build ONNX Runtime with CUDA "
                "or use TensorRT directly.\n"
            )
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        raise SystemExit(f"ERROR: Unknown ONNX provider '{provider}'")

    session = ort.InferenceSession(str(model_path), sess_options=sess_options, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session, input_name, output_name, available


def _onnx_infer(
    session: Any,
    input_name: str,
    output_name: str,
    img_bgr_uint8: Any,
    *,
    rss_interval_sec: float,
) -> tuple[Any, float, float, float, float]:
    import cv2  # type: ignore

    t0 = time.perf_counter()
    padded, _pad_hw, orig_hw = _pad_to_multiple_of_8(img_bgr_uint8)
    onnx_in = _bgr_uint8_to_onnx_input(padded)
    t1 = time.perf_counter()
    infer_start = time.perf_counter()
    outputs, rss_peak = _run_with_rss_sampler(
        lambda: session.run([output_name], {input_name: onnx_in}),
        interval_sec=rss_interval_sec,
    )
    infer_end = time.perf_counter()
    pred_bgr = _onnx_output_to_bgr_uint8(outputs[0], orig_hw)
    # Ensure final output matches original dimensions exactly
    if pred_bgr.shape[:2] != (orig_hw[0], orig_hw[1]):
        pred_bgr = cv2.resize(pred_bgr, (orig_hw[1], orig_hw[0]), interpolation=cv2.INTER_LINEAR)
    t3 = time.perf_counter()
    return pred_bgr, (t1 - t0), (infer_end - infer_start), (t3 - infer_end), rss_peak


def _load_torch_model(model_path: Path, device: str) -> Any:
    try:
        import torch  # type: ignore
    except Exception as e:
        raise SystemExit(
            "ERROR: torch is not available.\n"
            "  - Make sure you activated the correct conda env.\n"
            f"  - Import error: {e}"
        )

    sys.path.insert(0, str(SCRIPT_DIR / "BasicSR"))
    try:
        from basicsr.archs.NAFNet_arch import NAFNet  # type: ignore
    except Exception as e:
        raise SystemExit(
            "ERROR: Could not import BasicSR/NAFNet.\n"
            "  - Make sure BasicSR is installed (or in repo) and conda env is active.\n"
            f"  - Import error: {e}"
        )

    model = NAFNet(
        img_channel=3,
        width=32,
        middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2],
    )

    torch_device = torch.device(device)
    checkpoint = torch.load(str(model_path), map_location=torch_device, weights_only=True)
    if isinstance(checkpoint, dict) and "params" in checkpoint:
        model.load_state_dict(checkpoint["params"])
    elif isinstance(checkpoint, dict) and "params_ema" in checkpoint:
        model.load_state_dict(checkpoint["params_ema"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(torch_device)
    model.eval()
    return model


def _torch_infer(
    model: Any,
    device: str,
    img_bgr_uint8: Any,
    *,
    rss_interval_sec: float,
) -> tuple[Any, float, float, float, float]:
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    import torch  # type: ignore

    t0 = time.perf_counter()
    padded, _pad_hw, orig_hw = _pad_to_multiple_of_8(img_bgr_uint8)

    # BGR -> RGB, normalize to [0, 1], HWC -> CHW -> NCHW
    img_rgb = padded[:, :, ::-1].copy()
    img_rgb = img_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    t1 = time.perf_counter()

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    infer_start = time.perf_counter()

    def _forward() -> Any:
        with torch.no_grad():
            return model(tensor)

    out, rss_peak = _run_with_rss_sampler(
        _forward,
        interval_sec=rss_interval_sec,
    )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    infer_end = time.perf_counter()

    out = out.squeeze(0).detach().cpu().clamp(0, 1).numpy()
    out = (np.transpose(out, (1, 2, 0)) * 255.0).astype(np.uint8)
    out_bgr = out[:, :, ::-1]
    out_bgr = out_bgr[: orig_hw[0], : orig_hw[1]]
    if out_bgr.shape[:2] != (orig_hw[0], orig_hw[1]):
        out_bgr = cv2.resize(out_bgr, (orig_hw[1], orig_hw[0]), interpolation=cv2.INTER_LINEAR)
    t3 = time.perf_counter()
    return out_bgr, (t1 - t0), (infer_end - infer_start), (t3 - infer_end), rss_peak


def _write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _summarize(rows: list[Row], *, baseline_rss_mb: float) -> dict[str, Any]:
    if not rows:
        return {}
    by_split: dict[str, list[Row]] = {}
    for r in rows:
        by_split.setdefault(r.split, []).append(r)

    def _stats(values: list[float]) -> dict[str, float]:
        values = [v for v in values if v == v]  # drop NaN
        if not values:
            return {}
        values_sorted = sorted(values)
        n = len(values_sorted)
        p50 = values_sorted[n // 2]
        p90 = values_sorted[int(n * 0.9) - 1 if n > 1 else 0]
        return {
            "count": float(n),
            "min": float(values_sorted[0]),
            "p50": float(p50),
            "p90": float(p90),
            "max": float(values_sorted[-1]),
            "mean": float(sum(values_sorted) / n),
        }

    summary: dict[str, Any] = {
        "backend": rows[0].backend,
        "total_images": len(rows),
        "baseline_rss_mb": baseline_rss_mb,
        "splits": {},
    }
    for split, split_rows in by_split.items():
        summary["splits"][split] = {
            "images": len(split_rows),
            "timing_total_s": _stats([r.total_s for r in split_rows]),
            "timing_inference_s": _stats([r.inference_s for r in split_rows]),
            "rss_before_mb": _stats([r.rss_before_mb for r in split_rows]),
            "rss_after_mb": _stats([r.rss_after_mb for r in split_rows]),
            "rss_peak_during_infer_mb": _stats([r.rss_peak_during_infer_mb for r in split_rows]),
            "psnr_pred_gt": _stats([r.psnr_pred_gt for r in split_rows]),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run dataset inference and create (LQ | PRED | GT) triptychs with RAM/time logs."
    )
    parser.add_argument(
        "--backend",
        choices=["onnx", "torch"],
        default="torch",
        help="Inference backend (default: torch).",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=SCRIPT_DIR / "datasets" / "realestate",
        help="Dataset root containing train/val folders with lq/gt subfolders.",
    )
    parser.add_argument(
        "--splits",
        default="train,val",
        help="Comma-separated splits to process (default: train,val).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=SCRIPT_DIR / "dataset_triptychs",
        help="Output directory for predictions + triptychs + logs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N images (0 = all).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images where the triptych output already exists.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=90,
        help="JPEG quality for outputs (default: 90).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup runs on the first image (default: 1).",
    )
    parser.add_argument(
        "--ram-sample-ms",
        type=int,
        default=25,
        help="RSS sampling interval during inference, in ms (default: 25; 0 disables).",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="Print progress every N images (default: 10).",
    )
    parser.add_argument(
        "--no-conda-check",
        action="store_true",
        help="Bypass conda environment requirement check.",
    )
    parser.add_argument(
        "--expected-conda-env",
        default=None,
        help="If set, warn when CONDA_DEFAULT_ENV differs.",
    )

    # ONNX options
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=SCRIPT_DIR / "mobile_models" / "nafnet_realestate.onnx",
        help="Path to ONNX model.",
    )
    parser.add_argument(
        "--onnx-provider",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="ONNXRuntime provider (default: auto).",
    )
    parser.add_argument(
        "--onnx-threads",
        type=int,
        default=0,
        help="ONNXRuntime threads (0 = default).",
    )

    # Torch options
    parser.add_argument(
        "--torch-model",
        type=Path,
        default=SCRIPT_DIR / "BasicSR" / "experiments" / "NAFNet_RealEstate_Fast" / "models" / "net_g_12000.pth",
        help="Path to PyTorch checkpoint.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="Torch device (default: cuda).",
    )

    args = parser.parse_args()

    _require_conda(args.no_conda_check, args.expected_conda_env)

    try:
        import psutil  # type: ignore  # noqa: F401
    except Exception:
        print(
            "WARNING: psutil is not available; RAM numbers may be inaccurate (using resource.ru_maxrss fallback).",
            file=sys.stderr,
        )

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        raise SystemExit("ERROR: --splits must not be empty")

    # Collect dataset pairs
    pairs = _collect_pairs(args.dataset_root, splits)
    if args.limit and args.limit > 0:
        pairs = pairs[: args.limit]
    if not pairs:
        raise SystemExit("ERROR: No paired images found.")

    out_dir = args.out_dir / args.backend / _now_compact()
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = out_dir / "metrics.csv"
    summary_json = out_dir / "summary.json"

    print("=" * 72)
    print("DATASET TRIPTYCH RUN")
    print("=" * 72)
    print(f"Backend:       {args.backend}")
    print(f"Dataset root:  {args.dataset_root}")
    print(f"Splits:        {', '.join(splits)}")
    print(f"Images:        {len(pairs)}")
    print(f"Output dir:    {out_dir}")
    print(f"RSS baseline:  {_get_rss_mb():.1f} MB")
    rss_interval_sec = (args.ram_sample_ms / 1000.0) if args.ram_sample_ms and args.ram_sample_ms > 0 else 0.0

    # Load backend
    load_start = time.perf_counter()
    rss_before_load = _get_rss_mb()

    session = None
    input_name = ""
    output_name = ""
    torch_model = None
    pred_label = ""

    if args.backend == "onnx":
        if not args.onnx_model.exists():
            raise SystemExit(f"ERROR: ONNX model not found: {args.onnx_model}")
        external_data = args.onnx_model.with_suffix(args.onnx_model.suffix + ".data")
        if external_data.exists():
            print(f"ONNX external data: {external_data.name} (present)")
        session, input_name, output_name, available_providers = _load_onnx_session(
            args.onnx_model,
            provider=args.onnx_provider,
            num_threads=(args.onnx_threads or None),
        )
        selected_provider = args.onnx_provider
        if selected_provider == "auto":
            selected_provider = "cuda" if "CUDAExecutionProvider" in available_providers else "cpu"
        pred_label = "ONNX"
        if selected_provider == "cuda":
            pred_label = "ONNX (CUDA)"
        print(f"ONNX providers: {available_providers}")
        print(f"ONNX selected:  {selected_provider}")
    else:
        # Torch
        if args.device == "cuda":
            try:
                import torch  # type: ignore

                if not torch.cuda.is_available():
                    print("CUDA not available; falling back to CPU.", file=sys.stderr)
                    args.device = "cpu"
            except Exception:
                pass
        if not args.torch_model.exists():
            raise SystemExit(f"ERROR: Torch model not found: {args.torch_model}")
        torch_model = _load_torch_model(args.torch_model, device=args.device)
        pred_label = f"TORCH ({args.device.upper()})"

    load_s = time.perf_counter() - load_start
    rss_after_load = _get_rss_mb()
    print(f"Load time:     {load_s:.2f}s")
    print(f"RSS +model:    {rss_after_load:.1f} MB (+{rss_after_load - rss_before_load:.1f} MB)")

    # Warmup with first image (to stabilize allocator behavior)
    if args.warmup and args.warmup > 0:
        import cv2  # type: ignore

        split0, lq0, _gt0 = pairs[0]
        img0 = cv2.imread(str(lq0), cv2.IMREAD_COLOR)
        if img0 is None:
            raise SystemExit(f"ERROR: Could not read warmup image: {lq0}")
        print(f"Warmup:        {args.warmup} run(s) on {split0}/{lq0.name}")
        for _ in range(args.warmup):
            if args.backend == "onnx":
                _onnx_infer(
                    session,
                    input_name,
                    output_name,
                    img0,
                    rss_interval_sec=rss_interval_sec,
                )
            else:
                _torch_infer(torch_model, args.device, img0, rss_interval_sec=rss_interval_sec)

    # Process images
    import cv2  # type: ignore

    rows: list[Row] = []
    start_all = time.perf_counter()
    peak_rss_seen = _get_rss_mb()

    for idx, (split, lq_path, gt_path) in enumerate(pairs):
        img_lq = cv2.imread(str(lq_path), cv2.IMREAD_COLOR)
        img_gt = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)
        if img_lq is None or img_gt is None:
            print(f"[{idx+1}/{len(pairs)}] ERROR reading {lq_path.name} or GT; skipping", file=sys.stderr)
            continue

        h, w = img_lq.shape[:2]
        if img_gt.shape[:2] != (h, w):
            img_gt = cv2.resize(img_gt, (w, h), interpolation=cv2.INTER_AREA)
        mp = (h * w) / 1_000_000.0

        pred_dir = out_dir / split / "pred"
        trip_dir = out_dir / split / "triptych"
        pred_dir.mkdir(parents=True, exist_ok=True)
        trip_dir.mkdir(parents=True, exist_ok=True)

        pred_path = pred_dir / lq_path.name
        trip_path = trip_dir / f"{idx:04d}_{lq_path.stem}.jpg"

        if args.skip_existing and trip_path.exists():
            continue

        rss_before = _get_rss_mb()

        t0 = time.perf_counter()
        if args.backend == "onnx":
            img_pred, t_pre, t_inf, t_post, rss_peak_during_infer = _onnx_infer(
                session,
                input_name,
                output_name,
                img_lq,
                rss_interval_sec=rss_interval_sec,
            )
        else:
            img_pred, t_pre, t_inf, t_post, rss_peak_during_infer = _torch_infer(
                torch_model,
                args.device,
                img_lq,
                rss_interval_sec=rss_interval_sec,
            )
        t1 = time.perf_counter()

        rss_after = _get_rss_mb()
        peak_rss_seen = max(peak_rss_seen, rss_after if rss_after == rss_after else peak_rss_seen)

        # Save predicted output
        cv2.imwrite(str(pred_path), img_pred, [cv2.IMWRITE_JPEG_QUALITY, int(args.jpeg_quality)])

        # Create and save triptych
        stitched, imwrite_params = _stitch_triptych(
            img_lq,
            img_pred,
            img_gt,
            pred_label=pred_label,
            divider_px=6,
            jpeg_quality=int(args.jpeg_quality),
        )
        cv2.imwrite(str(trip_path), stitched, imwrite_params)

        psnr = _psnr_uint8(img_pred, img_gt)
        lq_b, lq_g, lq_r = _mean_bgr(img_lq)
        pr_b, pr_g, pr_r = _mean_bgr(img_pred)
        gt_b, gt_g, gt_r = _mean_bgr(img_gt)

        row = Row(
            backend=args.backend,
            split=split,
            filename=lq_path.name,
            width=w,
            height=h,
            megapixels=mp,
            preprocess_s=t_pre,
            inference_s=t_inf,
            postprocess_s=t_post,
            total_s=(t1 - t0),
            rss_before_mb=rss_before,
            rss_after_mb=rss_after,
            rss_peak_during_infer_mb=rss_peak_during_infer,
            psnr_pred_gt=psnr,
            mean_lq_b=lq_b,
            mean_lq_g=lq_g,
            mean_lq_r=lq_r,
            mean_pred_b=pr_b,
            mean_pred_g=pr_g,
            mean_pred_r=pr_r,
            mean_gt_b=gt_b,
            mean_gt_g=gt_g,
            mean_gt_r=gt_r,
        )
        rows.append(row)

        do_print = (idx < 5) or ((idx + 1) % int(args.print_every) == 0) or (idx == len(pairs) - 1)
        if do_print:
            peak_str = f"{rss_peak_during_infer:.0f}MB" if rss_peak_during_infer == rss_peak_during_infer else "n/a"
            print(
                f"[{idx+1:4d}/{len(pairs)}] {split}/{lq_path.name} "
                f"{w}x{h} ({mp:.2f}MP) | inf {t_inf:.3f}s | RSS {rss_after:.0f}MB | peak {peak_str} | PSNR {psnr:.2f}"
            )

    elapsed_all = time.perf_counter() - start_all

    if not rows:
        raise SystemExit("ERROR: No images were processed successfully.")

    _write_csv(metrics_csv, rows)
    summary = _summarize(rows, baseline_rss_mb=rss_before_load)
    summary.update(
        {
            "output_dir": str(out_dir),
            "dataset_root": str(args.dataset_root),
            "selected_splits": splits,
            "load_time_s": load_s,
            "ram_sample_ms": args.ram_sample_ms,
            "model_rss_mb": rss_after_load,
            "peak_rss_seen_mb": peak_rss_seen,
            "elapsed_wall_s": elapsed_all,
        }
    )
    if args.backend == "onnx":
        summary.update(
            {
                "onnx_model": str(args.onnx_model),
                "onnx_provider": args.onnx_provider,
                "onnx_threads": int(args.onnx_threads),
            }
        )
    else:
        summary.update({"torch_model": str(args.torch_model), "torch_device": args.device})
    summary_json.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)
    print(f"Processed:     {len(rows)} image(s)")
    print(f"Wall time:     {elapsed_all:.1f}s")
    print(f"Peak RSS seen: {peak_rss_seen:.1f} MB")
    print(f"Metrics CSV:   {metrics_csv}")
    print(f"Summary JSON:  {summary_json}")


if __name__ == "__main__":
    main()
