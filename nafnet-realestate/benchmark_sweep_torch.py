#!/usr/bin/env python3
"""
Benchmark sweep for PyTorch NAFNet inference speed + RAM vs resolution + precision.

What it does:
  - Picks N paired samples from datasets/realestate/<split>/{lq,gt}
  - For each target long-side resolution (e.g. 1024/1536/2048/full):
      - Resizes LQ+GT to that resolution (preserving aspect)
      - Runs FP32 inference and FP16 inference (separately timed)
      - Saves:
          - resized inputs/gt
          - predictions (fp32, fp16)
          - quadtych image: (LQ | FP32 | FP16 | GT)
          - metrics.csv + summary.json
  - Writes a sweep_summary.json with ratios across resolutions.

Notes:
  - Requires the conda env used for training/inference (torch + opencv + psutil).
  - For most APIs: keep the process/model warm; do not reload per request.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent


def _now_compact() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _is_image_path(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png"}


def _require_conda() -> None:
    if os.environ.get("CONDA_PREFIX"):
        return
    raise SystemExit(
        "ERROR: Run this inside your conda env.\n"
        "  Example: `conda activate nafnet`\n"
        "  (This env should have torch + opencv + psutil installed.)"
    )


def _get_rss_mb() -> float:
    import psutil  # type: ignore

    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


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
        for lq in lq_paths:
            gt = gt_by_stem.get(lq.stem)
            if gt is None:
                missing += 1
                continue
            pairs.append((split, lq, gt))
    if missing:
        print(f"WARNING: {missing} LQ files had no matching GT (skipped).", file=sys.stderr)
    return pairs


def _resize_long_side(img_bgr: Any, long_side: int) -> Any:
    import cv2  # type: ignore

    h, w = img_bgr.shape[:2]
    current = max(h, w)
    if current == long_side:
        return img_bgr
    scale = long_side / float(current)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=interp)


def _pad_to_multiple_of_8(img_bgr: Any) -> tuple[Any, tuple[int, int], tuple[int, int]]:
    import cv2  # type: ignore

    h, w = img_bgr.shape[:2]
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h == 0 and pad_w == 0:
        return img_bgr, (0, 0), (h, w)
    padded = cv2.copyMakeBorder(img_bgr, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    return padded, (pad_h, pad_w), (h, w)


def _bgr_uint8_to_tensor(img_bgr_uint8: Any, device: str, dtype: Any, *, channels_last: bool) -> Any:
    import numpy as np  # type: ignore
    import torch  # type: ignore

    img_rgb = img_bgr_uint8[:, :, ::-1].copy()
    img_rgb = img_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)
    if channels_last:
        tensor = tensor.contiguous(memory_format=torch.channels_last)
    return tensor.to(device=device, dtype=dtype)


def _tensor_to_bgr_uint8(out_nchw: Any, orig_hw: tuple[int, int]) -> Any:
    import numpy as np  # type: ignore

    out = out_nchw
    if out.ndim == 4:
        out = out[0]
    out = out.detach().float().cpu().clamp(0, 1).numpy()
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


def _stitch_quad(
    img_lq: Any,
    img_fp32: Any,
    img_fp16: Any,
    img_gt: Any,
    *,
    divider_px: int,
    jpeg_quality: int,
) -> tuple[Any, list[int]]:
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    base_h = img_lq.shape[0]

    def _resize_to_h(img: Any) -> Any:
        h, w = img.shape[:2]
        if h == base_h:
            return img
        scale = base_h / h
        new_w = max(1, int(round(w * scale)))
        return cv2.resize(img, (new_w, base_h), interpolation=cv2.INTER_AREA)

    img_fp32 = _resize_to_h(img_fp32)
    img_fp16 = _resize_to_h(img_fp16)
    img_gt = _resize_to_h(img_gt)

    divider = np.ones((base_h, divider_px, 3), dtype=np.uint8) * 128
    stitched = np.hstack([img_lq, divider, img_fp32, divider, img_fp16, divider, img_gt])

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.8
    thickness = 4

    def _label_at(x_offset: int, text: str) -> None:
        x = int(x_offset + 30)
        y = 70
        cv2.putText(stitched, text, (x, y), font, font_scale, (0, 0, 0), thickness + 6)
        cv2.putText(stitched, text, (x, y), font, font_scale, (255, 255, 255), thickness)

    x0 = 0
    x1 = img_lq.shape[1] + divider_px
    x2 = x1 + img_fp32.shape[1] + divider_px
    x3 = x2 + img_fp16.shape[1] + divider_px
    _label_at(x0, "INPUT (LQ)")
    _label_at(x1, "FP32")
    _label_at(x2, "FP16")
    _label_at(x3, "GT")

    return stitched, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]


def _load_model(model_path: Path, device: str) -> Any:
    import torch  # type: ignore

    sys.path.insert(0, str(SCRIPT_DIR / "BasicSR"))
    from basicsr.archs.NAFNet_arch import NAFNet  # type: ignore

    model = NAFNet(
        img_channel=3,
        width=32,
        middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2],
    )

    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "params" in checkpoint:
        model.load_state_dict(checkpoint["params"])
    elif isinstance(checkpoint, dict) and "params_ema" in checkpoint:
        model.load_state_dict(checkpoint["params_ema"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model.to(device)


def _save_fp16_checkpoint(fp32_model_path: Path, out_path: Path) -> None:
    import torch  # type: ignore

    out_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(str(fp32_model_path), map_location="cpu", weights_only=True)

    def _to_fp16_state(state: Any) -> Any:
        if not isinstance(state, dict):
            raise ValueError("Expected state dict")
        converted: dict[str, Any] = {}
        for k, v in state.items():
            if torch.is_tensor(v) and v.is_floating_point():
                converted[k] = v.half()
            else:
                converted[k] = v
        return converted

    if isinstance(checkpoint, dict) and "params" in checkpoint:
        checkpoint = dict(checkpoint)
        checkpoint["params"] = _to_fp16_state(checkpoint["params"])
    elif isinstance(checkpoint, dict) and "params_ema" in checkpoint:
        checkpoint = dict(checkpoint)
        checkpoint["params_ema"] = _to_fp16_state(checkpoint["params_ema"])
    elif isinstance(checkpoint, dict):
        checkpoint = _to_fp16_state(checkpoint)
    else:
        raise ValueError("Unsupported checkpoint format")

    torch.save(checkpoint, str(out_path))


@dataclass(frozen=True)
class Row:
    split: str
    filename: str
    long_side: str
    width: int
    height: int
    megapixels: float
    fp32_infer_s: float
    fp16_infer_s: float
    fp32_total_s: float
    fp16_total_s: float
    rss_before_mb: float
    rss_after_mb: float
    gpu_peak_mb: float
    psnr_fp32_gt: float
    psnr_fp16_gt: float


def _write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))


def _stats(values: list[float]) -> dict[str, float]:
    values = [v for v in values if v == v]  # drop NaN
    if not values:
        return {}
    s = sorted(values)
    n = len(s)
    p50 = s[n // 2]
    p90 = s[int(n * 0.9) - 1 if n > 1 else 0]
    return {
        "count": float(n),
        "min": float(s[0]),
        "p50": float(p50),
        "p90": float(p90),
        "max": float(s[-1]),
        "mean": float(sum(s) / n),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep benchmark across resolution + FP32/FP16.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=SCRIPT_DIR / "datasets" / "realestate",
        help="Dataset root containing train/val folders with lq/gt subfolders.",
    )
    parser.add_argument("--splits", default="val", help="Comma-separated splits (default: val).")
    parser.add_argument(
        "--long-sides",
        default="1024,1536,2048,full",
        help="Comma-separated long-side targets, use 'full' to keep original (default: 1024,1536,2048,full).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2,
        help="Number of images per split to benchmark per resolution (default: 2).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=SCRIPT_DIR / "preview_sweeps",
        help="Output root (default: nafnet-realestate/preview_sweeps).",
    )
    parser.add_argument(
        "--torch-model",
        type=Path,
        default=SCRIPT_DIR / "BasicSR" / "experiments" / "NAFNet_RealEstate_Fast" / "models" / "net_g_12000.pth",
        help="FP32 model checkpoint path.",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup runs per precision+resolution (default: 2).")
    parser.add_argument("--jpeg-quality", type=int, default=80, help="JPEG quality (default: 80).")
    parser.add_argument("--divider-px", type=int, default=6, help="Divider width (default: 6).")
    parser.add_argument(
        "--channels-last",
        action="store_true",
        help="Use channels_last tensors for potential conv speedups.",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable common CUDA speed flags (cudnn.benchmark, TF32).",
    )
    parser.add_argument(
        "--save-fp16-checkpoint",
        action="store_true",
        help="Also write a fp16 checkpoint copy under the run output.",
    )
    args = parser.parse_args()

    _require_conda()

    try:
        import cv2  # type: ignore  # noqa: F401
    except Exception as e:
        raise SystemExit(f"ERROR: opencv-python is required (cv2 import failed): {e}")

    import torch  # type: ignore

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.", file=sys.stderr)
        args.device = "cpu"

    if args.optimize and args.device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    long_sides_raw = [s.strip() for s in args.long_sides.split(",") if s.strip()]
    if not splits:
        raise SystemExit("ERROR: --splits must not be empty")
    if not long_sides_raw:
        raise SystemExit("ERROR: --long-sides must not be empty")

    long_sides: list[int | None] = []
    for s in long_sides_raw:
        if s.lower() in {"full", "0", "orig", "original"}:
            long_sides.append(None)
        else:
            long_sides.append(int(s))

    if not args.torch_model.exists():
        raise SystemExit(f"ERROR: Torch model not found: {args.torch_model}")

    pairs_all = _collect_pairs(args.dataset_root, splits)
    if not pairs_all:
        raise SystemExit("ERROR: No paired images found.")

    run_dir = args.out_dir / _now_compact()
    run_dir.mkdir(parents=True, exist_ok=True)

    fp16_ckpt_path = run_dir / "models" / f"{args.torch_model.stem}_fp16.pth"
    if args.save_fp16_checkpoint:
        print(f"Writing fp16 checkpoint: {fp16_ckpt_path}")
        _save_fp16_checkpoint(args.torch_model, fp16_ckpt_path)

    print("=" * 72)
    print("SWEEP BENCHMARK (TORCH)")
    print("=" * 72)
    print(f"Device:        {args.device}")
    print(f"Model (fp32):  {args.torch_model}")
    print(f"Splits:        {', '.join(splits)}")
    print(f"Long-sides:    {', '.join(long_sides_raw)}")
    print(f"Limit/split:   {args.limit}")
    print(f"Output:        {run_dir}")
    print(f"RSS baseline:  {_get_rss_mb():.1f} MB")
    if args.device == "cuda":
        print(f"GPU:           {torch.cuda.get_device_name(0)}")

    rss_baseline = _get_rss_mb()

    rows_all: list[Row] = []
    sweep_summary: dict[str, Any] = {
        "device": args.device,
        "torch_model_fp32": str(args.torch_model),
        "fp16_checkpoint": str(fp16_ckpt_path) if fp16_ckpt_path.exists() else None,
        "splits": splits,
        "long_sides": long_sides_raw,
        "limit_per_split": args.limit,
        "channels_last": bool(args.channels_last),
        "optimize": bool(args.optimize),
        "rss_baseline_mb": rss_baseline,
        "results": {},
    }

    import cv2  # type: ignore

    for long_side in long_sides:
        label = "full" if long_side is None else str(long_side)
        res_dir = run_dir / label
        res_dir.mkdir(parents=True, exist_ok=True)

        # Select sample pairs per split (stable ordering).
        sample_pairs: list[tuple[str, Path, Path]] = []
        for split in splits:
            split_pairs = [p for p in pairs_all if p[0] == split]
            sample_pairs.extend(split_pairs[: max(0, int(args.limit))])
        if not sample_pairs:
            continue

        # Precompute resized LQ/GT and write them once so quadtych uses identical inputs.
        lq_dir = res_dir / "inputs_lq"
        gt_dir = res_dir / "targets_gt"
        lq_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)

        resized: list[tuple[str, str, Path, Path]] = []  # (split, filename, lq_resized, gt_resized)
        for split, lq_path, gt_path in sample_pairs:
            img_lq = cv2.imread(str(lq_path), cv2.IMREAD_COLOR)
            img_gt = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)
            if img_lq is None or img_gt is None:
                continue
            if long_side is not None:
                img_lq = _resize_long_side(img_lq, long_side)
                img_gt = _resize_long_side(img_gt, long_side)
            # Force exact match dims for metrics/visuals
            if img_gt.shape[:2] != img_lq.shape[:2]:
                img_gt = cv2.resize(img_gt, (img_lq.shape[1], img_lq.shape[0]), interpolation=cv2.INTER_AREA)

            out_lq = lq_dir / f"{split}_{lq_path.name}"
            out_gt = gt_dir / f"{split}_{gt_path.name}"
            cv2.imwrite(str(out_lq), img_lq, [cv2.IMWRITE_JPEG_QUALITY, int(args.jpeg_quality)])
            cv2.imwrite(str(out_gt), img_gt, [cv2.IMWRITE_JPEG_QUALITY, int(args.jpeg_quality)])
            resized.append((split, lq_path.name, out_lq, out_gt))

        if not resized:
            continue

        # ----------------------------
        # FP32 pass
        # ----------------------------
        torch.cuda.empty_cache() if args.device == "cuda" else None
        gc.collect()
        fp32_model = _load_model(args.torch_model, device=args.device)
        if args.channels_last:
            fp32_model = fp32_model.to(memory_format=torch.channels_last)
        fp32_model.eval()

        fp32_pred_dir = res_dir / "pred_fp32"
        fp32_pred_dir.mkdir(parents=True, exist_ok=True)

        # Warmup on first image shape for stable numbers
        warm_img = cv2.imread(str(resized[0][2]), cv2.IMREAD_COLOR)
        if warm_img is not None and args.warmup > 0:
            warm_padded, _pad, _orig_hw = _pad_to_multiple_of_8(warm_img)
            warm_tensor = _bgr_uint8_to_tensor(
                warm_padded, args.device, torch.float32, channels_last=args.channels_last
            )
            for _ in range(int(args.warmup)):
                if args.device == "cuda":
                    torch.cuda.synchronize()
                with torch.inference_mode():
                    _ = fp32_model(warm_tensor)
                if args.device == "cuda":
                    torch.cuda.synchronize()

        fp32_times: dict[tuple[str, str], tuple[float, float]] = {}  # (split, filename) -> (infer, total)
        fp32_outputs: dict[tuple[str, str], Any] = {}
        gpu_peak_mb = 0.0

        for split, filename, lq_resized_path, gt_resized_path in resized:
            img_lq = cv2.imread(str(lq_resized_path), cv2.IMREAD_COLOR)
            if img_lq is None:
                continue
            padded, _pad_hw, orig_hw = _pad_to_multiple_of_8(img_lq)
            inp = _bgr_uint8_to_tensor(padded, args.device, torch.float32, channels_last=args.channels_last)

            if args.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                t_inf0 = time.perf_counter()
                out = fp32_model(inp)
                if args.device == "cuda":
                    torch.cuda.synchronize()
                t_inf1 = time.perf_counter()
            pred = _tensor_to_bgr_uint8(out, orig_hw)
            t1 = time.perf_counter()

            infer_s = t_inf1 - t_inf0
            total_s = t1 - t0
            fp32_times[(split, filename)] = (infer_s, total_s)
            fp32_outputs[(split, filename)] = pred

            if args.device == "cuda":
                gpu_peak_mb = max(gpu_peak_mb, torch.cuda.max_memory_allocated() / (1024 * 1024))

            cv2.imwrite(
                str(fp32_pred_dir / f"{split}_{filename}"),
                pred,
                [cv2.IMWRITE_JPEG_QUALITY, int(args.jpeg_quality)],
            )

        # Cleanup FP32 model to reduce VRAM before FP16 pass
        del fp32_model
        if args.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        # ----------------------------
        # FP16 pass
        # ----------------------------
        fp16_pred_dir = res_dir / "pred_fp16"
        fp16_pred_dir.mkdir(parents=True, exist_ok=True)

        fp16_model = _load_model(args.torch_model, device=args.device)
        fp16_model = fp16_model.half()
        if args.channels_last:
            fp16_model = fp16_model.to(memory_format=torch.channels_last)
        fp16_model.eval()

        if warm_img is not None and args.warmup > 0:
            warm_padded, _pad, _orig_hw = _pad_to_multiple_of_8(warm_img)
            warm_tensor = _bgr_uint8_to_tensor(
                warm_padded, args.device, torch.float16, channels_last=args.channels_last
            )
            for _ in range(int(args.warmup)):
                if args.device == "cuda":
                    torch.cuda.synchronize()
                with torch.inference_mode():
                    _ = fp16_model(warm_tensor)
                if args.device == "cuda":
                    torch.cuda.synchronize()

        fp16_times: dict[tuple[str, str], tuple[float, float]] = {}
        fp16_outputs: dict[tuple[str, str], Any] = {}

        for split, filename, lq_resized_path, gt_resized_path in resized:
            img_lq = cv2.imread(str(lq_resized_path), cv2.IMREAD_COLOR)
            if img_lq is None:
                continue
            padded, _pad_hw, orig_hw = _pad_to_multiple_of_8(img_lq)
            inp = _bgr_uint8_to_tensor(padded, args.device, torch.float16, channels_last=args.channels_last)

            if args.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                t_inf0 = time.perf_counter()
                out = fp16_model(inp)
                if args.device == "cuda":
                    torch.cuda.synchronize()
                t_inf1 = time.perf_counter()
            pred = _tensor_to_bgr_uint8(out, orig_hw)
            t1 = time.perf_counter()

            infer_s = t_inf1 - t_inf0
            total_s = t1 - t0
            fp16_times[(split, filename)] = (infer_s, total_s)
            fp16_outputs[(split, filename)] = pred

            if args.device == "cuda":
                gpu_peak_mb = max(gpu_peak_mb, torch.cuda.max_memory_allocated() / (1024 * 1024))

            cv2.imwrite(
                str(fp16_pred_dir / f"{split}_{filename}"),
                pred,
                [cv2.IMWRITE_JPEG_QUALITY, int(args.jpeg_quality)],
            )

        del fp16_model
        if args.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        # ----------------------------
        # Quadtychs + metrics
        # ----------------------------
        quad_dir = res_dir / "quadtych"
        quad_dir.mkdir(parents=True, exist_ok=True)

        rows: list[Row] = []
        rss_before = _get_rss_mb()
        for split, filename, lq_resized_path, gt_resized_path in resized:
            img_lq = cv2.imread(str(lq_resized_path), cv2.IMREAD_COLOR)
            img_gt = cv2.imread(str(gt_resized_path), cv2.IMREAD_COLOR)
            if img_lq is None or img_gt is None:
                continue
            key = (split, filename)
            pred32 = fp32_outputs.get(key)
            pred16 = fp16_outputs.get(key)
            if pred32 is None or pred16 is None:
                continue

            stitched, params = _stitch_quad(
                img_lq,
                pred32,
                pred16,
                img_gt,
                divider_px=int(args.divider_px),
                jpeg_quality=int(args.jpeg_quality),
            )
            quad_path = quad_dir / f"{split}_{Path(filename).stem}.jpg"
            cv2.imwrite(str(quad_path), stitched, params)

            h, w = img_lq.shape[:2]
            mp = (h * w) / 1_000_000.0
            fp32_inf, fp32_total = fp32_times.get(key, (float("nan"), float("nan")))
            fp16_inf, fp16_total = fp16_times.get(key, (float("nan"), float("nan")))
            psnr32 = _psnr_uint8(pred32, img_gt)
            psnr16 = _psnr_uint8(pred16, img_gt)

            rows.append(
                Row(
                    split=split,
                    filename=filename,
                    long_side=label,
                    width=w,
                    height=h,
                    megapixels=mp,
                    fp32_infer_s=fp32_inf,
                    fp16_infer_s=fp16_inf,
                    fp32_total_s=fp32_total,
                    fp16_total_s=fp16_total,
                    rss_before_mb=rss_before,
                    rss_after_mb=_get_rss_mb(),
                    gpu_peak_mb=gpu_peak_mb,
                    psnr_fp32_gt=psnr32,
                    psnr_fp16_gt=psnr16,
                )
            )

        if not rows:
            continue

        metrics_csv = res_dir / "metrics.csv"
        _write_csv(metrics_csv, rows)

        summary = {
            "long_side": label,
            "images": len(rows),
            "resolution_avg_megapixels": _stats([r.megapixels for r in rows]),
            "fp32_infer_s": _stats([r.fp32_infer_s for r in rows]),
            "fp16_infer_s": _stats([r.fp16_infer_s for r in rows]),
            "fp32_total_s": _stats([r.fp32_total_s for r in rows]),
            "fp16_total_s": _stats([r.fp16_total_s for r in rows]),
            "psnr_fp32_gt": _stats([r.psnr_fp32_gt for r in rows]),
            "psnr_fp16_gt": _stats([r.psnr_fp16_gt for r in rows]),
            "rss_mb": {
                "baseline": rss_baseline,
                "mean_after": float(sum(r.rss_after_mb for r in rows) / len(rows)),
                "peak_seen": float(max(r.rss_after_mb for r in rows)),
            },
            "gpu_peak_mb": float(gpu_peak_mb),
        }
        if summary["fp32_infer_s"] and summary["fp16_infer_s"]:
            fp32_mean = summary["fp32_infer_s"]["mean"]
            fp16_mean = summary["fp16_infer_s"]["mean"]
            if fp16_mean and fp16_mean > 0:
                summary["fp16_speedup_vs_fp32"] = float(fp32_mean / fp16_mean)
        (res_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        sweep_summary["results"][label] = summary
        rows_all.extend(rows)

        fp32_mean = summary.get("fp32_infer_s", {}).get("mean", float("nan"))
        fp16_mean = summary.get("fp16_infer_s", {}).get("mean", float("nan"))
        speedup = summary.get("fp16_speedup_vs_fp32", float("nan"))
        print(
            f"[longside {label:>4}] images={len(rows):>2} avgMP={summary['resolution_avg_megapixels']['mean']:.2f} "
            f"fp32={fp32_mean:.3f}s fp16={fp16_mean:.3f}s speedup={speedup:.2f}x "
            f"RSS_peak={summary['rss_mb']['peak_seen']:.0f}MB GPU_peak={gpu_peak_mb:.0f}MB"
        )

    # Add cross-resolution ratios vs full-resolution baseline (if available).
    full_summary = sweep_summary["results"].get("full")
    if isinstance(full_summary, dict):
        full_fp32 = full_summary.get("fp32_infer_s", {}).get("mean")
        full_fp16 = full_summary.get("fp16_infer_s", {}).get("mean")
        for label, res in sweep_summary["results"].items():
            if not isinstance(res, dict):
                continue
            if label == "full":
                continue
            fp32_mean = res.get("fp32_infer_s", {}).get("mean")
            fp16_mean = res.get("fp16_infer_s", {}).get("mean")
            if full_fp32 and fp32_mean and fp32_mean > 0:
                res["fp32_speedup_vs_full"] = float(full_fp32 / fp32_mean)
            if full_fp16 and fp16_mean and fp16_mean > 0:
                res["fp16_speedup_vs_full"] = float(full_fp16 / fp16_mean)

    (run_dir / "sweep_summary.json").write_text(json.dumps(sweep_summary, indent=2))
    if rows_all:
        _write_csv(run_dir / "all_metrics.csv", rows_all)

    print("\nDone.")
    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
