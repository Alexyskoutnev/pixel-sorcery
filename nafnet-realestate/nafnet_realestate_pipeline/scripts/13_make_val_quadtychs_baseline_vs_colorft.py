#!/usr/bin/env python3
"""
Create quadtychs (INPUT | BASELINE | COLORFT | GT) from existing triptych runs.

Expected inputs (defaults):
  - baseline triptychs:
      dataset_triptychs/torch/torch/baseline_12000_val/val/triptych/*.jpg
  - colorft triptychs:
      dataset_triptychs/torch/torch/colorft_3h_latest_val/val/triptych/*.jpg

Output (default):
  - hf_artifacts/quadtych_colorft_3h_latest_vs_baseline_val/*.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def split_triptych_bgr(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    w3 = w // 3
    return img[:, :w3], img[:, w3 : 2 * w3], img[:, 2 * w3 : 3 * w3]


def put_label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.1
    thickness = 2
    x, y = 20, 48
    cv2.putText(out, text, (x, y), font, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(out, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--baseline",
        type=Path,
        default=Path("dataset_triptychs/torch/torch/baseline_12000_val/val/triptych"),
        help="Baseline triptych folder",
    )
    ap.add_argument(
        "--colorft",
        type=Path,
        default=Path("dataset_triptychs/torch/torch/colorft_3h_latest_val/val/triptych"),
        help="ColorFT triptych folder",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("hf_artifacts/quadtych_colorft_3h_latest_vs_baseline_val"),
        help="Output quadtych folder",
    )
    ap.add_argument("--jpeg-quality", type=int, default=90)
    args = ap.parse_args()

    if not args.baseline.is_dir():
        raise SystemExit(f"Baseline triptych dir not found: {args.baseline}")
    if not args.colorft.is_dir():
        raise SystemExit(f"ColorFT triptych dir not found: {args.colorft}")

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_files = sorted(args.baseline.glob("*.jpg"))
    if not baseline_files:
        raise SystemExit(f"No triptychs found in: {args.baseline}")

    for p in baseline_files:
        q = args.colorft / p.name
        if not q.is_file():
            print(f"Skipping (missing colorft): {p.name}")
            continue

        img_b = cv2.imread(str(p), cv2.IMREAD_COLOR)
        img_f = cv2.imread(str(q), cv2.IMREAD_COLOR)
        if img_b is None or img_f is None:
            print(f"Skipping (read failure): {p.name}")
            continue

        lq_b, pred_b, gt_b = split_triptych_bgr(img_b)
        lq_f, pred_f, gt_f = split_triptych_bgr(img_f)

        # Prefer GT from baseline (should match); fall back to colorft.
        gt = gt_b if gt_b.size else gt_f
        lq = lq_b if lq_b.size else lq_f

        panels = [
            put_label(lq, "INPUT (LQ)"),
            put_label(pred_b, "BASELINE (12000)"),
            put_label(pred_f, "COLORFT (3h)"),
            put_label(gt, "GT"),
        ]

        h = min(pan.shape[0] for pan in panels)
        panels = [pan[:h, :] for pan in panels]
        quad = np.concatenate(panels, axis=1)

        out_path = out_dir / p.name
        cv2.imwrite(
            str(out_path),
            quad,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)],
        )

    print(f"Wrote quadtychs to: {out_dir}")


if __name__ == "__main__":
    main()

