#!/usr/bin/env python3
"""
Generate meta_info files to oversample curated "bad" examples without duplicating images on disk.

BasicSR's PairedImageDataset supports `meta_info_file`:
  - each line's first token is treated as the GT filename
  - the same filename is used to derive the LQ filename via `filename_tmpl`

We use this to:
  1) list all train GT filenames once
  2) append additional repeats of selected "bad" train filenames (from `bad/` triptych list)
  3) (optional) create a val "color stress" list from "bad" images that belong to val
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _list_images(folder: Path) -> list[str]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted([p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])


def _parse_bad_triptych_ids(bad_dir: Path) -> tuple[set[str], set[str]]:
    """Return (train_ids, val_ids) as GT filenames (e.g. '0006.jpg')."""
    train_ids: set[str] = set()
    val_ids: set[str] = set()

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    for p in bad_dir.iterdir():
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        name = p.stem  # e.g. "0525_0005"
        if "_" not in name:
            continue
        prefix, local_id = name.split("_", 1)
        if not prefix.isdigit() or not local_id.isdigit():
            continue
        gt_name = f"{local_id}{p.suffix.lower()}"
        if int(prefix) >= 520:
            val_ids.add(gt_name)
        else:
            train_ids.add(gt_name)

    return train_ids, val_ids


def _write_meta(path: Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # BasicSR only reads the first whitespace-separated token.
    path.write_text("".join(f"{n}\n" for n in names), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-gt-dir", type=Path, default=Path("datasets/realestate/train/gt"))
    ap.add_argument("--val-gt-dir", type=Path, default=Path("datasets/realestate/val/gt"))
    ap.add_argument("--bad-dir", type=Path, default=Path("../bad"))
    ap.add_argument("--repeat", type=int, default=8, help="Extra repeats for bad *train* IDs (default: 8).")
    ap.add_argument("--out-train", type=Path, default=Path("nafnet_realestate_pipeline/meta/train_color_focus.txt"))
    ap.add_argument("--out-val", type=Path, default=Path("nafnet_realestate_pipeline/meta/val_color_stress.txt"))
    args = ap.parse_args()

    if args.repeat < 0:
        raise SystemExit("--repeat must be >= 0")

    train_gt_dir = args.train_gt_dir
    val_gt_dir = args.val_gt_dir
    bad_dir = args.bad_dir

    if not train_gt_dir.is_dir():
        raise SystemExit(f"Train GT dir not found: {train_gt_dir}")
    if not val_gt_dir.is_dir():
        raise SystemExit(f"Val GT dir not found: {val_gt_dir}")
    if not bad_dir.is_dir():
        raise SystemExit(f"Bad dir not found: {bad_dir}")

    train_all = _list_images(train_gt_dir)
    val_all = _list_images(val_gt_dir)
    train_bad, val_bad = _parse_bad_triptych_ids(bad_dir)

    missing_train_bad = sorted([n for n in train_bad if n not in set(train_all)])
    missing_val_bad = sorted([n for n in val_bad if n not in set(val_all)])
    if missing_train_bad:
        print(f"WARNING: bad/train ids missing from train GT dir (ignoring): {missing_train_bad[:10]}")
    if missing_val_bad:
        print(f"WARNING: bad/val ids missing from val GT dir (ignoring): {missing_val_bad[:10]}")

    train_focus = list(train_all)
    for _ in range(args.repeat):
        for n in sorted(train_bad):
            if n in set(train_all):
                train_focus.append(n)

    _write_meta(args.out_train, train_focus)

    # Val color stress list: only the curated val IDs, once each.
    val_stress = [n for n in sorted(val_bad) if n in set(val_all)]
    _write_meta(args.out_val, val_stress)

    print("Wrote:")
    print(f"  train meta: {args.out_train}  (base={len(train_all)} + repeats={len(train_focus)-len(train_all)})")
    print(f"  val stress: {args.out_val}    (count={len(val_stress)})")


if __name__ == "__main__":
    main()
