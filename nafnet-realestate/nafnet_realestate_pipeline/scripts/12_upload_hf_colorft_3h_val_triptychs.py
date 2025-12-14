#!/usr/bin/env python3
"""
Upload ColorFT 3h validation triptychs (LQ | PRED | GT) to Hugging Face.

Auth:
  - Set `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`) in your environment.

Uploads to:
  - SebRincon/nafnet-realestate (repo_type=model)

Source folder (default):
  nafnet-realestate/dataset_triptychs/torch/torch/colorft_3h_latest_val/val/triptych/
"""

from __future__ import annotations

import os
from pathlib import Path


def _token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise SystemExit("Missing HF token: set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN).")
    return token


def main() -> None:
    from huggingface_hub import HfApi

    repo_id = os.environ.get("HF_REPO_ID", "SebRincon/nafnet-realestate")
    repo_type = "model"

    project_root = Path(__file__).resolve().parents[2]
    triptych_dir = Path(
        os.environ.get(
            "TRIPTYCH_DIR",
            str(
                project_root
                / "dataset_triptychs"
                / "torch"
                / "torch"
                / "colorft_3h_latest_val"
                / "val"
                / "triptych"
            ),
        )
    )
    if not triptych_dir.is_dir():
        raise SystemExit(f"Triptych dir not found: {triptych_dir}")

    images = sorted(triptych_dir.glob("*.jpg"))
    if not images:
        raise SystemExit(f"No .jpg files found in: {triptych_dir}")

    dst_prefix = os.environ.get("HF_PATH_PREFIX", "eval/colorft_3h_latest_val_triptychs")

    api = HfApi(token=_token())
    for i, p in enumerate(images, start=1):
        path_in_repo = f"{dst_prefix}/{p.name}"
        print(f"[{i}/{len(images)}] Uploading: {p.name}")
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=f"Add ColorFT 3h val triptychs ({p.name})",
        )

    print("Done.")


if __name__ == "__main__":
    main()

