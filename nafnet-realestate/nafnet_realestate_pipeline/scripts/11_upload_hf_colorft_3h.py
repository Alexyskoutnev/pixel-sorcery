#!/usr/bin/env python3
"""
Upload ColorFT 3h artifacts to Hugging Face.

Auth:
  - Set `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`) in your environment.

Uploads to:
  - SebRincon/nafnet-realestate (repo_type=model)

Artifacts expected under `nafnet-realestate/hf_artifacts/`:
  - nafnet_realestate_colorft_3h.pth
  - nafnet_realestate_colorft_3h_fp32.onnx
  - nafnet_realestate_colorft_3h_fp32.onnx.data
  - nafnet_realestate_colorft_3h_fp16.onnx
  - nafnet_realestate_colorft_3h_fp16.onnx.data
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
    artifacts = project_root / "hf_artifacts"
    if not artifacts.is_dir():
        raise SystemExit(f"Artifacts folder not found: {artifacts}")

    files = [
        artifacts / "nafnet_realestate_colorft_3h.pth",
        artifacts / "nafnet_realestate_colorft_3h_fp32.onnx",
        artifacts / "nafnet_realestate_colorft_3h_fp32.onnx.data",
        artifacts / "nafnet_realestate_colorft_3h_fp16.onnx",
        artifacts / "nafnet_realestate_colorft_3h_fp16.onnx.data",
    ]
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise SystemExit("Missing artifact files:\n  " + "\n  ".join(missing))

    api = HfApi(token=_token())
    for p in files:
        print(f"Uploading: {p.name}")
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=p.name,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=f"Add ColorFT 3h artifacts: {p.name}",
        )

    print("Done.")


if __name__ == "__main__":
    main()

