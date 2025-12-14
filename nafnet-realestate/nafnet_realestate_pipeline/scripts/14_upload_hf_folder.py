#!/usr/bin/env python3
"""
Upload a local folder to Hugging Face under a path prefix.

Auth:
  - Set `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`) in your environment.
  - Alternatively, run `huggingface-cli login` once and omit the token env var.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    from huggingface_hub import HfApi

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="Local folder to upload")
    ap.add_argument("--repo-id", default=os.environ.get("HF_REPO_ID", "SebRincon/nafnet-realestate"))
    ap.add_argument("--repo-type", default="model")
    ap.add_argument("--dst-prefix", required=True, help="Path prefix in the HF repo (folder)")
    ap.add_argument("--commit-message", default="Upload eval artifacts")
    args = ap.parse_args()

    if not args.src.is_dir():
        raise SystemExit(f"Source folder not found: {args.src}")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    api = HfApi(token=token)

    api.upload_folder(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        folder_path=str(args.src),
        path_in_repo=args.dst_prefix,
        commit_message=args.commit_message,
    )
    print("Done.")


if __name__ == "__main__":
    main()

