#!/usr/bin/env python3
"""
Generate simple HTML galleries with pre-signed S3 URLs for viewing images in a private bucket.

Creates:
  - gallery_train.html
  - gallery_val.html
  - manifest_train.csv
  - manifest_val.csv

Requires:
  - awscli in PATH (uses `aws s3api list-objects-v2` and `aws s3 presign`)
  - AWS credentials in environment (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION)
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import subprocess
import sys
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Obj:
    key: str
    size: int


def _run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{p.stderr.strip()}")
    return p.stdout


def _list_objects(bucket: str, prefix: str) -> list[Obj]:
    objs: list[Obj] = []
    token: str | None = None
    while True:
        cmd = [
            "aws",
            "s3api",
            "list-objects-v2",
            "--bucket",
            bucket,
            "--prefix",
            prefix,
            "--page-size",
            "1000",
        ]
        if token:
            cmd += ["--starting-token", token]
        out = _run(cmd)
        data = json.loads(out)
        for item in data.get("Contents", []) or []:
            key = item.get("Key")
            size = int(item.get("Size", 0))
            if not key:
                continue
            if key.lower().endswith((".jpg", ".jpeg", ".png")):
                objs.append(Obj(key=key, size=size))
        token = data.get("NextContinuationToken")
        if not token:
            break
    return sorted(objs, key=lambda o: o.key)


def _presign(bucket: str, key: str, expires: int) -> str:
    s3_uri = f"s3://{bucket}/{key}"
    out = _run(["aws", "s3", "presign", s3_uri, "--expires-in", str(expires)])
    return out.strip()


def _write_manifest_csv(path: Path, rows: list[tuple[str, int, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "size_bytes", "url"])
        for r in rows:
            w.writerow(r)


def _write_gallery_html(path: Path, title: str, rows: list[tuple[str, int, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    items = []
    for key, size, url in rows:
        fname = Path(key).name
        items.append(
            f"""
            <div class="card">
              <div class="meta">
                <div class="name">{html.escape(fname)}</div>
                <div class="sub">{html.escape(key)} · {size/1024/1024:.2f} MB</div>
                <div class="links">
                  <a href="{html.escape(url)}" target="_blank" rel="noreferrer">open</a>
                  <a href="{html.escape(url)}" download>download</a>
                </div>
              </div>
              <img loading="lazy" src="{html.escape(url)}" alt="{html.escape(fname)}" />
            </div>
            """.strip()
        )

    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 0; background: #0b0f19; color: #e6e8ef; }}
    header {{ position: sticky; top: 0; background: rgba(11,15,25,0.9); backdrop-filter: blur(8px); border-bottom: 1px solid rgba(255,255,255,0.08); padding: 14px 18px; }}
    h1 {{ font-size: 16px; margin: 0 0 4px; }}
    .hint {{ font-size: 12px; opacity: 0.8; }}
    .wrap {{ padding: 18px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(520px, 1fr)); gap: 14px; }}
    .card {{ border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; overflow: hidden; background: rgba(255,255,255,0.03); }}
    .meta {{ padding: 10px 12px; border-bottom: 1px solid rgba(255,255,255,0.06); }}
    .name {{ font-weight: 600; }}
    .sub {{ font-size: 12px; opacity: 0.75; margin-top: 2px; word-break: break-all; }}
    .links a {{ font-size: 12px; color: #9ac1ff; text-decoration: none; margin-right: 10px; }}
    img {{ display: block; width: 100%; height: auto; background: #111827; }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <div class="hint">Pre-signed links expire; re-generate if images stop loading.</div>
  </header>
  <div class="wrap">
    <div class="grid">
      {"".join(items)}
    </div>
  </div>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate HTML gallery from private S3 prefix using pre-signed URLs.")
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--prefix-train", required=True, help="Prefix containing train images (e.g. uploads/.../train/triptych/)")
    ap.add_argument("--prefix-val", required=True, help="Prefix containing val images (e.g. uploads/.../val/triptych/)")
    ap.add_argument("--expires-in", type=int, default=604800, help="Seconds until URL expiry (default: 7 days).")
    ap.add_argument("--out-dir", type=Path, default=Path("nafnet-realestate/s3_gallery"))
    args = ap.parse_args()

    if not shutil.which("aws"):
        raise SystemExit("ERROR: `aws` CLI not found in PATH.")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    def _do(split: str, prefix: str) -> None:
        print(f"[{split}] listing objects under s3://{args.bucket}/{prefix} ...", file=sys.stderr)
        objs = _list_objects(args.bucket, prefix)
        print(f"[{split}] found {len(objs)} images; presigning ...", file=sys.stderr)
        rows: list[tuple[str, int, str]] = []
        for i, o in enumerate(objs, 1):
            url = _presign(args.bucket, o.key, args.expires_in)
            rows.append((o.key, o.size, url))
            if i <= 3 or i % 100 == 0 or i == len(objs):
                print(f"[{split}] {i}/{len(objs)}", file=sys.stderr)
        _write_manifest_csv(out_dir / f"manifest_{split}.csv", rows)
        _write_gallery_html(out_dir / f"gallery_{split}.html", f"S3 Triptychs ({split})", rows)

    _do("train", args.prefix_train)
    _do("val", args.prefix_val)

    print(f"Done. Open:\n  {out_dir/'gallery_train.html'}\n  {out_dir/'gallery_val.html'}")


if __name__ == "__main__":
    main()
