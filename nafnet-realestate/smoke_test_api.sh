#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Smoke test for nafnet-realestate/api_server.py.

Usage:
  conda activate nafnet
  bash nafnet-realestate/smoke_test_api.sh [--tier 1024|2048|full] [--port 8009] [--image path.jpg]

Notes:
  - Starts the API server locally, submits 1 image, streams SSE events, downloads output, then stops the server.
  - Set WARMUP_ITERS=0 to reduce server startup time for this smoke test.

Examples:
  WARMUP_ITERS=0 bash nafnet-realestate/smoke_test_api.sh --tier 1024 --image nafnet-realestate/test_input/0000.jpg
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TIER="1024"
PORT="8009"
HOST="127.0.0.1"
IMAGE="${SCRIPT_DIR}/test_input/0000.jpg"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tier)
      TIER="${2:-}"; shift 2 ;;
    --port)
      PORT="${2:-}"; shift 2 ;;
    --host)
      HOST="${2:-}"; shift 2 ;;
    --image)
      IMAGE="${2:-}"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "ERROR: Activate your conda env first (example: \`conda activate nafnet\`)." >&2
  exit 1
fi

if [[ ! -f "$IMAGE" ]]; then
  echo "ERROR: Image not found: $IMAGE" >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "ERROR: jq is required for this smoke test (install jq)." >&2
  exit 1
fi

BASE_URL="http://${HOST}:${PORT}"
LOG="$(mktemp)"

echo "[smoke] starting server on ${BASE_URL} ..."
WARMUP_ITERS="${WARMUP_ITERS:-0}" \
CUDNN_BENCHMARK="${CUDNN_BENCHMARK:-0}" \
python "${SCRIPT_DIR}/api_server.py" --host "$HOST" --port "$PORT" >"$LOG" 2>&1 &
PID=$!

cleanup() {
  kill "$PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for _ in $(seq 1 400); do
  if curl -fsS "${BASE_URL}/healthz" >/dev/null 2>&1; then
    break
  fi
  sleep 0.25
  if ! kill -0 "$PID" >/dev/null 2>&1; then
    echo "[smoke] server exited early:" >&2
    tail -200 "$LOG" >&2 || true
    exit 1
  fi
done

echo "[smoke] creating job (tier=${TIER}) ..."
RESP="$(curl -fsS -X POST \
  -F "tier=${TIER}" \
  -F "jpeg_quality=95" \
  -F "images=@${IMAGE}" \
  "${BASE_URL}/v1/jobs")"

JOB_ID="$(echo "$RESP" | jq -r '.job_id')"
ITEM_ID="$(echo "$RESP" | jq -r '.items[0].item_id')"

echo "[smoke] job_id=${JOB_ID}"
echo "[smoke] item_id=${ITEM_ID}"

echo "[smoke] SSE events:"
curl -fsS -N "${BASE_URL}/v1/jobs/${JOB_ID}/events" || true

echo "[smoke] downloading output ..."
OUT_DIR="${REPO_ROOT}/nafnet-realestate/api_smoke_outputs/${JOB_ID}"
mkdir -p "$OUT_DIR"
OUT_PATH="${OUT_DIR}/${ITEM_ID}.jpg"
curl -fsS "${BASE_URL}/v1/items/${ITEM_ID}" -o "$OUT_PATH"
file "$OUT_PATH"

echo "[smoke] job summary:"
curl -fsS "${BASE_URL}/v1/jobs/${JOB_ID}" | jq -C .

echo "[smoke] done -> ${OUT_PATH}"
