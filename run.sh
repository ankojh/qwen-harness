#!/usr/bin/env bash
# One command to set up and run the harness.
#
#   ./run.sh          build the UI and serve it from FastAPI on 127.0.0.1:8765
#   ./run.sh dev      backend + Vite dev server (hot reload) on 127.0.0.1:5173
#   PORT=8010 ./run.sh   use a different backend port (8765 is the default)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8765}"
MODE="${1:-run}"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required (https://docs.astral.sh/uv/). Install it, then rerun." >&2
  exit 1
fi
if ! command -v npm >/dev/null 2>&1; then
  echo "error: npm/Node.js is required for the frontend." >&2
  exit 1
fi

echo "==> backend dependencies (uv sync)"
cd "$ROOT/backend"
uv sync

echo "==> frontend dependencies (npm install)"
cd "$ROOT/frontend"
npm install

if [[ "$MODE" == "dev" ]]; then
  echo "==> backend on http://127.0.0.1:${PORT}"
  (
    cd "$ROOT/backend"
    exec .venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port "$PORT"
  ) &
  BACK_PID=$!
  cleanup() {
    kill "$BACK_PID" 2>/dev/null || true
  }
  trap cleanup EXIT INT TERM

  echo "==> Vite dev server on http://127.0.0.1:5173 (proxying /api to ${PORT})"
  VITE_API_TARGET="http://127.0.0.1:${PORT}" npm run dev -- --host 127.0.0.1
  exit 0
fi

if [[ "$MODE" != "run" ]]; then
  echo "error: unknown mode '$MODE' (use no argument or 'dev')" >&2
  exit 1
fi

echo "==> building the frontend"
npm run build

echo "==> serving http://127.0.0.1:${PORT}  (if the port is busy, use PORT=8010 ./run.sh)"
cd "$ROOT/backend"
exec .venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port "$PORT"
