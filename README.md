# Qwen Harness

A small, local web UI for running and **inspecting** Qwen3.5-4B-4bit with MLX-LM on an Apple-silicon Mac. It shows the exact prompt and tokens, every active setting, the raw output, the prompt cache, and a live runtime event stream. It is a learning tool: readable code and visible behavior come before cleverness.

## Requirements

- macOS on Apple silicon (tested on M3 Max, 48 GB)
- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- Node.js 18+ with npm
- The weights in `~/.cache/huggingface` (`mlx-community/Qwen3.5-4B-4bit`), or an internet connection for the first Load

## Quick start

```bash
./run.sh
# open http://127.0.0.1:8765
```

`run.sh` runs `uv sync`, `npm install`, `npm run build`, then starts uvicorn on `127.0.0.1:8765`. The built UI is served by FastAPI, so there is one origin and no CORS.

If port 8765 is already taken (another local app), pick another port:

```bash
PORT=8010 ./run.sh
```

## Dev mode (hot reload)

```bash
./run.sh dev
# Vite UI:  http://127.0.0.1:5173   (proxies /api to the backend on 8765)
# backend:  http://127.0.0.1:8765
```

## Checks

```bash
cd backend
.venv/bin/python -m pytest -q      # unit tests, no model
.venv/bin/python verify.py         # end-to-end checks against the real model

cd ../frontend
npm run build
```

The model is loaded only when you press **Load** (or the first `/api/model/load`). Nothing runs in the background.

## What it is

- One FastAPI process, one dedicated MLX worker thread, one SSE event stream.
- Stateless conversation: the browser sends the whole session with Preview and Generate.
- `Generate` is idempotent on `client_request_id`; a second request while one runs returns 409.
- Every control is grounded in a checked fact about this model and mlx-lm. Controls that would do nothing are shown disabled or marked **inactive**, with the reason.

## What it is not

No agents, tools, retrieval, batching, speculative decoding, hidden extra model calls, or power-setting changes. Thinking text is sampled output, not a view of the model's internal computation. See `GUIDE.md`.
