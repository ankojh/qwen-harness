"""FastAPI wiring: HTTP routes, the SSE stream and the one-generation guard.

The event loop never touches MLX arrays.  Every operation that does -- load,
unload, prefill, decode, cache clear -- is submitted to a
``ThreadPoolExecutor(max_workers=1)``.  That keeps mlx's thread-local
generation stream consistent and makes "one active generation" structural
rather than a convention.

``POST /api/generate`` is idempotent on ``client_request_id`` and returns the
existing generation for a repeat.  A different request while one is running is
rejected with 409.  ``GET /api/events`` replays from ``Last-Event-ID`` or sends
a ``resync`` event when the id is older than the buffer.  Reconnecting can
never start work: it is a GET.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError
from starlette.middleware.trustedhost import TrustedHostMiddleware

from config import AppConfig, PRESETS, defaults_dict, fields_schema
from events import EventLog, format_sse
from generation import GenerationStore, run_generation
from model import (
    LoadedModel,
    apply_mlx_cache_limit,
    capabilities,
    load_model,
    memory_info,
    read_mlx_cache_limit,
    unload_model,
)
from prompt import Session, build_prompt
from prompt_cache import PromptCacheStore

FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"


class AppState:
    def __init__(self):
        self.events = EventLog()
        self.cache = PromptCacheStore(emit=self.events.emit)
        self.generations = GenerationStore()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
        self.config = AppConfig()
        self.loaded: LoadedModel | None = None
        self.active_generation: str | None = None
        self.cancel = threading.Event()
        self.counters = {"forward_calls_total": 0}
        self.request_ids: dict[str, str] = {}
        self.loop: asyncio.AbstractEventLoop | None = None
        # Remember MLX's own buffer-cache limit so clearing the setting restores it.
        self.mlx_default_cache_limit = read_mlx_cache_limit()
        self.mlx_cache_limit = self.mlx_default_cache_limit

    # -- called on the event loop -------------------------------------------
    def finish_generation(self, gen_id: str) -> None:
        if self.active_generation == gen_id:
            self.active_generation = None
        record = self.generations.get(gen_id) or {}
        self.events.emit(
            "generation.finished",
            {
                "generation_id": gen_id,
                "finish": record.get("finish"),
                "stats": record.get("stats"),
                "channels": record.get("channels"),
                "tokens_generated": record.get("tokens_generated", 0),
                "prompt_hash": record.get("prompt_hash"),
                "effective_settings": record.get("effective_settings"),
                "cache_verify": record.get("cache_verify"),
            },
        )

    def schedule_finish(self, gen_id: str) -> None:
        """Called from the MLX worker thread at the end of every generation."""
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self.finish_generation, gen_id)


def _field_meta() -> dict[str, dict[str, Any]]:
    return {field["name"]: field for field in fields_schema()}


def _compat_key(ctx: AppState) -> tuple:
    loaded = ctx.loaded
    identity = loaded.load_config if loaded else {
        "model_path": ctx.config.model_path,
        "revision": ctx.config.revision,
        "adapter_path": ctx.config.adapter_path,
    }
    return PromptCacheStore.compat_key(
        model_path=identity["model_path"],
        revision=identity.get("revision"),
        adapter_path=identity.get("adapter_path"),
        kv_bits=ctx.config.kv_bits,
        kv_group_size=ctx.config.kv_group_size,
        quantized_kv_start=ctx.config.quantized_kv_start,
    )


def _public_state(ctx: AppState) -> dict[str, Any]:
    loaded = ctx.loaded
    info = loaded.info if loaded else None
    model_fields = ["model_path", "revision", "adapter_path", "lazy_load"]
    configured = {name: getattr(ctx.config, name) for name in model_fields}
    stale = bool(loaded and loaded.load_config != configured)
    memory = memory_info()
    memory["mlx_cache_limit_bytes"] = ctx.mlx_cache_limit
    memory["mlx_default_cache_limit_bytes"] = ctx.mlx_default_cache_limit
    cache_summary = ctx.cache.summary(
        limit_bytes=int(ctx.config.cache_max_gb * 1_000_000_000)
    )
    cache_summary["active_bytes"] = memory["active_bytes"]
    return {
        "config": ctx.config.model_dump(mode="json"),
        "defaults": defaults_dict(),
        "fields": fields_schema(),
        "presets": PRESETS,
        "capabilities": capabilities(ctx.config, info),
        "model": info,
        "model_loaded": loaded is not None,
        "model_stale": stale,
        "cache": cache_summary,
        "memory": memory,
        "events": ctx.events.summary(),
        "active_generation": ctx.generations.progress(ctx.active_generation),
        "counters": dict(ctx.counters),
        "generations_total": ctx.generations.count(),
        "server": {"pid": __import__("os").getpid(), "bind": "127.0.0.1"},
    }


def _apply_config(ctx: AppState, new_config: AppConfig) -> dict[str, Any]:
    old_data = ctx.config.model_dump(mode="json")
    new_data = new_config.model_dump(mode="json")
    meta = _field_meta()
    for name in AppConfig.model_fields:
        if old_data[name] == new_data[name]:
            continue
        ctx.events.emit(
            "config.changed",
            {
                "field": name,
                "old": old_data[name],
                "new": new_data[name],
                "applies": meta.get(name, {}).get("applies"),
            },
        )
    kv_before = (
        ctx.config.kv_bits,
        ctx.config.kv_group_size,
        ctx.config.quantized_kv_start,
    )
    limit_changed = old_data["mlx_cache_limit_gb"] != new_data["mlx_cache_limit_gb"]
    ctx.config = new_config
    kv_after = (new_config.kv_bits, new_config.kv_group_size, new_config.quantized_kv_start)
    if kv_before != kv_after:
        ctx.cache.clear("kv_* setting changed")
    if limit_changed:
        applied = apply_mlx_cache_limit(
            new_config.mlx_cache_limit_gb, ctx.mlx_default_cache_limit
        )
        ctx.mlx_cache_limit = applied["new_bytes"]
        ctx.events.emit("runtime.mlx_cache_limit", applied)
    return _public_state(ctx)


class PreviewRequest(BaseModel):
    session: Session
    include_pieces: bool = False


class GenerateRequest(BaseModel):
    client_request_id: str
    session: Session


@asynccontextmanager
async def lifespan(app: FastAPI):
    ctx = AppState()
    ctx.loop = asyncio.get_running_loop()
    ctx.events.bind_loop(ctx.loop)
    app.state.ctx = ctx
    ctx.events.emit(
        "server.started",
        {"bind": "127.0.0.1", "mlx_default_cache_limit_bytes": ctx.mlx_default_cache_limit},
    )
    yield
    ctx.events.emit("server.stopped", {})
    ctx.executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="Qwen Harness", lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])
api = APIRouter(prefix="/api")


def _ctx(request: Request) -> AppState:
    return request.app.state.ctx


# --------------------------------------------------------------------- state
@api.get("/state")
async def get_state(request: Request):
    return _public_state(_ctx(request))


@api.put("/config")
async def put_config(payload: dict[str, Any], request: Request):
    ctx = _ctx(request)
    if ctx.active_generation is not None:
        raise HTTPException(
            status_code=409,
            detail="a generation is running; stop it before changing config",
        )
    merged = ctx.config.model_dump(mode="json")
    merged.update(payload)
    try:
        new_config = AppConfig(**merged)
    except ValidationError as error:
        raise HTTPException(status_code=422, detail=error.errors())
    return _apply_config(ctx, new_config)


@api.post("/config/reset")
async def reset_config(request: Request):
    ctx = _ctx(request)
    if ctx.active_generation is not None:
        raise HTTPException(status_code=409, detail="a generation is running")
    return _apply_config(ctx, AppConfig())


# --------------------------------------------------------------------- model
async def _unload(ctx: AppState) -> dict[str, Any]:
    loaded = ctx.loaded
    if loaded is None:
        return {"message": "no model was loaded"}
    ctx.loaded = None
    ctx.cache.clear("model unloaded")
    ctx.events.emit("model.unloading", {"path": loaded.info.get("path")})
    stats = await asyncio.get_running_loop().run_in_executor(
        ctx.executor, unload_model, loaded
    )
    ctx.events.emit("model.unloaded", stats)
    return stats


async def _load(ctx: AppState) -> dict[str, Any]:
    if ctx.loaded is not None:
        raise HTTPException(status_code=409, detail="a model is already loaded")
    ctx.cache.clear("model load")
    ctx.events.emit(
        "model.loading",
        {
            "model_path": ctx.config.model_path,
            "revision": ctx.config.revision,
            "adapter_path": ctx.config.adapter_path,
            "lazy_load": ctx.config.lazy_load,
        },
    )
    config = ctx.config.model_copy(deep=True)
    try:
        loaded = await asyncio.get_running_loop().run_in_executor(
            ctx.executor, load_model, config
        )
    except Exception as error:  # noqa: BLE001 - shown in the UI and log
        ctx.events.emit(
            "model.error",
            {"model_path": config.model_path, "message": f"{type(error).__name__}: {error}"},
        )
        raise HTTPException(
            status_code=500, detail=f"model load failed: {type(error).__name__}: {error}"
        )
    ctx.loaded = loaded
    ctx.events.emit(
        "model.loaded",
        {
            "model_path": loaded.info.get("path"),
            "resolved_path": loaded.info.get("resolved_path"),
            "seconds": round(loaded.load_seconds, 3),
            "weight_bytes": loaded.info.get("weight_bytes"),
            "active_bytes": memory_info()["active_bytes"],
            "cache_types": loaded.info.get("cache_types"),
        },
    )
    return _public_state(ctx)


@api.post("/model/load")
async def model_load(request: Request):
    ctx = _ctx(request)
    if ctx.active_generation is not None:
        raise HTTPException(status_code=409, detail="a generation is running")
    return await _load(ctx)


@api.post("/model/unload")
async def model_unload(request: Request):
    ctx = _ctx(request)
    if ctx.active_generation is not None:
        raise HTTPException(status_code=409, detail="a generation is running")
    await _unload(ctx)
    return _public_state(ctx)


@api.post("/model/reload")
async def model_reload(request: Request):
    ctx = _ctx(request)
    if ctx.active_generation is not None:
        raise HTTPException(status_code=409, detail="a generation is running")
    await _unload(ctx)
    return await _load(ctx)


# -------------------------------------------------------------------- prompt
@api.post("/prompt/preview")
async def preview_prompt(payload: PreviewRequest, request: Request):
    ctx = _ctx(request)
    if ctx.loaded is None:
        raise HTTPException(status_code=409, detail="load the model first")
    report = build_prompt(
        payload.session,
        ctx.config,
        ctx.loaded.tokenizer,
        ctx.loaded.info,
        capabilities(ctx.config, ctx.loaded.info),
        cache_store=ctx.cache,
        compat_key=_compat_key(ctx),
        include_pieces=payload.include_pieces,
    )
    return report


# ---------------------------------------------------------------- generation
@api.post("/generate")
async def generate(payload: GenerateRequest, request: Request):
    ctx = _ctx(request)
    if ctx.loaded is None:
        raise HTTPException(status_code=409, detail="load the model first")

    existing = ctx.request_ids.get(payload.client_request_id)
    if existing is not None:
        record = ctx.generations.get(existing) or {}
        return {
            "generation_id": existing,
            "deduplicated": True,
            "status": record.get("status"),
        }
    if ctx.active_generation is not None:
        raise HTTPException(
            status_code=409,
            detail=f"generation {ctx.active_generation} is already running",
        )

    gen_id = uuid.uuid4().hex[:12]
    session = payload.session.model_copy(deep=True)
    record = {
        "id": gen_id,
        "client_request_id": payload.client_request_id,
        "created_at": time.time(),
        "status": "queued",
        "phase": "queued",
        "session": session.model_dump(mode="json"),
        "prompt": None,
        "effective_settings": None,
        "tokens": [],
        "tokens_generated": 0,
        "channels": {"thinking": "", "answer": "", "raw": ""},
        "finish": None,
        "error": None,
        "stats": None,
        "prompt_hash": None,
    }
    ctx.generations.create(record)
    ctx.request_ids[payload.client_request_id] = gen_id
    ctx.active_generation = gen_id
    ctx.cancel = threading.Event()
    ctx.events.emit(
        "generation.started",
        {
            "generation_id": gen_id,
            "client_request_id": payload.client_request_id,
            "mode": session.mode,
            "messages": len(session.messages),
            "max_tokens": ctx.config.max_tokens,
        },
    )

    ctx.executor.submit(
        run_generation,
        gen_id=gen_id,
        session=session,
        config=ctx.config.model_copy(deep=True),
        loaded=ctx.loaded,
        cache_store=ctx.cache,
        generations=ctx.generations,
        emit=ctx.events.emit,
        cancel=ctx.cancel,
        counters=ctx.counters,
        compat_key=_compat_key(ctx),
        on_finished=ctx.schedule_finish,
    )
    return {"generation_id": gen_id, "deduplicated": False, "status": "queued"}


@api.post("/generate/stop")
async def stop_generation(request: Request):
    ctx = _ctx(request)
    if ctx.active_generation is None:
        return {"stopping": False, "reason": "no active generation"}
    ctx.cancel.set()
    ctx.events.emit(
        "generation.stop_requested", {"generation_id": ctx.active_generation}
    )
    return {"stopping": True, "generation_id": ctx.active_generation}


@api.get("/generations/{gen_id}")
async def get_generation(gen_id: str, request: Request):
    record = _ctx(request).generations.get(gen_id)
    if record is None:
        raise HTTPException(status_code=404, detail="unknown generation id")
    return record


@api.post("/cache/clear")
async def clear_cache(request: Request):
    ctx = _ctx(request)
    ctx.cache.clear("manual clear")
    return _public_state(ctx)


# ----------------------------------------------------------------------- SSE
@api.get("/events")
async def events(
    request: Request,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
):
    ctx = _ctx(request)
    parsed: int | None = None
    if last_event_id:
        try:
            parsed = int(last_event_id)
        except ValueError:
            parsed = None
    queue, backlog = ctx.events.subscribe(parsed)

    async def stream():
        try:
            for event in backlog:
                yield format_sse(event)
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
                    continue
                yield format_sse(event)
        finally:
            ctx.events.unsubscribe(queue)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


app.include_router(api)


# -------------------------------------------------------------------- static
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
else:

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return (
            "<h1>Qwen Harness</h1>"
            "<p>The frontend has not been built yet. Run <code>./run.sh</code> "
            "from the project root (it installs, builds and serves), or "
            "<code>./run.sh dev</code> for the Vite dev server.</p>"
        )
