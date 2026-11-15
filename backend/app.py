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
import copy
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import mlx.core as mx
from fastapi import APIRouter, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from mlx_lm.models.cache import make_prompt_cache
from pydantic import BaseModel, ValidationError
from starlette.middleware.trustedhost import TrustedHostMiddleware

from config import RESERVED_TEMPLATE_KWARGS, AppConfig, PRESETS, defaults_dict, fields_schema
from events import EventLog, format_sse
from generation import GenerationStore, prefill_to, run_generation
from model import (
    LoadedModel,
    apply_mlx_cache_limit,
    capabilities,
    load_model,
    memory_info,
    read_mlx_cache_limit,
    unload_model,
)
from prompt import (
    Session,
    _token_is_special,
    _token_spans,
    _token_text,
    build_prompt,
    checkpoint_boundary,
    relabel_token_ids,
)
from prompt_cache import PromptCacheStore, describe_cache_layers

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


class CheckpointRequest(BaseModel):
    session: Session
    after_message: int


class RelabelRequest(BaseModel):
    position: int
    text: str


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


def _read_cache_entry(entry: Any, tokenizer: Any, layer_types: list[str] | None) -> dict[str, Any]:
    """Read-only inspection of one stored cache entry.

    Runs on the MLX worker thread: it reads tokenizer state and MLX array
    *metadata* (shapes, dtypes, sizes) only, never the values inside, and
    never copies the cache.
    """
    token_ids = list(entry.tokens)
    tokens = []
    for token_id in token_ids:
        piece = tokenizer.convert_ids_to_tokens(token_id)
        tokens.append(
            {
                "id": token_id,
                "piece": piece,
                "text": _token_text(tokenizer, token_id, piece),
                "special": _token_is_special(tokenizer, token_id),
            }
        )
    layers = describe_cache_layers(entry.cache, layer_types)
    kv_layers = [layer for layer in layers if layer["kind"] == "full_attention"]
    recurrent_layers = [
        layer for layer in layers if layer["kind"] == "linear_attention"
    ]
    kv_allocated = sum(layer["allocated_bytes"] for layer in kv_layers)
    kv_used = sum(layer["used_bytes"] for layer in kv_layers)
    allocated_tokens = next(
        (layer["allocated_tokens"] for layer in kv_layers if layer["allocated_tokens"]),
        None,
    )
    kv_per_token_per_layer = None
    if kv_layers and allocated_tokens:
        kv_per_token_per_layer = round(
            kv_allocated / allocated_tokens / len(kv_layers), 2
        )
    relabels = []
    for item in entry.relabels:
        relabels.append(
            {
                "position": item["position"],
                "original_id": item["original_id"],
                "original_piece": tokenizer.convert_ids_to_tokens(
                    item["original_id"]
                ),
                "new_id": item["new_id"],
            }
        )
    return {
        "id": entry.id,
        "kind": entry.kind,
        "after_message": entry.after_message,
        "token_ids": token_ids,
        "tokens": tokens,
        "text": tokenizer.decode(token_ids),
        "spans": _token_spans(token_ids, tokenizer, None),
        "relabels": relabels,
        "layers": layers,
        "totals": {
            "recurrent_bytes": sum(
                layer["allocated_bytes"] for layer in recurrent_layers
            ),
            "kv_used_bytes": kv_used,
            "kv_allocated_bytes": kv_allocated,
            "kv_bytes_per_token_per_layer": kv_per_token_per_layer,
        },
    }


@api.get("/cache/tokens")
async def cache_tokens(request: Request):
    """Token ids per entry; plain ints under the store lock, safe on the loop."""
    ctx = _ctx(request)
    entries = ctx.cache.token_lists()
    # Token labels are ordinary CPU data. Keep the current cache readable even
    # before a request exists, without inspecting or copying MLX layer state.
    if ctx.loaded is not None:
        for entry in entries:
            entry["text_excerpt"] = ctx.loaded.tokenizer.decode(entry["token_ids"][:128])
            entry["excerpt_truncated"] = len(entry["token_ids"]) > 128
    return {"entries": entries}


@api.get("/cache/entries/{entry_id}")
async def cache_entry(entry_id: int, request: Request):
    ctx = _ctx(request)
    if ctx.active_generation is not None:
        raise HTTPException(
            status_code=409,
            detail="a generation is running; inspect cache entries when it finishes",
        )
    if ctx.loaded is None:
        raise HTTPException(status_code=409, detail="load the model first")
    entry = ctx.cache.get_entry(entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"unknown cache entry {entry_id}")
    layer_types = (ctx.loaded.info or {}).get("layer_types") or None
    return await asyncio.get_running_loop().run_in_executor(
        ctx.executor, _read_cache_entry, entry, ctx.loaded.tokenizer, layer_types
    )


# ---------------------------------------------------------------- cache edit
@api.delete("/cache/entries/{entry_id}")
async def delete_cache_entry(entry_id: int, request: Request):
    ctx = _ctx(request)
    if ctx.active_generation is not None:
        raise HTTPException(
            status_code=409,
            detail="a generation is running; wait for it to finish before editing the cache",
        )
    info = ctx.cache.remove(entry_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"unknown cache entry {entry_id}")
    return info


@api.post("/cache/entries/{entry_id}/relabel")
async def relabel_entry(entry_id: int, payload: RelabelRequest, request: Request):
    ctx = _ctx(request)
    if ctx.active_generation is not None:
        raise HTTPException(
            status_code=409,
            detail="a generation is running; wait for it to finish before editing the cache",
        )
    if ctx.loaded is None:
        raise HTTPException(status_code=409, detail="load the model first")
    entry = ctx.cache.get_entry(entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"unknown cache entry {entry_id}")
    if payload.position < 0 or payload.position >= len(entry.tokens):
        raise HTTPException(
            status_code=400,
            detail=(
                f"position {payload.position} is outside the entry "
                f"({len(entry.tokens)} tokens)"
            ),
        )
    ids = relabel_token_ids(ctx.loaded.tokenizer, payload.text)
    if len(ids) != 1:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{payload.text!r} is {len(ids)} tokens; relabel replaces "
                "exactly one token"
            ),
        )
    info = ctx.cache.relabel(entry_id, payload.position, ids[0])
    if info is None:
        raise HTTPException(status_code=404, detail=f"unknown cache entry {entry_id}")
    return info


@api.post("/cache/entries/{entry_id}/restore")
async def restore_entry(entry_id: int, request: Request):
    ctx = _ctx(request)
    if ctx.active_generation is not None:
        raise HTTPException(
            status_code=409,
            detail="a generation is running; wait for it to finish before editing the cache",
        )
    info = ctx.cache.restore(entry_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"unknown cache entry {entry_id}")
    return info


def _save_checkpoint(ctx: AppState, payload: CheckpointRequest) -> dict[str, Any]:
    """Snapshot the cache at the end of message ``after_message``.

    Runs on the MLX worker thread: builds the prompt, takes the boundary the
    automatic checkpoint logic uses (``checkpoint_boundary``), reuses whatever
    prefix the cache offers, prefills up to that boundary with the shared
    chunked prefill, and inserts an entry of kind ``manual``.
    """
    loaded = ctx.loaded
    tokenizer = loaded.tokenizer
    caps = capabilities(ctx.config, loaded.info)
    compat_key = _compat_key(ctx)
    report = build_prompt(
        payload.session,
        ctx.config,
        tokenizer,
        loaded.info,
        caps,
        cache_store=ctx.cache,
        compat_key=compat_key,
        include_pieces=False,
    )
    if not report["ok"]:
        raise HTTPException(
            status_code=400, detail=f"the prompt is not valid: {report['error']}"
        )
    tokens = report["token_ids"]
    if not tokens:
        raise HTTPException(
            status_code=400,
            detail="the prompt is empty; there is nothing to checkpoint",
        )

    # The boundary: the same computation build_prompt uses for its checkpoint
    # table, applied to the requested message.
    thinking_supported = bool(
        caps.get("enable_thinking", {}).get("supported", True)
    )
    template_kwargs = {
        key: value
        for key, value in ctx.config.template_kwargs.items()
        if key not in RESERVED_TEMPLATE_KWARGS
    }
    template_messages = []
    for item in report["submitted_messages"]:
        message: dict[str, Any] = {"role": item["role"], "content": item["content"]}
        if item["reasoning_content"]:
            message["reasoning_content"] = item["reasoning_content"]
        template_messages.append(message)
    if not 1 <= payload.after_message <= len(template_messages):
        raise HTTPException(
            status_code=400,
            detail=(
                f"after_message {payload.after_message} is out of range: the "
                f"request has {len(template_messages)} enabled message(s)"
            ),
        )
    boundary = checkpoint_boundary(
        template_messages=template_messages,
        token_ids=tokens,
        config=ctx.config,
        tokenizer=tokenizer,
        thinking_supported=thinking_supported,
        template_kwargs=template_kwargs,
        after_message=payload.after_message,
    )
    if not boundary["ok"]:
        raise HTTPException(
            status_code=400,
            detail=(
                f"no boundary after message {payload.after_message}: "
                f"{boundary['reason']}"
            ),
        )
    position = boundary["position"]

    # Reuse whatever prefix the cache offers, then prefill up to the boundary.
    cache = None
    reused = 0
    if ctx.config.cache_reuse:
        cache, reused, lookup = ctx.cache.lookup(tokens, compat_key)
    else:
        lookup = {
            "entry_id": None,
            "kind": None,
            "entry_tokens": 0,
            "reused_tokens": 0,
            "relabeled": False,
            "reason": "cache_reuse is off",
        }
        ctx.cache.note_lookup(lookup)
    ctx.events.emit(
        "cache.lookup",
        {"generation_id": None, "prompt_tokens": len(tokens), **lookup},
    )

    if cache is None:
        cache = make_prompt_cache(
            loaded.model,
            max_kv_size=(
                ctx.config.max_kv_size
                if not loaded.info.get("has_make_cache")
                else None
            ),
        )
        reused = 0

    prefill = prefill_to(
        model=loaded.model,
        tokens=tokens,
        cache=cache,
        start=reused,
        end=position,
        config=ctx.config,
        emit=ctx.events.emit,
        cancel=threading.Event(),
        counters=ctx.counters,
        stream=mx.default_stream(mx.default_device()),
        gen_id=None,
    )

    # Snapshot and store (or mark used when an identical entry exists).
    key = tokens[:position]
    existing = ctx.cache.touch_exact(key, compat_key)
    used_existing = existing is not None
    if existing is not None:
        info = existing
        ctx.events.emit(
            "cache.checkpoint",
            {
                "generation_id": None,
                "after_message": payload.after_message,
                "position": position,
                "used_existing": True,
                "detail": (
                    "an identical entry is already stored; marked used "
                    "instead of copying the cache again"
                ),
                **existing,
            },
        )
    else:
        snapshot = copy.deepcopy(cache)
        info = ctx.cache.insert(
            key,
            snapshot,
            "manual",
            compat_key,
            max_entries=ctx.config.cache_max_entries,
            max_bytes=int(ctx.config.cache_max_gb * 1_000_000_000),
            after_message=payload.after_message,
        )
        ctx.events.emit(
            "cache.checkpoint",
            {
                "generation_id": None,
                "after_message": payload.after_message,
                "position": position,
                "used_existing": False,
                **info,
            },
        )
    return {
        "after_message": payload.after_message,
        "position": position,
        "kind": "manual",
        "used_existing": used_existing,
        "prefill": {"reused_tokens": reused, **prefill},
        **info,
    }


@api.post("/cache/checkpoint")
async def save_checkpoint(payload: CheckpointRequest, request: Request):
    ctx = _ctx(request)
    if ctx.active_generation is not None:
        raise HTTPException(
            status_code=409,
            detail="a generation is running; wait for it to finish before saving a checkpoint",
        )
    if ctx.loaded is None:
        raise HTTPException(status_code=409, detail="load the model first")
    if payload.after_message < 1:
        raise HTTPException(
            status_code=400, detail="after_message is 1-based and must be >= 1"
        )
    return await asyncio.get_running_loop().run_in_executor(
        ctx.executor, _save_checkpoint, ctx, payload
    )


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
