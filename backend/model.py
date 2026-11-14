"""Model lifecycle and the read-only facts the UI shows about it.

Loading is the only place mlx_lm reads weights.  Everything that describes the
model -- architecture, layer mix, quantization baked into the weights, tokenizer
facts, chat-template variables, cache types, whether the cache can be trimmed --
is collected once here and served read-only.

``capabilities()`` is the answer to "will this control actually do something?".
It combines model facts (for example: Qwen3.5 defines make_cache, so mlx-lm
ignores max_kv_size) with the current config (for example: top_p does nothing
while temperature is 0).  The UI disables or marks controls from this map
instead of silently ignoring settings.
"""

from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load as mlx_load
from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache

from config import AppConfig, inactive_reasons
from prompt import template_variables
from prompt_cache import cache_types


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    info: dict[str, Any]
    load_config: dict[str, Any]
    load_seconds: float


def _local_dir(model_path: str, revision: str | None) -> str | None:
    """Resolve a repo id to the cached snapshot directory (small files only)."""
    if os.path.isdir(os.path.expanduser(model_path)):
        return os.path.expanduser(model_path)
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(
            repo_id=model_path,
            revision=revision,
            allow_patterns=["*.json", "*.jinja", "*.txt", "*.model"],
        )
    except Exception:  # noqa: BLE001 - info is best effort
        return None


def _read_config(local_dir: str | None) -> dict[str, Any]:
    if not local_dir:
        return {}
    path = os.path.join(local_dir, "config.json")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def device_info() -> dict[str, Any]:
    info = {key: _json_safe(value) for key, value in mx.device_info().items()}
    return info


def memory_info() -> dict[str, int]:
    return {
        "active_bytes": int(mx.get_active_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
        "cache_bytes": int(mx.get_cache_memory()),
    }


def read_mlx_cache_limit() -> int:
    """MLX has no getter for the buffer-cache limit; set_cache_limit returns
    the previous value, so set and immediately restore it."""
    previous = int(mx.set_cache_limit(0))
    mx.set_cache_limit(previous)
    return previous


def apply_mlx_cache_limit(limit_gb: float | None, default_bytes: int) -> dict[str, int]:
    """Apply config.mlx_cache_limit_gb now; None restores MLX's own default.

    This is an allocator-wide setting (how many freed buffers MLX keeps for
    reuse), safe to change between generations.
    """
    requested = default_bytes if limit_gb is None else int(limit_gb * 1_000_000_000)
    previous = int(mx.set_cache_limit(requested))
    return {"previous_bytes": previous, "new_bytes": requested, "mlx_default_bytes": default_bytes}


def _weight_bytes(model: Any) -> int:
    total = 0
    for _, value in tree_flatten(model.parameters()):
        if isinstance(value, mx.array):
            total += value.nbytes
    return total


def load_model(config: AppConfig) -> LoadedModel:
    started = time.perf_counter()
    model, tokenizer = mlx_load(
        config.model_path,
        adapter_path=config.adapter_path or None,
        lazy=config.lazy_load,
        revision=config.revision,
    )
    if not config.lazy_load:
        # mlx_lm.load already evaluates non-lazy weights; doing it again is a
        # cheap no-op and makes the timing boundary explicit.
        mx.eval(model.parameters())
    seconds = time.perf_counter() - started

    local_dir = _local_dir(config.model_path, config.revision)
    raw_config = _read_config(local_dir)
    text_config = raw_config.get("text_config", raw_config)

    cache = make_prompt_cache(model)
    cache_type_counts = cache_types(cache)
    trimmable = can_trim_prompt_cache(cache)
    del cache

    layer_types = text_config.get("layer_types") or []
    layer_mix: dict[str, int] = {}
    for layer_type in layer_types:
        layer_mix[layer_type] = layer_mix.get(layer_type, 0) + 1

    template = getattr(tokenizer, "chat_template", None)
    info = {
        "path": config.model_path,
        "resolved_path": local_dir,
        "architecture": (raw_config.get("architectures") or [None])[0],
        "model_type": raw_config.get("model_type"),
        "num_layers": text_config.get("num_hidden_layers"),
        "layer_mix": layer_mix,
        "layer_types": layer_types,
        "hidden_size": text_config.get("hidden_size"),
        "vocab_size": text_config.get("vocab_size") or len(tokenizer),
        "max_position_embeddings": text_config.get("max_position_embeddings"),
        "quantization": raw_config.get("quantization"),
        "weight_bytes": _weight_bytes(model),
        "has_make_cache": hasattr(model, "make_cache"),
        "cache_types": cache_type_counts,
        "cache_trimmable": trimmable,
        "eos_token_ids": sorted(int(item) for item in tokenizer.eos_token_ids),
        "has_thinking": bool(tokenizer.has_thinking),
        "think_start": tokenizer.think_start,
        "think_end": tokenizer.think_end,
        "detokenizer": type(tokenizer.detokenizer).__name__,
        "chat_template": template,
        "chat_template_variables": template_variables(template),
        "load_seconds": round(seconds, 3),
        "loaded_at": time.time(),
        "device": device_info(),
    }

    load_config = {
        "model_path": config.model_path,
        "revision": config.revision,
        "adapter_path": config.adapter_path,
        "lazy_load": config.lazy_load,
    }
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        info=info,
        load_config=load_config,
        load_seconds=seconds,
    )


def unload_model(loaded: LoadedModel) -> dict[str, Any]:
    before = int(mx.get_active_memory())
    del loaded.model
    gc.collect()
    mx.clear_cache()
    return {
        "active_bytes_before": before,
        "active_bytes_after": int(mx.get_active_memory()),
    }


def capabilities(config: AppConfig, info: dict[str, Any] | None) -> dict[str, Any]:
    """Per-field support / inactive reasons for the config UI."""
    caps: dict[str, Any] = {}

    caps["trust_remote_code"] = {
        "supported": False,
        "reason": (
            "Disabled: trust_remote_code executes Python from the model repo and "
            "MLX Qwen models do not need it."
        ),
    }

    if info:
        if info.get("has_make_cache"):
            caps["max_kv_size"] = {
                "supported": False,
                "reason": (
                    "This model defines make_cache, so mlx-lm ignores max_kv_size "
                    "(make_prompt_cache returns model.make_cache())."
                ),
            }
        attention_layers = int(info.get("cache_types", {}).get("KVCache", 0)) + int(
            info.get("cache_types", {}).get("QuantizedKVCache", 0)
        )
        if attention_layers == 0:
            caps["kv_bits"] = {
                "supported": False,
                "reason": "This model has no KV-cache attention layers to quantize.",
            }
        caps["speculative_decoding"] = {
            "supported": False,
            "reason": (
                "Not implemented. It also needs a draft model and a rewindable cache; "
                "this model's cache is "
                + ("trimmable." if info.get("cache_trimmable") else "not trimmable.")
            ),
        }

        template = config.chat_template_override or info.get("chat_template")
        variables = template_variables(template)
        if "enable_thinking" not in variables:
            caps["enable_thinking"] = {
                "supported": False,
                "reason": (
                    "The active chat template never reads enable_thinking, so "
                    "toggling it would have no effect."
                ),
            }
        budget = config.context_budget
        model_max = info.get("max_position_embeddings")
        if model_max and budget > model_max:
            caps["context_budget"] = {
                "supported": True,
                "warning": (
                    f"context_budget ({budget}) exceeds the model's "
                    f"max_position_embeddings ({model_max}); generation will refuse."
                ),
            }

    for name, reason in inactive_reasons(config).items():
        existing = caps.get(name)
        if existing and not existing.get("supported", True):
            continue
        caps[name] = {"supported": True, "inactive": True, "reason": reason}

    return caps
