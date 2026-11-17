"""One Generate request from start to finish, on the single MLX worker thread.

Why the manual prefill?  ``generate_step`` would prefill the prompt itself, but
we want to (a) split prefill at the cache checkpoint, (b) emit one event per
chunk with real timings, and (c) check the cancel Event between chunks.  So we
feed ``tokens[reused:-1]`` ourselves, then hand ``generate_step`` just the last
prompt token.  From there mlx-lm's own loop runs and yields every token.

Why do logits processors get wrapped?  mlx-lm builds the processor's token
history from the tokens passed to ``generate_step`` plus generated ones.  With
cache reuse we feed only the final prompt token, so penalties would silently
lose the earlier prompt.  The wrapper prepends ``tokens[:-1]`` (the last token
is the first one generate_step sees) so processors always see the full history.

Why one cache key of prompt + yielded tokens?  ``generate_step`` feeds a token
into the model *before* yielding it, so after n yielded tokens the cache holds
prompt + those n tokens.  That is exactly what we store, and we verify it
against ``KVCache.offset`` before inserting.
"""

from __future__ import annotations

import copy
import threading
import time
import traceback
from typing import Any, Callable

import mlx.core as mx
from mlx_lm.generate import (
    generate_step,
    generation_stream,
    maybe_quantize_kv_cache,
    wired_limit,
)
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_logits_processors, make_sampler

from config import AppConfig
from model import LoadedModel, capabilities, memory_info
from parsing import StopMatcher, ThinkingSplitter
from prompt import Session, build_prompt
from prompt_cache import PromptCacheStore


class GenerationStore:
    """Thread-safe store of generation records (the worker writes, HTTP reads)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._records: dict[str, dict[str, Any]] = {}

    def create(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._records[record["id"]] = record

    def update(self, gen_id: str, **fields: Any) -> None:
        with self._lock:
            self._records[gen_id].update(fields)

    def append_token(
        self, gen_id: str, record: dict[str, Any], segments: list[tuple[str, str]]
    ) -> None:
        with self._lock:
            entry = self._records[gen_id]
            entry["tokens"].append(record)
            entry["tokens_generated"] = len(entry["tokens"])
            entry["channels"]["raw"] += record["text"]
            for channel, text in segments:
                entry["channels"][channel] += text

    def finish_channels(
        self,
        gen_id: str,
        extra_raw: str,
        tail_segments: list[tuple[str, str]],
    ) -> None:
        """Attach detokenizer-finalize text and flushed parser text.

        ``extra_raw`` belongs to the last token (the streaming detokenizer was
        holding it back), so it is appended there.  Raw therefore stays equal to
        the concatenation of per-token text.
        """
        with self._lock:
            entry = self._records[gen_id]
            if extra_raw:
                entry["channels"]["raw"] += extra_raw
                if entry["tokens"]:
                    entry["tokens"][-1]["text"] += extra_raw
                else:
                    entry["raw_tail"] = entry.get("raw_tail", "") + extra_raw
            for channel, text in tail_segments:
                entry["channels"][channel] += text
                if entry["tokens"]:
                    entry["tokens"][-1]["segments"].append(
                        {"channel": channel, "text": text}
                    )
                    if entry["tokens"][-1]["channel"] is None:
                        entry["tokens"][-1]["channel"] = channel

    def get(self, gen_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._records.get(gen_id)
            return copy.deepcopy(record) if record is not None else None

    def progress(self, gen_id: str | None) -> dict[str, Any] | None:
        if gen_id is None:
            return None
        with self._lock:
            record = self._records.get(gen_id)
            if record is None:
                return None
            return {
                "id": gen_id,
                "status": record.get("status"),
                "phase": record.get("phase"),
                "tokens_generated": record.get("tokens_generated", 0),
                "prompt_hash": record.get("prompt_hash"),
                "started_at": record.get("started_at"),
            }

    def get_raw(self, gen_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._records.get(gen_id)

    def count(self) -> int:
        with self._lock:
            return len(self._records)


def _top_logprobs(logprobs: mx.array, count: int, tokenizer: Any) -> list[dict]:
    """Top-k alternatives for one token (a small top-k over the 248k vocab)."""
    if count <= 0:
        return []
    size = logprobs.shape[0]
    count = min(count, int(size))
    if count == 0:
        return []
    indices = mx.argpartition(-logprobs, kth=count - 1)[:count]
    values = logprobs[indices]
    order = mx.argsort(-values)
    indices = indices[order].tolist()
    values = values[order].tolist()
    return [
        {
            "id": int(token_id),
            "piece": tokenizer.convert_ids_to_tokens(int(token_id)),
            "logprob": round(float(value), 6),
        }
        for token_id, value in zip(indices, values)
    ]


def _processor_names(config: AppConfig) -> list[str]:
    names = []
    if config.logit_bias:
        names.append("logit_bias")
    if config.repetition_penalty != 1.0:
        names.append("repetition_penalty")
    if config.presence_penalty != 0:
        names.append("presence_penalty")
    if config.frequency_penalty != 0:
        names.append("frequency_penalty")
    return names


def _effective_settings(
    config: AppConfig,
    loaded: LoadedModel,
    eos_ids: set[int],
    max_new: int,
    caps: dict[str, Any],
) -> dict[str, Any]:
    inactive = {
        name: entry["reason"]
        for name, entry in caps.items()
        if entry.get("inactive") and entry.get("reason")
    }
    return {
        "sampler": {
            "temp": config.temperature,
            "top_p": config.top_p,
            "min_p": config.min_p,
            "min_tokens_to_keep": config.min_tokens_to_keep,
            "top_k": config.top_k,
            "xtc_probability": config.xtc_probability,
            "xtc_threshold": config.xtc_threshold,
        },
        "greedy": config.temperature == 0,
        "logits_processors": _processor_names(config),
        "kv": {
            "kv_bits": config.kv_bits,
            "kv_group_size": config.kv_group_size,
            "quantized_kv_start": config.quantized_kv_start,
        },
        "prefill_step_size": config.prefill_step_size,
        "max_new_tokens": max_new,
        "max_tokens": config.max_tokens,
        "context_budget": config.context_budget,
        "seed": config.seed,
        "top_logprobs": config.top_logprobs,
        "eos_token_ids": sorted(eos_ids),
        "stop_sequences": list(config.stop_sequences),
        "cache_reuse": config.cache_reuse,
        "cache_checkpoints": config.cache_checkpoints,
        "checkpoint_at": None,
        "checkpoint_positions": [],
        "inactive": inactive,
        "model": loaded.load_config,
    }


def prefill_to(
    *,
    model: Any,
    tokens: list[int],
    cache: Any,
    start: int,
    end: int,
    config: AppConfig,
    emit: Callable[[str, dict[str, Any]], Any],
    cancel: threading.Event | None,
    counters: dict[str, int],
    stream: Any,
    gen_id: str | None = None,
    planned_checkpoints: list[dict[str, Any]] | None = None,
    cache_store: PromptCacheStore | None = None,
    compat_key: tuple | None = None,
) -> dict[str, Any]:
    """Feed ``tokens[start:end]`` to the model in ``prefill_step_size`` chunks.

    The chunked prefill loop, shared by ``run_generation`` and the manual
    checkpoint endpoint, so both feed the model identically.  Each chunk ends
    exactly at the next planned checkpoint (a clean token boundary), emits
    one ``prefill.chunk`` event, and snapshots planned checkpoints exactly
    where a generation would.  Returns prefill stats plus a cancelled flag.
    """
    planned = planned_checkpoints or []
    prefill_started = time.perf_counter()
    prefill_seconds = 0.0
    chunks = 0
    processed = 0
    cancelled = False
    with mx.stream(stream):
        offset = start
        while offset < end:
            if cancel is not None and cancel.is_set():
                cancelled = True
                break
            step = min(config.prefill_step_size, end - offset)
            # End this chunk exactly at the next checkpoint so the snapshot
            # is a clean token boundary.
            next_position = next(
                (
                    item["position"]
                    for item in planned
                    if item["position"] is not None and item["position"] > offset
                ),
                None,
            )
            if next_position is not None and next_position < offset + step:
                step = next_position - offset
            chunk_started = time.perf_counter()
            model(mx.array(tokens[offset : offset + step], dtype=mx.int32)[None], cache=cache)
            maybe_quantize_kv_cache(
                cache,
                quantized_kv_start=config.quantized_kv_start,
                kv_group_size=config.kv_group_size,
                kv_bits=config.kv_bits,
            )
            mx.eval([item.state for item in cache])
            mx.clear_cache()
            chunk_seconds = time.perf_counter() - chunk_started
            prefill_seconds += chunk_seconds
            offset += step
            processed += step
            chunks += 1
            counters["forward_calls_total"] += 1
            emit(
                "prefill.chunk",
                {
                    "generation_id": gen_id,
                    "chunk_tokens": step,
                    "processed_tokens": processed,
                    "total_tokens": end - start,
                    "reused_tokens": start,
                    "seconds": round(chunk_seconds, 4),
                    "tokens_per_second": (
                        round(step / chunk_seconds, 1) if chunk_seconds else None
                    ),
                },
            )
            for entry in planned:
                if entry["position"] != offset:
                    continue
                position = entry["position"]
                key = tokens[:position]
                existing = cache_store.touch_exact(key, compat_key)
                if existing is not None:
                    emit(
                        "cache.checkpoint",
                        {
                            "generation_id": gen_id,
                            "after_message": entry["after_message"],
                            "position": position,
                            "used_existing": True,
                            "detail": (
                                "an identical entry is already stored; marked used "
                                "instead of copying the cache again"
                            ),
                            **existing,
                        },
                    )
                    continue
                snapshot_started = time.perf_counter()
                snapshot = copy.deepcopy(cache)
                info = cache_store.insert(
                    key,
                    snapshot,
                    "checkpoint",
                    compat_key,
                    max_entries=config.cache_max_entries,
                    max_bytes=int(config.cache_max_gb * 1_000_000_000),
                    after_message=entry["after_message"],
                )
                emit(
                    "cache.checkpoint",
                    {
                        "generation_id": gen_id,
                        "after_message": entry["after_message"],
                        "position": position,
                        "used_existing": False,
                        "seconds": round(
                            time.perf_counter() - snapshot_started, 4
                        ),
                        **info,
                    },
                )
    return {
        "processed_tokens": processed,
        "chunks": chunks,
        "seconds": round(prefill_seconds, 4),
        "wall_seconds": round(time.perf_counter() - prefill_started, 4),
        "tokens_per_second": (
            round(processed / prefill_seconds, 1) if prefill_seconds else None
        ),
        "cancelled": cancelled,
    }


def run_generation(
    *,
    gen_id: str,
    session: Session,
    config: AppConfig,
    loaded: LoadedModel,
    cache_store: PromptCacheStore,
    generations: GenerationStore,
    emit: Callable[[str, dict[str, Any]], Any],
    cancel: threading.Event,
    counters: dict[str, int],
    compat_key: tuple,
    on_finished: Callable[[str], None],
) -> None:
    started = time.time()
    tokenizer = loaded.tokenizer
    model = loaded.model
    caps = capabilities(config, loaded.info)

    generations.update(gen_id, status="running", phase="prompt", started_at=started)

    try:
        prompt_started = time.perf_counter()
        report = build_prompt(
            session,
            config,
            tokenizer,
            loaded.info,
            caps,
            cache_store=cache_store,
            compat_key=compat_key,
            include_pieces=True,
        )
        prompt_seconds = time.perf_counter() - prompt_started

        tokens = report["token_ids"]
        generations.update(
            gen_id,
            prompt=report,
            prompt_hash=report["prompt_hash"],
            phase="prefill" if report["ok"] else "done",
            prompt_seconds=round(prompt_seconds, 4),
        )

        if not report["ok"]:
            generations.update(
                gen_id,
                finish={
                    "reason": "context_overflow",
                    "detail": report["error"],
                    "budget": report["budget"],
                },
                stats={"prompt_build_seconds": round(prompt_seconds, 4)},
            )
            return
        if not tokens:
            generations.update(
                gen_id,
                finish={
                    "reason": "error",
                    "detail": "the prompt is empty; there is nothing to feed",
                },
            )
            return

        eos_ids = set(int(item) for item in tokenizer.eos_token_ids)
        eos_ids |= set(config.extra_eos_token_ids)
        eos_pieces = {
            token_id: tokenizer.convert_ids_to_tokens(token_id)
            for token_id in eos_ids
        }

        max_new = min(config.max_tokens, config.context_budget - len(tokens))
        settings = _effective_settings(config, loaded, eos_ids, max_new, caps)
        settings["checkpoint_at"] = report["checkpoint_at"]
        settings["checkpoint_positions"] = [
            entry["position"] for entry in report["checkpoints"] if entry["ok"]
        ]
        generations.update(gen_id, effective_settings=settings)

        # ------------------------------------------------------------ sampler
        special_tokens = sorted(eos_ids)
        for marker in (loaded.info.get("think_start"), loaded.info.get("think_end")):
            if marker:
                marker_id = tokenizer.convert_tokens_to_ids(marker)
                if isinstance(marker_id, int) and marker_id >= 0:
                    special_tokens.append(marker_id)
        sampler = make_sampler(
            temp=config.temperature,
            top_p=config.top_p,
            min_p=config.min_p,
            min_tokens_to_keep=config.min_tokens_to_keep,
            top_k=config.top_k,
            xtc_probability=config.xtc_probability,
            xtc_threshold=config.xtc_threshold,
            xtc_special_tokens=sorted(set(special_tokens)),
        )

        processors = make_logits_processors(
            logit_bias=config.logit_bias or None,
            repetition_penalty=(
                None if config.repetition_penalty == 1.0 else config.repetition_penalty
            ),
            repetition_context_size=config.repetition_context_size,
            presence_penalty=(
                None if config.presence_penalty == 0 else config.presence_penalty
            ),
            presence_context_size=config.presence_context_size,
            frequency_penalty=(
                None if config.frequency_penalty == 0 else config.frequency_penalty
            ),
            frequency_context_size=config.frequency_context_size,
        )

        # generate_step's first token is tokens[-1], so the prefix is all but it.
        prefix = mx.array(tokens[:-1], dtype=mx.int32)

        def wrap_processor(processor):
            def wrapped(seen: mx.array, logits: mx.array) -> mx.array:
                seen = seen.astype(mx.int32)
                full = (
                    mx.concatenate([prefix, seen]) if seen.size else prefix
                )
                return processor(full, logits)

            return wrapped

        wrapped_processors = [wrap_processor(p) for p in processors]

        # ------------------------------------------------------------- cache
        cache = None
        reused = 0
        if config.cache_reuse:
            cache, reused, lookup = cache_store.lookup(tokens, compat_key)
        else:
            lookup = {
                "entry_id": None,
                "kind": None,
                "entry_tokens": 0,
                "reused_tokens": 0,
                "relabeled": False,
                "reason": "cache_reuse is off",
            }
            cache_store.note_lookup(lookup)
        # Keep the actual lookup with its request; later checkpoint operations
        # also update the cache store's global last_lookup.
        generations.update(gen_id, cache_lookup=dict(lookup))
        emit(
            "cache.lookup",
            {
                "generation_id": gen_id,
                "prompt_tokens": len(tokens),
                **lookup,
            },
        )

        if cache is None:
            cache = make_prompt_cache(
                model,
                max_kv_size=(
                    config.max_kv_size
                    if not loaded.info.get("has_make_cache")
                    else None
                ),
            )
            reused = 0

        # ----------------------------------------------------------- prefill
        feed_end = len(tokens) - 1
        # Accepted boundaries that fall after the reused prefix and before the
        # final prompt token (which generate_step feeds itself).
        planned_checkpoints: list[dict[str, Any]] = []
        if config.cache_reuse and config.cache_checkpoints != "off":
            planned_checkpoints = sorted(
                (
                    entry
                    for entry in report["checkpoints"]
                    if entry["ok"]
                    and entry["position"] is not None
                    and reused < entry["position"] <= feed_end
                ),
                key=lambda item: item["position"],
            )
        prefill_started = time.perf_counter()
        mx.reset_peak_memory()
        if config.seed is not None:
            mx.random.seed(config.seed)

        prefill = prefill_to(
            model=model,
            tokens=tokens,
            cache=cache,
            start=reused,
            end=feed_end,
            config=config,
            emit=emit,
            cancel=cancel,
            counters=counters,
            stream=generation_stream,
            gen_id=gen_id,
            planned_checkpoints=planned_checkpoints,
            cache_store=cache_store,
            compat_key=compat_key,
        )
        cancelled = prefill["cancelled"]
        prefill_seconds = prefill["seconds"]
        chunks = prefill["chunks"]
        processed = prefill["processed_tokens"]
        prefill_total_seconds = time.perf_counter() - prefill_started
        generations.update(gen_id, phase="decode" if not cancelled else "done")

        if cancelled:
            generations.update(
                gen_id,
                finish={
                    "reason": "cancelled",
                    "detail": "stopped during prefill",
                    "phase": "prefill",
                    "tokens_generated": 0,
                },
                stats={
                    "prompt_build_seconds": round(prompt_seconds, 4),
                    "prefill": {
                        "reused_tokens": reused,
                        "processed_tokens": processed,
                        "chunks": chunks,
                        "seconds": round(prefill_seconds, 4),
                        "wall_seconds": round(prefill_total_seconds, 4),
                        "tokens_per_second": (
                            round(processed / prefill_seconds, 1)
                            if prefill_seconds
                            else None
                        ),
                    },
                    "memory": memory_info(),
                    "forward_calls": counters["forward_calls_total"],
                },
            )
            return

        # ------------------------------------------------------------ decode
        detokenizer = tokenizer.detokenizer
        stop_matcher = StopMatcher(config.stop_sequences)
        splitter = ThinkingSplitter(
            loaded.info.get("think_start"),
            loaded.info.get("think_end"),
            report["starts_in_thinking"],
        )

        # ``generate_step`` feeds a token into the model *before* yielding it, so
        # the cache also contains a yielded EOS token (which we never record as
        # output).  Keep the full list for the cache key and offset check.
        yielded_ids: list[int] = []
        token_count = 0
        finish: dict[str, Any] | None = None
        time_to_first_token: float | None = None
        last_token_time: float | None = None
        decode_started = time.perf_counter()

        gen = generate_step(
            mx.array(tokens[-1:], dtype=mx.int32),
            model,
            max_tokens=max_new,
            sampler=sampler,
            logits_processors=wrapped_processors or None,
            prompt_cache=cache,
            prefill_step_size=config.prefill_step_size,
            kv_bits=config.kv_bits,
            kv_group_size=config.kv_group_size,
            quantized_kv_start=config.quantized_kv_start,
        )

        with wired_limit(model, [generation_stream]):
            emit(
                "runtime.wired_limit",
                {
                    "max_recommended_working_set_size": int(
                        mx.device_info().get("max_recommended_working_set_size", 0)
                    ),
                    "note": (
                        "generation runs inside mlx_lm.generate.wired_limit, which "
                        "temporarily raises the wired limit to the recommended size"
                    ),
                },
            )
            try:
                for token_id, logprobs in gen:
                    now = time.perf_counter()
                    if time_to_first_token is None:
                        time_to_first_token = now - decode_started
                        # The model call on the last prompt token, which
                        # produced this first token.
                        counters["forward_calls_total"] += 1
                    # generate_step works one step ahead: before yielding a
                    # token it has already fed it to the model to compute the
                    # next one.  So every yielded token = one more model call,
                    # and a run of n tokens makes n + 1 decode calls.
                    counters["forward_calls_total"] += 1
                    token_id = int(token_id)
                    yielded_ids.append(token_id)
                    piece = tokenizer.convert_ids_to_tokens(token_id)

                    if token_id in eos_ids:
                        finish = {
                            "reason": "eos",
                            "detail": f"token {token_id} ({eos_pieces.get(token_id)})",
                            "token_id": token_id,
                            "piece": eos_pieces.get(token_id),
                        }
                        break

                    detokenizer.add_token(token_id)
                    text = detokenizer.last_segment
                    safe, match = stop_matcher.feed(text)
                    segments = splitter.feed(safe)
                    channel = None
                    if segments:
                        channel = segments[0][0]
                    elif text:
                        channel = "marker"

                    logprob_value = float(logprobs[token_id].item())
                    record = {
                        "index": token_count,
                        "id": token_id,
                        "piece": piece,
                        "text": text,
                        "segments": [
                            {"channel": name, "text": chunk}
                            for name, chunk in segments
                        ],
                        "channel": channel,
                        "logprob": round(logprob_value, 6),
                        "top_logprobs": _top_logprobs(
                            logprobs, config.top_logprobs, tokenizer
                        ),
                        "time_ms": round((now - decode_started) * 1000, 2),
                        "stop": match is not None,
                    }
                    generations.append_token(gen_id, record, segments)
                    emit(
                        "token",
                        {"generation_id": gen_id, **record},
                    )
                    token_count += 1
                    last_token_time = now

                    if match is not None:
                        finish = {
                            "reason": "stop_sequence",
                            "detail": (
                                f"matched {match.sequence!r} at character "
                                f"offset {match.offset}"
                            ),
                            "sequence": match.sequence,
                            "offset": match.offset,
                        }
                        break
                    if token_count >= max_new:
                        finish = {
                            "reason": "max_tokens",
                            "detail": (
                                "max_tokens"
                                if config.max_tokens <= config.context_budget - len(tokens)
                                else "remaining context budget"
                            ),
                            "limit": max_new,
                        }
                        break
                    if cancel.is_set():
                        finish = {
                            "reason": "cancelled",
                            "detail": "stopped during decode",
                            "phase": "decode",
                        }
                        break
            finally:
                gen.close()

        decode_seconds = time.perf_counter() - decode_started

        # Release text the streaming detokenizer held back, then the parsers.
        detokenizer.finalize()
        finalize_text = detokenizer.last_segment
        tail_segments: list[tuple[str, str]] = []
        if finalize_text and (finish is None or finish["reason"] != "stop_sequence"):
            safe, match = stop_matcher.feed(finalize_text)
            tail_segments.extend(splitter.feed(safe))
            if match is not None:
                finish = {
                    "reason": "stop_sequence",
                    "detail": (
                        f"matched {match.sequence!r} at character "
                        f"offset {match.offset}"
                    ),
                    "sequence": match.sequence,
                    "offset": match.offset,
                }
        if finish is None or finish["reason"] != "stop_sequence":
            tail_segments.extend(splitter.feed(stop_matcher.flush()))
        tail_segments.extend(splitter.flush())
        generations.finish_channels(gen_id, finalize_text, tail_segments)

        if finish is None:
            finish = {"reason": "error", "detail": "generation ended without a reason"}

        decode_rate = None
        if token_count > 1 and last_token_time is not None and time_to_first_token:
            span = last_token_time - decode_started - time_to_first_token
            if span > 0:
                decode_rate = round((token_count - 1) / span, 1)

        stats = {
            "prompt_build_seconds": round(prompt_seconds, 4),
            "prefill": {
                "reused_tokens": reused,
                "processed_tokens": processed,
                "chunks": chunks,
                "seconds": round(prefill_seconds, 4),
                "wall_seconds": round(prefill_total_seconds, 4),
                "tokens_per_second": (
                    round(processed / prefill_seconds, 1) if prefill_seconds else None
                ),
            },
            "decode": {
                "generated_tokens": token_count,
                "seconds": round(decode_seconds, 4),
                "tokens_per_second": decode_rate,
                "time_to_first_token_seconds": (
                    round(time_to_first_token, 4)
                    if time_to_first_token is not None
                    else None
                ),
            },
            "memory": memory_info(),
            "prompt_cache_bytes": cache_store.summary()["total_bytes"],
            "forward_calls": counters["forward_calls_total"],
            # prefill chunks + (last prompt token) + one lookahead per yielded token
            "forward_calls_this_run": chunks + (len(yielded_ids) + 1 if yielded_ids else 0),
        }

        # ----------------------------------------------------------- caching
        if config.cache_reuse and finish["reason"] != "error":
            key_tokens = tokens + yielded_ids
            offsets = [
                int(item.offset)
                for item in cache
                if hasattr(item, "offset") and item.offset is not None
            ]
            verified = all(offset == len(key_tokens) for offset in offsets)
            verification = {
                "generation_id": gen_id,
                "expected_offset": len(key_tokens),
                "kv_offsets": offsets,
                "ok": verified,
            }
            emit("cache.verify", verification)
            # Kept on the record so the UI and verify.py can show the check.
            generations.update(gen_id, cache_verify=verification)
            if verified:
                cache_store.insert(
                    key_tokens,
                    cache,
                    "after-generation",
                    compat_key,
                    max_entries=config.cache_max_entries,
                    max_bytes=int(config.cache_max_gb * 1_000_000_000),
                )

        generations.update(gen_id, finish=finish, stats=stats)

    except Exception as error:  # noqa: BLE001 - reported in the record, never retried
        generations.update(
            gen_id,
            finish={"reason": "error", "detail": f"{type(error).__name__}: {error}"},
            error=traceback.format_exc()[-6000:],
            stats={"prompt_build_seconds": 0},
        )
    finally:
        generations.update(gen_id, status="finished", phase="done", finished_at=time.time())
        on_finished(gen_id)
