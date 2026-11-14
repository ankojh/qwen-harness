"""A small, readable prompt-cache store built on mlx-lm cache primitives.

Each entry is a list of mlx-lm cache objects plus the exact token sequence it
represents.  Reuse is only possible when a stored token sequence is an exact
prefix of the new prompt (or when the model's cache supports trimming, which
Qwen3.5's does not: 24 ArraysCache layers return is_trimmable() == False).

The same idea exists in ``mlx_lm.models.cache.LRUPromptCache`` (used by
``mlx_lm.server``); it is written out here so the reuse decision can be
reported token by token in the UI.

Entries are keyed by a compatibility tuple, because a cache built with
different KV quantization or a different model cannot be reused.  The store is
in memory only and lives in ``app.py``; every insert/evict is emitted as an
event.
"""

from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache

Emit = Callable[[str, dict[str, Any]], Any]


def _locked(method):
    """Run one store method under the store lock.

    The MLX worker inserts entries while the event loop reads for /api/state or
    a preview, so the entry list and byte counter need a lock.  It is an RLock
    because insert/summary read counters that other locked methods touch.
    """

    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


@dataclass
class CacheEntry:
    id: int
    tokens: list[int]
    cache: list[Any]
    kind: str  # "checkpoint" or "after-generation"
    compat_key: tuple
    nbytes: int
    arrays_bytes: int = 0  # recurrent state of the 24 linear-attention layers
    kv_bytes: int = 0  # KV of the 8 full-attention layers
    after_message: int | None = None  # only for checkpoints
    created: float = field(default_factory=time.time)
    last_used: int = 0


BYTE_NOTE = (
    "Accounted bytes are summed cache.nbytes. MLX shares buffers between "
    "identical arrays: three deepcopies of one state used no extra memory in a "
    "measurement on this machine, so real usage can be lower than accounted. "
    "Snapshots at different positions each hold their own recurrent state and "
    "KV buffers."
)


def cache_nbytes(cache: list[Any]) -> int:
    return sum(int(getattr(item, "nbytes", 0)) for item in cache)


def cache_byte_split(cache: list[Any]) -> tuple[int, int]:
    """(recurrent-state bytes, KV bytes) for one cache snapshot.

    Qwen3.5's cache is 24 ``ArraysCache`` layers plus 8 ``KVCache`` (or
    ``QuantizedKVCache``) layers.  The recurrent state is a fixed size; the KV
    grows with the token position, which is why the estimate can scale it.
    """
    arrays = 0
    kv = 0
    for item in cache:
        nbytes = int(getattr(item, "nbytes", 0))
        if type(item).__name__.endswith("ArraysCache"):
            arrays += nbytes
        else:
            kv += nbytes
    return arrays, kv


def cache_types(cache: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in cache:
        name = type(item).__name__
        counts[name] = counts.get(name, 0) + 1
    return counts


def _array_metas(name: str, arr: Any) -> list[dict[str, Any]]:
    """Shape/dtype/size metadata for one array or a quantized list of arrays."""
    if arr is None:
        return []
    if isinstance(arr, (list, tuple)):
        return [
            {
                "name": f"{name}[{position}]",
                "shape": list(item.shape),
                "dtype": str(item.dtype),
                "nbytes": int(item.nbytes),
            }
            for position, item in enumerate(arr)
        ]
    return [
        {
            "name": name,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "nbytes": int(arr.nbytes),
        }
    ]


def _allocated_tokens(keys: Any) -> int | None:
    """Token capacity of a KV layer (axis 2 of the keys buffer)."""
    if isinstance(keys, (list, tuple)):
        return int(keys[0].shape[2]) if keys else None
    if keys is not None and hasattr(keys, "shape") and len(keys.shape) > 2:
        return int(keys.shape[2])
    return None


def describe_cache_layers(
    cache: list[Any], layer_types: list[str] | None = None
) -> list[dict[str, Any]]:
    """Per-layer metadata for one stored cache snapshot.

    Read-only inspection of mlx-lm cache objects: shapes, dtypes and byte
    sizes only, never the values inside.  Must run on the MLX worker thread
    (``ctx.executor``), because it touches MLX array objects.  For KVCache
    layers ``used_tokens`` is the offset and ``allocated_tokens`` the buffer
    capacity (a multiple of ``KVCache.step``); ArraysCache layers have no
    per-token split and report null token fields.
    """
    rows: list[dict[str, Any]] = []
    for index, layer in enumerate(cache):
        cls = type(layer).__name__
        kind = "linear_attention" if cls.endswith("ArraysCache") else "full_attention"
        if layer_types and index < len(layer_types):
            # The model config says which layer index is which; prefer it.
            kind = (
                "linear_attention"
                if layer_types[index] == "linear_attention"
                else "full_attention"
            )
        arrays: list[dict[str, Any]] = []
        keys = getattr(layer, "keys", None)
        if keys is not None:
            arrays.extend(_array_metas("keys", keys))
            arrays.extend(_array_metas("values", getattr(layer, "values", None)))
            allocated_tokens = _allocated_tokens(keys)
            used_tokens = int(getattr(layer, "offset", 0) or 0)
            allocated_bytes = int(layer.nbytes)
            used_bytes = (
                round(allocated_bytes * used_tokens / allocated_tokens)
                if allocated_tokens
                else 0
            )
        else:
            state = getattr(layer, "cache", None)
            if isinstance(state, (list, tuple)):
                for position, arr in enumerate(state):
                    if arr is not None:
                        arrays.extend(_array_metas(f"state[{position}]", arr))
            used_tokens = None
            allocated_tokens = None
            allocated_bytes = int(layer.nbytes)
            used_bytes = allocated_bytes
        rows.append(
            {
                "index": index,
                "class": cls,
                "kind": kind,
                "arrays": arrays,
                "used_tokens": used_tokens,
                "allocated_tokens": allocated_tokens,
                "used_bytes": used_bytes,
                "allocated_bytes": allocated_bytes,
            }
        )
    return rows


class PromptCacheStore:
    def __init__(self, emit: Emit | None = None):
        self._entries: list[CacheEntry] = []
        self._next_id = 0
        self._clock = 0
        self._total_bytes = 0
        self._evictions = 0
        self._emit = emit or (lambda type, data: None)
        self.last_lookup: dict[str, Any] | None = None
        # The MLX worker inserts while the event loop reads for /api/state.
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ keys
    @staticmethod
    def compat_key(
        *,
        model_path: str,
        revision: str | None,
        adapter_path: str | None,
        kv_bits: int | None,
        kv_group_size: int,
        quantized_kv_start: int,
    ) -> tuple:
        return (
            model_path,
            revision,
            adapter_path,
            kv_bits,
            kv_group_size,
            quantized_kv_start,
        )

    # --------------------------------------------------------------- lookup
    @staticmethod
    def _common_prefix(a: list[int], b: list[int]) -> int:
        size = min(len(a), len(b))
        index = 0
        while index < size and a[index] == b[index]:
            index += 1
        return index

    @_locked
    def _find(self, tokens: list[int], compat_key: tuple) -> dict[str, Any]:
        """Pick the best *usable* entry.

        A longer entry is not automatically better: on a non-trimmable cache it
        can only be used when it is an exact prefix.  So score every entry by
        how many tokens it can actually contribute, and keep the longest such
        entry (ties go to the most recently used).  The longest *matching*
        entry is kept separately only to explain a refusal.
        """
        limit = max(len(tokens) - 1, 0)
        best: CacheEntry | None = None
        best_reuse = 0
        closest: CacheEntry | None = None
        closest_common = -1

        for entry in self._entries:
            if entry.compat_key != compat_key:
                continue
            common = self._common_prefix(entry.tokens, tokens)
            if common > closest_common or (
                common == closest_common
                and (closest is None or entry.last_used > closest.last_used)
            ):
                closest, closest_common = entry, common

            reuse = min(common, limit)
            if reuse <= 0:
                continue
            if reuse < len(entry.tokens) and not can_trim_prompt_cache(entry.cache):
                # This entry diverges before its end and cannot be shortened.
                continue
            if (
                best is None
                or reuse > best_reuse
                or (reuse == best_reuse and entry.last_used > best.last_used)
            ):
                best, best_reuse = entry, reuse

        if best is not None:
            trim_by = 0 if best_reuse == len(best.tokens) else len(best.tokens) - best_reuse
            return {
                "entry": best,
                "reuse_tokens": best_reuse,
                "trim_by": trim_by,
                "reason": "",
            }

        if closest is None:
            return {
                "entry": None,
                "reuse_tokens": 0,
                "trim_by": 0,
                "reason": "no compatible cache entry",
            }

        if limit == 0:
            reason = "the prompt has only one token and it must be fed to the model"
        elif closest_common <= 0:
            reason = (
                f"closest entry #{closest.id} diverges at token 0 and the cache "
                "is not trimmable"
            )
        else:
            reason = (
                f"closest entry #{closest.id} ({closest.kind}, "
                f"{len(closest.tokens)} tokens) diverges at token {closest_common} "
                "and this model's cache is not trimmable"
            )
        return {
            "entry": closest,
            "reuse_tokens": 0,
            "trim_by": 0,
            "reason": reason,
        }

    def peek(self, tokens: list[int], compat_key: tuple) -> dict[str, Any]:
        """Dry-run prediction for the prompt inspector; does not touch LRU."""
        found = self._find(tokens, compat_key)
        entry: CacheEntry | None = found["entry"]
        trimmed = found["trim_by"] > 0
        if entry is None:
            reason = found["reason"]
        elif found["reuse_tokens"] == 0:
            reason = found["reason"]
        elif trimmed:
            reason = (
                f"would reuse {found['reuse_tokens']} of {len(tokens)} tokens from "
                f"{entry.kind} entry #{entry.id} after trimming {found['trim_by']}"
            )
        else:
            reason = (
                f"would reuse {found['reuse_tokens']} of {len(tokens)} tokens from "
                f"{entry.kind} entry #{entry.id}"
            )
        return {
            "entry_id": entry.id if entry else None,
            "kind": entry.kind if entry else None,
            "entry_tokens": len(entry.tokens) if entry else 0,
            "reused_tokens": found["reuse_tokens"],
            "would_trim": found["trim_by"],
            "reason": reason,
        }

    @_locked
    def lookup(
        self, tokens: list[int], compat_key: tuple
    ) -> tuple[list[Any] | None, int, dict[str, Any]]:
        """Return (cache copy, reused tokens, info).  Updates LRU on a hit."""
        found = self._find(tokens, compat_key)
        entry: CacheEntry | None = found["entry"]
        reuse = found["reuse_tokens"]

        if entry is None or reuse <= 0:
            info = {
                "entry_id": entry.id if entry else None,
                "kind": entry.kind if entry else None,
                "entry_tokens": len(entry.tokens) if entry else 0,
                "reused_tokens": 0,
                "reason": found["reason"],
            }
            self.last_lookup = info
            return None, 0, info

        cache = copy.deepcopy(entry.cache)
        trim_by = found["trim_by"]
        if trim_by:
            trim_prompt_cache(cache, trim_by)

        self._clock += 1
        entry.last_used = self._clock
        info = {
            "entry_id": entry.id,
            "kind": entry.kind,
            "entry_tokens": len(entry.tokens),
            "reused_tokens": reuse,
            "trimmed": trim_by,
            "reason": (
                f"reused {reuse} tokens from {entry.kind} entry #{entry.id}"
                + (f" after trimming {trim_by}" if trim_by else "")
            ),
        }
        self.last_lookup = info
        return cache, reuse, info

    # --------------------------------------------------------------- insert
    @_locked
    def insert(
        self,
        tokens: list[int],
        cache: list[Any],
        kind: str,
        compat_key: tuple,
        *,
        max_entries: int,
        max_bytes: int,
        after_message: int | None = None,
    ) -> dict[str, Any]:
        entry_bytes = cache_nbytes(cache)
        arrays_bytes, kv_bytes = cache_byte_split(cache)

        # Replace an identical entry (same compatible settings and token
        # sequence) instead of keeping two copies.  The kind may differ: a
        # checkpoint and an after-generation entry for the same tokens hold
        # the same cache state.
        for index, existing in enumerate(self._entries):
            if existing.tokens == tokens and existing.compat_key == compat_key:
                self._total_bytes -= existing.nbytes
                self._entries.pop(index)
                break

        self._next_id += 1
        self._clock += 1
        entry = CacheEntry(
            id=self._next_id,
            tokens=list(tokens),
            cache=cache,
            kind=kind,
            compat_key=compat_key,
            nbytes=entry_bytes,
            arrays_bytes=arrays_bytes,
            kv_bytes=kv_bytes,
            after_message=after_message,
            last_used=self._clock,
        )
        self._entries.append(entry)
        self._total_bytes += entry_bytes

        info = {
            "entry_id": entry.id,
            "kind": kind,
            "after_message": after_message,
            "tokens": len(tokens),
            "nbytes": entry_bytes,
            "arrays_bytes": arrays_bytes,
            "kv_bytes": kv_bytes,
            "entries": len(self._entries),
            "total_bytes": self._total_bytes,
            "evictions_total": self._evictions,
        }
        self._emit("cache.insert", info)

        evicted = []
        while len(self._entries) > max_entries or (
            self._total_bytes > max_bytes and self._entries
        ):
            oldest = min(self._entries, key=lambda item: item.last_used)
            self._entries.remove(oldest)
            self._total_bytes -= oldest.nbytes
            self._evictions += 1
            eviction = {
                "entry_id": oldest.id,
                "kind": oldest.kind,
                "after_message": oldest.after_message,
                "tokens": len(oldest.tokens),
                "nbytes": oldest.nbytes,
                "arrays_bytes": oldest.arrays_bytes,
                "kv_bytes": oldest.kv_bytes,
                "reason": "entry" if len(self._entries) + 1 > max_entries else "bytes",
                "evictions_total": self._evictions,
            }
            evicted.append(eviction)
            self._emit("cache.evict", eviction)

        if evicted:
            info["evicted"] = evicted
            info["entries"] = len(self._entries)
            info["total_bytes"] = self._total_bytes
        return info

    @_locked
    def touch_exact(self, tokens: list[int], compat_key: tuple) -> dict[str, Any] | None:
        """Mark an existing identical entry used; return it, or None.

        Used while prefilling: if a checkpoint for exactly these tokens is
        already stored, there is no need to deep-copy the cache again.
        """
        for entry in self._entries:
            if entry.compat_key == compat_key and entry.tokens == tokens:
                self._clock += 1
                entry.last_used = self._clock
                return {
                    "entry_id": entry.id,
                    "kind": entry.kind,
                    "after_message": entry.after_message,
                    "tokens": len(entry.tokens),
                    "nbytes": entry.nbytes,
                    "arrays_bytes": entry.arrays_bytes,
                    "kv_bytes": entry.kv_bytes,
                }
        return None

    @_locked
    def estimate(self, positions: list[int], compat_key: tuple) -> dict[str, Any] | None:
        """Estimate snapshot bytes from a measured entry, without inference.

        The recurrent state is constant; KV scales with the token position.
        The most recent compatible entry is the measurement source.
        """
        sources = [
            entry
            for entry in self._entries
            if entry.compat_key == compat_key and entry.tokens
        ]
        if not sources or not positions:
            return None
        source = max(sources, key=lambda item: (item.created, item.id))
        kv_per_token = source.kv_bytes / len(source.tokens)
        rows = []
        for position in positions:
            kv_bytes = int(kv_per_token * position)
            rows.append(
                {
                    "position": position,
                    "arrays_bytes": source.arrays_bytes,
                    "kv_bytes": kv_bytes,
                    "estimated_bytes": source.arrays_bytes + kv_bytes,
                }
            )
        return {
            "derived_from": {
                "entry_id": source.id,
                "kind": source.kind,
                "tokens": len(source.tokens),
                "arrays_bytes": source.arrays_bytes,
                "kv_bytes": source.kv_bytes,
            },
            "rows": rows,
            "total_bytes": sum(row["estimated_bytes"] for row in rows),
            "note": (
                "Estimate only, scaled from a measured entry. No model run is "
                "performed to produce it."
            ),
        }

    # ---------------------------------------------------------------- clear
    @_locked
    def clear(self, reason: str) -> int:
        count = len(self._entries)
        self._entries.clear()
        self._total_bytes = 0
        self.last_lookup = None
        self._emit("cache.clear", {"reason": reason, "entries": count})
        return count

    # -------------------------------------------------------------- summary
    @_locked
    def token_lists(self) -> list[dict[str, Any]]:
        """Exact token ids per entry.  Plain Python ints, no MLX objects."""
        return [
            {
                "id": entry.id,
                "kind": entry.kind,
                "after_message": entry.after_message,
                "token_ids": list(entry.tokens),
            }
            for entry in self._entries
        ]

    @_locked
    def get_entry(self, entry_id: int) -> CacheEntry | None:
        for entry in self._entries:
            if entry.id == entry_id:
                return entry
        return None

    @_locked
    def summary(self, limit_bytes: int | None = None) -> dict[str, Any]:
        now = time.time()
        entries = [
            {
                "id": entry.id,
                "kind": entry.kind,
                "after_message": entry.after_message,
                "tokens": len(entry.tokens),
                "nbytes": entry.nbytes,
                "arrays_bytes": entry.arrays_bytes,
                "kv_bytes": entry.kv_bytes,
                "age_seconds": round(now - entry.created, 2),
                "last_used": entry.last_used,
                "compat": list(entry.compat_key),
            }
            for entry in self._entries
        ]
        arrays_total = sum(entry.arrays_bytes for entry in self._entries)
        kv_total = sum(entry.kv_bytes for entry in self._entries)
        layer_types: dict[str, int] = {}
        if self._entries:
            layer_types = cache_types(self._entries[-1].cache)
        return {
            "entries": entries,
            "entries_count": len(entries),
            "total_bytes": self._total_bytes,
            "accounted_bytes": self._total_bytes,
            "arrays_bytes": arrays_total,
            "kv_bytes": kv_total,
            "limit_bytes": limit_bytes,
            "utilization": (
                round(self._total_bytes / limit_bytes, 4) if limit_bytes else None
            ),
            "evictions_total": self._evictions,
            "layer_types": layer_types,
            "last_lookup": self.last_lookup,
            "note": BYTE_NOTE,
        }
