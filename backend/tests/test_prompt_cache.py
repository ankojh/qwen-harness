"""Unit tests for the in-memory prompt-cache store.

No model is loaded here.  The store only needs the small slice of the mlx-lm
cache interface that ``can_trim_prompt_cache`` / ``trim_prompt_cache`` use:
``is_trimmable()``, ``trim(n)`` and ``nbytes``.  ``FakeCache`` provides those,
so every reuse decision can be tested exactly and quickly.
"""

import copy

from prompt_cache import PromptCacheStore, cache_nbytes, cache_types

KEY = PromptCacheStore.compat_key(
    model_path="model-a",
    revision=None,
    adapter_path=None,
    kv_bits=None,
    kv_group_size=64,
    quantized_kv_start=5000,
)
OTHER_KEY = PromptCacheStore.compat_key(
    model_path="model-a",
    revision=None,
    adapter_path=None,
    kv_bits=4,  # different KV quantization = incompatible cache
    kv_group_size=64,
    quantized_kv_start=5000,
)


class FakeCache:
    """Minimal stand-in for one mlx-lm cache layer."""

    def __init__(self, offset: int = 0, nbytes: int = 100, trimmable: bool = False):
        self.offset = offset
        self.nbytes = nbytes
        self._trimmable = trimmable

    def is_trimmable(self) -> bool:
        return self._trimmable

    def trim(self, num_tokens: int) -> int:
        trimmed = min(num_tokens, self.offset)
        self.offset -= trimmed
        return trimmed


def fake(tokens: list[int], *, nbytes: int = 100, trimmable: bool = False) -> list:
    return [FakeCache(offset=len(tokens), nbytes=nbytes, trimmable=trimmable)]


def store_with_events() -> tuple[PromptCacheStore, list[tuple[str, dict]]]:
    events: list[tuple[str, dict]] = []
    return PromptCacheStore(emit=lambda kind, data: events.append((kind, data))), events


def insert(store, tokens, *, kind="checkpoint", key=KEY, max_entries=8, max_bytes=10_000, after_message=None):
    return store.insert(
        list(tokens),
        fake(tokens),
        kind,
        key,
        max_entries=max_entries,
        max_bytes=max_bytes,
        after_message=after_message,
    )


# --------------------------------------------------------------------- lookup
def test_empty_store_does_not_reuse():
    store = PromptCacheStore()
    cache, reused, info = store.lookup([1, 2, 3], KEY)
    assert cache is None
    assert reused == 0
    assert "no compatible cache entry" in info["reason"]


def test_exact_prefix_is_reused_from_a_copy():
    store = PromptCacheStore()
    insert(store, [1, 2, 3, 4])

    cache, reused, info = store.lookup([1, 2, 3, 4, 5], KEY)
    assert reused == 4
    assert cache is not None
    assert cache[0].offset == 4
    assert info["entry_id"] == 1
    assert "reused 4 tokens" in info["reason"]

    # The stored cache must not be the object handed out.
    assert cache[0] is not store._entries[0].cache[0]
    cache[0].offset = 999
    assert store._entries[0].cache[0].offset == 4


def test_whole_prompt_is_never_reused():
    """generate_step needs one prompt token to feed, so reuse stops one short."""
    store = PromptCacheStore()
    store.insert(
        [1, 2, 3],
        fake([1, 2, 3], trimmable=True),
        "checkpoint",
        KEY,
        max_entries=8,
        max_bytes=10_000,
    )
    cache, reused, info = store.lookup([1, 2, 3], KEY)
    assert reused == 2
    assert info["trimmed"] == 1
    assert cache is not None and cache[0].offset == 2


def test_full_prompt_entry_on_a_fixed_cache_is_unusable():
    """A non-trimmable entry that ends exactly at the prompt cannot be cut."""
    store = PromptCacheStore()
    insert(store, [1, 2, 3])  # trimmable=False
    cache, reused, info = store.lookup([1, 2, 3], KEY)
    assert cache is None and reused == 0
    assert "not trimmable" in info["reason"]


def test_non_trimmable_divergence_refuses_reuse():
    store = PromptCacheStore()
    insert(store, [1, 2, 3, 4, 5])  # trimmable=False

    cache, reused, info = store.lookup([1, 2, 9, 9], KEY)
    assert cache is None
    assert reused == 0
    assert "diverges at token 2" in info["reason"]
    assert "not trimmable" in info["reason"]


def test_trimmable_divergence_trims_to_common_prefix():
    store = PromptCacheStore()
    store.insert(
        [1, 2, 3, 4, 5],
        fake([1, 2, 3, 4, 5], trimmable=True),
        "after-generation",
        KEY,
        max_entries=8,
        max_bytes=10_000,
    )

    cache, reused, info = store.lookup([1, 2, 9, 9], KEY)
    assert reused == 2
    assert info["trimmed"] == 3
    assert cache is not None and cache[0].offset == 2


def test_shorter_exact_prefix_beats_longer_diverging_entry():
    """A checkpoint must win over a longer after-generation entry it prefixes."""
    store = PromptCacheStore()
    insert(store, [1, 2, 3], kind="checkpoint")
    insert(store, [1, 2, 3, 9, 9], kind="after-generation")

    cache, reused, info = store.lookup([1, 2, 3, 4, 5], KEY)
    assert reused == 3
    assert info["entry_id"] == 1
    assert cache is not None and cache[0].offset == 3
    assert store.peek([1, 2, 3, 4, 5], KEY)["reused_tokens"] == 3


def test_longest_prefix_wins():
    store = PromptCacheStore()
    insert(store, [1, 2, 3])
    insert(store, [1, 2, 3, 4, 5])

    _, reused, info = store.lookup([1, 2, 3, 4, 5, 6], KEY)
    assert reused == 5
    assert info["entry_id"] == 2


def test_compat_key_isolates_entries():
    store = PromptCacheStore()
    insert(store, [1, 2, 3])
    cache, reused, info = store.lookup([1, 2, 3, 4], OTHER_KEY)
    assert cache is None and reused == 0
    assert "no compatible cache entry" in info["reason"]
    assert store.summary()["last_lookup"]["reused_tokens"] == 0


def test_peek_does_not_touch_lru_or_lookup_state():
    store = PromptCacheStore()
    insert(store, [1, 2, 3])
    before = copy.deepcopy(store.summary())
    prediction = store.peek([1, 2, 3, 4], KEY)
    assert prediction["reused_tokens"] == 3
    assert "would reuse 3" in prediction["reason"]
    assert store.summary() == before


# -------------------------------------------------------------- eviction/LRU
def test_entry_limit_evicts_least_recently_used():
    store, events = store_with_events()
    insert(store, [1, 2], max_entries=2)
    insert(store, [3, 4], max_entries=2)

    # Touch entry #1 so entry #2 becomes the LRU one.
    store.lookup([1, 2, 5], KEY)

    info = insert(store, [6, 7], max_entries=2)
    ids = [entry["id"] for entry in store.summary()["entries"]]
    assert ids == [1, 3]
    assert info["evicted"][0]["entry_id"] == 2
    assert info["evicted"][0]["reason"] == "entry"
    assert [kind for kind, _ in events].count("cache.evict") == 1


def test_byte_limit_evicts_until_under_budget():
    store, events = store_with_events()
    insert(store, [1, 2], max_entries=8, max_bytes=250)
    insert(store, [3, 4], max_entries=8, max_bytes=250)
    info = insert(store, [5, 6], max_entries=8, max_bytes=250)

    assert [entry["id"] for entry in store.summary()["entries"]] == [2, 3]
    assert store.summary()["total_bytes"] == 200
    assert info["evicted"][0]["reason"] == "bytes"
    assert [kind for kind, _ in events].count("cache.evict") == 1


def test_identical_insert_replaces_instead_of_duplicating():
    store = PromptCacheStore()
    insert(store, [1, 2, 3])
    insert(store, [1, 2, 3])
    summary = store.summary()
    assert summary["entries_count"] == 1
    assert summary["total_bytes"] == 100


# -------------------------------------------------------------- housekeeping
def test_clear_empties_the_store_and_emits():
    store, events = store_with_events()
    insert(store, [1, 2, 3])
    assert store.clear("model unloaded") == 1
    assert store.summary()["entries_count"] == 0
    assert store.summary()["total_bytes"] == 0
    assert events[-1] == ("cache.clear", {"reason": "model unloaded", "entries": 1})


def test_cache_helpers_count_layers_and_bytes():
    cache = [FakeCache(nbytes=10), FakeCache(nbytes=25)]
    assert cache_nbytes(cache) == 35
    assert cache_types(cache) == {"FakeCache": 2}


# --------------------------------------------------- recurrent/KV split + estimate
class ArraysCache(FakeCache):
    """Stands in for one linear-attention recurrent-state layer."""


class KVCache(FakeCache):
    """Stands in for one full-attention KV layer."""


def layered_cache(tokens, arrays_nbytes=1000, kv_nbytes=500):
    return [
        ArraysCache(offset=len(tokens), nbytes=arrays_nbytes),
        KVCache(offset=len(tokens), nbytes=kv_nbytes),
    ]


def test_summary_splits_recurrent_and_kv_bytes():
    store = PromptCacheStore()
    store.insert(
        [1, 2, 3, 4],
        layered_cache([1, 2, 3, 4]),
        "checkpoint",
        KEY,
        max_entries=8,
        max_bytes=100_000,
        after_message=1,
    )
    summary = store.summary(limit_bytes=10_000)
    row = summary["entries"][0]
    assert row["after_message"] == 1
    assert row["arrays_bytes"] == 1000
    assert row["kv_bytes"] == 500
    assert summary["arrays_bytes"] == 1000
    assert summary["kv_bytes"] == 500
    assert summary["accounted_bytes"] == 1500
    assert summary["limit_bytes"] == 10_000
    assert summary["utilization"] == 0.15
    assert "three deepcopies" in summary["note"]


def test_identical_tokens_dedupe_across_kinds():
    store = PromptCacheStore()
    insert(store, [1, 2, 3], kind="checkpoint")
    store.insert(
        [1, 2, 3],
        fake([1, 2, 3], nbytes=250),
        "after-generation",
        KEY,
        max_entries=8,
        max_bytes=10_000,
    )
    summary = store.summary()
    assert summary["entries_count"] == 1
    assert summary["total_bytes"] == 250
    assert summary["entries"][0]["kind"] == "after-generation"


def test_touch_exact_marks_used_and_returns_none_for_other_tokens():
    store = PromptCacheStore()
    entry_info = insert(store, [1, 2, 3], after_message=2)
    before = store.summary()["entries"][0]["last_used"]
    touched = store.touch_exact([1, 2, 3], KEY)
    assert touched["entry_id"] == entry_info["entry_id"]
    assert touched["after_message"] == 2
    assert store.summary()["entries"][0]["last_used"] > before
    assert store.touch_exact([1, 2, 9], KEY) is None
    assert store.touch_exact([1, 2, 3], OTHER_KEY) is None


def test_estimate_scales_kv_from_a_measured_entry():
    store = PromptCacheStore()
    store.insert(
        list(range(10)),
        layered_cache(range(10), arrays_nbytes=1000, kv_nbytes=500),
        "checkpoint",
        KEY,
        max_entries=8,
        max_bytes=100_000,
        after_message=1,
    )
    estimate = store.estimate([20, 30], KEY)
    assert estimate is not None
    rows = estimate["rows"]
    assert [row["position"] for row in rows] == [20, 30]
    assert rows[0]["arrays_bytes"] == 1000
    assert rows[0]["kv_bytes"] == 1000  # 50 bytes/token * 20
    assert rows[1]["kv_bytes"] == 1500
    assert estimate["total_bytes"] == 4500
    assert estimate["derived_from"]["entry_id"] == 1
    assert store.estimate([20], OTHER_KEY) is None
    assert store.estimate([], KEY) is None


def test_evictions_are_counted():
    store, _ = store_with_events()
    insert(store, [1, 2], max_entries=1, max_bytes=10_000)
    insert(store, [3, 4], max_entries=1, max_bytes=10_000)
    assert store.summary()["evictions_total"] == 1
