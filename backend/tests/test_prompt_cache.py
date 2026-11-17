"""Unit tests for the in-memory prompt-cache store.

No model is loaded here.  The store only needs the small slice of the mlx-lm
cache interface that ``can_trim_prompt_cache`` / ``trim_prompt_cache`` use:
``is_trimmable()``, ``trim(n)`` and ``nbytes``.  ``FakeCache`` provides those,
so every reuse decision can be tested exactly and quickly.
"""

import copy

from prompt_cache import (
    PromptCacheStore,
    cache_nbytes,
    cache_types,
    describe_cache_layers,
)

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


def test_compare_reports_exact_prefix_and_final_token_limit():
    store = PromptCacheStore()
    insert(store, [1, 2, 3], after_message=4)

    row = store.compare([1, 2, 3, 4], KEY)[0]
    assert row["id"] == 1
    assert row["kind"] == "checkpoint"
    assert row["after_message"] == 4
    assert row["token_ids"] == [1, 2, 3]
    assert row["shared_prefix"] == 3
    assert row["reusable_tokens"] == 3
    assert row["compatible"] is True
    assert row["trimmable"] is False

    # Equality is not a full reuse: generate_step must process the final input
    # token, and this fixed state cannot be rewound to leave it out.
    equal = store.compare([1, 2, 3], KEY)[0]
    assert equal["shared_prefix"] == 3
    assert equal["reusable_tokens"] == 0
    assert "cannot rewind" in equal["reason"]


def test_compare_reports_mismatch_and_cache_compatibility():
    store = PromptCacheStore()
    insert(store, [1, 2, 3, 4], key=KEY)
    insert(store, [1, 2, 8], key=OTHER_KEY)

    rows = store.compare([1, 2, 9, 10], KEY)
    assert len(rows) == 2
    matching_key, incompatible = rows
    assert matching_key["shared_prefix"] == 2
    assert matching_key["reusable_tokens"] == 0
    assert "first different token at position 2" in matching_key["reason"].lower()
    assert incompatible["compatible"] is False
    assert incompatible["reusable_tokens"] == 0
    assert "different model or cache settings" in incompatible["reason"].lower()


def test_compare_reports_trimmable_partial_state_as_usable():
    store = PromptCacheStore()
    store.insert(
        [1, 2, 3, 4], fake([1, 2, 3, 4], trimmable=True), "checkpoint", KEY,
        max_entries=8, max_bytes=10_000,
    )

    row = store.compare([1, 2, 9, 10], KEY)[0]
    assert row["shared_prefix"] == 2
    assert row["reusable_tokens"] == 2
    assert row["trimmable"] is True
    assert "positions 0–1" in row["reason"]


def test_relabel_changes_match_assessment_and_restore_recovers_it():
    store = PromptCacheStore()
    insert(store, [1, 2, 3], after_message=1)

    store.relabel(1, 1, 20)
    relabeled = store.compare([1, 2, 3, 4], KEY)[0]
    assert relabeled["token_ids"] == [1, 20, 3]
    assert relabeled["relabels"] == 1
    assert relabeled["shared_prefix"] == 1
    assert relabeled["reusable_tokens"] == 0

    restored = store.restore(1)
    assert restored["restored"] == 1
    original = store.compare([1, 2, 3, 4], KEY)[0]
    assert original["token_ids"] == [1, 2, 3]
    assert original["relabels"] == 0
    assert original["reusable_tokens"] == 3


def test_relabel_and_restore_emit_events_with_entry_ids_and_counts():
    store, events = store_with_events()
    insert(store, [1, 2, 3])
    events.clear()  # Ignore the setup insertion event.

    relabeled = store.relabel(1, 1, 20)
    restored = store.restore(1)
    assert store.relabel(999, 0, 30) is None
    assert store.restore(999) is None

    assert events == [
        (
            "cache.relabel",
            {
                "entry_id": 1,
                "kind": "checkpoint",
                "position": 1,
                "original_id": 2,
                "new_id": 20,
                "relabels": 1,
            },
        ),
        (
            "cache.restore",
            {
                "entry_id": 1,
                "kind": "checkpoint",
                "restored": 1,
                "relabels": 0,
            },
        ),
    ]
    assert relabeled["entry_id"] == restored["entry_id"] == 1
    assert relabeled["relabels"] == 1
    assert restored["restored"] == 1


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


# ------------------------------------------------------------- token_lists()
def test_token_lists_reports_all_entries_without_mlx():
    store = PromptCacheStore()
    insert(store, [1, 2, 3], kind="checkpoint", after_message=1)
    insert(store, [1, 2, 3, 4, 5], kind="after-generation", after_message=2)

    lists = store.token_lists()
    assert [item["id"] for item in lists] == [1, 2]
    assert lists[0] == {
        "id": 1,
        "kind": "checkpoint",
        "after_message": 1,
        "token_ids": [1, 2, 3],
    }
    assert lists[1]["token_ids"] == [1, 2, 3, 4, 5]
    assert lists[1]["after_message"] == 2


def test_token_lists_is_empty_for_an_empty_store():
    assert PromptCacheStore().token_lists() == []


# --------------------------------------------------------------------- remove
def test_remove_deletes_the_entry_updates_totals_and_emits():
    store, events = store_with_events()
    insert(store, [1, 2, 3], after_message=1)
    insert(store, [4, 5], after_message=2)

    info = store.remove(1)
    assert info["entry_id"] == 1
    assert info["after_message"] == 1
    assert info["tokens"] == 3
    assert info["entries"] == 1
    assert info["total_bytes"] == 100
    assert [entry["id"] for entry in store.summary()["entries"]] == [2]
    assert store.summary()["total_bytes"] == 100
    assert events[-1][0] == "cache.delete"
    assert events[-1][1]["entry_id"] == 1


def test_remove_returns_none_for_an_unknown_id():
    store = PromptCacheStore()
    insert(store, [1, 2, 3])
    assert store.remove(999) is None
    assert store.summary()["entries_count"] == 1


# ----------------------------------------------------------- relabel/restore
def test_relabel_replaces_the_token_and_records_it():
    store = PromptCacheStore()
    info = insert(store, [1, 2, 3])
    entry_id = info["entry_id"]

    result = store.relabel(entry_id, 1, 99)
    assert result == {
        "entry_id": entry_id,
        "kind": "checkpoint",
        "position": 1,
        "original_id": 2,
        "new_id": 99,
        "relabels": 1,
    }
    entry = store.get_entry(entry_id)
    assert entry.tokens == [1, 99, 3]
    assert entry.relabels == [{"position": 1, "original_id": 2, "new_id": 99}]


def test_relabelling_the_same_position_twice_keeps_the_first_original():
    store = PromptCacheStore()
    info = insert(store, [1, 2, 3])
    entry_id = info["entry_id"]

    store.relabel(entry_id, 1, 99)
    result = store.relabel(entry_id, 1, 7)

    entry = store.get_entry(entry_id)
    assert entry.tokens == [1, 7, 3]
    assert entry.relabels == [{"position": 1, "original_id": 2, "new_id": 7}]
    assert result["original_id"] == 2
    assert result["new_id"] == 7
    assert result["relabels"] == 1


def test_restore_puts_originals_back_and_clears_relabels():
    store = PromptCacheStore()
    info = insert(store, [1, 2, 3])
    entry_id = info["entry_id"]
    store.relabel(entry_id, 0, 50)
    store.relabel(entry_id, 2, 60)

    result = store.restore(entry_id)
    assert result == {
        "entry_id": entry_id,
        "kind": "checkpoint",
        "restored": 2,
        "relabels": 0,
    }
    entry = store.get_entry(entry_id)
    assert entry.tokens == [1, 2, 3]
    assert entry.relabels == []


def test_relabel_and_restore_return_none_for_an_unknown_id():
    store = PromptCacheStore()
    insert(store, [1, 2, 3])
    assert store.relabel(999, 0, 1) is None
    assert store.restore(999) is None


def test_a_relabelled_entry_is_matched_by_its_new_tokens():
    store = PromptCacheStore()
    info = store.insert(
        [1, 2, 3, 4],
        fake([1, 2, 3, 4], trimmable=True),
        "checkpoint",
        KEY,
        max_entries=8,
        max_bytes=10_000,
    )
    entry_id = info["entry_id"]

    store.relabel(entry_id, 1, 99)

    # The old sequence no longer matches past the relabelled position.
    _, reused_old, info_old = store.lookup([1, 2, 3, 4, 5], KEY)
    assert reused_old == 1

    # The new sequence matches in full and reports the entry as relabeled.
    cache, reused_new, info_new = store.lookup([1, 99, 3, 4, 5], KEY)
    assert reused_new == 4
    assert cache is not None
    assert info_new["entry_id"] == entry_id
    assert info_new["relabeled"] is True

    prediction = store.peek([1, 99, 3, 4, 5], KEY)
    assert prediction["reused_tokens"] == 4
    assert prediction["relabeled"] is True


def test_lookup_miss_reports_relabeled_false():
    store = PromptCacheStore()
    _, reused, info = store.lookup([1, 2, 3], KEY)
    assert reused == 0
    assert info["relabeled"] is False


def test_summary_reports_relabels_count():
    store = PromptCacheStore()
    info = insert(store, [1, 2, 3])
    entry_id = info["entry_id"]
    store.relabel(entry_id, 0, 50)
    store.relabel(entry_id, 1, 60)

    row = store.summary()["entries"][0]
    assert row["relabels"] == 2


# ----------------------------------------------------------------- get_entry
def test_get_entry_returns_the_matching_entry_or_none():
    store = PromptCacheStore()
    info = insert(store, [1, 2, 3], after_message=1)

    entry = store.get_entry(info["entry_id"])
    assert entry is not None
    assert entry.tokens == [1, 2, 3]
    assert entry.after_message == 1

    assert store.get_entry(info["entry_id"] + 999) is None


# --------------------------------------------------- describe_cache_layers()
class FakeArray:
    """Shape/dtype/nbytes only -- enough for the metadata rows, no values."""

    def __init__(self, shape, dtype, nbytes):
        self.shape = shape
        self.dtype = dtype
        self.nbytes = nbytes


class FakeArraysLayer:
    """A linear-attention layer: a ``.cache`` list, no ``keys``/``values``."""

    def __init__(self, nbytes=2_146_304):
        self.cache = [FakeArray((1, 4, 1024), "float16", nbytes)]
        self.nbytes = nbytes


class FakeKVLayer:
    """A full-attention layer: ``keys``/``values`` buffers sized by tokens."""

    def __init__(self, offset, allocated_tokens, bytes_per_token=4096, dtype="float16"):
        per_array = (bytes_per_token // 2) * allocated_tokens
        self.offset = offset
        self.keys = FakeArray((1, 4, allocated_tokens, 256), dtype, per_array)
        self.values = FakeArray((1, 4, allocated_tokens, 256), dtype, per_array)
        self.nbytes = self.keys.nbytes + self.values.nbytes


def test_describe_cache_layers_arrays_layer_has_no_token_fields():
    rows = describe_cache_layers(
        [FakeArraysLayer()], layer_types=["linear_attention"]
    )
    row = rows[0]
    assert row["index"] == 0
    assert row["kind"] == "linear_attention"
    assert row["used_tokens"] is None
    assert row["allocated_tokens"] is None
    assert row["used_bytes"] == row["allocated_bytes"] == 2_146_304
    assert [a["name"] for a in row["arrays"]] == ["state[0]"]
    assert row["arrays"][0]["nbytes"] == 2_146_304


def test_describe_cache_layers_kv_layer_tracks_used_and_allocated_tokens():
    # 8 KVCache layers at [1, 4, N, 256], 4096 bytes/token/layer (keys+values).
    layer = FakeKVLayer(offset=100, allocated_tokens=256)
    rows = describe_cache_layers([layer], layer_types=["full_attention"])
    row = rows[0]
    assert row["kind"] == "full_attention"
    assert row["used_tokens"] == 100
    assert row["allocated_tokens"] == 256
    assert row["allocated_bytes"] == 4096 * 256
    assert row["used_bytes"] == 4096 * 100
    assert {a["name"] for a in row["arrays"]} == {"keys", "values"}


def test_describe_cache_layers_handles_an_unused_fresh_entry():
    layer = FakeKVLayer(offset=0, allocated_tokens=256)
    row = describe_cache_layers([layer], layer_types=["full_attention"])[0]
    assert row["used_tokens"] == 0
    assert row["used_bytes"] == 0
    assert row["allocated_bytes"] == 4096 * 256


def test_describe_cache_layers_infers_kind_from_class_name_without_layer_types():
    class QuantizedArraysCache(FakeArraysLayer):
        pass

    rows = describe_cache_layers([QuantizedArraysCache(), FakeKVLayer(0, 256)])
    assert rows[0]["kind"] == "linear_attention"
    assert rows[1]["kind"] == "full_attention"


def test_describe_cache_layers_reports_every_layer_in_order():
    cache = [FakeArraysLayer() for _ in range(3)] + [
        FakeKVLayer(offset=10, allocated_tokens=256) for _ in range(2)
    ]
    layer_types = ["linear_attention"] * 3 + ["full_attention"] * 2
    rows = describe_cache_layers(cache, layer_types=layer_types)
    assert [row["index"] for row in rows] == [0, 1, 2, 3, 4]
    assert [row["kind"] for row in rows] == [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
        "full_attention",
    ]
