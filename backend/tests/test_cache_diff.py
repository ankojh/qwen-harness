"""Unit tests for the token-level cache entry vs prompt diff.  No model."""

from cache_diff import add_text, diff_tokens


class FakeTokenizer:
    def decode(self, ids: list[int]) -> str:
        return " ".join(f"t{token}" for token in ids)


def ops(diff):
    return [(hunk["op"], hunk["left"], hunk["right"]) for hunk in diff["hunks"]]


def test_entry_that_is_a_prefix_is_fully_reusable():
    diff = diff_tokens([1, 2, 3], [1, 2, 3, 4, 5], trimmable=False)
    assert diff["common_prefix"] == 3
    assert diff["reusable_tokens"] == 3
    assert diff["prefill_tokens"] == 2
    assert ops(diff) == [("prefix", [0, 3], [0, 3]), ("insert", [3, 3], [3, 5])]


def test_divergence_before_entry_end_is_not_reusable_without_trim():
    # Reasoning dropped from history: the entry holds 7, 8 the prompt does not.
    cached = [1, 2, 7, 8, 3, 4]
    prompt = [1, 2, 3, 4, 5, 6]
    diff = diff_tokens(cached, prompt, trimmable=False, min_equal_tokens=1)
    assert diff["common_prefix"] == 2
    assert diff["reusable_tokens"] == 0
    assert diff["prefill_tokens"] == 6
    assert "not trimmable" in diff["reason"]
    assert ops(diff) == [
        ("prefix", [0, 2], [0, 2]),
        ("delete", [2, 4], [2, 2]),
        ("equal", [4, 6], [2, 4]),  # matches again, but can't be spliced in
        ("insert", [6, 6], [4, 6]),
    ]


def test_short_equal_runs_fold_into_the_change():
    # A lone shared "newline" token (5) inside unrelated text is noise.
    cached = [1, 2, 7, 5, 8, 9, 9, 9, 9]
    prompt = [1, 2, 3, 5, 4, 9, 9, 9, 9]
    diff = diff_tokens(cached, prompt, trimmable=False)
    assert ops(diff) == [
        ("prefix", [0, 2], [0, 2]),
        ("replace", [2, 5], [2, 5]),
        ("equal", [5, 9], [5, 9]),  # long enough to keep
    ]


def test_trailing_short_equal_joins_a_one_sided_change():
    diff = diff_tokens([1, 7, 7, 7, 5], [1, 5], trimmable=False)
    assert ops(diff) == [("prefix", [0, 1], [0, 1]), ("replace", [1, 5], [1, 2])]


def test_trimmable_cache_reuses_the_common_prefix():
    diff = diff_tokens([1, 2, 7, 8], [1, 2, 3], trimmable=True)
    assert diff["reusable_tokens"] == 2
    assert "after trimming 2" in diff["reason"]


def test_last_prompt_token_is_always_fed():
    diff = diff_tokens([1, 2, 3], [1, 2, 3], trimmable=True)
    assert diff["common_prefix"] == 3
    assert diff["reusable_tokens"] == 2
    assert diff["prefill_tokens"] == 1


def test_identical_but_untrimmable_entry_is_not_reusable():
    # Reusing 2 of 3 would need the entry cut back by one token.
    diff = diff_tokens([1, 2, 3], [1, 2, 3], trimmable=False)
    assert diff["reusable_tokens"] == 0
    assert ops(diff) == [("prefix", [0, 3], [0, 3])]


def test_divergence_at_token_zero():
    diff = diff_tokens([9, 9], [1, 2], trimmable=False)
    assert diff["common_prefix"] == 0
    assert diff["reason"] == "diverges at token 0"
    assert ops(diff) == [("replace", [0, 2], [0, 2])]


def test_large_remainder_is_one_block():
    diff = diff_tokens([1, 5, 6, 7], [1, 8, 9, 10], trimmable=False, max_diff_tokens=4)
    assert diff["truncated"]
    assert ops(diff) == [("prefix", [0, 1], [0, 1]), ("replace", [1, 4], [1, 4])]


def test_add_text_decodes_each_side():
    cached = [1, 2, 7]
    prompt = [1, 2, 3, 4]
    diff = add_text(diff_tokens(cached, prompt, trimmable=False), cached, prompt, FakeTokenizer())
    prefix, replace = diff["hunks"]
    assert prefix["left_text"] == prefix["right_text"] == "t1 t2"
    assert replace["op"] == "replace"
    assert replace["left_text"] == "t7"
    assert replace["right_text"] == "t3 t4"
