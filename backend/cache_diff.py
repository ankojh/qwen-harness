"""Token-level diff between a stored cache entry and the prompt about to be sent.

The cache side is read-only: an entry is model state computed from a token
sequence, so the only thing to compare is that sequence.  The diff answers two
questions the inspector cannot answer from counts alone:

* how far the entry and the prompt agree (the common prefix, which is the only
  part a non-trimmable cache can reuse), and
* what differs after that, so a miss can be traced to its cause (for example
  the chat template dropping reasoning from an appended assistant turn).

Regions that match again *after* the divergence are reported as ``equal`` but
are never reusable: a cache can only be shortened at its end, never spliced.

Everything here is tokenizer-only.  No model call is made.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

# Above this many tokens after the common prefix (both sides together) the
# remainder is shown as one replaced block instead of running SequenceMatcher,
# which is roughly quadratic on dissimilar input.
MAX_DIFF_TOKENS = 16_000

# Equal runs shorter than this after the divergence are folded into the
# surrounding change.  Without it, unrelated text lines up on single newline
# or punctuation tokens and the diff shatters into noise.
MIN_EQUAL_TOKENS = 4


def _merge_short_equal(hunks: list[dict[str, Any]], min_equal: int) -> list[dict[str, Any]]:
    """Fold short ``equal`` hunks into the change before them; join changes.

    Runs on the opcodes after the common prefix, which always start with a
    change (the first tokens differ), so every short equal run has a change
    to join.  Ranges stay exact; each change is named by the sides it covers.
    """
    merged: list[dict[str, Any]] = []
    for hunk in hunks:
        short = hunk["op"] == "equal" and hunk["left"][1] - hunk["left"][0] < min_equal
        if (hunk["op"] != "equal" or short) and merged and merged[-1]["op"] != "equal":
            last = merged[-1]
            last["left"] = [last["left"][0], hunk["left"][1]]
            last["right"] = [last["right"][0], hunk["right"][1]]
        else:
            merged.append(dict(hunk))
    for hunk in merged:
        if hunk["op"] == "equal":
            continue
        left = hunk["left"][1] > hunk["left"][0]
        right = hunk["right"][1] > hunk["right"][0]
        hunk["op"] = "replace" if left and right else "delete" if left else "insert"
    return merged


def common_prefix(a: list[int], b: list[int]) -> int:
    size = min(len(a), len(b))
    index = 0
    while index < size and a[index] == b[index]:
        index += 1
    return index


def diff_tokens(
    cached: list[int],
    prompt: list[int],
    *,
    trimmable: bool,
    max_diff_tokens: int = MAX_DIFF_TOKENS,
    min_equal_tokens: int = MIN_EQUAL_TOKENS,
) -> dict[str, Any]:
    """Hunks over token ranges, ``[start, end)`` on each side.

    ``op`` is ``prefix`` (the common prefix), then SequenceMatcher's ``equal``,
    ``replace``, ``delete`` (only in the cache entry) and ``insert`` (only in
    the prompt), with equal runs shorter than ``min_equal_tokens`` folded into
    the changes around them.  ``reusable_tokens`` applies the same rule as the store:
    the last prompt token is always fed to the model, and an entry that
    diverges before its end is unusable unless the cache can be trimmed.
    """
    prefix = common_prefix(cached, prompt)
    reusable = min(prefix, max(len(prompt) - 1, 0))
    if reusable > 0 and reusable < len(cached) and not trimmable:
        reusable = 0

    if reusable > 0:
        reason = (
            f"{reusable} of {len(prompt)} prompt tokens reusable"
            + (f" after trimming {len(cached) - reusable}" if reusable < len(cached) else "")
        )
    elif prefix == 0:
        reason = "diverges at token 0"
    elif prefix < len(cached) and not trimmable:
        reason = (
            f"diverges at token {prefix} of {len(cached)} and this model's cache "
            "is not trimmable"
        )
    else:
        reason = "the prompt has only one token and it must be fed to the model"

    hunks: list[dict[str, Any]] = []
    if prefix:
        hunks.append({"op": "prefix", "left": [0, prefix], "right": [0, prefix]})

    left_rest = cached[prefix:]
    right_rest = prompt[prefix:]
    truncated = None
    if len(left_rest) + len(right_rest) > max_diff_tokens:
        truncated = (
            f"{len(left_rest) + len(right_rest)} tokens differ after the prefix "
            f"(limit {max_diff_tokens}); shown as one block"
        )
        hunks.append(
            {
                "op": "replace",
                "left": [prefix, len(cached)],
                "right": [prefix, len(prompt)],
            }
        )
    elif left_rest or right_rest:
        matcher = SequenceMatcher(None, left_rest, right_rest, autojunk=False)
        opcodes = [
            {
                "op": op,
                "left": [prefix + i1, prefix + i2],
                "right": [prefix + j1, prefix + j2],
            }
            for op, i1, i2, j1, j2 in matcher.get_opcodes()
        ]
        hunks.extend(_merge_short_equal(opcodes, min_equal_tokens))

    return {
        "entry_tokens": len(cached),
        "prompt_tokens": len(prompt),
        "common_prefix": prefix,
        "reusable_tokens": reusable,
        "prefill_tokens": len(prompt) - reusable,
        "trimmable": trimmable,
        "reason": reason,
        "truncated": truncated,
        "hunks": hunks,
    }


def add_text(diff: dict[str, Any], cached: list[int], prompt: list[int], tokenizer: Any) -> dict[str, Any]:
    """Decode each hunk's two sides.

    Spans are decoded separately, so a character split across a hunk boundary
    can show as U+FFFD on either side; the token ranges stay exact.
    """
    for hunk in diff["hunks"]:
        left = cached[hunk["left"][0] : hunk["left"][1]]
        right = prompt[hunk["right"][0] : hunk["right"][1]]
        hunk["left_text"] = tokenizer.decode(left) if left else ""
        if hunk["op"] in ("prefix", "equal"):
            hunk["right_text"] = hunk["left_text"]  # same ids, same text
        else:
            hunk["right_text"] = tokenizer.decode(right) if right else ""
    return diff
