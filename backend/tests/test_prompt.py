"""Prompt-boundary tests with a tiny fake tokenizer/template.

No model is involved: ``FakeTokenizer`` renders messages with a deliberately
simple template and tokenizes by character code.  That is enough to exercise
the checkpoint-boundary rules, including the two real Qwen3.5 cases:
a template that raises for a system-only prefix, and reasoning that the
template drops once a later user message exists.
"""

from config import AppConfig
from prompt import Session, build_prompt, checkpoint_boundary, relabel_token_ids

MODEL_INFO = {
    "max_position_embeddings": 10_000,
    "chat_template": "fake template",
    "think_start": "<think>",
    "think_end": "</think>",
}


class FakeTokenizer:
    chat_template = "fake template"

    def encode(self, text, add_special_tokens=False):
        return [ord(char) + 100 for char in text]

    def convert_ids_to_tokens(self, token_id):
        return chr(token_id - 100)

    def convert_tokens_to_ids(self, token):
        # This fake has no "<|im_start|>" token at all, like a tokenizer with
        # a different template: unknown-token id, same as real tokenizers.
        return -1

    def render(self, messages, add_generation_prompt):
        if not any(message["role"] == "user" for message in messages):
            raise ValueError("No user query found in messages")
        parts = []
        for index, message in enumerate(messages):
            content = message["content"]
            later_user = any(
                other["role"] == "user" for other in messages[index + 1 :]
            )
            if (
                message["role"] == "assistant"
                and message.get("reasoning_content")
                and not later_user
            ):
                # Simulates a template that keeps reasoning only at the end.
                content = f"<think>{message['reasoning_content']}</think>{content}"
            parts.append(f"<{message['role']}>{content}</{message['role']}>")
        if add_generation_prompt:
            parts.append("<|assistant|>")
        return "".join(parts)

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
        chat_template=None,
        **kwargs,
    ):
        text = self.render(messages, add_generation_prompt)
        return self.encode(text) if tokenize else text


def conversation() -> Session:
    return Session(
        mode="messages",
        messages=[
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1", "reasoning_content": "secret"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"},
        ],
    )


def test_every_message_boundaries_report_accepts_and_rejects():
    config = AppConfig(cache_checkpoints="every_message", max_tokens=10, context_budget=1000)
    report = build_prompt(
        conversation(),
        config,
        FakeTokenizer(),
        model_info=MODEL_INFO,
        capabilities={},
    )
    assert report["ok"] is True, report["error"]
    checkpoints = report["checkpoints"]
    assert [entry["after_message"] for entry in checkpoints] == [1, 2, 3, 4, 5]
    assert checkpoints[0]["ok"] is True
    # Prefix [user, assistant-with-reasoning] renders the reasoning because no
    # later user exists; the full prompt drops it, so it is not a prefix.
    assert checkpoints[1]["ok"] is False
    assert "not a token prefix" in checkpoints[1]["reason"]
    assert checkpoints[2]["ok"] is True
    assert checkpoints[3]["ok"] is True
    assert checkpoints[4]["ok"] is True

    accepted = [entry["position"] for entry in checkpoints if entry["ok"]]
    assert accepted == sorted(accepted)
    assert report["checkpoint_at"] == accepted[-1]


def test_last_message_only_checks_the_final_boundary():
    config = AppConfig(cache_checkpoints="last_message", max_tokens=10, context_budget=1000)
    report = build_prompt(
        conversation(),
        config,
        FakeTokenizer(),
        model_info=MODEL_INFO,
        capabilities={},
    )
    assert [entry["after_message"] for entry in report["checkpoints"]] == [5]
    assert report["checkpoints"][0]["ok"] is True
    assert report["checkpoint_at"] == report["checkpoints"][0]["position"]


def test_system_only_prefix_is_reported_skipped():
    config = AppConfig(
        cache_checkpoints="every_message",
        system_prompt="You are terse.",
        max_tokens=10,
        context_budget=1000,
    )
    report = build_prompt(
        conversation(),
        config,
        FakeTokenizer(),
        model_info=MODEL_INFO,
        capabilities={},
    )
    first = next(entry for entry in report["checkpoints"] if entry["after_message"] == 1)
    assert first["ok"] is False
    assert "No user query found" in first["reason"]
    # Later boundaries are still computed.
    assert any(entry["ok"] for entry in report["checkpoints"])


def test_drop_oldest_messages_keeps_earlier_drops_dropped():
    full = build_prompt(
        conversation(),
        AppConfig(max_tokens=10, context_budget=1000),
        FakeTokenizer(),
        model_info=MODEL_INFO,
        capabilities={},
    )
    # A budget that fits only the last message forces several drops in a row.
    config = AppConfig(
        overflow_policy="drop_oldest_messages",
        max_tokens=10,
        context_budget=len(full["token_ids"]) - 25,
    )
    report = build_prompt(
        conversation(), config, FakeTokenizer(), model_info=MODEL_INFO, capabilities={}
    )
    assert report["ok"] is True, report["error"]
    dropped = [item["message_index"] for item in report["dropped_messages"]]
    assert len(dropped) >= 2
    kept = [message["content"] for message in report["submitted_messages"]]
    for index in dropped:
        assert conversation().messages[index - 1].content not in kept
    assert len(report["token_ids"]) + 10 <= config.context_budget


def test_off_and_raw_have_no_checkpoints_with_a_reason():
    off = build_prompt(
        conversation(),
        AppConfig(cache_checkpoints="off", max_tokens=10, context_budget=1000),
        FakeTokenizer(),
        model_info=MODEL_INFO,
        capabilities={},
    )
    assert off["checkpoints"] == []
    assert "off" in off["checkpoint_reason"]

    raw = build_prompt(
        Session(mode="raw", raw_text="hello", messages=[]),
        AppConfig(cache_checkpoints="every_message", max_tokens=10, context_budget=1000),
        FakeTokenizer(),
        model_info=MODEL_INFO,
        capabilities={},
    )
    assert raw["checkpoints"] == []
    assert "raw mode" in raw["checkpoint_reason"]


# --------------------------------------------------------------------------
# Per-token pieces and spans (``include_pieces=True``)
# --------------------------------------------------------------------------
class SpanTokenizer:
    """A fake tokenizer with real ``<|im_start|>``-style special tokens.

    Unlike ``FakeTokenizer`` (one token per character), this one tokenizes
    known markers -- ``<|im_start|>``, ``<|im_end|>``, role names, ``<think>``
    tags and newlines -- as whole tokens, which is what ``_token_spans``
    needs to find message boundaries. Everything else still falls back to
    one token per character, so it stays simple and symmetric.
    """

    chat_template = "qwen-like"
    KNOWN = [
        "<|im_start|>",
        "<|im_end|>",
        "<think>",
        "</think>",
        "\n",
        "system",
        "user",
        "assistant",
    ]
    SPECIAL_COUNT = 4  # the markers, not the role names or the newline

    def _char_id(self, char: str) -> int:
        return ord(char) + 1000

    def encode(self, text, add_special_tokens=False):
        ids = []
        i = 0
        while i < len(text):
            match = next((k for k in self.KNOWN if text.startswith(k, i)), None)
            if match:
                ids.append(self.KNOWN.index(match) + 1)
                i += len(match)
            else:
                ids.append(self._char_id(text[i]))
                i += 1
        return ids

    def convert_ids_to_tokens(self, token_id):
        if 1 <= token_id <= len(self.KNOWN):
            return self.KNOWN[token_id - 1]
        return chr(token_id - 1000)

    def convert_tokens_to_ids(self, token):
        if token in self.KNOWN:
            return self.KNOWN.index(token) + 1
        return -1

    def decode(self, ids):
        return "".join(self.convert_ids_to_tokens(i) for i in ids)

    def is_special_token(self, token_id):
        return 1 <= token_id <= self.SPECIAL_COUNT

    def render(self, messages, add_generation_prompt):
        parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n<think>\n")
        return "".join(parts)

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
        chat_template=None,
        **kwargs,
    ):
        text = self.render(messages, add_generation_prompt)
        return self.encode(text) if tokenize else text


SPAN_MODEL_INFO = {
    "max_position_embeddings": 10_000,
    "chat_template": "qwen-like",
    "think_start": "<think>",
    "think_end": "</think>",
}


def test_pieces_carry_decoded_text_and_special_flag():
    session = Session(
        mode="messages", messages=[{"role": "user", "content": "hi"}]
    )
    report = build_prompt(
        session,
        AppConfig(max_tokens=10, context_budget=1000),
        SpanTokenizer(),
        model_info=SPAN_MODEL_INFO,
        capabilities={},
        include_pieces=True,
    )
    assert report["ok"] is True, report["error"]
    pieces = report["token_pieces"]
    assert pieces is not None
    tokenizer = SpanTokenizer()
    for piece, token_id in zip(pieces, report["token_ids"]):
        assert piece["id"] == token_id
        assert piece["piece"] == tokenizer.convert_ids_to_tokens(token_id)
        assert piece["text"] == tokenizer.decode([token_id])

    by_piece = {p["piece"]: p["special"] for p in pieces}
    assert by_piece["<|im_start|>"] is True
    assert by_piece["<|im_end|>"] is True
    assert by_piece["user"] is False
    assert by_piece["\n"] is False
    assert by_piece["h"] is False  # a plain content character


def test_spans_match_submitted_messages_and_trailing_opener():
    session = Session(
        mode="messages",
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "again"},
        ],
    )
    config = AppConfig(
        system_prompt="be terse", max_tokens=10, context_budget=1000
    )
    report = build_prompt(
        session,
        config,
        SpanTokenizer(),
        model_info=SPAN_MODEL_INFO,
        capabilities={},
        include_pieces=True,
    )
    assert report["ok"] is True, report["error"]
    submitted = report["submitted_messages"]
    spans = report["spans"]
    assert spans is not None
    # 4 submitted messages (system + 3) plus the trailing assistant opener.
    assert len(spans) == len(submitted) + 1

    for span, message in zip(spans, submitted):
        assert span["kind"] == "message"
        assert span["index"] == message["index"]
        assert span["role"] == message["role"]
        assert span["source"] == message["source"]
        assert span["end"] > span["start"]

    opener = spans[-1]
    assert opener["kind"] == "opener"
    assert opener["index"] is None
    assert opener["role"] == "assistant"
    assert opener["source"] == "template"
    assert opener["end"] == len(report["token_ids"])

    # Spans tile the whole prompt with no gaps or overlaps.
    assert spans[0]["start"] == 0
    for left, right in zip(spans, spans[1:]):
        assert left["end"] == right["start"]


def test_spans_are_none_in_raw_mode():
    report = build_prompt(
        Session(mode="raw", raw_text="hello", messages=[]),
        AppConfig(max_tokens=10, context_budget=1000),
        SpanTokenizer(),
        model_info=SPAN_MODEL_INFO,
        capabilities={},
        include_pieces=True,
    )
    assert report["token_pieces"] is not None
    assert report["spans"] is None


def test_pieces_and_spans_are_none_without_include_pieces():
    session = Session(
        mode="messages", messages=[{"role": "user", "content": "hi"}]
    )
    report = build_prompt(
        session,
        AppConfig(max_tokens=10, context_budget=1000),
        SpanTokenizer(),
        model_info=SPAN_MODEL_INFO,
        capabilities={},
    )
    assert report["token_pieces"] is None
    assert report["spans"] is None


def test_token_text_and_special_fall_back_without_decode_support():
    """FakeTokenizer has no decode()/is_special_token(); _token_text and
    _token_is_special must fall back instead of raising, and spans must come
    back None rather than crash on a tokenizer with no "<|im_start|>" id."""
    report = build_prompt(
        conversation(),
        AppConfig(max_tokens=10, context_budget=1000),
        FakeTokenizer(),
        model_info=MODEL_INFO,
        capabilities={},
        include_pieces=True,
    )
    assert report["ok"] is True, report["error"]
    pieces = report["token_pieces"]
    assert pieces is not None
    for piece in pieces:
        assert piece["text"] == piece["piece"]  # decode failed -> fell back
        assert piece["special"] is False
    assert report["spans"] is None


# --------------------------------------------------------------------------
# relabel_token_ids() -- the one-token rule for a cache relabel
# --------------------------------------------------------------------------
class WordTokenizer:
    """A fake tokenizer that treats each whitespace-separated run as one token.

    Real tokenizers often map a common word (with its leading space) to a
    single id; this fake is enough to exercise the one-token rule without a
    real vocabulary.
    """

    VOCAB = [" Alice", " Bob", "My", " name", " is", "."]

    def encode(self, text, add_special_tokens=False):
        ids = []
        i = 0
        while i < len(text):
            match = next((w for w in self.VOCAB if text.startswith(w, i)), None)
            if match:
                ids.append(self.VOCAB.index(match) + 1)
                i += len(match)
            else:
                ids.append(1000 + ord(text[i]))
                i += 1
        return ids


def test_relabel_token_ids_accepts_a_single_token_string():
    assert relabel_token_ids(WordTokenizer(), " Alice") == [1]


def test_relabel_token_ids_reports_every_token_for_a_multi_token_string():
    ids = relabel_token_ids(WordTokenizer(), "My name")
    assert len(ids) == 2
    assert ids != [ids[0]]


# --------------------------------------------------------------------------
# checkpoint_boundary() -- the manual-checkpoint endpoint's boundary choice
# --------------------------------------------------------------------------
def _span_prompt(messages, add_generation_prompt=True):
    template_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
    tokenizer = SpanTokenizer()
    text = tokenizer.render(template_messages, add_generation_prompt=add_generation_prompt)
    token_ids = tokenizer.encode(text)
    return template_messages, token_ids, tokenizer


def test_checkpoint_boundary_returns_the_position_after_the_message():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok"},
    ]
    template_messages, token_ids, tokenizer = _span_prompt(messages)
    config = AppConfig(max_tokens=10, context_budget=1000)

    expected_text = tokenizer.render(template_messages[:1], add_generation_prompt=False)
    expected_position = len(tokenizer.encode(expected_text))

    boundary = checkpoint_boundary(
        template_messages=template_messages,
        token_ids=token_ids,
        config=config,
        tokenizer=tokenizer,
        thinking_supported=True,
        template_kwargs={},
        after_message=1,
    )
    assert boundary["ok"] is True
    assert boundary["position"] == expected_position
    assert expected_position < len(token_ids)


def test_checkpoint_boundary_rejects_a_boundary_covering_the_whole_prompt():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok"},
    ]
    # No generation prompt in the full text, so a boundary rendered the same
    # way (add_generation_prompt=False) after the last message equals the
    # whole prompt exactly, leaving no token to feed.
    template_messages, token_ids, tokenizer = _span_prompt(
        messages, add_generation_prompt=False
    )
    config = AppConfig(max_tokens=10, context_budget=1000)
    boundary = checkpoint_boundary(
        template_messages=template_messages,
        token_ids=token_ids,
        config=config,
        tokenizer=tokenizer,
        thinking_supported=True,
        template_kwargs={},
        after_message=len(template_messages),
    )
    assert boundary["ok"] is False
    assert "covers the whole prompt" in boundary["reason"]


def test_checkpoint_boundary_rejects_a_token_mismatch():
    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1", "reasoning_content": "secret"},
        {"role": "user", "content": "u2"},
    ]
    template_messages = [
        {"role": m["role"], "content": m["content"], **(
            {"reasoning_content": m["reasoning_content"]} if "reasoning_content" in m else {}
        )}
        for m in messages
    ]
    tokenizer = FakeTokenizer()
    text = tokenizer.render(template_messages, add_generation_prompt=True)
    token_ids = tokenizer.encode(text)
    config = AppConfig(max_tokens=10, context_budget=1000)

    # Prefix [user, assistant-with-reasoning] renders the reasoning because no
    # later user exists in that slice, but the full prompt drops it -- not a
    # token prefix of the full prompt.
    boundary = checkpoint_boundary(
        template_messages=template_messages,
        token_ids=token_ids,
        config=config,
        tokenizer=tokenizer,
        thinking_supported=True,
        template_kwargs={},
        after_message=2,
    )
    assert boundary["ok"] is False
    assert "not a token prefix" in boundary["reason"]


def test_build_prompt_checkpoint_table_matches_checkpoint_boundary():
    """Regression: build_prompt's own checkpoint table must report the same
    positions checkpoint_boundary would compute for each boundary."""
    config = AppConfig(cache_checkpoints="every_message", max_tokens=10, context_budget=1000)
    report = build_prompt(
        conversation(),
        config,
        FakeTokenizer(),
        model_info=MODEL_INFO,
        capabilities={},
    )
    assert report["ok"] is True, report["error"]

    template_messages = [
        {
            "role": m.role,
            "content": m.content,
            **({"reasoning_content": m.reasoning_content} if m.reasoning_content else {}),
        }
        for m in conversation().messages
    ]

    for entry in report["checkpoints"]:
        direct = checkpoint_boundary(
            template_messages=template_messages,
            token_ids=report["token_ids"],
            config=config,
            tokenizer=FakeTokenizer(),
            thinking_supported=True,
            template_kwargs={},
            after_message=entry["after_message"],
        )
        # build_prompt additionally rejects a boundary that does not advance
        # past the previous accepted one; checkpoint_boundary alone does not
        # know about that rule, so compare only when both agree it is a real
        # token prefix.
        if direct["ok"] and entry["ok"]:
            assert direct["position"] == entry["position"]
        elif not direct["ok"]:
            assert entry["ok"] is False
