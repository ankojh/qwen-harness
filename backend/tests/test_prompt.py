"""Prompt-boundary tests with a tiny fake tokenizer/template.

No model is involved: ``FakeTokenizer`` renders messages with a deliberately
simple template and tokenizes by character code.  That is enough to exercise
the checkpoint-boundary rules, including the two real Qwen3.5 cases:
a template that raises for a system-only prefix, and reasoning that the
template drops once a later user message exists.
"""

from config import AppConfig
from prompt import Session, build_prompt

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
