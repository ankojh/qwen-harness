"""Session -> messages -> chat template -> tokens, with a full report.

``build_prompt`` is the only place that turns user input into a prompt.  Both
Preview and Generate call it, so the inspector can never show something
different from what was submitted.

The report deliberately contains every transformation: the system prompt that
was inserted, messages dropped for the budget, reasoning the chat template
removed, unused template variables, whether the tokenizer's own tokenization
matched the rendered text, and where a cache checkpoint could go.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

import jinja2
import jinja2.meta
from pydantic import BaseModel, Field

from config import RESERVED_TEMPLATE_KWARGS, AppConfig


class Message(BaseModel):
    role: Literal["system", "user", "assistant"] = "user"
    content: str = ""
    reasoning_content: str | None = None
    enabled: bool = True


class Session(BaseModel):
    mode: Literal["messages", "raw"] = "messages"
    messages: list[Message] = Field(default_factory=list)
    raw_text: str = ""


# --------------------------------------------------------------------------
# Template inspection
# --------------------------------------------------------------------------
def template_variables(template_text: str | None) -> list[str]:
    """Names the Jinja template reads (best effort: syntax errors give [])."""
    if not template_text:
        return []
    try:
        parsed = jinja2.Environment().parse(template_text)
        return sorted(jinja2.meta.find_undeclared_variables(parsed))
    except jinja2.TemplateSyntaxError:
        return []


def _reasoning_text(message: Message, think_start: str | None, think_end: str | None) -> str:
    if message.reasoning_content:
        return message.reasoning_content
    if think_start and think_end and think_start in message.content:
        start = message.content.find(think_start) + len(think_start)
        end = message.content.find(think_end, start)
        if end != -1:
            return message.content[start:end]
    return ""


def _preview(text: str, limit: int = 60) -> str:
    flat = " ".join(text.split())
    return flat[:limit] + ("…" if len(flat) > limit else "")


def _render(
    messages: list[dict[str, Any]],
    config: AppConfig,
    tokenizer: Any,
    *,
    add_generation_prompt: bool,
    pass_thinking: bool,
    template_kwargs: dict[str, Any],
) -> str:
    kwargs: dict[str, Any] = dict(template_kwargs)
    if pass_thinking:
        kwargs["enable_thinking"] = config.enable_thinking
    if config.chat_template_override:
        kwargs["chat_template"] = config.chat_template_override
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **kwargs,
    )


def _tokenize(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _token_text(tokenizer: Any, token_id: int, piece: str) -> str:
    """The token's own decoded text (best effort; falls back to the piece)."""
    try:
        return tokenizer.decode([token_id])
    except Exception:  # noqa: BLE001 - a display fallback, never fatal
        return piece or ""


def _token_is_special(tokenizer: Any, token_id: int) -> bool:
    """True for special or added tokens (``<|im_start|>``, ``<think>``, ...)."""
    try:
        if tokenizer.is_special_token(token_id):
            return True
    except Exception:  # noqa: BLE001 - tokenizer variants differ
        pass
    decoder = getattr(tokenizer, "added_tokens_decoder", None) or {}
    return token_id in decoder


def _token_spans(
    token_ids: list[int],
    tokenizer: Any,
    submitted: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    """Split a token list at each ``<|im_start|>`` into per-message spans.

    With ``submitted`` the spans are matched to messages in order and the
    trailing assistant opener (when present) is the last span; any mismatch
    (raw mode, an unusual template) returns None.  Without ``submitted``
    (cache entries) roles are derived from the role token after the opener
    and a trailing assistant segment without ``<|im_end|>`` is the opener.
    Positions are half-open.
    """
    im_start = None
    try:
        im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    except Exception:  # noqa: BLE001 - reported as "no spans" rather than failing
        return None
    if not isinstance(im_start, int) or im_start < 0:
        return None
    starts = [i for i, token_id in enumerate(token_ids) if token_id == im_start]
    if not starts:
        return None

    def segment_role(start: int, end: int) -> str | None:
        if end - start < 2:
            return None
        return (tokenizer.convert_ids_to_tokens(token_ids[start + 1]) or "").strip()

    def has_end(start: int, end: int) -> bool:
        if end <= start:
            return False
        return (tokenizer.convert_ids_to_tokens(token_ids[end - 1]) or "").strip() == "<|im_end|>"

    spans: list[dict[str, Any]] = []
    total = len(starts)
    for k, start in enumerate(starts):
        end = starts[k + 1] if k + 1 < total else len(token_ids)
        role = segment_role(start, end)
        if submitted is not None:
            if k < len(submitted):
                message = submitted[k]
                if role is not None and role != message["role"]:
                    return None
                spans.append(
                    {
                        "index": message["index"],
                        "role": message["role"],
                        "source": message["source"],
                        "start": start,
                        "end": end,
                        "kind": "message",
                    }
                )
            else:
                # A trailing span beyond the submitted messages must be the
                # assistant opener the template appends.
                if k != len(submitted) or role != "assistant":
                    return None
                spans.append(
                    {
                        "index": None,
                        "role": "assistant",
                        "source": "template",
                        "start": start,
                        "end": end,
                        "kind": "opener",
                    }
                )
        else:
            kind = "message"
            if k == total - 1 and role == "assistant" and not has_end(start, end):
                kind = "opener"
            spans.append(
                {
                    "index": k + 1,
                    "role": role or "unknown",
                    "source": "cache",
                    "start": start,
                    "end": end,
                    "kind": kind,
                }
            )
    if submitted is not None and not (
        len(spans) == len(submitted)
        or (len(spans) == len(submitted) + 1 and spans[-1]["kind"] == "opener")
    ):
        return None
    return spans


def _hash_tokens(token_ids: list[int]) -> str:
    payload = ",".join(str(token_id) for token_id in token_ids)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _prefix_position(
    prefix_ids: list[int], full_ids: list[int]
) -> tuple[int | None, str | None]:
    """Where a rendered message prefix lands in the full prompt token list."""
    shared = min(len(prefix_ids), len(full_ids))
    for index in range(shared):
        if prefix_ids[index] != full_ids[index]:
            return (
                None,
                "not a token prefix of the full prompt; first mismatch at token "
                f"{index}",
            )
    if len(prefix_ids) >= len(full_ids):
        return None, "prefix covers the whole prompt, leaving no token to feed"
    return len(prefix_ids), None


def _build_message_list(
    session: Session, config: AppConfig
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Return (template messages, submitted report, transformation lines)."""
    submitted: list[dict[str, Any]] = []
    transformations: list[str] = []

    if config.system_prompt:
        submitted.append(
            {
                "index": 1,
                "role": "system",
                "content": config.system_prompt,
                "reasoning_content": None,
                "source": "config.system_prompt",
            }
        )
        transformations.append(
            "inserted config.system_prompt as message 1 with role system"
        )

    for message in session.messages:
        if not message.enabled:
            transformations.append(
                f"skipped disabled message (role {message.role}, "
                f"{len(message.content)} chars)"
            )
            continue
        submitted.append(
            {
                "index": len(submitted) + 1,
                "role": message.role,
                "content": message.content,
                "reasoning_content": message.reasoning_content,
                "source": "session",
            }
        )

    template_messages: list[dict[str, Any]] = []
    for item in submitted:
        message: dict[str, Any] = {
            "role": item["role"],
            "content": item["content"],
        }
        if item["reasoning_content"]:
            message["reasoning_content"] = item["reasoning_content"]
        template_messages.append(message)

    if not any(item["role"] == "user" for item in submitted):
        transformations.append("no enabled user message is present")

    return template_messages, submitted, transformations


def _validity(
    input_tokens: int,
    config: AppConfig,
    model_max: int | None,
) -> tuple[bool, str | None, dict[str, Any]]:
    remaining = config.context_budget - input_tokens - config.max_tokens
    budget = {
        "context_budget": config.context_budget,
        "input_tokens": input_tokens,
        "output_reservation": config.max_tokens,
        "remaining": remaining,
        "model_max_position_embeddings": model_max,
    }
    if model_max is not None and config.context_budget > model_max:
        return (
            False,
            f"context_budget ({config.context_budget}) exceeds the model's "
            f"max_position_embeddings ({model_max})",
            budget,
        )
    if remaining < 0:
        return (
            False,
            f"prompt ({input_tokens} tokens) + output reservation "
            f"({config.max_tokens}) exceeds context_budget "
            f"({config.context_budget}) by {-remaining} tokens",
            budget,
        )
    return True, None, budget


def build_prompt(
    session: Session,
    config: AppConfig,
    tokenizer: Any,
    model_info: dict[str, Any] | None = None,
    capabilities: dict[str, Any] | None = None,
    *,
    cache_store: Any = None,
    compat_key: tuple | None = None,
    include_pieces: bool = False,
) -> dict[str, Any]:
    model_info = model_info or {}
    capabilities = capabilities or {}
    model_max = model_info.get("max_position_embeddings")
    think_start = model_info.get("think_start")
    think_end = model_info.get("think_end")

    template_vars = template_variables(
        config.chat_template_override or model_info.get("chat_template")
    )
    thinking_supported = bool(
        capabilities.get("enable_thinking", {}).get("supported", True)
    )

    reserved: list[str] = []
    template_kwargs: dict[str, Any] = {}
    for key, value in config.template_kwargs.items():
        if key in RESERVED_TEMPLATE_KWARGS:
            reserved.append(key)
        else:
            template_kwargs[key] = value

    unused = sorted(
        key
        for key in template_kwargs
        if template_vars and key not in template_vars
    )

    transformations: list[str] = []
    dropped: list[dict[str, Any]] = []
    submitted: list[dict[str, Any]] = []
    template_messages: list[dict[str, Any]] = []
    text = ""
    token_ids: list[int] = []
    checkpoint_at: int | None = None
    tokenize_mismatch: dict[str, Any] | None = None
    template_report: dict[str, Any] = {
        "source": "override" if config.chat_template_override else "model",
        "variables": template_vars,
        "passed_kwargs": sorted(template_kwargs) + (
            ["enable_thinking"] if thinking_supported else []
        ),
        "unused_kwargs": unused,
        "reserved_kwargs": reserved,
        "thinking_supported": thinking_supported,
        "enable_thinking_passed": thinking_supported,
    }
    if reserved:
        transformations.append(
            "dropped reserved template variables: " + ", ".join(reserved)
        )
    if unused:
        transformations.append(
            "template does not read these variables: " + ", ".join(unused)
        )

    if session.mode == "raw":
        text = session.raw_text
        token_ids = _tokenize(tokenizer, text)
        transformations.append("chat template bypassed (raw mode)")
        transformations.append("system_prompt and template settings were not used")
        valid, error, budget = _validity(len(token_ids), config, model_max)
        if not valid and config.overflow_policy == "drop_oldest_messages":
            transformations.append(
                "raw mode cannot drop messages; overflow is reported"
            )
    else:
        template_messages, submitted, transformations = _build_message_list(
            session, config
        )

        try:
            text = _render(
                template_messages,
                config,
                tokenizer,
                add_generation_prompt=config.add_generation_prompt,
                pass_thinking=thinking_supported,
                template_kwargs=template_kwargs,
            )
        except Exception as error:  # noqa: BLE001 - surfaced to the user verbatim
            report = {
                "ok": False,
                "error": f"chat template failed: {error}",
                "mode": session.mode,
                "text": "",
                "token_ids": [],
                "token_pieces": [] if include_pieces else None,
                "spans": None,
                "submitted_messages": submitted,
                "dropped_messages": [],
                "transformations": transformations,
                "checkpoint_at": None,
                "checkpoints": [],
                "checkpoint_reason": (
                    "the chat template failed before any boundary could be computed"
                ),
                "checkpoint_estimate": None,
                "checkpoint_estimate_reason": "the prompt could not be built",
                "budget": {
                    "context_budget": config.context_budget,
                    "input_tokens": 0,
                    "output_reservation": config.max_tokens,
                    "remaining": config.context_budget - config.max_tokens,
                    "model_max_position_embeddings": model_max,
                },
                "starts_in_thinking": False,
                "prompt_hash": _hash_tokens([]),
                "cache_prediction": None,
                "cache_entries": [],
                "template": template_report,
            }
            return report

        token_ids = _tokenize(tokenizer, text)
        valid, error, budget = _validity(len(token_ids), config, model_max)

        if not valid and config.overflow_policy == "drop_oldest_messages":
            candidates = [
                index
                for index, message in enumerate(session.messages)
                if message.enabled and message.role != "system"
            ]
            base_transformations = list(transformations)
            dropped_indices: set[int] = set()
            while not valid and candidates:
                drop_index = candidates.pop(0)
                dropped_indices.add(drop_index)
                message = session.messages[drop_index]
                dropped.append(
                    {
                        "message_index": drop_index + 1,
                        "role": message.role,
                        "preview": _preview(message.content),
                    }
                )
                # Rebuild from the original session minus *every* message dropped
                # so far, then re-render and re-count.
                remaining_session = Session(
                    mode="messages",
                    messages=[
                        item
                        for position, item in enumerate(session.messages)
                        if position not in dropped_indices
                    ],
                )
                template_messages, submitted, _ = _build_message_list(
                    remaining_session, config
                )
                transformations = base_transformations + [
                    f"dropped oldest message {item['message_index']} (role "
                    f"{item['role']}) to fit the budget"
                    for item in dropped
                ]
                text = _render(
                    template_messages,
                    config,
                    tokenizer,
                    add_generation_prompt=config.add_generation_prompt,
                    pass_thinking=thinking_supported,
                    template_kwargs=template_kwargs,
                )
                token_ids = _tokenize(tokenizer, text)
                valid, error, budget = _validity(
                    len(token_ids), config, model_max
                )

        # Compare our tokenization with apply_chat_template(tokenize=True).
        try:
            template_ids = list(
                tokenizer.apply_chat_template(
                    template_messages,
                    tokenize=True,
                    add_generation_prompt=config.add_generation_prompt,
                    **(
                        {"enable_thinking": config.enable_thinking}
                        if thinking_supported
                        else {}
                    ),
                    **template_kwargs,
                    **(
                        {"chat_template": config.chat_template_override}
                        if config.chat_template_override
                        else {}
                    ),
                )
            )
        except Exception:  # noqa: BLE001 - reported, not hidden
            template_ids = []
        if template_ids != token_ids:
            mismatch = next(
                (
                    index
                    for index, (left, right) in enumerate(zip(template_ids, token_ids))
                    if left != right
                ),
                min(len(template_ids), len(token_ids)),
            )
            tokenize_mismatch = {
                "template_tokens": len(template_ids),
                "encode_tokens": len(token_ids),
                "first_mismatch": mismatch,
            }
            template_report["tokenize_mismatch"] = tokenize_mismatch
            transformations.append(
                "tokenizer.encode(text) does not equal "
                "apply_chat_template(tokenize=True); the submission uses encode()"
            )

        # Did the template drop reasoning from historical assistant messages?
        for item, template_message in zip(submitted, template_messages):
            if item["role"] != "assistant":
                continue
            reasoning = _reasoning_text(
                Message(
                    role="assistant",
                    content=item["content"],
                    reasoning_content=item["reasoning_content"],
                ),
                think_start,
                think_end,
            )
            if not reasoning.strip():
                continue
            probe = reasoning.strip()[:64]
            if probe not in text:
                transformations.append(
                    f"template removed reasoning from assistant message "
                    f"{item['index']} ({len(reasoning)} chars)"
                )

    # ---------------------------------------------------------- checkpoints
    # A checkpoint is a position where prefill may snapshot the cache: after
    # message i, rendered with the same template arguments and WITHOUT the
    # assistant generation prompt.  It is only usable when it is a real token
    # prefix of the final prompt and leaves a token to feed to generate_step.
    checkpoints: list[dict[str, Any]] = []
    checkpoint_reason: str | None = None
    if session.mode == "raw":
        checkpoint_reason = (
            "raw mode bypasses the chat template, so there are no message boundaries"
        )
    elif config.cache_checkpoints == "off":
        checkpoint_reason = "cache_checkpoints is off"
    elif not config.add_generation_prompt:
        checkpoint_reason = (
            "add_generation_prompt is off; there is no assistant header to "
            "checkpoint before"
        )
    elif not submitted:
        checkpoint_reason = "there are no submitted messages to checkpoint"
    else:
        total = len(submitted)
        indices = (
            list(range(1, total + 1))
            if config.cache_checkpoints == "every_message"
            else [total]
        )
        previous = 0
        for index in indices:
            entry: dict[str, Any] = {
                "after_message": index,
                "position": None,
                "ok": False,
                "reason": None,
            }
            try:
                prefix_text = _render(
                    template_messages[:index],
                    config,
                    tokenizer,
                    add_generation_prompt=False,
                    pass_thinking=thinking_supported,
                    template_kwargs=template_kwargs,
                )
                prefix_ids = _tokenize(tokenizer, prefix_text)
            except Exception as exc:  # noqa: BLE001 - a skipped boundary is reported
                entry["reason"] = (
                    f"chat template failed for this prefix: "
                    f"{type(exc).__name__}: {exc}"
                )
                checkpoints.append(entry)
                continue
            position, reason = _prefix_position(prefix_ids, token_ids)
            if position is None:
                entry["reason"] = reason
            elif position <= previous:
                entry["reason"] = (
                    f"position {position} is not longer than the previous accepted "
                    f"position {previous}"
                )
            else:
                entry.update(position=position, ok=True)
                previous = position
            checkpoints.append(entry)
        checkpoint_at = previous or None
        if checkpoint_at is None:
            checkpoint_reason = (
                "no candidate boundary was accepted; see the skipped reasons"
            )

    starts_in_thinking = False
    if think_start and think_end:
        starts_in_thinking = text.rfind(think_start) > text.rfind(think_end)

    pieces = None
    spans = None
    if include_pieces:
        pieces = []
        for token_id in token_ids:
            piece = tokenizer.convert_ids_to_tokens(token_id)
            pieces.append(
                {
                    "id": token_id,
                    "piece": piece,
                    "text": _token_text(tokenizer, token_id, piece),
                    "special": _token_is_special(tokenizer, token_id),
                }
            )
        if session.mode == "messages":
            spans = _token_spans(token_ids, tokenizer, submitted)

    cache_prediction = None
    cache_entries: list[dict[str, Any]] = []
    checkpoint_estimate = None
    checkpoint_estimate_reason: str | None = None
    if cache_store is None or compat_key is None:
        checkpoint_estimate_reason = "no prompt cache is available"
    else:
        cache_prediction = cache_store.peek(token_ids, compat_key)
        cache_entries = cache_store.compare(token_ids, compat_key)
        planned = [
            entry["position"]
            for entry in checkpoints
            if entry["ok"]
            and entry["position"] is not None
            and entry["position"] > cache_prediction["reused_tokens"]
        ]
        if not config.cache_reuse:
            checkpoint_estimate_reason = "cache_reuse is off"
        elif not planned:
            checkpoint_estimate_reason = "no new checkpoints are planned"
        else:
            checkpoint_estimate = cache_store.estimate(planned, compat_key)
            checkpoint_estimate_reason = (
                None
                if checkpoint_estimate is not None
                else "unknown until a snapshot is measured"
            )

    report = {
        "ok": valid,
        "error": error,
        "mode": session.mode,
        "text": text,
        "token_ids": token_ids,
        "token_pieces": pieces,
        "spans": spans,
        "submitted_messages": submitted,
        "dropped_messages": dropped,
        "transformations": transformations,
        "checkpoint_at": checkpoint_at,
        "checkpoints": checkpoints,
        "checkpoint_reason": checkpoint_reason,
        "checkpoint_estimate": checkpoint_estimate,
        "checkpoint_estimate_reason": checkpoint_estimate_reason,
        "budget": budget,
        "starts_in_thinking": starts_in_thinking,
        "prompt_hash": _hash_tokens(token_ids),
        "cache_prediction": cache_prediction,
        "cache_entries": cache_entries,
        "template": template_report,
    }
    return report
