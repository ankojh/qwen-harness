"""AppConfig: the single source of truth for every setting in the harness.

The UI does not hard-code controls.  Each field below carries metadata in
``json_schema_extra`` -- the group it belongs to, what it does, its tradeoff,
and when it takes effect -- and ``/api/state`` serves that metadata.  Adding a
field here therefore adds a control to the UI (and a key to the JSON editor)
without a frontend change.

"applies" is one of:
  immediate  -- read by the next action that needs it (preview, cache policy,
                allocator setup)
  next_gen   -- captured when Generate is pressed
  reload     -- only used when the model is loaded
  unsupported-- shown disabled with a reason from model.capabilities()

``inactive_reasons()`` explains settings that are valid but would do nothing
with the current values (for example top_p while temperature is 0).  The UI
shows those as inactive instead of pretending they apply.
"""

from __future__ import annotations

import types
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, field_validator

GROUPS = ["model", "prompt", "context", "sampling", "runtime"]

RESERVED_TEMPLATE_KWARGS = {
    "messages",
    "tokenize",
    "return_dict",
    "add_generation_prompt",
    "chat_template",
    "enable_thinking",
}


def _meta(
    group: str,
    label: str,
    purpose: str,
    tradeoff: str,
    applies: str,
    *,
    library_default: str | None = None,
    kind: str | None = None,
    options: list[Any] | None = None,
) -> dict[str, Any]:
    """Build the json_schema_extra dict for one field."""
    return {
        "group": group,
        "label": label,
        "purpose": purpose,
        "tradeoff": tradeoff,
        "applies": applies,
        "library_default": library_default,
        "kind": kind,
        "options": options,
    }


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # ---------------------------------------------------------------- model
    model_path: str = Field(
        "mlx-community/Qwen3.5-4B-4bit",
        json_schema_extra=_meta(
            "model",
            "Model path",
            "Hugging Face repo id or local folder passed to mlx_lm.load.",
            "A repo that is not in ~/.cache/huggingface triggers a download on Load. "
            "Other sizes and families change speed, memory and supported features.",
            "reload",
        ),
    )
    revision: str | None = Field(
        None,
        json_schema_extra=_meta(
            "model",
            "Revision",
            "Branch, tag or commit of the repository to load.",
            "Pin a commit if you want reproducible loads.",
            "reload",
            kind="text_nullable",
        ),
    )
    adapter_path: str | None = Field(
        None,
        json_schema_extra=_meta(
            "model",
            "Adapter path",
            "Local folder with a LoRA adapter applied on top of the base weights.",
            "Adapters change behaviour but not the base weight quantization.",
            "reload",
            kind="text_nullable",
        ),
    )
    lazy_load: bool = Field(
        False,
        json_schema_extra=_meta(
            "model",
            "Lazy load",
            "If true, weights are materialized on first use instead of during Load.",
            "Moves load time into the first generation, so the two timings are no "
            "longer separate. Off by default.",
            "reload",
            library_default="False",
        ),
    )
    trust_remote_code: Literal[False] = Field(
        False,
        json_schema_extra=_meta(
            "model",
            "Trust remote code",
            "Would execute Python shipped inside the model repository.",
            "Always disabled: MLX Qwen models do not need it.",
            "unsupported",
            library_default="False",
        ),
    )

    # --------------------------------------------------------------- prompt
    system_prompt: str = Field(
        "",
        json_schema_extra=_meta(
            "prompt",
            "System prompt",
            "When non-empty, inserted as message 0 with role system.",
            "The prompt inspector lists this insertion explicitly; nothing is added secretly.",
            "next_gen",
            kind="textarea",
        ),
    )
    enable_thinking: bool = Field(
        True,
        json_schema_extra=_meta(
            "prompt",
            "Enable thinking",
            "Passed to the chat template as enable_thinking.",
            "ON ends the prompt inside <think> so reasoning is generated; OFF injects "
            "an empty think block. Disabled if the template never reads the variable.",
            "next_gen",
            library_default="True when the tokenizer has thinking tokens",
        ),
    )
    add_generation_prompt: bool = Field(
        True,
        json_schema_extra=_meta(
            "prompt",
            "Add generation prompt",
            "Append the assistant turn header (<|im_start|>assistant).",
            "Turning it off asks the model to continue the last message instead of "
            "starting an assistant turn; output usually looks odd. It is allowed so "
            "you can see the difference.",
            "next_gen",
            library_default="True",
        ),
    )
    template_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        json_schema_extra=_meta(
            "prompt",
            "Template variables",
            "Extra variables for the Jinja chat template (for example tools).",
            "Keys this template never reads are reported as unused. Reserved keys "
            "(messages, tokenize, ...) are dropped and reported.",
            "next_gen",
            kind="json",
        ),
    )
    chat_template_override: str | None = Field(
        None,
        json_schema_extra=_meta(
            "prompt",
            "Chat template override",
            "Jinja text that replaces the model's own chat template. Empty = use the model's.",
            "You can break the prompt format; the report always shows the rendered "
            "result, the variables the template reads, and unused variables.",
            "next_gen",
            kind="textarea_nullable",
        ),
    )

    # -------------------------------------------------------------- context
    context_budget: int = Field(
        32768,
        ge=1,
        le=262144,
        json_schema_extra=_meta(
            "context",
            "Context budget",
            "Upper bound for prompt tokens plus the output reservation.",
            "Larger budgets cost KV memory (about 32 KB per token on this model's 8 "
            "attention layers at fp16) and prefill time. Must not exceed the model's "
            "max_position_embeddings (262144).",
            "next_gen",
        ),
    )
    overflow_policy: Literal["report", "drop_oldest_messages"] = Field(
        "report",
        json_schema_extra=_meta(
            "context",
            "Overflow policy",
            "What to do when prompt + reservation exceeds the budget.",
            "report refuses to run and shows the numbers. drop_oldest_messages removes "
            "the oldest enabled non-system messages until it fits and lists each one. "
            "Raw mode always reports.",
            "next_gen",
            kind="select",
            options=["report", "drop_oldest_messages"],
        ),
    )
    max_tokens: int = Field(
        8192,
        ge=1,
        le=262144,
        json_schema_extra=_meta(
            "context",
            "Max tokens",
            "Output reservation and hard cap on generated tokens.",
            "Thinking uses many tokens; the model card suggests 32768 for hard problems.",
            "next_gen",
        ),
    )
    stop_sequences: list[str] = Field(
        default_factory=list,
        json_schema_extra=_meta(
            "context",
            "Stop sequences",
            "Text matched on the raw decoded stream, thinking included.",
            "A match is held back, excluded from the answer, and still visible in raw. "
            "One sequence per line.",
            "next_gen",
            kind="lines",
        ),
    )
    extra_eos_token_ids: list[int] = Field(
        default_factory=list,
        json_schema_extra=_meta(
            "context",
            "Extra EOS token ids",
            "Token ids added to the tokenizer's own EOS set.",
            "Generation stops when any EOS id is sampled. The tokenizer's set is shown "
            "read-only next to this field.",
            "next_gen",
            kind="int_list",
        ),
    )

    # ------------------------------------------------------------- sampling
    temperature: float = Field(
        1.0,
        ge=0.0,
        le=5.0,
        json_schema_extra=_meta(
            "sampling",
            "Temperature",
            "0 means greedy argmax; above 0 samples from the distribution.",
            "At 0 mlx-lm returns the greedy sampler and ignores every other sampling option.",
            "next_gen",
            library_default="0.0",
        ),
    )
    top_p: float = Field(
        0.95,
        ge=0.0,
        le=1.0,
        json_schema_extra=_meta(
            "sampling",
            "Top-p",
            "Nucleus sampling: keep the smallest token set whose probability sums to p.",
            "Only applied when 0 < p < 1; 0 disables it and 1 keeps everything.",
            "next_gen",
            library_default="0.0 (off)",
        ),
    )
    top_k: int = Field(
        20,
        ge=0,
        json_schema_extra=_meta(
            "sampling",
            "Top-k",
            "Keep only the k most likely tokens before sampling.",
            "0 disables it.",
            "next_gen",
            library_default="0 (off)",
        ),
    )
    min_p: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        json_schema_extra=_meta(
            "sampling",
            "Min-p",
            "Drop tokens below min_p times the top token's probability.",
            "0 disables it; typical values are 0.01-0.2.",
            "next_gen",
            library_default="0.0",
        ),
    )
    min_tokens_to_keep: int = Field(
        1,
        ge=1,
        json_schema_extra=_meta(
            "sampling",
            "Min tokens to keep",
            "Minimum number of tokens min-p is not allowed to remove.",
            "Only used when min_p > 0.",
            "next_gen",
            library_default="1",
        ),
    )
    xtc_probability: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        json_schema_extra=_meta(
            "sampling",
            "XTC probability",
            "Chance per token to apply exclude-top-choices sampling.",
            "EOS and think-marker tokens are passed as protected ids so XTC cannot "
            "remove them. 0 disables it.",
            "next_gen",
            library_default="0.0",
        ),
    )
    xtc_threshold: float = Field(
        0.1,
        ge=0.0,
        le=0.5,
        json_schema_extra=_meta(
            "sampling",
            "XTC threshold",
            "Probability threshold XTC uses to choose which top tokens to exclude.",
            "Only used when xtc_probability > 0.",
            "next_gen",
            library_default="0.1",
        ),
    )
    repetition_penalty: float = Field(
        1.0,
        ge=0.0,
        le=10.0,
        json_schema_extra=_meta(
            "sampling",
            "Repetition penalty",
            "Sign-aware multiplicative penalty on tokens in the recent window.",
            "1.0 means off. It looks at the last repetition_context_size tokens, not "
            "the whole output.",
            "next_gen",
            library_default="None (off)",
        ),
    )
    repetition_context_size: int = Field(
        20,
        ge=1,
        json_schema_extra=_meta(
            "sampling",
            "Repetition context",
            "Window size for the repetition penalty.",
            "Larger windows also penalize older repeats.",
            "next_gen",
            library_default="20",
        ),
    )
    presence_penalty: float = Field(
        1.5,
        ge=-2.0,
        le=2.0,
        json_schema_extra=_meta(
            "sampling",
            "Presence penalty",
            "Subtract this from a logit once a token appears in the window.",
            "mlx-lm's window-based definition differs from OpenAI's whole-output one. "
            "0 means off.",
            "next_gen",
            library_default="None (off)",
        ),
    )
    presence_context_size: int = Field(
        20,
        ge=1,
        json_schema_extra=_meta(
            "sampling",
            "Presence context",
            "Window size for the presence penalty.",
            "Only the last N tokens are considered.",
            "next_gen",
            library_default="20",
        ),
    )
    frequency_penalty: float = Field(
        0.0,
        ge=-2.0,
        le=2.0,
        json_schema_extra=_meta(
            "sampling",
            "Frequency penalty",
            "Subtract this for every occurrence of a token in the window.",
            "0 means off.",
            "next_gen",
            library_default="None (off)",
        ),
    )
    frequency_context_size: int = Field(
        20,
        ge=1,
        json_schema_extra=_meta(
            "sampling",
            "Frequency context",
            "Window size for the frequency penalty.",
            "Only the last N tokens are considered.",
            "next_gen",
            library_default="20",
        ),
    )
    logit_bias: dict[int, float] = Field(
        default_factory=dict,
        json_schema_extra=_meta(
            "sampling",
            "Logit bias",
            'Additive bias per token id, for example {"248044": -5}.',
            "Ids are checked against the vocabulary size when the prompt is built.",
            "next_gen",
            kind="json",
        ),
    )
    seed: int | None = Field(
        None,
        json_schema_extra=_meta(
            "sampling",
            "Seed",
            "Calls mx.random.seed before each generation.",
            "Cache reuse changes prefill chunking, which can change floating-point "
            "results slightly even with the same seed.",
            "next_gen",
            kind="number_nullable",
        ),
    )

    # -------------------------------------------------------------- runtime
    prefill_step_size: int = Field(
        2048,
        ge=1,
        le=65536,
        json_schema_extra=_meta(
            "runtime",
            "Prefill step size",
            "Prompt tokens per forward pass.",
            "Larger is faster but raises peak memory and makes Stop react less often.",
            "next_gen",
            library_default="2048",
        ),
    )
    kv_bits: Literal[2, 3, 4, 5, 6, 8] | None = Field(
        None,
        json_schema_extra=_meta(
            "runtime",
            "KV bits",
            "Quantize the KV cache to this many bits (None = no quantization).",
            "Only the 8 full-attention layers change. Stored caches are keyed by this "
            "value, so changing it clears the prompt cache.",
            "next_gen",
            library_default="None",
            kind="select",
            options=[None, 2, 3, 4, 5, 6, 8],
        ),
    )
    kv_group_size: Literal[32, 64, 128] = Field(
        64,
        json_schema_extra=_meta(
            "runtime",
            "KV group size",
            "Group size used for KV cache quantization.",
            "Only used when kv_bits is set.",
            "next_gen",
            library_default="64",
            kind="select",
            options=[32, 64, 128],
        ),
    )
    quantized_kv_start: int = Field(
        5000,
        ge=0,
        json_schema_extra=_meta(
            "runtime",
            "Quantized KV start",
            "Start quantizing once the cache passes this many tokens.",
            "Only used when kv_bits is set. Earlier quantization saves more memory and "
            "can cost quality.",
            "next_gen",
            library_default="5000",
        ),
    )
    max_kv_size: int | None = Field(
        None,
        ge=1,
        json_schema_extra=_meta(
            "runtime",
            "Max KV size",
            "Rotating KV cache size for models that do not define make_cache.",
            "mlx-lm ignores this when the model defines make_cache, which Qwen3.5 does.",
            "next_gen",
            kind="number_nullable",
        ),
    )
    cache_reuse: bool = Field(
        True,
        json_schema_extra=_meta(
            "runtime",
            "Cache reuse",
            "Look up and store prompt caches across runs.",
            "Reuse skips prefill for a matching prefix; off makes every run a cold "
            "prefill.",
            "immediate",
        ),
    )
    cache_checkpoints: Literal["off", "last_message", "every_message"] = Field(
        "last_message",
        json_schema_extra=_meta(
            "runtime",
            "Cache checkpoints",
            "Where prefill stores prompt-cache snapshots: before the assistant "
            "header of the last message, after every message, or not at all.",
            "Each snapshot holds the full recurrent state of the 24 linear-attention "
            "layers (about 51.5 MB, constant) plus the attention KV up to that "
            "position (about 33.5 KB per token at fp16). every_message lets an edit "
            "to message k reuse everything up to message k-1, at the cost of one "
            "snapshot per message; long conversations hit cache_max_gb and evict. "
            "off stores only the after-generation entry.",
            "immediate",
            kind="select",
            options=["off", "last_message", "every_message"],
        ),
    )
    cache_max_entries: int = Field(
        8,
        ge=1,
        json_schema_extra=_meta(
            "runtime",
            "Cache max entries",
            "Maximum number of stored prompt-cache entries (LRU).",
            "The least recently used entry is dropped past this limit.",
            "immediate",
        ),
    )
    cache_max_gb: float = Field(
        4.0,
        ge=0.0,
        json_schema_extra=_meta(
            "runtime",
            "Cache max GB",
            "Byte budget for stored prompt caches (sum of cache.nbytes).",
            "Lower values free memory but lose reuse.",
            "immediate",
        ),
    )
    mlx_cache_limit_gb: float | None = Field(
        None,
        ge=0.0,
        json_schema_extra=_meta(
            "runtime",
            "MLX cache limit GB",
            "MLX allocator buffer-cache limit (mx.set_cache_limit). Empty = MLX default.",
            "This is not the prompt cache. Lower values return memory to the OS sooner "
            "at some speed cost.",
            "immediate",
            kind="number_nullable",
        ),
    )
    top_logprobs: int = Field(
        5,
        ge=0,
        le=20,
        json_schema_extra=_meta(
            "runtime",
            "Top logprobs",
            "Number of alternative tokens recorded for every generated token.",
            "Costs a top-k over the 248k vocabulary per token. 0 disables it.",
            "next_gen",
        ),
    )

    # ------------------------------------------------------------ validators
    @field_validator("stop_sequences")
    @classmethod
    def _clean_stop_sequences(cls, value: list[str]) -> list[str]:
        cleaned = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("stop sequences must be strings")
            if item == "":
                raise ValueError("a stop sequence cannot be empty")
            cleaned.append(item)
        return cleaned

    @field_validator("extra_eos_token_ids")
    @classmethod
    def _clean_eos(cls, value: list[int]) -> list[int]:
        cleaned = []
        for item in value:
            if item < 0:
                raise ValueError("token ids must be non-negative")
            cleaned.append(int(item))
        return cleaned


# --------------------------------------------------------------------------
# Presets.  Buttons in the UI apply these through the same validated
# PUT /api/config path as manual edits, so the config.changed events look
# identical.
# --------------------------------------------------------------------------
PRESETS = [
    {
        "id": "qwen_thinking",
        "name": "Qwen thinking (general)",
        "description": "Qwen model-card settings for thinking mode.",
        "values": {
            "enable_thinking": True,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
            "frequency_penalty": 0.0,
        },
    },
    {
        "id": "qwen_non_thinking",
        "name": "Qwen non-thinking (general)",
        "description": "Qwen model-card settings with thinking disabled.",
        "values": {
            "enable_thinking": False,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
            "frequency_penalty": 0.0,
        },
    },
    {
        "id": "greedy",
        "name": "Greedy",
        "description": "Temperature 0: always take the most likely token.",
        "values": {"temperature": 0.0},
    },
]


def defaults_dict() -> dict[str, Any]:
    """The default config as plain JSON-compatible data."""
    return AppConfig().model_dump(mode="json")


def inactive_reasons(config: AppConfig) -> dict[str, str]:
    """Valid settings that would do nothing right now, with the reason why."""
    reasons: dict[str, str] = {}

    if config.temperature == 0:
        for name in (
            "top_p",
            "top_k",
            "min_p",
            "min_tokens_to_keep",
            "xtc_probability",
            "xtc_threshold",
        ):
            reasons[name] = (
                "temperature is 0, so mlx-lm uses the greedy sampler and ignores "
                "every sampling option"
            )
    else:
        if not 0 < config.top_p < 1:
            reasons["top_p"] = "top_p only applies when 0 < top_p < 1; it is off now"
        if config.top_k == 0:
            reasons["top_k"] = "top_k = 0 disables the top-k filter"
        if config.min_p == 0:
            reasons["min_p"] = "min_p = 0 disables the min-p filter"
            reasons["min_tokens_to_keep"] = "only used by min-p, which is off"
        if config.xtc_probability == 0:
            reasons["xtc_probability"] = "0 means XTC is never applied"
            reasons["xtc_threshold"] = "only used when XTC is applied"

    if not config.logit_bias:
        reasons["logit_bias"] = "empty, so no bias is applied"
    if config.repetition_penalty == 1.0:
        reasons["repetition_penalty"] = "1.0 means the penalty is off"
        reasons["repetition_context_size"] = "only used by the repetition penalty"
    if config.presence_penalty == 0:
        reasons["presence_penalty"] = "0 means the penalty is off"
        reasons["presence_context_size"] = "only used by the presence penalty"
    if config.frequency_penalty == 0:
        reasons["frequency_penalty"] = "0 means the penalty is off"
        reasons["frequency_context_size"] = "only used by the frequency penalty"

    if config.kv_bits is None:
        reasons["kv_group_size"] = "only used when kv_bits is set"
        reasons["quantized_kv_start"] = "only used when kv_bits is set"

    if not config.cache_reuse:
        for name in (
            "cache_checkpoints",
            "cache_max_entries",
            "cache_max_gb",
        ):
            reasons[name] = "cache_reuse is off"

    if config.top_logprobs == 0:
        reasons["top_logprobs"] = "0 means alternatives are not recorded"

    return reasons


# --------------------------------------------------------------------------
# Schema for the generic frontend.  Type names are deliberately small and
# stable: the UI only needs to know which widget to draw.
# --------------------------------------------------------------------------
def _unwrap(annotation: Any) -> tuple[Any, bool]:
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        return (args[0] if args else str), True
    return annotation, False


def _type_name(annotation: Any) -> tuple[str, list[Any] | None]:
    inner, _ = _unwrap(annotation)
    origin = get_origin(inner)
    if origin is Literal:
        values = list(get_args(inner))
        kinds = {type(v) for v in values}
        if kinds == {bool}:
            return "boolean", values
        if kinds == {int}:
            return "integer", values
        return "string", values
    if inner is bool:
        return "boolean", None
    if inner is int:
        return "integer", None
    if inner is float:
        return "number", None
    if inner is str:
        return "string", None
    if inner is dict or origin is dict:
        return "object", None
    if inner is list or origin is list:
        return "array", None
    return "string", None


def fields_schema() -> list[dict[str, Any]]:
    """Serving copy of every field's metadata, ordered for the config panel."""
    fields: list[dict[str, Any]] = []
    for group in GROUPS:
        for name, info in AppConfig.model_fields.items():
            extra = dict(info.json_schema_extra or {})
            if extra.get("group") != group:
                continue
            type_name, literal_values = _type_name(info.annotation)
            _, nullable = _unwrap(info.annotation)
            default = info.get_default(call_default_factory=True)
            fields.append(
                {
                    "name": name,
                    "label": extra.get("label", name.replace("_", " ").title()),
                    "group": group,
                    "type": type_name,
                    "nullable": nullable,
                    "kind": extra.get("kind") or type_name,
                    "options": extra.get("options") or literal_values,
                    "default": default,
                    "purpose": extra.get("purpose", ""),
                    "tradeoff": extra.get("tradeoff", ""),
                    "applies": extra.get("applies", "next_gen"),
                    "library_default": extra.get("library_default"),
                }
            )
    return fields
