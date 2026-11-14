import pytest
from pydantic import ValidationError

from config import AppConfig, PRESETS, defaults_dict, fields_schema, inactive_reasons


def test_defaults_round_trip():
    config = AppConfig()
    assert config.model_dump(mode="json") == defaults_dict()
    assert config.model_path == "mlx-community/Qwen3.5-4B-4bit"
    assert config.temperature == 1.0
    assert config.presence_penalty == 1.5
    assert config.kv_bits is None


def test_fields_schema_is_complete_and_ordered():
    fields = fields_schema()
    names = [field["name"] for field in fields]
    assert len(names) == len(set(names))
    assert "model_path" in names and "top_p" in names and "cache_reuse" in names
    groups = [field["group"] for field in fields]
    assert groups == sorted(groups, key=["model", "prompt", "context", "sampling", "runtime"].index)
    for field in fields:
        assert field["purpose"]
        assert field["tradeoff"]
        assert field["applies"] in {"immediate", "next_gen", "reload", "unsupported"}


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        AppConfig(not_a_field=1)


def test_bounds_are_enforced():
    with pytest.raises(ValidationError):
        AppConfig(temperature=-1)
    with pytest.raises(ValidationError):
        AppConfig(top_p=1.5)
    with pytest.raises(ValidationError):
        AppConfig(kv_bits=7)
    with pytest.raises(ValidationError):
        AppConfig(max_tokens=0)
    with pytest.raises(ValidationError):
        AppConfig(trust_remote_code=True)
    with pytest.raises(ValidationError):
        AppConfig(stop_sequences=[""])


def test_logit_bias_keys_become_ints():
    config = AppConfig(logit_bias={"248044": -5.0, "12": 1.5})
    assert config.logit_bias == {248044: -5.0, 12: 1.5}


def test_presets_validate():
    for preset in PRESETS:
        values = defaults_dict()
        values.update(preset["values"])
        AppConfig(**values)


def test_inactive_reasons_for_greedy_and_kv():
    greedy = AppConfig(temperature=0.0)
    reasons = inactive_reasons(greedy)
    for name in ("top_p", "top_k", "min_p", "xtc_probability"):
        assert name in reasons
        assert "greedy" in reasons[name]
    # penalties are logits processors, so they still apply under greedy
    assert "presence_penalty" not in reasons

    kv_off = AppConfig(kv_bits=None)
    assert "kv_group_size" in inactive_reasons(kv_off)
    assert "quantized_kv_start" in inactive_reasons(kv_off)

    cache_off = AppConfig(cache_reuse=False)
    assert "cache_checkpoints" in inactive_reasons(cache_off)
    assert "cache_max_entries" in inactive_reasons(cache_off)


def test_inactive_reasons_when_enabled_are_gone():
    config = AppConfig(
        temperature=0.7,
        top_p=0.9,
        top_k=20,
        min_p=0.05,
        repetition_penalty=1.1,
        presence_penalty=0.5,
        frequency_penalty=0.5,
        logit_bias={1: 2.0},
        kv_bits=4,
        cache_reuse=True,
        top_logprobs=5,
    )
    reasons = inactive_reasons(config)
    for name in ("top_p", "top_k", "min_p", "kv_group_size", "cache_max_entries"):
        assert name not in reasons
