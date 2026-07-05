import pytest

from mlops_agents.observability.pricing import estimate_cost, normalize


def test_normalize_strips_date_suffix():
    assert normalize("gpt-5.4-mini-2025-11-01") == "gpt-5.4-mini"


def test_normalize_strips_provider_prefix():
    assert normalize("openai/gpt-5.4-mini") == "gpt-5.4-mini"


def test_normalize_strips_both():
    assert normalize("openai/gpt-5.4-mini-2025-11-01") == "gpt-5.4-mini"


def test_normalize_plain_name_unchanged():
    assert normalize("gpt-5.4-mini") == "gpt-5.4-mini"


def test_unknown_model_returns_none():
    assert estimate_cost("gpt-unknown-xyz", 1000, 1000) is None


def test_mini_cost_per_million():
    # gpt-5.4-mini: $0.75 input + $4.50 output per 1M
    cost = estimate_cost("gpt-5.4-mini", 1_000_000, 1_000_000)
    assert cost == pytest.approx(5.25)


def test_nano_cost_per_million():
    # gpt-5.4-nano: $0.20 input + $1.25 output per 1M
    cost = estimate_cost("gpt-5.4-nano", 1_000_000, 1_000_000)
    assert cost == pytest.approx(1.45)


def test_mini_small_call():
    # 1000 input + 200 output tokens
    cost = estimate_cost("gpt-5.4-mini", 1000, 200)
    assert cost == pytest.approx(1000 * 0.75 / 1_000_000 + 200 * 4.50 / 1_000_000)


def test_date_suffix_still_matches():
    cost_plain = estimate_cost("gpt-5.4-mini", 1000, 1000)
    cost_dated = estimate_cost("gpt-5.4-mini-2025-11-01", 1000, 1000)
    assert cost_plain == pytest.approx(cost_dated)


def test_provider_prefix_still_matches():
    cost_plain = estimate_cost("gpt-5.4-mini", 1000, 1000)
    cost_prefixed = estimate_cost("openai/gpt-5.4-mini", 1000, 1000)
    assert cost_plain == pytest.approx(cost_prefixed)


def test_cached_tokens_zero_cost_when_not_published():
    # cached_input_per_1m = 0.0 in YAML, so cached tokens add nothing
    cost_without = estimate_cost("gpt-5.4-mini", 1000, 1000)
    cost_with = estimate_cost("gpt-5.4-mini", 1000, 1000, cached_input_tokens=500)
    assert cost_without == pytest.approx(cost_with)
