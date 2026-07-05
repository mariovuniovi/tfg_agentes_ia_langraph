"""Tests for the LLM factory — OpenAI native vs OpenAI-compatible endpoints."""
from __future__ import annotations

from unittest.mock import patch

from mlops_agents.utils import llm


def _init_kwargs(agent: str) -> dict:
    """Capture the kwargs get_llm passes to ChatOpenAI."""
    with patch.object(llm, "ChatOpenAI") as mock_cls:
        llm.get_llm(agent)
    return mock_cls.call_args.kwargs


def test_native_openai_uses_reasoning_api_when_configured():
    """On the native endpoint, an agent with reasoning_effort uses the Responses API."""
    with patch.object(llm, "get_agent_config", return_value={"model": "gpt-5.4-mini", "reasoning_effort": "medium"}), \
         patch.object(llm.settings, "openai_base_url", ""), \
         patch.object(llm.settings, "openai_model_override", ""):
        kwargs = _init_kwargs("planner")
    assert kwargs["model"] == "gpt-5.4-mini"
    assert kwargs["use_responses_api"] is True
    assert kwargs["reasoning"] == {"effort": "medium", "summary": "auto"}
    assert "base_url" not in kwargs


def test_github_models_sets_base_url_and_overrides_model():
    """A base_url routes to the compatible endpoint, the override forces its model,
    and the reasoning API is disabled (unsupported there)."""
    with patch.object(llm, "get_agent_config", return_value={"model": "gpt-5.4-mini", "reasoning_effort": "medium"}), \
         patch.object(llm.settings, "openai_base_url", "https://models.github.ai/inference"), \
         patch.object(llm.settings, "openai_model_override", "openai/gpt-4.1-mini"):
        kwargs = _init_kwargs("planner")
    assert kwargs["base_url"] == "https://models.github.ai/inference"
    assert kwargs["model"] == "openai/gpt-4.1-mini"
    assert "use_responses_api" not in kwargs
    assert "reasoning" not in kwargs
