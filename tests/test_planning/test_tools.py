import pytest
from mlops_agents.planning.tools import build_planner_tools, _view_to_tool_dict
from mlops_agents.planning.trace import ToolTrace
from mlops_agents.config.settings import settings
import mlops_agents.experience.pool as pool_mod
from unittest.mock import patch, MagicMock


@pytest.fixture
def fresh_trace():
    return ToolTrace()


@pytest.fixture
def fake_pool():
    p = MagicMock()
    p.find_similar.return_value = []
    return p


def test_list_available_models_filters_by_problem_type(fresh_trace):
    tools = build_planner_tools({}, {}, "forecasting", fresh_trace)
    list_tool = next(t for t in tools if t.name == "list_available_models")
    result = list_tool.invoke({})
    assert isinstance(result, list)
    assert all(m["problem_type"] == "forecasting" for m in result)
    assert "list_available_models" in fresh_trace.called_tools
    assert len(fresh_trace.listed_model_keys) > 0
    assert fresh_trace.tool_call_count == 1


def test_inspect_model_details_returns_error_for_unknown_key(fresh_trace):
    tools = build_planner_tools({}, {}, "forecasting", fresh_trace)
    inspect_tool = next(t for t in tools if t.name == "inspect_model_details")
    result = inspect_tool.invoke({"model_key": "nonexistent_model"})
    assert "error" in result
    assert "nonexistent_model" in result["error"]


def test_per_tool_inspect_cap_counts_calls_not_unique_keys(fresh_trace, monkeypatch):
    """Cap is on call count, not unique keys — repeated inspects of the same model still
    burn the budget."""
    monkeypatch.setattr(settings, "planner_max_inspect_calls", 2)
    tools = build_planner_tools({}, {}, "forecasting", fresh_trace)
    inspect_tool = next(t for t in tools if t.name == "inspect_model_details")
    inspect_tool.invoke({"model_key": "ets"})  # call 1
    inspect_tool.invoke({"model_key": "ets"})  # call 2 — same key, still counts
    # Third call hits cap
    result = inspect_tool.invoke({"model_key": "ets"})
    assert "max inspect_model_details calls" in result["error"]
    assert fresh_trace.inspect_model_details_count == 2


def test_global_max_tool_calls_short_circuits(fresh_trace, monkeypatch):
    monkeypatch.setattr(settings, "planner_max_tool_calls", 2)
    tools = build_planner_tools({}, {}, "forecasting", fresh_trace)
    list_tool = next(t for t in tools if t.name == "list_available_models")
    rules_tool = next(t for t in tools if t.name == "retrieve_ml_knowledge")
    list_tool.invoke({})
    rules_tool.invoke({})
    # 3rd call should be rejected
    result = list_tool.invoke({})
    assert "max_tool_calls exceeded" in result["error"]


def test_retrieve_similar_experiences_clamps_top_k(fresh_trace, monkeypatch):
    """Patch at the USAGE site (planning.tools.ExperiencePool), not the source module.
    `from ... import ExperiencePool` rebinds the symbol into tools.py's namespace, so
    patching pool_mod has no effect on the already-imported reference."""
    monkeypatch.setattr(settings, "planner_max_retrieved", 5)
    captured = {}

    def fake_pool_factory(path):
        p = MagicMock()

        def fake_find(profile, problem_type, k):
            captured["k"] = k
            return []
        p.find_similar.side_effect = fake_find
        return p

    monkeypatch.setattr("mlops_agents.planning.tools.ExperiencePool", fake_pool_factory)
    tools = build_planner_tools({}, {}, "forecasting", fresh_trace)
    retrieve_tool = next(t for t in tools if t.name == "retrieve_similar_experiences")
    retrieve_tool.invoke({"top_k": 50})  # ask for way too many
    assert captured["k"] == 5  # clamped


def test_dedup_across_repeated_calls(fresh_trace):
    tools = build_planner_tools({}, {}, "forecasting", fresh_trace)
    list_tool = next(t for t in tools if t.name == "list_available_models")
    list_tool.invoke({})
    # Calling again would normally double-count — verify dedup
    assert fresh_trace.called_tools == ["list_available_models"]
    initial_models = set(fresh_trace.listed_model_keys)
    list_tool.invoke({})
    # Same models, still deduped
    assert set(fresh_trace.listed_model_keys) == initial_models
