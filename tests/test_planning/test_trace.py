"""Tests for ToolTrace model."""
from mlops_agents.planning.trace import ToolTrace


def test_tooltrace_defaults():
    t = ToolTrace()
    assert t.called_tools == []
    assert t.tool_call_count == 0
    assert t.raw_observations == []


def test_tooltrace_model_dump_roundtrip():
    t = ToolTrace()
    t.called_tools = ["a"]
    t.tool_call_count = 1
    t.raw_observations = [{"tool": "a"}]
    d = t.model_dump()
    assert d["tool_call_count"] == 1
    t2 = ToolTrace.model_validate(d)
    assert t2.called_tools == ["a"]
