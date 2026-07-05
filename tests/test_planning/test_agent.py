"""Test build_planner_agent factory."""
from unittest.mock import MagicMock, patch

from mlops_agents.planning.agent import build_planner_agent
from mlops_agents.planning.tools import build_planner_tools
from mlops_agents.planning.trace import ToolTrace


def test_build_planner_agent_returns_compiled_graph():
    trace = ToolTrace()
    tools = build_planner_tools({}, {}, "forecasting", trace)
    # No real LLM call — just verify the agent builds without exception
    with patch("mlops_agents.utils.llm.get_llm") as mock_get_llm:
        mock_get_llm.return_value = MagicMock()
        agent = build_planner_agent(tools)
        assert agent is not None
        # CompiledStateGraph has an invoke method
        assert hasattr(agent, "invoke")
