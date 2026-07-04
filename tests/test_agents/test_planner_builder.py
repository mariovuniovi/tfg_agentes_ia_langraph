"""build_planner_agent now lives in mlops_agents.planning.agent (see planning/agent.py).
The shim agents/planner.py no longer exports it — these tests now point to the new module."""
from unittest.mock import patch, MagicMock

from mlops_agents.planning.agent import build_planner_agent
from mlops_agents.planning.tools import build_planner_tools
from mlops_agents.planning.trace import ToolTrace


def test_build_planner_agent_returns_agent():
    trace = ToolTrace()
    tools = build_planner_tools({}, {}, "regression", trace)

    with patch("mlops_agents.planning.agent.get_llm") as mock_get_llm, \
         patch("mlops_agents.planning.agent.create_agent") as mock_create_agent:
        mock_get_llm.return_value = MagicMock()
        mock_create_agent.return_value = "AGENT"

        result = build_planner_agent(tools)

    mock_get_llm.assert_called_once_with("planner")
    mock_create_agent.assert_called_once()
    assert result == "AGENT"
