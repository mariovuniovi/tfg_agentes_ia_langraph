"""Unit tests for the data validation agent build function."""

from unittest.mock import patch



@patch("mlops_agents.data_validation.agent.get_llm")
def test_build_data_agent_returns_compiled_graph(mock_get_llm):
    """build_data_agent() should return a compiled react agent without errors."""
    from unittest.mock import MagicMock
    mock_get_llm.return_value = MagicMock()

    from mlops_agents.data_validation.agent import build_data_agent

    agent = build_data_agent()
    assert agent is not None
