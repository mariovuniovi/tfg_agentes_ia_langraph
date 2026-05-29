from unittest.mock import patch, MagicMock


def test_build_planner_agent_uses_structured_output():
    with patch("mlops_agents.agents.planner.get_llm") as mock_get_llm:
        fake_llm = MagicMock()
        fake_llm.with_structured_output.return_value = "STRUCTURED"
        mock_get_llm.return_value = fake_llm

        from mlops_agents.agents.planner import build_planner_agent
        result = build_planner_agent()

    mock_get_llm.assert_called_once_with("planner", max_tokens=16000)
    fake_llm.with_structured_output.assert_called_once()
    assert result == "STRUCTURED"
