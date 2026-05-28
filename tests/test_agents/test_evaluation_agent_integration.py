"""Integration test for the evaluation agent — requires GITHUB_TOKEN."""
import pytest
from langchain_core.messages import HumanMessage


@pytest.mark.integration
@pytest.mark.slow
def test_evaluation_agent_responds_to_classification_metrics() -> None:
    """Real LLM call — verifies the evaluator agent responds to a classification scenario."""
    from mlops_agents.agents.evaluation_agent import build_evaluation_agent

    agent = build_evaluation_agent()
    result = agent.invoke({
        "messages": [HumanMessage(content=(
            "Problem type: classification\n"
            "Training run ID: test-run-nonexistent\n"
            "Training metrics: {\"macro_f1\": 0.87, \"accuracy\": 0.91}\n"
            "NOTE: MLflow stores F1 under 'macro_f1'. Call get_best_run with "
            "metric='macro_f1' (ascending=False). "
            "If get_best_run returns an error because the run does not exist in this "
            "test environment, base your recommendation on the provided metrics alone "
            "and state that evaluation_passed=True if macro_f1 >= 0.75."
        ))]
    })

    messages = result.get("messages", [])
    assert len(messages) > 1, "Agent must produce at least one response beyond the input"
    last = messages[-1]
    assert hasattr(last, "content") and last.content.strip(), (
        "Last message must have non-empty content"
    )
    content_lower = last.content.lower()
    assert any(kw in content_lower for kw in ("evaluat", "promot", "approv", "f1", "metric", "pass")), (
        "Response should reference evaluation outcome or metrics"
    )
