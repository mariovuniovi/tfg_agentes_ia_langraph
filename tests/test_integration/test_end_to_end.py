"""End-to-end integration tests — require real LLM and MLflow connections.

Mark: @pytest.mark.integration
Run with: uv run pytest -m integration
Skip in CI with: uv run pytest -m "not integration"
"""

import pytest


@pytest.mark.integration
@pytest.mark.slow
def test_full_pipeline_runs_on_sample_dataset(sample_csv):
    """Full pipeline should run end-to-end on a sample CSV without errors.

    Requires: GITHUB_TOKEN env var set, MLflow running at MLFLOW_TRACKING_URI.
    """
    from langchain_core.messages import HumanMessage

    from mlops_agents.config.constants import GRAPH_RECURSION_LIMIT
    from mlops_agents.graphs.mlops_graph import graph

    config = {
        "configurable": {"thread_id": "integration-test-1"},
        "recursion_limit": GRAPH_RECURSION_LIMIT,
    }
    initial_state = {
        "messages": [HumanMessage(content=f"Validate dataset: {sample_csv}")],
        "next": "",
        "processed_dataset_path": str(sample_csv),
        "validation_passed": False,
        "validation_report": {},
        "trained_model_path": "",
        "training_run_id": "",
        "training_metrics": {},
        "evaluation_passed": False,
        "evaluation_report": {},
        "best_model_uri": "",
        "deployment_decision": "pending",
        "deployment_status": "",
        "error_message": "",
        "agent_attempt_counts": {},
    }

    result = graph.invoke(initial_state, config=config)
    assert result is not None
    assert len(result["messages"]) > 1
