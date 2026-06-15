import json
from unittest.mock import patch
import pytest
from langgraph.types import Command
from mlops_agents.graphs.mlops_graph import graph


@pytest.mark.integration
def test_approval_routes_to_deployer():
    """After human approves at Gate 2, the deployer node MUST run."""
    cfg = {"configurable": {"thread_id": "deploy-test"}}
    state = {
        "training_run_id": "abc123",
        "validation_passed": True,
        "dataset_approved": True,
        "training_plan": {"selected_model": "seasonal_naive"},
        "evaluation_passed": True,
        "evaluation_report_audit": {"summary": "ok"},
        "evaluation_report": {},
        "candidate_metrics": {},
        "champion_metrics": {},
        "thresholds_applied": {},
        "deployment_decision": "pending",
        "deployment_approved": None,
        "agent_attempt_counts": {},
    }
    # Patch at the bound location (mlops_agents.deployment.deployer), not the source module
    with patch("mlops_agents.deployment.deployer.register_model") as reg, \
         patch("mlops_agents.deployment.deployer.set_model_alias") as alias:
        reg.invoke.return_value = json.dumps({"model_name": "seasonal_naive", "version": "1"})
        alias.invoke.return_value = "ok"
        # First astream pass: stops at deployment_approval HITL
        for _ in graph.stream(state, cfg, stream_mode="updates"):
            pass
        # Resume with approval
        for _ in graph.stream(Command(resume={"approved": True, "comment": ""}), cfg, stream_mode="updates"):
            pass
        final = graph.get_state(cfg).values
    assert final["deployment_status"] == "deployed", (
        "Deployer node did not run after human approval — "
        "deployment_approval_node likely overwrote deployment_decision."
    )
