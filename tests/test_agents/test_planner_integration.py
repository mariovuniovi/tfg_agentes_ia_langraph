"""Integration test for the full planner flow — requires GITHUB_TOKEN + --llm flag."""
import pytest
from pathlib import Path

from mlops_agents.planning.context import build_planner_validation_context
from mlops_agents.contracts.planner import PlannerOutput
from mlops_agents.prompts import get_prompt
from mlops_agents.utils.llm import get_llm
from langchain_core.messages import HumanMessage, SystemMessage


@pytest.mark.integration
@pytest.mark.llm
def test_planner_produces_valid_plan_for_regression(tmp_path: Path) -> None:
    """Real LLM call — produces a valid plan for a medium regression dataset.

    NOTE: The new planner is a tool-using ReAct agent (planning.node.planner_node).
    This integration test drives the LLM directly via the old structured-output path
    for quick validation of the prompt; end-to-end node tests belong in test_planning/.
    """
    profile = {
        "schema_version": 1,
        "problem_type": "regression",
        "n_rows": "small",
        "n_features": "small",
        "missing_rate": "none",
        "n_categorical_features": "none",
        "n_numerical_features": "few",
        "target_distribution": "near_normal",
    }
    task_metadata = {"target_column": "target"}

    ctx = build_planner_validation_context(profile, task_metadata, "regression")
    assert len(ctx.available_model_keys) > 0

    llm = get_llm("planner").with_structured_output(PlannerOutput)
    prompt = get_prompt("planner").template
    output: PlannerOutput = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=str(ctx)),
    ])

    # planning_analysis is non-empty
    assert len(output.planning_analysis) > 50
