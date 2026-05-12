"""Integration test for the full planner flow — requires GITHUB_TOKEN + --llm flag."""
import pytest
from pathlib import Path

from mlops_agents.agents.planner import (
    _check_evidence_references,
    _check_plan_exhaustiveness,
    build_planner_context,
    PlannerError,
)
from mlops_agents.contracts.planner import PlannerOutput
from mlops_agents.experience.pool import ExperiencePool
from mlops_agents.prompts import get_prompt
from mlops_agents.utils.llm import get_llm
from langchain_core.messages import HumanMessage, SystemMessage


@pytest.mark.integration
@pytest.mark.llm
def test_planner_produces_valid_plan_for_regression(tmp_path: Path) -> None:
    """Real LLM call — produces a valid plan for a medium regression dataset."""
    pool = ExperiencePool(tmp_path / "test.db")

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

    ctx = build_planner_context(profile, task_metadata, "regression", pool)
    assert len(ctx.available_models) > 0

    llm = get_llm("planner").with_structured_output(PlannerOutput)
    prompt = get_prompt("planner").template
    output: PlannerOutput = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=ctx.model_dump_json(indent=2)),
    ])

    # Stage 3: evidence references
    _check_evidence_references(output.evidence_used, ctx)
    # Stage 4: exhaustiveness
    _check_plan_exhaustiveness(output.plan, ctx.available_models)

    # All candidate model_keys must be in available_models
    for cand in output.plan.candidates:
        assert cand.model_key in ctx.available_models, (
            f"{cand.model_key} not in available_models"
        )

    # All rejected models must have non-empty reason
    for rej in output.plan.models_not_recommended:
        assert rej.reason.strip(), f"{rej.model_key} has empty reason"

    # Every available model accounted for
    accounted = (
        {c.model_key for c in output.plan.candidates}
        | {r.model_key for r in output.plan.models_not_recommended}
    )
    assert accounted == set(ctx.available_models)

    # planning_analysis is non-empty
    assert len(output.planning_analysis) > 50
