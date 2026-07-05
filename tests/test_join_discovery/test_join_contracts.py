import pytest
from pydantic import ValidationError

from mlops_agents.contracts.join_discovery import BaseDatasetSelection, JoinPlan


def test_join_plan_valid() -> None:
    base = BaseDatasetSelection(
        dataset_name="energy",
        confidence="high",
        reason="contains target column",
    )
    plan = JoinPlan(base_dataset=base)
    assert plan.mode == "inferred"
    assert plan.selected_joins == []


def test_base_dataset_selection_requires_reason() -> None:
    with pytest.raises(ValidationError):
        BaseDatasetSelection(
            dataset_name="energy",
            confidence="high",
            reason="",  # min_length=1 should fail
        )
