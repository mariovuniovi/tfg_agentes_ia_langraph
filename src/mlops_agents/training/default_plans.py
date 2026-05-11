"""Generate a default TrainingPlan from the registry when no agent is involved."""
from __future__ import annotations
from typing import Any, Union
from mlops_agents.contracts.profile import DatasetProfile
from mlops_agents.contracts.training import TrainingPlan, TrainingPlanCandidate, TrialBudget
from mlops_agents.models.loader import ModelSpec, get_models_for


def _row_count_lower_bound(bucket: str) -> int:
    return {"very_small": 0, "small": 500, "medium": 1000, "large": 50_000}.get(bucket, 0)


def _is_eligible(model: ModelSpec, profile: Union[DatasetProfile, dict[str, Any]]) -> bool:
    requires = model.requires or {}
    n_rows_bucket = profile.n_rows if isinstance(profile, DatasetProfile) else profile.get("n_rows", "small")
    if "min_rows" in requires:
        if _row_count_lower_bound(n_rows_bucket) < requires["min_rows"]:
            return False
    return True


def default_training_plan(problem_type: str, dataset_profile: Union[DatasetProfile, dict[str, Any]]) -> TrainingPlan:
    """All eligible models from the registry with deterministic registry-order priority."""
    eligible = [m for m in get_models_for(problem_type) if _is_eligible(m, dataset_profile)]
    candidates = [
        TrainingPlanCandidate(
            priority=i + 1,
            model_key=m.model_key,
            initial_hyperparameters=m.default_params,
            reason="default plan: registry-eligible",
        )
        for i, m in enumerate(eligible)
    ]
    if not candidates:
        raise ValueError(f"No eligible models for problem_type={problem_type}")
    return TrainingPlan(
        problem_type=problem_type,
        candidates=candidates,
        trial_budget=TrialBudget(allocation_strategy="equal", total_trials=60),
    )
