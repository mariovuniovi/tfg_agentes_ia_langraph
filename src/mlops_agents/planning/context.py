"""build_planner_validation_context — deterministic ground-truth context built ONCE
before the planner agent's retry loop. Independent of agent behavior."""
from typing import Any

from mlops_agents.config.settings import settings
from mlops_agents.contracts.planner import (
    ExperienceSummary, PlannerValidationContext,
)
from mlops_agents.experience.pool import ExperiencePool
from mlops_agents.experience.retrieval import derive_relevance_tier
from mlops_agents.knowledge.reader import match_rules
from mlops_agents.models.loader import get_models_for
from mlops_agents.experience.retrieval import to_experience_summary  # moved out of agents/planner in Task 2.3


def build_planner_validation_context(
    dataset_profile: dict[str, Any],
    task_metadata: dict[str, Any],
    problem_type: str,
) -> PlannerValidationContext:
    """Build the deterministic context that validation uses to check agent output.

    Pre-fetches similar experiences at depth `planner_max_retrieved` so conflict
    detection always sees what the agent could have cited.
    """
    specs = get_models_for(problem_type)
    pool = ExperiencePool(settings.experience_db_path)
    views = pool.find_similar(dataset_profile, problem_type, settings.planner_max_retrieved)
    experiences = [to_experience_summary(v) for v in views]

    rule_input = {**dataset_profile, **task_metadata, "problem_type": problem_type}
    matched = match_rules(rule_input)
    rules_as_dicts = [{
        "rule_id": r.rule_id,
        "prefer": r.prefer,
        "avoid_or_deprioritize": r.avoid_or_deprioritize,
        "recommend": r.recommend,
        "summary": r.reason,
    } for r in matched]

    return PlannerValidationContext(
        problem_type=problem_type,
        task_metadata=task_metadata,
        available_model_keys=[s.model_key for s in specs],
        available_model_specs=specs,
        similar_experiences=experiences,
        matched_rules=rules_as_dicts,
        rules_by_id={r["rule_id"]: r for r in rules_as_dicts},
    )
