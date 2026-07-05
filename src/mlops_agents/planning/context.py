"""Planner validation context — deterministic ground-truth built ONCE
before the planner agent's retry loop. Independent of agent behavior."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from mlops_agents.config.settings import settings
from mlops_agents.contracts.planner import ExperienceSummary
from mlops_agents.experience.pool import ExperiencePool
from mlops_agents.experience.retrieval import to_experience_summary  # moved out of agents/planner in Task 2.3
from mlops_agents.knowledge.reader import match_rules
from mlops_agents.models.loader import ModelSpec, get_models_for


class PlannerValidationContext(BaseModel):
    """Deterministic ground-truth context — independent of agent behavior.

    Lives in planning/ (not contracts/) because it carries domain objects
    (ModelSpec) and contracts/ must stay free of domain imports.
    """

    problem_type: str
    task_metadata: dict[str, Any]
    available_model_keys: list[str]
    available_model_specs: list[ModelSpec]
    similar_experiences: list[ExperienceSummary]
    matched_rules: list[dict[str, Any]]
    rules_by_id: dict[str, dict[str, Any]]

    model_config = {"arbitrary_types_allowed": True}  # for ModelSpec


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
