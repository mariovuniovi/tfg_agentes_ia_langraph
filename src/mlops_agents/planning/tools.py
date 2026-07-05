"""Planner agent tools — closure-bound deterministic wrappers around existing helpers.

Each tool records observations to a shared ToolTrace so validation can later
verify the agent only cited what it actually retrieved (hybrid validation A1)."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.tools import tool

from mlops_agents.config.settings import settings
from mlops_agents.experience.pool import ExperiencePool
from mlops_agents.experience.retrieval import derive_relevance_tier
from mlops_agents.experience.schema import RetrievalView
from mlops_agents.knowledge.reader import match_rules
from mlops_agents.models.loader import get_model, get_models_for
from mlops_agents.planning.trace import ToolTrace

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

_MAX_CALLS_ERR = {
    "error": "max_tool_calls exceeded — terminate and produce final PlannerOutput"
}
_MAX_INSPECT_ERR = {
    "error": ("max inspect_model_details calls reached — produce final PlannerOutput "
              "using available info or call other tools")
}


def _view_to_tool_dict(view: RetrievalView) -> dict[str, Any]:
    """Compact, agent-friendly serialization of a RetrievalView."""
    return {
        "experience_id": view.task_id,
        "similarity_score": view.similarity_ratio,
        "relevance_tier": derive_relevance_tier(view.similarity_ratio),
        "matched_buckets": list(view.matched_buckets),
        "mismatched_buckets": list(view.mismatched_buckets),
        "target_scale_note": view.target_scale_note,
        "dataset_summary": view.experience_summary or "",
        "models_tested": [c.model_key for c in view.models_tested],
        "best_model": view.selected_solution.model_key,
        "primary_metric": view.metric_to_optimize,
        "score": view.selected_solution.validation_score,
    }


def build_planner_tools(
    dataset_profile: dict[str, Any],
    task_metadata: dict[str, Any],
    problem_type: str,
    trace: ToolTrace,
) -> list[BaseTool]:
    """Build closure-bound planner tools that record observations to the shared trace.

    `problem_type` is bound at closure time — the agent cannot override it.
    `dataset_profile` and `task_metadata` are also closed over (no per-call args)."""

    def _gate(tool_name: str | None = None) -> bool:
        """Return False if call should be rejected. Enforces global ceiling.
        Per-tool inspect cap is checked separately inside inspect_model_details (uses
        trace.inspect_model_details_count — call count, NOT len(inspected_model_keys))."""
        if trace.tool_call_count >= settings.planner_max_tool_calls:
            return False
        trace.tool_call_count += 1
        return True

    def _dedup(field: list[str], new_items: set[str]) -> list[str]:
        return sorted(set(field) | new_items)

    @tool
    def list_available_models() -> list[dict[str, Any]] | dict[str, str]:
        """List all models in the registry for the current problem type. Returns one entry
        per model with headline fields (model_key, family, complexity_rank,
        supports_exogenous, supports_missing, use_when, avoid_when). Call this once at the
        start of planning. Models not in this list cannot be recommended."""
        if not _gate():
            return _MAX_CALLS_ERR
        specs = get_models_for(problem_type)
        out = [s.summary_dict() for s in specs]
        trace.called_tools = _dedup(trace.called_tools, {"list_available_models"})
        trace.listed_model_keys = _dedup(trace.listed_model_keys, {s["model_key"] for s in out})
        trace.raw_observations.append({"tool": "list_available_models", "result_count": len(out)})
        return out

    @tool
    def retrieve_similar_experiences(top_k: int = 5) -> list[dict[str, Any]] | dict[str, str]:
        """Retrieve the top-k most similar past training experiences for the current dataset.
        Similarity is deterministic (bucket-based, no embeddings). Each result includes
        experience_id, similarity_score, relevance_tier, matched_buckets, mismatched_buckets,
        target_scale_note, best_model, primary_metric, score, dataset_summary.
        Use these to inform candidate selection. Call this once unless you need a wider net.
        top_k is clamped to [1, planner_max_retrieved] so it never exceeds the
        deterministic validation context depth."""
        if not _gate():
            return _MAX_CALLS_ERR
        top_k = max(1, min(top_k, settings.planner_max_retrieved))
        pool = ExperiencePool(settings.experience_db_path)
        views = pool.find_similar(dataset_profile, problem_type, top_k)
        out = [_view_to_tool_dict(v) for v in views]
        trace.called_tools = _dedup(trace.called_tools, {"retrieve_similar_experiences"})
        trace.retrieved_experience_ids = _dedup(
            trace.retrieved_experience_ids, {o["experience_id"] for o in out}
        )
        trace.raw_observations.append({
            "tool": "retrieve_similar_experiences", "top_k": top_k, "returned": len(out),
        })
        return out

    @tool
    def retrieve_ml_knowledge() -> list[dict[str, Any]] | dict[str, str]:
        """Retrieve static ML rules that match the current dataset profile + task metadata.
        Each rule returns rule_id, prefer, avoid_or_deprioritize, recommend, summary.
        Call this once."""
        if not _gate():
            return _MAX_CALLS_ERR
        # NOTE: if task_metadata keys collide with profile keys, task_metadata wins.
        rule_input = {**dataset_profile, **task_metadata, "problem_type": problem_type}
        matched = match_rules(rule_input)
        out: list[dict[str, Any]] = [{
            "rule_id": r.rule_id,
            "prefer": r.prefer,
            "avoid_or_deprioritize": r.avoid_or_deprioritize,
            "recommend": r.recommend,
            "summary": r.reason,
        } for r in matched]
        trace.called_tools = _dedup(trace.called_tools, {"retrieve_ml_knowledge"})
        trace.retrieved_rule_ids = _dedup(trace.retrieved_rule_ids, {r["rule_id"] for r in out})
        trace.raw_observations.append({"tool": "retrieve_ml_knowledge", "returned": len(out)})
        return out

    @tool
    def inspect_model_details(model_key: str) -> dict[str, Any]:
        """Get full registry metadata for one model. Use sparingly — only when
        list_available_models doesn't give you enough info to decide. Hard cap of 3
        inspect CALLS per planner run (repeated calls on same key still burn budget).
        Returns {"error": ...} if model_key unknown."""
        # Per-tool cap check BEFORE incrementing global budget
        if trace.inspect_model_details_count >= settings.planner_max_inspect_calls:
            return _MAX_INSPECT_ERR
        if not _gate("inspect_model_details"):
            return _MAX_CALLS_ERR
        trace.inspect_model_details_count += 1
        try:
            spec = get_model(model_key)
        except KeyError:
            trace.raw_observations.append({
                "tool": "inspect_model_details", "model_key": model_key, "error": "unknown",
            })
            return {"error": f"unknown model_key: {model_key!r}"}
        out = spec.details_dict()
        trace.called_tools = _dedup(trace.called_tools, {"inspect_model_details"})
        trace.inspected_model_keys = _dedup(trace.inspected_model_keys, {model_key})
        trace.raw_observations.append({"tool": "inspect_model_details", "model_key": model_key})
        return out

    return [list_available_models, retrieve_similar_experiences,
            retrieve_ml_knowledge, inspect_model_details]
