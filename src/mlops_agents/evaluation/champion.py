"""Single source of truth for the human-readable champion model name."""
from typing import Any


def resolve_champion_model_name(state: dict[str, Any]) -> str:
    """Resolve the champion model name via a 4-step fallback chain.

    1. state["evaluation_report_audit"]["champion_model"]
    2. state["champion_candidate"]["model_key"]
    3. state["training_plan"]["selected_model"]
    4. state["training_run_id"][:8]
    """
    audit = state.get("evaluation_report_audit") or {}
    if isinstance(audit, dict) and audit.get("champion_model"):
        return str(audit["champion_model"])

    candidate = state.get("champion_candidate") or {}
    if isinstance(candidate, dict) and candidate.get("model_key"):
        return str(candidate["model_key"])

    plan = state.get("training_plan") or {}
    if isinstance(plan, dict) and plan.get("selected_model"):
        return str(plan["selected_model"])

    run_id = state.get("training_run_id") or ""
    if run_id:
        return str(run_id)[:8]

    return "unknown"
