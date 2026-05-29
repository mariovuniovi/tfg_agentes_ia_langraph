"""Deterministic evaluation & promotion decision.

Reads candidate training metrics from state, fetches the current MLflow champion,
applies fixed thresholds, and returns evaluation_passed + a structured result dict.
No LLM involved — pure number comparison.
"""
from __future__ import annotations

from typing import Any

from mlflow.tracking import MlflowClient

from mlops_agents.config.settings import settings
from mlops_agents.utils.logging import get_logger

logger = get_logger(__name__)


def _metric_for_problem_type(problem_type: str) -> tuple[str, bool]:
    """Return (metric_name, ascending) for the given problem type."""
    if problem_type == "classification":
        return "macro_f1", False
    if problem_type in ("regression", "forecasting"):
        return "rmse", True
    raise ValueError(f"Unknown problem_type: {problem_type!r}")


def _thresholds_for(problem_type: str) -> dict[str, float]:
    """Return absolute thresholds applied to candidate metrics."""
    if problem_type == "classification":
        return {"accuracy_min": 0.80, "macro_f1_min": 0.75}
    if problem_type == "regression":
        return {"r2_min": 0.70}
    # forecasting has no absolute thresholds — only relative comparison vs champion
    return {}


def _apply_thresholds(
    problem_type: str,
    candidate: dict[str, Any],
    champion: dict[str, Any],
) -> bool:
    """Apply absolute thresholds AND relative-vs-champion comparison."""
    metric, ascending = _metric_for_problem_type(problem_type)

    cand_value = candidate.get(metric)
    if cand_value is None:
        return False

    # Absolute thresholds
    thresholds = _thresholds_for(problem_type)
    if problem_type == "classification":
        if candidate.get("accuracy", 0.0) < thresholds["accuracy_min"]:
            return False
        if candidate.get("macro_f1", 0.0) < thresholds["macro_f1_min"]:
            return False
    elif problem_type == "regression":
        if candidate.get("r2", -1.0) < thresholds["r2_min"]:
            return False

    # Relative comparison
    champ_value = champion.get(metric) if champion else None
    if champ_value is None:
        return True
    return cand_value <= champ_value if ascending else cand_value >= champ_value


def _get_client() -> MlflowClient:
    return MlflowClient(tracking_uri=settings.mlflow_tracking_uri)


def _fetch_current_champion(metric: str, ascending: bool) -> dict[str, Any]:
    """Return the top run's metrics dict for the given metric/direction, or {} if none."""
    client = _get_client()
    experiment = client.get_experiment_by_name(settings.mlflow_experiment_name)
    if experiment is None:
        return {}
    direction = "ASC" if ascending else "DESC"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} {direction}"],
        max_results=1,
    )
    if not runs:
        return {}
    return dict(runs[0].data.metrics)


def evaluate_promotion(state: dict[str, Any]) -> dict[str, Any]:
    """Deterministic promotion decision.

    Returns a dict of state updates: evaluation_passed, candidate_metrics,
    champion_metrics, thresholds_applied.
    """
    problem_type = state["problem_type"]
    metric, ascending = _metric_for_problem_type(problem_type)
    candidate = dict(state.get("training_metrics") or {})
    champion = _fetch_current_champion(metric, ascending)
    passed = _apply_thresholds(problem_type, candidate, champion)
    thresholds = _thresholds_for(problem_type)

    logger.info(
        f"[evaluation] problem_type={problem_type} metric={metric} "
        f"candidate={candidate.get(metric)} champion={champion.get(metric)} "
        f"passed={passed}"
    )

    return {
        "evaluation_passed": passed,
        "candidate_metrics": candidate,
        "champion_metrics": champion,
        "thresholds_applied": thresholds,
        # Preserve legacy SSE/frontend wire shape — same keys the old evaluator wrote.
        "evaluation_report": {
            "candidate_metrics": candidate,
            "candidate_run_id": state.get("training_run_id", ""),
            "baseline_metrics": champion,
        },
    }
