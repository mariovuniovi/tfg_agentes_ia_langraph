"""Deterministic evaluation & promotion decision.

Reads candidate training metrics from state, fetches the current MLflow champion,
applies fixed thresholds, and returns evaluation_passed + a structured result dict.
No LLM involved — pure number comparison.
"""
from __future__ import annotations

from typing import Any

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
