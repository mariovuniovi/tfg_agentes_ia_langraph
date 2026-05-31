"""Weighted-overlap retrieval for experience records."""
from __future__ import annotations
import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal
from mlops_agents.experience.schema import CandidateResultView, RetrievalView, SelectedSolutionView

if TYPE_CHECKING:
    from mlops_agents.experience.pool import ExperiencePool

RETRIEVAL_WEIGHTS: dict[str, int] = {
    "n_rows": 3, "n_series": 3, "history_length": 3, "horizon_difficulty": 3, "seasonality_detected": 3,
    "class_balance": 2, "n_classes": 2, "target_distribution": 2, "exogenous_features_available": 2,
    "frequency": 2, "trend_detected": 2, "stationarity": 2,
    "n_features": 1, "missing_rate": 1, "n_categorical_features": 1, "n_numerical_features": 1,
}

MAX_SCORE_BY_PROBLEM_TYPE: dict[str, int] = {
    "classification": 13, "regression": 11, "forecasting": 29,
}


def derive_relevance_tier(similarity_score: float) -> Literal["high", "medium", "low"]:
    """Map a similarity score to a coarse relevance tier for UI display.

    Thresholds match spec: high >= 0.7, medium 0.4-0.7, low < 0.4.
    """
    if similarity_score >= 0.7:
        return "high"
    if similarity_score >= 0.4:
        return "medium"
    return "low"


def _parse_ts(iso: str | None) -> float:
    if not iso:
        return 0.0
    try:
        return datetime.fromisoformat(iso).timestamp()
    except ValueError:
        return 0.0


def _build_view(row: Any, cand_rows: list, score: int, ratio: float, matched: list) -> RetrievalView | None:
    profile = json.loads(row["dataset_profile_json"])
    candidates = [
        CandidateResultView(model_key=r["model_key"], status=r["status"],
                            best_score=r["best_score"], complexity_rank=r["complexity_rank"],
                            error_type=r["error_type"])
        for r in cand_rows
    ]
    if not row["selected_model_key"] or row["validation_score"] is None:
        return None
    sol = SelectedSolutionView(
        model_key=row["selected_model_key"],
        validation_score=row["validation_score"],
        validation_std=row["validation_std"],
        complexity_rank=next((c.complexity_rank for c in candidates
                              if c.model_key == row["selected_model_key"]), 0) or 0,
    )
    return RetrievalView(
        task_id=row["task_id"], dataset_name=row["dataset_name"],
        dataset_profile=profile, models_tested=candidates, selected_solution=sol,
        experience_summary=row["experience_summary"],
        similarity_score=score, similarity_ratio=ratio, matched_fields=matched,
        metric_to_optimize=row["metric_to_optimize"],
    )


def compare_target_scales(
    profile_target_std: float | None,
    experience_target_std: float | None,
) -> str | None:
    """Return a human-readable scale warning when target stds differ by an order
    of magnitude or more. Returns None when both sides have similar scales or
    when either side is missing/zero (graceful for legacy ExperienceRecords)."""
    if profile_target_std is None or experience_target_std is None:
        return None
    if profile_target_std <= 0 or experience_target_std <= 0:
        return None
    ratio = max(profile_target_std, experience_target_std) / min(profile_target_std, experience_target_std)
    if ratio < 10:
        return None
    direction = "larger" if profile_target_std > experience_target_std else "smaller"
    return (
        f"candidate target std ({profile_target_std:.3g}) is ~{ratio:.0f}× {direction} "
        f"than experience target std ({experience_target_std:.3g}); raw metric values "
        f"may not be directly comparable"
    )


def find_similar_impl(pool: "ExperiencePool", profile: dict[str, Any], problem_type: str, k: int) -> list[RetrievalView]:
    max_score = MAX_SCORE_BY_PROBLEM_TYPE.get(problem_type, 10)
    with pool._conn() as conn:
        rows = conn.execute(
            "SELECT * FROM experiences WHERE problem_type = ? ORDER BY created_at DESC",
            (problem_type,),
        ).fetchall()
    scored = []
    for row in rows:
        cp = json.loads(row["dataset_profile_json"])
        score = 0
        matched = ["problem_type"]
        for field, weight in RETRIEVAL_WEIGHTS.items():
            pv, cv = profile.get(field), cp.get(field)
            if pv is not None and cv is not None and pv == cv:
                score += weight
                matched.append(field)
        scored.append((score, _parse_ts(row["created_at"]), round(score / max_score, 3), matched, row))
    scored.sort(key=lambda x: (-x[0], -x[1]))
    views = []
    for score, ts, ratio, matched, row in scored[:k]:
        with pool._conn() as conn:
            cand_rows = conn.execute(
                "SELECT * FROM candidate_results WHERE task_id = ?", (row["task_id"],)
            ).fetchall()
        v = _build_view(row, cand_rows, score, ratio, matched)
        if v is not None:
            views.append(v)
    return views
