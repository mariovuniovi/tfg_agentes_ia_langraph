"""Join discovery tools — profiling, candidate evaluation, and plan execution."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

from mlops_agents.config.settings import settings
from mlops_agents.contracts.join_discovery import ColumnProfile, RawDatasetProfile
from mlops_agents.utils.logging import get_logger

logger = get_logger(__name__)


def profile_raw_datasets(raw_paths: dict[str, str]) -> list[RawDatasetProfile]:
    """Profile raw datasets deterministically. Called before the agent starts — not a tool.

    Stats are computed on the full dataset. head_rows contains the first
    settings.data_validator_profile_nrows rows for the agent to inspect.

    Args:
        raw_paths: mapping of dataset_name → file path

    Returns:
        List of RawDatasetProfile objects.
    """
    head_n = settings.data_validator_profile_nrows
    profiles: list[RawDatasetProfile] = []

    for dataset_name, path_str in raw_paths.items():
        path = Path(path_str)
        if not path.exists():
            logger.warning(f"[profile] file not found: {path_str}")
            continue

        df = pd.read_csv(path)
        n_rows = len(df)
        col_profiles: list[ColumnProfile] = []

        for col in df.columns:
            series = df[col]
            non_null = series.dropna()
            null_rate = round(float(series.isnull().sum()) / max(n_rows, 1), 4)
            unique_count = int(non_null.nunique())
            unique_ratio = round(unique_count / max(len(non_null), 1), 4)

            min_val = max_val = None
            if (pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series)) and not non_null.empty:
                min_val = str(non_null.min())
                max_val = str(non_null.max())

            col_profiles.append(ColumnProfile(
                column_name=col,
                dtype=str(series.dtype),
                non_null_count=int(len(non_null)),
                null_rate=null_rate,
                unique_count=unique_count,
                unique_ratio=unique_ratio,
                min_value=min_val,
                max_value=max_val,
            ))

        profiles.append(RawDatasetProfile(
            dataset_name=dataset_name,
            path=path_str,
            n_rows=n_rows,
            n_columns=len(df.columns),
            columns=col_profiles,
            head_rows=df.head(head_n).to_dict(orient="records"),
        ))

    logger.info(f"[profile] profiled {len(profiles)} datasets, {sum(p.n_columns for p in profiles)} total columns")
    return profiles


@tool
def evaluate_join_candidates(candidates_json: str, raw_paths_json: str) -> str:
    """Evaluate a list of proposed join candidates with deterministic coverage/cardinality metrics.

    Args:
        candidates_json: JSON array of candidate dicts. Each entry needs only:
            {"candidate_id": str, "left_dataset": str, "left_column": str,
             "right_dataset": str, "right_column": str}
        raw_paths_json: JSON object {"dataset_name": "path/to/file.csv", ...}

    Returns:
        JSON with {evaluations: [JoinCandidateEvaluation], errors: []}.
    """
    from mlops_agents.contracts.join_discovery import JoinCandidateEvaluation

    candidates: list[dict] = json.loads(candidates_json)
    raw_paths: dict[str, str] = json.loads(raw_paths_json)

    if len(candidates) > settings.data_validator_max_join_candidates:
        return json.dumps({
            "error": f"Too many candidates ({len(candidates)}). Max is {settings.data_validator_max_join_candidates}."
        })

    _df_cache: dict[str, pd.DataFrame] = {}

    def _load(name: str) -> pd.DataFrame | None:
        if name not in _df_cache:
            path = raw_paths.get(name)
            if not path or not Path(path).exists():
                return None
            _df_cache[name] = pd.read_csv(path)
        return _df_cache[name]

    def _normalize_series(s: pd.Series) -> set[str]:
        return {str(v).strip().lower() for v in s.dropna()}

    evaluations = []
    errors = []
    medium_t = settings.data_validator_row_explosion_medium_threshold
    high_t = settings.data_validator_row_explosion_high_threshold

    for cand in candidates:
        cid = cand["candidate_id"]
        left_ds = cand["left_dataset"]
        right_ds = cand["right_dataset"]
        left_col = cand["left_column"]
        right_col = cand["right_column"]

        left_df = _load(left_ds)
        right_df = _load(right_ds)

        if left_df is None:
            errors.append(f"{cid}: dataset '{left_ds}' not found")
            continue
        if right_df is None:
            errors.append(f"{cid}: dataset '{right_ds}' not found")
            continue
        if left_col not in left_df.columns:
            errors.append(f"{cid}: column '{left_col}' not in '{left_ds}'")
            continue
        if right_col not in right_df.columns:
            errors.append(f"{cid}: column '{right_col}' not in '{right_ds}'")
            continue

        left_vals = _normalize_series(left_df[left_col])
        right_vals = _normalize_series(right_df[right_col])
        intersection = left_vals & right_vals
        union = left_vals | right_vals

        left_coverage = len(intersection) / len(left_vals) if left_vals else 0.0
        right_coverage = len(intersection) / len(right_vals) if right_vals else 0.0
        jaccard = len(intersection) / len(union) if union else 0.0
        containment = max(left_coverage, right_coverage)

        left_non_null = left_df[left_col].dropna()
        right_non_null = right_df[right_col].dropna()
        left_unique_ratio = left_df[left_col].nunique() / max(len(left_non_null), 1)
        right_unique_ratio = right_df[right_col].nunique() / max(len(right_non_null), 1)

        UR_THRESHOLD = 0.98
        if left_unique_ratio >= UR_THRESHOLD and right_unique_ratio >= UR_THRESHOLD:
            rel = "one_to_one"
        elif left_unique_ratio >= UR_THRESHOLD:
            rel = "one_to_many"
        elif right_unique_ratio >= UR_THRESHOLD:
            rel = "many_to_one"
        else:
            rel = "many_to_many"

        # Row explosion estimation
        left_counts = left_df[left_col].astype(str).str.strip().str.lower().value_counts()
        right_counts = right_df[right_col].astype(str).str.strip().str.lower().value_counts()
        est_inner = sum(
            int(left_counts.get(k, 0)) * int(right_counts.get(k, 0))
            for k in intersection
        )
        est_left = sum(
            int(left_counts.get(k, 0)) * max(int(right_counts.get(k, 0)), 1)
            for k in left_vals
        )
        multiplier = est_left / max(len(left_df), 1)

        if multiplier >= high_t:
            risk = "high"
        elif multiplier >= medium_t:
            risk = "medium"
        else:
            risk = "low"

        warnings: list[str] = []
        if str(left_df[left_col].dtype) != str(right_df[right_col].dtype):
            warnings.append(f"dtype mismatch: {left_df[left_col].dtype} vs {right_df[right_col].dtype}")
        if left_coverage < settings.data_validator_min_left_coverage:
            warnings.append(f"low left_coverage: {left_coverage:.2%}")
        if containment < settings.data_validator_min_containment:
            warnings.append(f"low containment: {containment:.2%}")
        if rel == "many_to_many":
            warnings.append("many-to-many relationship detected")
        if left_df[left_col].isnull().any():
            warnings.append(f"left column '{left_col}' has nulls")
        if right_df[right_col].isnull().any():
            warnings.append(f"right column '{right_col}' has nulls")

        evaluations.append(JoinCandidateEvaluation(
            candidate_id=cid,
            left_dataset=left_ds,
            left_column=left_col,
            right_dataset=right_ds,
            right_column=right_col,
            left_distinct=len(left_vals),
            right_distinct=len(right_vals),
            intersection_count=len(intersection),
            left_coverage=round(left_coverage, 4),
            right_coverage=round(right_coverage, 4),
            jaccard=round(jaccard, 4),
            containment=round(containment, 4),
            left_unique_ratio=round(left_unique_ratio, 4),
            right_unique_ratio=round(right_unique_ratio, 4),
            inferred_relationship=rel,
            estimated_inner_rows=est_inner,
            estimated_left_rows=est_left,
            row_multiplier_left=round(multiplier, 4),
            join_explosion_risk=risk,
            warnings=warnings,
        ).model_dump())

    logger.info(f"[evaluate] evaluated {len(evaluations)} candidates, {len(errors)} errors")
    return json.dumps({"evaluations": evaluations, "errors": errors}, default=str)
