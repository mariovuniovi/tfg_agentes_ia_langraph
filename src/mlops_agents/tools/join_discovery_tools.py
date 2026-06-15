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

    try:
        candidates: list[dict] = json.loads(candidates_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"candidates_json is not valid JSON: {e}"})
    try:
        raw_paths: dict[str, str] = json.loads(raw_paths_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"raw_paths_json is not valid JSON: {e}"})

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


@tool
def execute_join_plan(
    selections_json: str,
    evaluations_json: str,
    raw_paths_json: str,
    output_path: str,
    target_schema_json: str = "",
) -> str:
    """Execute join plan from agent selections + prior evaluation results.

    Args:
        selections_json: Agent's decisions. Format:
            {
              "base_dataset": {"dataset_name": str, "confidence": "high"|"medium"|"low",
                               "covered_target_columns": [...], "missing_target_columns": [...],
                               "reason": str, "warnings": [...]},
              "selected": [{"candidate_id": str, "columns_to_add": [...],
                            "confidence_after_evaluation": "high"|"medium"|"low",
                            "reason": str, "warnings": [...]}],
              "rejected": [{"candidate_id": str, "reason": str}],
              "unresolved_ambiguities": [...],
              "warnings": [...]
            }
        evaluations_json: Raw output from evaluate_join_candidates — pass through unchanged.
            Format: {"evaluations": [...], "errors": [...]}
        raw_paths_json: JSON object {"dataset_name": "path/to/file.csv", ...}
        output_path: Destination path for the merged CSV.
        target_schema_json: Optional JSON schema — verifies required columns present after join.

    Returns:
        JSON with {success, output_path, row_count, columns, columns_added_by_join, warnings, join_plan} or {error}.
    """
    from mlops_agents.contracts.join_discovery import (
        JoinPlan, BaseDatasetSelection, SelectedJoin, RejectedJoinCandidate, JoinCandidateEvaluation,
    )

    try:
        selections: dict = json.loads(selections_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"selections_json is not valid JSON: {e}"})
    try:
        eval_data: dict = json.loads(evaluations_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"evaluations_json is not valid JSON: {e}. Pass the exact string returned by evaluate_join_candidates."})
    try:
        raw_paths: dict[str, str] = json.loads(raw_paths_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"raw_paths_json is not valid JSON: {e}"})

    # Build evaluation lookup by candidate_id
    eval_by_id: dict[str, dict] = {e["candidate_id"]: e for e in eval_data.get("evaluations", [])}

    # Validate all selected candidates have evaluations
    for sel in selections.get("selected", []):
        cid = sel["candidate_id"]
        if cid not in eval_by_id:
            return json.dumps({"error": f"Selected candidate '{cid}' was not evaluated — call evaluate_join_candidates first"})
        if eval_by_id[cid].get("left_coverage", 0.0) == 0.0:
            ev = eval_by_id[cid]
            return json.dumps({"error": f"Join blocked for {cid}: zero overlap between "
                                        f"'{ev['left_dataset']}.{ev['left_column']}' and "
                                        f"'{ev['right_dataset']}.{ev['right_column']}' — wrong join key pair"})

    # Build BaseDatasetSelection
    try:
        base_sel = BaseDatasetSelection(**selections["base_dataset"])
    except Exception as e:
        return json.dumps({"error": f"Invalid base_dataset: {e}"})

    # Build SelectedJoin list
    selected_joins: list[SelectedJoin] = []
    for i, sel in enumerate(selections.get("selected", []), start=1):
        cid = sel["candidate_id"]
        try:
            ev = JoinCandidateEvaluation(**eval_by_id[cid])
        except Exception as e:
            return json.dumps({"error": f"Invalid evaluation for {cid}: {e}"})
        selected_joins.append(SelectedJoin(
            step_id=i,
            candidate_id=cid,
            left_dataset=ev.left_dataset,
            left_column=ev.left_column,
            right_dataset=ev.right_dataset,
            right_column=ev.right_column,
            join_type="left",
            columns_added=sel.get("columns_to_add", []),
            evaluation=ev,
            confidence_after_evaluation=sel.get("confidence_after_evaluation", "medium"),
            reason=sel.get("reason", "selected"),
            warnings=sel.get("warnings", []),
        ))

    # Build RejectedJoinCandidate list
    rejected: list[RejectedJoinCandidate] = []
    for rej in selections.get("rejected", []):
        cid = rej["candidate_id"]
        ev_dict = eval_by_id.get(cid)
        ev = JoinCandidateEvaluation(**ev_dict) if ev_dict else None
        rejected.append(RejectedJoinCandidate(
            candidate_id=cid,
            left_dataset=ev.left_dataset if ev else "",
            left_column=ev.left_column if ev else "",
            right_dataset=ev.right_dataset if ev else "",
            right_column=ev.right_column if ev else "",
            reason=rej.get("reason", "rejected"),
            evaluation=ev,
        ))

    plan = JoinPlan(
        mode="inferred",
        base_dataset=base_sel,
        selected_joins=selected_joins,
        rejected_candidates=rejected,
        unresolved_ambiguities=selections.get("unresolved_ambiguities", []),
        warnings=selections.get("warnings", []),
    )

    # Execute merges
    base_name = plan.base_dataset.dataset_name
    base_path = raw_paths.get(base_name)
    if not base_path or not Path(base_path).exists():
        return json.dumps({"error": f"Base dataset '{base_name}' not found"})

    current_df = pd.read_csv(base_path)
    columns_added_by_join: dict[str, list[str]] = {}
    join_warnings: list[str] = []

    for step in plan.selected_joins:
        right_path = raw_paths.get(step.right_dataset)
        if not right_path or not Path(right_path).exists():
            return json.dumps({"error": f"Right dataset '{step.right_dataset}' not found at step {step.step_id}"})

        right_df = pd.read_csv(right_path)
        keep_cols = [step.right_column] + [c for c in step.columns_added if c in right_df.columns and c != step.right_column]
        right_before = len(right_df)
        right_df = right_df[keep_cols].drop_duplicates(subset=[step.right_column])
        if len(right_df) < right_before:
            join_warnings.append(
                f"Step {step.step_id}: {right_before - len(right_df)} duplicate key(s) dropped from "
                f"'{step.right_dataset}.{step.right_column}' — first occurrence retained"
            )

        # Avoid column collisions
        existing = set(current_df.columns)
        rename_map = {
            c: f"{c}_{step.right_dataset}"
            for c in keep_cols
            if c in existing and c != step.right_column
        }
        if rename_map:
            right_df = right_df.rename(columns=rename_map)
            join_warnings += [f"Column '{c}' renamed to '{rename_map[c]}' to avoid collision" for c in rename_map]

        current_df = current_df.merge(right_df, left_on=step.left_column, right_on=step.right_column, how="left")

        # Drop the right join key if it differs from the left — pandas keeps both after merge
        if step.right_column != step.left_column and step.right_column in current_df.columns:
            current_df = current_df.drop(columns=[step.right_column])

        columns_added_by_join[step.candidate_id] = [rename_map.get(c, c) for c in step.columns_added]

    # Verify required target columns present
    if target_schema_json:
        schema = json.loads(target_schema_json)
        required_cols = {c["name"] for c in schema.get("columns", []) if c.get("required", False)}
        missing = required_cols - set(current_df.columns)
        if missing:
            return json.dumps({"error": f"Required target columns missing after all joins: {sorted(missing)}"})

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    current_df.to_csv(output_path, index=False)

    result = {
        "success": True,
        "output_path": output_path,
        "row_count": len(current_df),
        "columns": current_df.columns.tolist(),
        "columns_added_by_join": columns_added_by_join,
        "warnings": join_warnings + plan.warnings,
        "join_plan": plan.model_dump(),  # echoed so data_validator_node can write it to state
    }
    logger.info(f"[execute_join_plan] {len(plan.selected_joins)} join(s) → {len(current_df)} rows, {len(current_df.columns)} cols → {output_path}")
    return json.dumps(result, default=str)
