# Agentic Join Discovery V1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an agentic join-discovery workflow to the data validator so it can infer how to merge multiple raw CSV datasets without requiring explicit column mappings in the schema.

**Architecture:** Deterministic pre-profile → agent proposes → deterministic tool measures → agent decides → deterministic executor runs → human reviews. Raw dataset profiles are computed deterministically in `_build_data_validator_context` and injected into the agent's initial payload; `evaluate_join_candidates` computes bidirectional coverage/cardinality metrics on agent-proposed pairs only; `execute_join_plan` runs the final merge deterministically. The human sees the full `JoinPlan` audit at the Dataset Approval gate.

**Tech Stack:** Python 3.12, pandas, pydantic v2, LangGraph `@tool`, TypeScript/React (Next.js), FastAPI SSE

---

## Non-goals for V1

Do not implement:
- exhaustive pairwise join search across all columns
- composite key discovery
- fuzzy string joins / entity resolution / embedding-based similarity
- date tolerance joins or graph optimization across arbitrary datasets
- automatic many-to-many joins
- inner joins (V1 is left-join only)
- SQL foreign-key introspection

V1 supports: single-column exact joins, CSV/raw dataframe joins, agent-proposed candidates, deterministic evaluation, left joins only (right-key deduplication enforced), zero-coverage block, human review.

---

## High-level workflow

```
data_validator_node (deterministic, before agent starts)
│
└── _build_data_validator_context()
      ├── reads raw file paths from state
      ├── calls _profile_raw_datasets() — pure Python, no @tool
      └── injects compact profiles + schema + paths into HumanMessage

data_validator agent (ReAct loop)
│
├── agent selects base_dataset
│     reasons over profiles in initial context
│
├── agent proposes ProposedJoinCandidate[]
│     candidates chosen to add missing target columns to base/current table
│
├── evaluate_join_candidates(candidates)   ← tool call
│     deterministic coverage/cardinality/row explosion metrics
│
├── agent selects JoinPlan
│     selected joins + rejected candidates + ambiguities + warnings
│
├── execute_join_plan(join_plan)           ← tool call
│     deterministic merge
│
├── validate_against_schema()             ← tool call
│
└── dataset_approval_node
      shows final dataset preview + JoinPlan audit
```

---

## Phase 0 — Pre-implementation inspection

Before coding, verify these files (corrected paths from Phase 0 inspection):

```
src/mlops_agents/agents/data_agent.py            ← data validator agent builder
src/mlops_agents/tools/data_tools.py             ← existing tools incl. merge_datasets
src/mlops_agents/state/agent_state.py            ← shared state TypedDict
src/mlops_agents/graphs/mlops_graph.py           ← _build_data_validator_context, data_validator_node
src/mlops_agents/graphs/approval_nodes.py        ← dataset_approval_node (HITL interrupt)
api/services/pipeline_helpers.py                 ← build_initial_state
api/services/pipeline.py                         ← SSE event stream
frontend/components/pipeline/DatasetApprovalCard.tsx   ← dataset approval UI
frontend/types/api.ts                            ← DataValidationInterrupt interface
```

Key findings already confirmed:
- `validate_against_schema` already reads `col_def.get("unique", False)` at line 115 — no tool change needed for the `is_key` → `unique` rename
- `DataValidationInterrupt` interface exists at `frontend/types/api.ts:36`
- HITL interrupt is in `graphs/approval_nodes.py:dataset_approval_node`, not a service file
- `contracts/` directory already has `profile.py`, `evidence.py`, `planner.py`, `training.py`

Do NOT remove existing `merge_datasets`. This feature must support both explicit and inferred modes.

---

## Phase 1 — Contracts

- [ ] Create `src/mlops_agents/contracts/join_discovery.py`

```python
from typing import Literal, Any
from pydantic import BaseModel, Field


class ColumnProfile(BaseModel):
    column_name: str
    dtype: str
    non_null_count: int
    null_rate: float
    unique_count: int
    unique_ratio: float
    min_value: str | None = None
    max_value: str | None = None


class RawDatasetProfile(BaseModel):
    dataset_name: str
    path: str
    n_rows: int
    n_columns: int
    columns: list[ColumnProfile]
    head_rows: list[dict] = Field(default_factory=list)  # head(profile_nrows) for agent inspection


class BaseDatasetSelection(BaseModel):
    dataset_name: str
    confidence: Literal["high", "medium", "low"]
    covered_target_columns: list[str] = Field(default_factory=list)
    missing_target_columns: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)


class JoinCandidateEvaluation(BaseModel):
    candidate_id: str
    left_dataset: str
    left_column: str
    right_dataset: str
    right_column: str
    left_distinct: int
    right_distinct: int
    intersection_count: int
    left_coverage: float
    right_coverage: float
    jaccard: float
    containment: float
    left_unique_ratio: float
    right_unique_ratio: float
    inferred_relationship: Literal[
        "one_to_one", "one_to_many", "many_to_one", "many_to_many", "unknown"
    ]
    estimated_inner_rows: int
    estimated_left_rows: int
    row_multiplier_left: float
    join_explosion_risk: Literal["low", "medium", "high"]
    warnings: list[str] = Field(default_factory=list)


class SelectedJoin(BaseModel):
    step_id: int
    candidate_id: str
    left_dataset: str
    left_column: str
    right_dataset: str
    right_column: str
    join_type: Literal["left"] = "left"
    columns_added: list[str] = Field(default_factory=list)
    evaluation: JoinCandidateEvaluation
    confidence_after_evaluation: Literal["high", "medium", "low"]
    reason: str = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)


class RejectedJoinCandidate(BaseModel):
    candidate_id: str
    left_dataset: str
    left_column: str
    right_dataset: str
    right_column: str
    reason: str = Field(min_length=1)
    evaluation: JoinCandidateEvaluation | None = None


class JoinPlan(BaseModel):
    mode: Literal["explicit", "inferred", "hybrid"] = "inferred"
    base_dataset: BaseDatasetSelection
    selected_joins: list[SelectedJoin] = Field(default_factory=list)
    rejected_candidates: list[RejectedJoinCandidate] = Field(default_factory=list)
    unresolved_ambiguities: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
```

Notes:
- `candidate_id` format: `join_001`, `join_002`, etc. — agent-generated, used as stable lookup key
- V1: single-column exact joins only, left join enforced
- `evaluate_join_candidates` accepts minimal 5-field dicts per candidate — no Pydantic type required from the agent

- [ ] Run `uv run pytest tests/test_join_discovery/test_join_contracts.py -v` (write this test first in Phase 11)

---

## Phase 2 — Settings

- [ ] Modify `src/mlops_agents/config/settings.py` — add to `Settings` class:

```python
data_validator_profile_nrows: int = 10
data_validator_max_join_candidates: int = 20
data_validator_row_explosion_medium_threshold: float = 1.25
data_validator_row_explosion_high_threshold: float = 2.0
data_validator_min_left_coverage: float = 0.8
data_validator_min_containment: float = 0.8
```

Field meanings:
- `profile_nrows`: number of head rows per raw file included in the profile payload for the agent to inspect — stats are always computed on the full dataset
- `max_join_candidates`: agent cannot ask the evaluator to check hundreds of joins
- `row_explosion_*`: thresholds for warn/block classification
- `min_left_coverage` / `min_containment`: used by validation guardrails, not for candidate generation

---

## Phase 3 — Deterministic raw dataset profiler

`profile_raw_datasets` is a **pure Python function** — no `@tool` decorator, no LangChain dependency. It is called deterministically inside `_build_data_validator_context` before the agent starts.

- [ ] Create `src/mlops_agents/tools/join_discovery_tools.py` with the profiler and the two tools:

```python
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
```

---

## Phase 4 — Evaluate join candidates tool

- [ ] Add `evaluate_join_candidates` to `src/mlops_agents/tools/join_discovery_tools.py`

```python
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
        if risk == "high":
            warnings.append(f"high row explosion risk: multiplier={multiplier:.2f}")
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
```

---

## Phase 5 — Execute join plan tool

- [ ] Add `execute_join_plan` to `src/mlops_agents/tools/join_discovery_tools.py`

The agent does NOT construct a full `JoinPlan` JSON. It passes a lightweight `selections_json` (its decisions) plus the raw `evaluations_json` it received from `evaluate_join_candidates` (passed through unchanged). The tool reconstructs the full `JoinPlan` internally and echoes it in the result.

```python
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

    selections: dict = json.loads(selections_json)
    eval_data: dict = json.loads(evaluations_json)
    raw_paths: dict[str, str] = json.loads(raw_paths_json)

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
            reason=sel.get("reason", ""),
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
            reason=rej.get("reason", ""),
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
```

---

## Phase 6 — Sample schema update + context enrichment

This phase handles two changes that were agreed separately:

### 6a — Rename `is_key` → `unique` in sample schemas

The tool `validate_against_schema` already reads `col_def.get("unique", False)` — the rename is in data files only.

- [ ] Edit `data/samples/forecasting/energy_forecast_schema.json`:
  - Replace `"is_key": true` with `"unique": true`
  - Remove all `"mapping_hint"` fields
  - Update `week_date` description: remove "(join key between the two raw files)" — keep it generic

Expected result for `week_date`:
```json
{
  "name": "week_date",
  "dtype": "datetime",
  "description": "Monday of the measurement week",
  "required": true,
  "nullable": false,
  "unique": true
}
```

- [ ] Check for any other sample schema files using `is_key` or `mapping_hint` and apply the same rename:
  - `grep -r "is_key\|mapping_hint" data/`

### 6b — Enrich `_build_data_validator_context` with deterministic pre-profiles

- [ ] Modify `src/mlops_agents/graphs/mlops_graph.py:_build_data_validator_context`:

Call `profile_raw_datasets` (the pure Python function, not a tool) and serialize the compact profiles into the context message. The agent receives everything it needs before its first reasoning step.

```python
def _build_data_validator_context(
    state: AgentState,
    *,
    schema_json: str = "{}",
    schema_path: str = "",
) -> HumanMessage:
    from mlops_agents.tools.join_discovery_tools import profile_raw_datasets as _profile

    paths: list[str] = state.get("dataset_paths") or []

    # Build name → path mapping; use filename stem as dataset name
    raw_paths = {Path(p).stem: p for p in paths}

    profiles_section = ""
    if len(paths) > 1:
        try:
            profiles = _profile(raw_paths)
            profiles_section = "\nRaw dataset profiles:\n" + json.dumps(
                [p.model_dump() for p in profiles], default=str, indent=2
            )
        except Exception as exc:
            profiles_section = f"\n(Could not pre-profile raw files: {exc})"

    single_file_note = (
        "\nNOTE: Only ONE file was uploaded. "
        "Do NOT call merge_datasets or execute_join_plan. "
        "After load_dataset, go directly to apply_column_mapping on this single file."
        if len(paths) == 1 else ""
    )
    return HumanMessage(content=(
        f"Raw files: {json.dumps(paths)}\n"
        f"Schema path: {schema_path}\n"
        f"Target schema:\n{schema_json}"
        f"{profiles_section}"
        f"{single_file_note}"
    ))
```

Note: The dataset name key used in `raw_paths` (filename stem) is the same name the agent must use when referencing datasets in `ProposedJoinCandidate.left_dataset` / `right_dataset`. Make this explicit in the prompt.

### 6c — Data validator prompt update

- [ ] Identify the data agent prompt YAML file (run `ls src/mlops_agents/prompts/`)
- [ ] Add this section to the data agent prompt:

```
## Join discovery workflow

When multiple raw datasets are provided and explicit join keys are not fully specified,
compact profiles for every raw dataset are already included in your initial context
under "Raw dataset profiles". Use them — do not call any profiling tool.

1. Select a base_dataset using the profiles. Its row count defines the final table row count — all other datasets are joined to it (left join, NAs for unmatched rows).
   - Prefer the dataset covering the most required target columns — this identifies the semantic "fact table".
   - Prefer the dataset that contains the target variable.
   - Prefer the dataset that contains the time/entity index of the final dataset.
   - Use row count as a tiebreaker when coverage is equal — prefer the larger dataset.
   Dataset names must exactly match the `dataset_name` field from the Raw dataset profiles in your context. Do not invent aliases, abbreviations, or guesses. If the profile says `"dataset_name": "energy_readings"`, every reference to that dataset in tool calls must use exactly `"energy_readings"`.
2. Identify missing target columns not covered by the base dataset.
3. Propose at most {data_validator_max_join_candidates} ProposedJoinCandidate objects to add those columns.
   Use the profiles (column names, dtypes, null_rate, unique_ratio, min/max values, head rows) to reason
   about plausible join keys — do not guess blindly.
4. Call evaluate_join_candidates() with only those proposed candidates.
5. Select a JoinPlan using the evaluation metrics.
6. Reject or request human clarification if:
   - No candidate has acceptable coverage/containment
   - Multiple candidates lead to different row counts
   - Selected join has many_to_many relationship
   - Row explosion risk is high
7. Call execute_join_plan() with:
   - selections_json: your decisions (base_dataset, selected candidate IDs with columns_to_add/reason, rejected candidate IDs with reason)
   - evaluations_json: the raw output you received from evaluate_join_candidates — pass it through unchanged
   Do NOT reconstruct evaluation metrics — the tool looks them up internally.
8. Include the JoinPlan in the dataset summary for human approval.

Do not exhaustively test every possible column pair.
You are responsible for reducing the search space by proposing semantically plausible candidates.
You must not select a final join unless evaluate_join_candidates confirms the join quality.

Always use left join. The base dataset defines the target rows and its row count must be preserved. Unmatched enrichment rows are filled with NAs. Inner joins are not supported in V1.
```

**Examples to add:**

Example 1:
```
Raw datasets:
energy_readings(week_date, kwh_consumed)
weather(week_date, avg_temp_c)

Target: week_date, kwh_consumed, avg_temp_c

Correct behavior:
- select energy_readings as base (contains target and time index)
- propose energy_readings.week_date LEFT JOIN weather.week_date
- evaluate candidate
- select join if coverage is high and row_multiplier is low
```

Example 2:
```
Raw datasets:
orders(order_id, customer_id, amount)
customers(customer_id, customer_name)

Target: order_id, customer_id, amount, customer_name

Correct behavior:
- select orders as base (defines transaction rows)
- propose orders.customer_id LEFT JOIN customers.customer_id
- expect many_to_one
- reject if customers.customer_id is not unique
```

---

## Phase 7 — Agent state changes

- [ ] Modify `src/mlops_agents/state/agent_state.py` — add to `AgentState`:

```python
# Join discovery outputs — written by data_validator_node after agent run
data_join_plan: dict | None
data_join_base_nrows: int | None  # row count of the base dataset before any joins
data_join_evaluations: list[dict]
```

- [ ] Modify `api/services/pipeline_helpers.py:build_initial_state` — add initial values:

```python
"data_join_plan": None,
"data_join_base_nrows": None,
"data_join_evaluations": [],
```

- [ ] Modify `src/mlops_agents/graphs/mlops_graph.py:data_validator_node` — after `agent.invoke(...)`, extract join outputs from tool messages and add them to `base_update`. Use the existing `_extract_tool_json` helper:

```python
# After the existing _extract_tool_json calls:
join_exec_result: dict = _extract_tool_json(result["messages"], "execute_join_plan")
eval_result: dict = _extract_tool_json(result["messages"], "evaluate_join_candidates")

data_join_plan = join_exec_result.get("join_plan")          # echoed by execute_join_plan
data_join_evaluations = eval_result.get("evaluations", [])

# Capture base dataset row count for audit (read from disk using the same stem→path mapping)
data_join_base_nrows: int | None = None
if data_join_plan:
    base_name = data_join_plan.get("base_dataset", {}).get("dataset_name")
    if base_name:
        raw_paths = {Path(p).stem: p for p in (state.get("dataset_paths") or [])}
        base_path = raw_paths.get(base_name)
        if base_path and Path(base_path).exists():
            try:
                data_join_base_nrows = len(pd.read_csv(base_path))
            except Exception:
                pass
```

Then include in `base_update`:
```python
"data_join_plan": data_join_plan,
"data_join_base_nrows": data_join_base_nrows,
"data_join_evaluations": data_join_evaluations,
```

- [ ] Modify `src/mlops_agents/agents/data_agent.py:build_data_agent` — register the two new tools (`evaluate_join_candidates` and `execute_join_plan`). `profile_raw_datasets` is a pure Python function, not a tool — do not register it here:

```python
from mlops_agents.tools.join_discovery_tools import (
    evaluate_join_candidates,
    execute_join_plan,
)

def build_data_agent():
    return create_agent(
        model=get_llm("data_validator"),
        tools=[
            load_dataset,
            merge_datasets,
            evaluate_join_candidates,
            execute_join_plan,
            apply_column_mapping,
            validate_against_schema,
            check_missing_values,
            check_data_quality,
            impute_missing_values,
            parse_datetime_column,
            detect_temporal_gaps,
        ],
        name="data_validator",
        system_prompt=get_prompt("data_agent").template,
    )
```

---

## Phase 8 — HITL / dataset approval payload

- [ ] Modify `src/mlops_agents/graphs/approval_nodes.py:dataset_approval_node`

Add `join_plan` and `join_evaluations` to the interrupt payload:

```python
approval = interrupt({
    "type": "data_validation",
    "question": "Review the processed dataset before training begins.",
    "attempt": attempt,
    "dataset_preview": preview,
    "validation_report": state.get("validation_report", {}),
    "join_plan": state.get("data_join_plan"),
    "join_evaluations": state.get("data_join_evaluations"),
    "join_base_nrows": state.get("data_join_base_nrows"),
})
```

The human approval UI shows the full JoinPlan when `join_plan` is present.

If no join plan was needed: `join_plan` will be `None` — the UI shows "No joins required — target dataset built from a single source."

---

## Phase 9 — Frontend types

- [ ] Modify `frontend/types/api.ts` — add the join discovery interfaces:

```ts
export interface ColumnProfile {
  column_name: string
  dtype: string
  non_null_count: number
  null_rate: number
  unique_count: number
  unique_ratio: number
  min_value?: string | null
  max_value?: string | null
}

export interface RawDatasetProfile {
  dataset_name: string
  path: string
  n_rows: number
  n_columns: number
  columns: ColumnProfile[]
  head_rows: Record<string, unknown>[]
}

export interface BaseDatasetSelection {
  dataset_name: string
  confidence: 'high' | 'medium' | 'low'
  covered_target_columns: string[]
  missing_target_columns: string[]
  reason: string
  warnings: string[]
}

export interface JoinCandidateEvaluation {
  candidate_id: string
  left_dataset: string
  left_column: string
  right_dataset: string
  right_column: string
  left_distinct: number
  right_distinct: number
  intersection_count: number
  left_coverage: number
  right_coverage: number
  jaccard: number
  containment: number
  left_unique_ratio: number
  right_unique_ratio: number
  inferred_relationship: string
  estimated_inner_rows: number
  estimated_left_rows: number
  row_multiplier_left: number
  join_explosion_risk: 'low' | 'medium' | 'high'
  warnings: string[]
}

export interface SelectedJoin {
  step_id: number
  candidate_id: string
  left_dataset: string
  left_column: string
  right_dataset: string
  right_column: string
  join_type: 'left'
  columns_added: string[]
  evaluation: JoinCandidateEvaluation
  confidence_after_evaluation: 'high' | 'medium' | 'low'
  reason: string
  warnings: string[]
}

export interface RejectedJoinCandidate {
  candidate_id: string
  left_dataset: string
  left_column: string
  right_dataset: string
  right_column: string
  reason: string
  evaluation?: JoinCandidateEvaluation | null
}

export interface JoinPlan {
  mode: 'explicit' | 'inferred' | 'hybrid'
  base_dataset: BaseDatasetSelection
  selected_joins: SelectedJoin[]
  rejected_candidates: RejectedJoinCandidate[]
  unresolved_ambiguities: string[]
  warnings: string[]
}
```

- [ ] Extend `DataValidationInterrupt` interface in `frontend/types/api.ts`:

```ts
export interface DataValidationInterrupt {
  // ... existing fields ...
  join_plan?: JoinPlan | null
  join_evaluations?: JoinCandidateEvaluation[]
  join_base_nrows?: number | null
}
```

---

## Phase 10 — Frontend UI component

- [ ] Create `frontend/components/pipeline/JoinPlanPanel.tsx`

```tsx
'use client'

interface JoinPlanPanelProps {
  joinPlan?: JoinPlan | null
  joinBaseNrows?: number | null
}

function formatPct(x: number): string {
  return `${(x * 100).toFixed(1)}%`
}

function formatMultiplier(x: number): string {
  return `${x.toFixed(2)}×`
}

function RiskBadge({ risk }: { risk: 'low' | 'medium' | 'high' }) {
  const colors = {
    low: 'bg-emerald-50 text-emerald-700 border-emerald-200',
    medium: 'bg-amber-50 text-amber-700 border-amber-200',
    high: 'bg-red-50 text-red-700 border-red-200',
  }
  return (
    <span className={`rounded border px-1.5 py-0.5 text-xs font-medium ${colors[risk]}`}>
      {risk} risk
    </span>
  )
}

export function JoinPlanPanel({ joinPlan, joinBaseNrows }: JoinPlanPanelProps) {
  if (!joinPlan) {
    return (
      <div className="mt-4 rounded border border-zinc-100 bg-zinc-50 px-3 py-2 text-xs text-zinc-500">
        No inferred join plan — target dataset built from a single source.
      </div>
    )
  }

  return (
    <div className="mt-4 space-y-3">
      <div className="text-xs font-semibold text-zinc-500 uppercase tracking-wide">
        Join Plan ({joinPlan.mode})
      </div>

      {/* Base dataset */}
      <div className="rounded border border-zinc-200 p-3 text-xs">
        <div className="font-medium text-zinc-700">
          Base dataset: <span className="font-mono">{joinPlan.base_dataset.dataset_name}</span>
          {joinBaseNrows != null && (
            <span className="ml-2 text-zinc-400">({joinBaseNrows.toLocaleString()} rows preserved)</span>
          )}
          <span className={`ml-2 rounded px-1.5 py-0.5 text-[10px] font-medium ${
            joinPlan.base_dataset.confidence === 'high' ? 'bg-emerald-50 text-emerald-700' :
            joinPlan.base_dataset.confidence === 'medium' ? 'bg-amber-50 text-amber-700' :
            'bg-red-50 text-red-700'
          }`}>{joinPlan.base_dataset.confidence} confidence</span>
        </div>
        <div className="mt-1 text-zinc-500">{joinPlan.base_dataset.reason}</div>
        {joinPlan.base_dataset.covered_target_columns.length > 0 && (
          <div className="mt-1 text-zinc-400">
            Covers: {joinPlan.base_dataset.covered_target_columns.join(', ')}
          </div>
        )}
        {joinPlan.base_dataset.missing_target_columns.length > 0 && (
          <div className="mt-1 text-amber-600">
            Missing before joins: {joinPlan.base_dataset.missing_target_columns.join(', ')}
          </div>
        )}
      </div>

      {/* Selected joins */}
      {joinPlan.selected_joins.length > 0 && (
        <div className="space-y-2">
          <div className="text-xs font-medium text-zinc-500">Selected joins</div>
          {joinPlan.selected_joins.map((join) => (
            <div key={join.step_id} className="rounded border border-zinc-200 p-3 text-xs">
              <div className="font-mono text-zinc-700">
                {join.left_dataset}.{join.left_column}
                <span className="mx-1 text-zinc-400">{join.join_type.toUpperCase()} JOIN</span>
                {join.right_dataset}.{join.right_column}
              </div>
              {join.columns_added.length > 0 && (
                <div className="mt-1 text-zinc-500">Adds: {join.columns_added.join(', ')}</div>
              )}
              <div className="mt-1 flex flex-wrap gap-3 text-zinc-500">
                <span>left coverage: {formatPct(join.evaluation.left_coverage)}</span>
                <span>right coverage: {formatPct(join.evaluation.right_coverage)}</span>
                <span>containment: {formatPct(join.evaluation.containment)}</span>
                <span>row multiplier: {formatMultiplier(join.evaluation.row_multiplier_left)}</span>
                <span>relationship: {join.evaluation.inferred_relationship}</span>
                <RiskBadge risk={join.evaluation.join_explosion_risk} />
              </div>
              <div className="mt-1 text-zinc-400">{join.reason}</div>
              {join.warnings.length > 0 && (
                <div className="mt-1 text-amber-600">⚠ {join.warnings.join(' · ')}</div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Rejected candidates */}
      {joinPlan.rejected_candidates.length > 0 && (
        <details className="text-xs">
          <summary className="cursor-pointer text-zinc-400 hover:text-zinc-600">
            {joinPlan.rejected_candidates.length} rejected candidate(s)
          </summary>
          <div className="mt-1 space-y-1 pl-2">
            {joinPlan.rejected_candidates.map((r) => (
              <div key={r.candidate_id} className="text-zinc-500">
                <span className="font-mono">{r.left_dataset}.{r.left_column} → {r.right_dataset}.{r.right_column}</span>
                {' — '}{r.reason}
              </div>
            ))}
          </div>
        </details>
      )}

      {/* Unresolved ambiguities */}
      {joinPlan.unresolved_ambiguities.length > 0 && (
        <div className="rounded border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-700">
          ⚠ Unresolved ambiguities: {joinPlan.unresolved_ambiguities.join(' · ')}
        </div>
      )}

      {/* Global warnings */}
      {joinPlan.warnings.length > 0 && (
        <div className="rounded border border-amber-100 bg-amber-50 px-3 py-2 text-xs text-amber-600">
          {joinPlan.warnings.map((w, i) => <div key={i}>⚠ {w}</div>)}
        </div>
      )}
    </div>
  )
}
```

- [ ] Import and use `JoinPlanPanel` in `frontend/components/pipeline/DatasetApprovalCard.tsx`
  - Add import: `import { JoinPlanPanel } from './JoinPlanPanel'`
  - Add `{ key: 'join_plan' as const, label: 'Join plan' }` to `tabs` (only if `interrupt.join_plan` is present)
  - Render `<JoinPlanPanel joinPlan={interrupt.join_plan} joinBaseNrows={interrupt.join_base_nrows} />` inside the matching tab content block

---

## Phase 11 — Tests

### Backend tests

- [ ] Create `tests/test_join_discovery/` directory with `__init__.py`

- [ ] Create `tests/test_join_discovery/test_join_contracts.py`:

```python
import pytest
from pydantic import ValidationError
from mlops_agents.contracts.join_discovery import JoinPlan, BaseDatasetSelection


def test_join_plan_valid():
    base = BaseDatasetSelection(
        dataset_name="energy",
        confidence="high",
        reason="contains target column",
    )
    plan = JoinPlan(base_dataset=base)
    assert plan.mode == "inferred"
    assert plan.selected_joins == []
```

- [ ] Create `tests/test_join_discovery/test_profile_raw_datasets.py`:

```python
import json
import tempfile
from pathlib import Path
import pandas as pd
import pytest
from mlops_agents.tools.join_discovery_tools import profile_raw_datasets


@pytest.fixture
def sample_csvs(tmp_path):
    energy = tmp_path / "energy.csv"
    weather = tmp_path / "weather.csv"
    energy.write_text("week_date,kwh_consumed\n2024-01-01,100\n2024-01-08,120\n")
    weather.write_text("week_date,avg_temp_c\n2024-01-01,10\n2024-01-08,12\n")
    return {"energy": str(energy), "weather": str(weather)}


def test_profile_returns_both_datasets(sample_csvs):
    profiles = profile_raw_datasets(sample_csvs)
    assert len(profiles) == 2
    names = {p.dataset_name for p in profiles}
    assert names == {"energy", "weather"}


def test_profile_column_stats(sample_csvs):
    profiles = profile_raw_datasets(sample_csvs)
    energy = next(p for p in profiles if p.dataset_name == "energy")
    assert energy.n_rows == 2
    assert energy.n_columns == 2
    week_col = next(c for c in energy.columns if c.column_name == "week_date")
    assert week_col.null_rate == 0.0
    assert week_col.unique_count == 2
    assert week_col.unique_ratio == 1.0


def test_profile_head_rows(sample_csvs):
    profiles = profile_raw_datasets(sample_csvs)
    energy = next(p for p in profiles if p.dataset_name == "energy")
    assert len(energy.head_rows) == 2  # only 2 rows in fixture
    assert "week_date" in energy.head_rows[0]
    assert "kwh_consumed" in energy.head_rows[0]
```

- [ ] Create `tests/test_join_discovery/test_evaluate_join_candidates.py`:

```python
import json
import tempfile
import pytest
from mlops_agents.tools.join_discovery_tools import evaluate_join_candidates


@pytest.fixture
def perfect_overlap_csvs(tmp_path):
    e = tmp_path / "energy.csv"
    w = tmp_path / "weather.csv"
    e.write_text("week_date,kwh_consumed\n2024-01-01,100\n2024-01-08,120\n")
    w.write_text("week_date,avg_temp_c\n2024-01-01,10\n2024-01-08,12\n")
    return {
        "paths": {"energy": str(e), "weather": str(w)},
        "candidate": {
            "candidate_id": "join_001",
            "left_dataset": "energy",
            "left_column": "week_date",
            "right_dataset": "weather",
            "right_column": "week_date",
        }
    }


def test_perfect_overlap_metrics(perfect_overlap_csvs):
    result = json.loads(evaluate_join_candidates.invoke({
        "candidates_json": json.dumps([perfect_overlap_csvs["candidate"]]),
        "raw_paths_json": json.dumps(perfect_overlap_csvs["paths"]),
    }))
    assert not result["errors"]
    ev = result["evaluations"][0]
    assert ev["left_coverage"] == 1.0
    assert ev["right_coverage"] == 1.0
    assert ev["jaccard"] == 1.0
    assert ev["row_multiplier_left"] == pytest.approx(1.0)
    assert ev["join_explosion_risk"] == "low"


def test_subset_coverage(tmp_path):
    base = tmp_path / "base.csv"
    enrichment = tmp_path / "enrichment.csv"
    base.write_text("id\n1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n")
    enrichment.write_text("id\n1\n2\n3\n4\n5\n6\n7\n8\n")
    result = json.loads(evaluate_join_candidates.invoke({
        "candidates_json": json.dumps([{
            "candidate_id": "join_001",
            "left_dataset": "base",
            "left_column": "id",
            "right_dataset": "enrichment",
            "right_column": "id",
        }]),
        "raw_paths_json": json.dumps({"base": str(base), "enrichment": str(enrichment)}),
    }))
    ev = result["evaluations"][0]
    assert ev["left_coverage"] == pytest.approx(0.8)
    assert ev["right_coverage"] == 1.0
    assert ev["jaccard"] == pytest.approx(0.8)
    assert ev["containment"] == 1.0  # max(0.8, 1.0)


def test_many_to_many_risk(tmp_path):
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    left.write_text("key\nA\nA\nA\nB\n")
    right.write_text("key\nA\nA\nA\nA\nB\n")
    result = json.loads(evaluate_join_candidates.invoke({
        "candidates_json": json.dumps([{
            "candidate_id": "join_001",
            "left_dataset": "left",
            "left_column": "key",
            "right_dataset": "right",
            "right_column": "key",
        }]),
        "raw_paths_json": json.dumps({"left": str(left), "right": str(right)}),
    }))
    ev = result["evaluations"][0]
    assert ev["inferred_relationship"] == "many_to_many"
    assert "many-to-many" in " ".join(ev["warnings"])
```

- [ ] Create `tests/test_join_discovery/test_execute_join_plan.py`:

```python
import json
import pytest
import pandas as pd
from mlops_agents.tools.join_discovery_tools import execute_join_plan


EVALUATION = {
    "candidate_id": "join_001",
    "left_dataset": "energy", "left_column": "week_date",
    "right_dataset": "weather", "right_column": "week_date",
    "left_distinct": 2, "right_distinct": 2, "intersection_count": 2,
    "left_coverage": 1.0, "right_coverage": 1.0, "jaccard": 1.0, "containment": 1.0,
    "left_unique_ratio": 1.0, "right_unique_ratio": 1.0,
    "inferred_relationship": "one_to_one",
    "estimated_inner_rows": 2, "estimated_left_rows": 2,
    "row_multiplier_left": 1.0, "join_explosion_risk": "low",
    "warnings": [],
}

SELECTIONS = {
    "base_dataset": {
        "dataset_name": "energy", "confidence": "high",
        "covered_target_columns": ["week_date", "kwh_consumed"],
        "missing_target_columns": ["avg_temp_c"],
        "reason": "contains target column", "warnings": [],
    },
    "selected": [{"candidate_id": "join_001", "columns_to_add": ["avg_temp_c"],
                  "confidence_after_evaluation": "high", "reason": "perfect overlap", "warnings": []}],
    "rejected": [],
    "unresolved_ambiguities": [],
    "warnings": [],
}


@pytest.fixture
def energy_csvs(tmp_path):
    energy = tmp_path / "energy.csv"
    weather = tmp_path / "weather.csv"
    energy.write_text("week_date,kwh_consumed\n2024-01-01,100\n2024-01-08,120\n")
    weather.write_text("week_date,avg_temp_c\n2024-01-01,10\n2024-01-08,12\n")
    return {"energy": str(energy), "weather": str(weather)}, str(tmp_path / "merged.csv")


def test_execute_join_plan_produces_merged_csv(energy_csvs):
    paths, output = energy_csvs
    result = json.loads(execute_join_plan.invoke({
        "selections_json": json.dumps(SELECTIONS),
        "evaluations_json": json.dumps({"evaluations": [EVALUATION], "errors": []}),
        "raw_paths_json": json.dumps(paths),
        "output_path": output,
    }))
    assert result["success"]
    df = pd.read_csv(output)
    assert "avg_temp_c" in df.columns
    assert len(df) == 2  # base rows preserved


def test_execute_join_plan_blocks_zero_coverage(energy_csvs):
    paths, output = energy_csvs
    zero_eval = {**EVALUATION, "left_coverage": 0.0, "intersection_count": 0}
    result = json.loads(execute_join_plan.invoke({
        "selections_json": json.dumps(SELECTIONS),
        "evaluations_json": json.dumps({"evaluations": [zero_eval], "errors": []}),
        "raw_paths_json": json.dumps(paths),
        "output_path": output,
    }))
    assert "error" in result
    assert "zero overlap" in result["error"]


def test_execute_join_plan_blocks_unevaluated_candidate(energy_csvs):
    paths, output = energy_csvs
    result = json.loads(execute_join_plan.invoke({
        "selections_json": json.dumps(SELECTIONS),
        "evaluations_json": json.dumps({"evaluations": [], "errors": []}),  # no evaluations
        "raw_paths_json": json.dumps(paths),
        "output_path": output,
    }))
    assert "error" in result
    assert "not evaluated" in result["error"]
```

- [ ] Run all tests: `uv run pytest tests/test_join_discovery/ -v`

### Frontend tests

- [ ] Create `frontend/__tests__/components/pipeline/JoinPlanPanel.test.tsx`:

```tsx
import { render, screen } from '@testing-library/react'
import { JoinPlanPanel } from '@/components/pipeline/JoinPlanPanel'
import type { JoinPlan } from '@/types/api'

const mockPlan: JoinPlan = {
  mode: 'inferred',
  base_dataset: {
    dataset_name: 'energy',
    confidence: 'high',
    covered_target_columns: ['week_date', 'kwh_consumed'],
    missing_target_columns: ['avg_temp_c'],
    reason: 'contains target column',
    warnings: [],
  },
  selected_joins: [{
    step_id: 1,
    candidate_id: 'join_001',
    left_dataset: 'energy',
    left_column: 'week_date',
    right_dataset: 'weather',
    right_column: 'week_date',
    join_type: 'left',
    columns_added: ['avg_temp_c'],
    evaluation: {
      candidate_id: 'join_001',
      left_dataset: 'energy', left_column: 'week_date',
      right_dataset: 'weather', right_column: 'week_date',
      left_distinct: 52, right_distinct: 52, intersection_count: 52,
      left_coverage: 1.0, right_coverage: 1.0, jaccard: 1.0, containment: 1.0,
      left_unique_ratio: 1.0, right_unique_ratio: 1.0,
      inferred_relationship: 'one_to_one',
      estimated_inner_rows: 52, estimated_left_rows: 52,
      row_multiplier_left: 1.0, join_explosion_risk: 'low',
      warnings: [],
    },
    confidence_after_evaluation: 'high',
    reason: 'perfect overlap on week_date',
    warnings: [],
  }],
  rejected_candidates: [],
  unresolved_ambiguities: [],
  warnings: [],
}

test('renders empty state when no plan', () => {
  render(<JoinPlanPanel joinPlan={null} />)
  expect(screen.getByText(/No inferred join plan/)).toBeInTheDocument()
})

test('renders base dataset selection with row count', () => {
  render(<JoinPlanPanel joinPlan={mockPlan} joinBaseNrows={52} />)
  expect(screen.getByText('energy')).toBeInTheDocument()
  expect(screen.getByText(/contains target column/)).toBeInTheDocument()
  expect(screen.getByText(/52/)).toBeInTheDocument()
})

test('renders selected join metrics', () => {
  render(<JoinPlanPanel joinPlan={mockPlan} />)
  expect(screen.getByText(/LEFT JOIN/)).toBeInTheDocument()
  expect(screen.getByText(/low risk/)).toBeInTheDocument()
})
```

---

## Phase 12 — Schema join_policy support

_(Previously Phase 13 — Phase 12 validation module was dropped; its required-columns check was moved into `execute_join_plan`.)_

- [ ] Extend `src/mlops_agents/graphs/mlops_graph.py:_validate_schema_contract` to allow (but not require) `join_policy`:

```python
join_policy = schema_data.get("join_policy", {})
valid_modes = {"explicit", "inferred", "hybrid"}
if join_policy:
    mode = join_policy.get("mode")
    if mode and mode not in valid_modes:
        raise ValueError(f"join_policy.mode must be one of {valid_modes}, got {mode!r}")
```

`join_policy` modes:
- `explicit`: use joins from schema — validate them, do not infer
- `infer`: agent chooses base and proposes joins
- `hybrid`: use explicit joins where provided, infer missing joins for remaining target columns

Backward compatibility (no `join_policy` in schema):
- one raw dataset → no join plan required
- multiple raw datasets + no explicit joins → `infer`
- multiple raw datasets + explicit joins → `explicit`

---

## Phase 13 — End-to-end smoke test

- [ ] Run the existing forecasting demo with two raw files and verify:
  - `_build_data_validator_context` calls `profile_raw_datasets` before the agent starts
  - The agent receives "Raw dataset profiles" in its initial context
  - The agent does NOT call any profiling tool
  - Agent proposes join candidates using the profiles from context
  - Agent calls `evaluate_join_candidates`
  - Agent calls `execute_join_plan` with `selections_json` + `evaluations_json`
  - Merged CSV is produced
  - `validate_against_schema` passes
  - Dataset Approval UI shows JoinPlan panel with base row count

- [ ] Smoke test with single dataset (iris) — verify no join tools are called

- [ ] Docker smoke test:
  ```bash
  docker compose up
  # Run forecasting pipeline with two raw files
  # Run classification pipeline with single file
  ```

---

## Acceptance criteria

1. If one raw dataset is uploaded, no join discovery is required and no join tools are called.
2. If multiple raw datasets are uploaded and explicit joins are absent, compact profiles are injected into the agent's initial context by `_build_data_validator_context`; the agent does not need to call any profiling tool.
3. The agent selects a base dataset and records a `BaseDatasetSelection`.
4. The agent proposes at most `settings.data_validator_max_join_candidates` candidates.
5. The deterministic evaluator computes: left coverage, right coverage, Jaccard, containment, uniqueness ratios, inferred relationship, estimated row multiplier, join explosion risk.
6. The agent cannot execute a join unless it was evaluated.
7. Many-to-many relationships (detected via uniqueness ratio) generate a warning; right-key deduplication is always applied before the merge, so row count is always preserved.
8. Joins with zero left coverage (no common values between key columns) are blocked at execution time.
9. The selected JoinPlan is saved in state as `data_join_plan`.
10. Dataset Approval UI shows: base dataset, selected join steps, metrics, rejected candidates, warnings.
11. The final processed dataset preserves the base dataset row count for left joins.
12. The final processed dataset contains all required target columns.
13. The JoinPlan appears in logs/audit payloads.
14. Existing explicit join schemas still work.
15. Existing single-dataset pipeline runs still work.
16. No exhaustive all-column-pair join search is introduced.
17. Backend tests pass: `uv run pytest tests/test_join_discovery/ -v`
18. Frontend tests pass: `cd frontend && npx vitest run`
19. Docker smoke run works with one dataset, two datasets (inferred join), two datasets (explicit join).

---

## Suggested implementation order

1. Contracts: `join_discovery.py` + contract tests
2. Settings
3. `profile_raw_datasets` pure function + tests
4. `evaluate_join_candidates` tool + tests
5. `execute_join_plan` tool + tests
6. Sample schema update (`is_key` → `unique`, remove `mapping_hint`)
7. `_build_data_validator_context` enrichment (call `profile_raw_datasets`, inject profiles)
8. Data validator prompt update (remove step 1 "call profiling tool", add profiles-in-context note)
9. Data validator state update + `build_data_agent` tool registration (2 tools only)
10. Dataset approval payload update (`approval_nodes.py`)
11. `build_initial_state` update (`pipeline_helpers.py`)
12. Frontend types
13. `JoinPlanPanel`
14. Frontend tests
15. End-to-end smoke
16. Documentation/thesis paragraph

## Branch

```
feature/agentic-join-discovery-v1
```

## Commit grouping

```
feat(join): add join discovery contracts and settings
feat(join): add raw dataset profiling tool
feat(join): add deterministic join candidate evaluator
feat(join): add join plan execution and validation
feat(data-validator): add agentic join discovery workflow + context enrichment
feat(ui): show inferred JoinPlan in dataset approval
test(join): add backend and frontend coverage for join discovery
docs: document Agentic Join Discovery V1
```

---

## Thesis paragraph

Agentic Join Discovery V1 allows the data validator agent to infer how raw datasets should be connected when explicit join keys are missing. The agent first selects a base dataset that best covers the target schema and defines the target granularity. It then proposes a limited set of semantically plausible join candidates. Deterministic tools evaluate only those candidates using coverage, containment, Jaccard overlap, cardinality and row-explosion metrics. The final JoinPlan is executed deterministically and exposed to the user during human approval. This design pattern — agent proposes, deterministic tool measures, agent decides, deterministic executor runs, human reviews — is the central architectural principle of the system and is illustrated concretely by this feature.
