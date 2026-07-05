# Planner Agent v2 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Model Planner from a single structured-output LLM call into a real tool-using ReAct agent with rich per-candidate rationale and a redesigned Planner tab that honestly explains decisions.

**Architecture:** Closure-bound `build_planner_tools(profile, task_meta, problem_type, trace)` factory feeds 4 deterministic tools into `create_agent(..., response_format=PlannerOutput)`. Hybrid validation: deterministic for invariants (registry, exhaustiveness), agent-observed for citation honesty. Hard vs soft conflict detection; only hard blocks. Two true agents (`data_validator`, `planner`) after this lands.

**Tech stack:** Python 3.12 + LangChain 1.2.14 (`create_agent` with `response_format`) + LangGraph + Pydantic v2 + FastAPI + Next.js 16 + React 19 + Tailwind v4 + vitest + pytest.

**Spec:** [docs/superpowers/specs/2026-05-31-planner-agent-v2-design.md](../specs/2026-05-31-planner-agent-v2-design.md)

**Prerequisite:** The prior frontend refactor (slice 5.6 smoke + final code review + `refactor-frontend-mlops-v1` tag) must be complete before Phase 7 starts. Phases 0–6 are backend-only and can begin immediately.

**Post-review patches applied to this plan (read once before starting):**
- Phase 0.1: YAML explicitness test now reads `registry.yaml` directly (the original `assert spec.supports_exogenous is not None` was a no-op due to Pydantic's `False` default).
- Phase 1.2: `ExperienceSummary.relevance_tier` ships with a TEMPORARY `= "low"` default — Task 2.3 removes it after callers populate it explicitly. No expected-failure window between Phase 1 and Phase 2.
- Phase 2.3: `_to_experience_summary` moves to `experience/retrieval.py` as the public `to_experience_summary`. `agents/planner.py` becomes a 2-line shim.
- Phase 3.1: `ToolTrace.inspect_model_details_count` (int) added — caps now count CALLS, not unique inspected keys.
- Phase 3.2: `monkeypatch.setattr` targets the USAGE site (`planning.tools.ExperiencePool`), not the source module (`from … import` rebinds the symbol).
- Phase 5.2: node test patches the four validation functions instead of rebinding `ToolTrace` in the module. Validation has its own tests.
- Frontend: `CandidateFull`/`RejectedModelFull` renamed to `CandidateRationale`/`RejectedModelRationale`; JSON keys `candidates_full`/`rejected_full` → `candidate_rationales`/`rejected_model_rationales`. Backend `_planner_output_record` uses the new keys.
- Phase 8: confirm `@/components/ui/Card` exists before writing components; use the project's existing card primitive convention.

---

## File map

```
src/mlops_agents/planning/                    ← NEW module (Phase 3–5)
  __init__.py
  tools.py            # build_planner_tools — 4 closure-bound @tool functions
  trace.py            # ToolTrace pydantic model
  context.py          # build_planner_validation_context()
  validation.py       # _check_plan_*, _detect_conflicts, detect_soft_conflicts
  agent.py            # build_planner_agent(tools) — wraps create_agent
  node.py             # planner_node() — entry + retry orchestration
  prompts.py          # message builders (planner input formatter)

src/mlops_agents/contracts/planner.py         ← MODIFIED (Phase 1)
src/mlops_agents/contracts/training.py        ← MODIFIED (CandidateSpec/RejectedModelSpec) (Phase 1)
src/mlops_agents/models/loader.py             ← MODIFIED (Phase 0)
src/mlops_agents/models/registry.yaml         ← MODIFIED (Phase 0 — backfill)
src/mlops_agents/training/profiler.py         ← MODIFIED (Phase 0)
src/mlops_agents/experience/schema.py         ← MODIFIED (Phase 0)
src/mlops_agents/experience/retrieval.py      ← MODIFIED (Phase 2)
src/mlops_agents/agents/taxonomy.py           ← NEW (Phase 0)
src/mlops_agents/agents/planner.py            ← THIN SHIM (Phase 5)
src/mlops_agents/agents/registry.py           ← MODIFIED (Phase 5)
src/mlops_agents/graphs/mlops_graph.py        ← MODIFIED (Phase 5)
src/mlops_agents/state/agent_state.py         ← MODIFIED (Phase 6)
src/mlops_agents/config/settings.py           ← MODIFIED (Phase 3)
src/mlops_agents/prompts/planner.yaml         ← REWRITTEN (Phase 5)

api/services/pipeline.py                      ← MODIFIED (Phase 6)

frontend/types/api.ts                         ← MODIFIED (Phase 7)
frontend/components/pipeline/PipelineStepper.tsx  ← MODIFIED (Phase 7)
frontend/components/pipeline/RunHeader.tsx    ← MODIFIED (Phase 7)
frontend/components/pipeline/ResultsDashboard.tsx ← MODIFIED (Phase 7+8)
frontend/components/pipeline/PlannerPanel.tsx ← NEW (Phase 8)
frontend/components/pipeline/planner/         ← NEW directory (Phase 8)
  PlannerSummaryHeader.tsx
  DecisionBasisCard.tsx
  ConflictPanel.tsx
  CandidateCard.tsx
  RejectedModelCard.tsx
  EvidenceQualityCard.tsx
  ExperienceCard.tsx
frontend/app/pipeline/page.tsx                ← MODIFIED (Phase 7)

tests/test_planning/                          ← NEW directory
tests/test_contracts/test_planner_schemas.py  ← NEW (Phase 1)
frontend/__tests__/components/pipeline/PlannerPanel.test.tsx  ← NEW (Phase 8)
frontend/__tests__/components/pipeline/planner/*.test.tsx     ← NEW per sub-component
```

---

# Phase 0 — Foundations (model/profile/taxonomy)

## Task 0.1 — ModelSpec fields + summary/details methods + registry backfill

**Files:**
- Modify: `src/mlops_agents/models/loader.py`
- Modify: `src/mlops_agents/models/registry.yaml`
- Create: `tests/test_models/test_loader_v2.py`

### Step 1 — Write failing test

`tests/test_models/test_loader_v2.py`:

```python
from mlops_agents.models.loader import get_model, get_models_for, ModelSpec

def test_modelspec_has_supports_exogenous_field():
    spec = get_model("ets")  # statistical forecasting model
    assert spec.supports_exogenous is False

def test_modelspec_has_supports_missing_field():
    spec = get_model("ets")
    assert spec.supports_missing is False

def test_summary_dict_shape():
    spec = get_model("ets")
    summary = spec.summary_dict()
    expected_keys = {"model_key", "problem_type", "family", "complexity_rank",
                     "supports_exogenous", "supports_missing", "use_when", "avoid_when"}
    assert expected_keys == set(summary.keys())

def test_details_dict_includes_search_space_name():
    spec = get_model("ets")
    details = spec.details_dict()
    assert "search_space" in details
    assert "default_params" in details
    assert "notes" in details
    assert details["model_key"] == "ets"

def test_every_model_declares_support_flags_in_registry_yaml():
    """No silent defaults: every registered model must EXPLICITLY declare both flags
    in registry.yaml. Checking spec.supports_exogenous on a constructed instance can't
    catch missing entries because the Pydantic default is False — so we read the raw YAML."""
    import yaml
    from pathlib import Path
    raw = yaml.safe_load(Path("src/mlops_agents/models/registry.yaml").read_text())
    # Adjust top-level key if registry.yaml uses a different shape (raw["models"] vs raw).
    models = raw.get("models", raw)
    for model_key, entry in models.items():
        assert "supports_exogenous" in entry, f"{model_key} missing supports_exogenous in YAML"
        assert "supports_missing" in entry, f"{model_key} missing supports_missing in YAML"
```

### Step 2 — Run test, expect fail

```
uv run pytest tests/test_models/test_loader_v2.py -v
```

Expected: `AttributeError: 'ModelSpec' object has no attribute 'supports_exogenous'`.

### Step 3 — Add fields to ModelSpec

In `src/mlops_agents/models/loader.py`, add two fields right after `notes`:

```python
class ModelSpec(BaseModel):
    model_key: str
    problem_type: Literal["classification", "regression", "forecasting"]
    family: str
    complexity_rank: int
    library: str
    factory: str
    search_space: SearchSpaceSpec
    default_params: dict[str, Any] = Field(default_factory=dict)
    requires: dict[str, Any] = Field(default_factory=dict)
    use_when: list[str] = Field(default_factory=list)
    avoid_when: list[str] = Field(default_factory=list)
    notes: str = ""
    supports_exogenous: bool = False
    supports_missing: bool = False

    def summary_dict(self) -> dict[str, Any]:
        """Headline fields for `list_available_models` planner tool."""
        return {
            "model_key": self.model_key,
            "problem_type": self.problem_type,
            "family": self.family,
            "complexity_rank": self.complexity_rank,
            "supports_exogenous": self.supports_exogenous,
            "supports_missing": self.supports_missing,
            "use_when": list(self.use_when),
            "avoid_when": list(self.avoid_when),
        }

    def details_dict(self) -> dict[str, Any]:
        """Full info for `inspect_model_details` planner tool."""
        return {
            "model_key": self.model_key,
            "problem_type": self.problem_type,
            "family": self.family,
            "complexity_rank": self.complexity_rank,
            "library": self.library,
            "supports_exogenous": self.supports_exogenous,
            "supports_missing": self.supports_missing,
            "search_space": self.search_space.name,
            "default_params": dict(self.default_params),
            "requires": dict(self.requires),
            "use_when": list(self.use_when),
            "avoid_when": list(self.avoid_when),
            "notes": self.notes,
        }
```

### Step 4 — Backfill `registry.yaml`

For every model entry in `src/mlops_agents/models/registry.yaml`, add `supports_exogenous: <bool>` and `supports_missing: <bool>` (`supports_missing` may already exist on some entries — leave them).

Forecasting models — typical mapping:
- `naive`, `seasonal_naive` → `supports_exogenous: false`, `supports_missing: false`
- `ets` → `supports_exogenous: false`, `supports_missing: false`
- `auto_arima` → `supports_exogenous: true`, `supports_missing: false` (statsforecast supports xreg)
- `*_forecaster` (tree ensembles like `extra_trees_forecaster`, `random_forest_forecaster`, `gbm_forecaster`, `lightgbm_forecaster`, `xgboost_forecaster`, `svr_forecaster`) → `supports_exogenous: true`, `supports_missing: false`
- `lightgbm_forecaster` → `supports_exogenous: true`, `supports_missing: true` (lightgbm handles NaN natively)

Classification/regression models — set both to `false` unless you know they handle missing values natively (lightgbm/xgboost classifiers can be `supports_missing: true`).

When unsure, default both to `false`. Conservative under-claiming is safe; the planner just won't recommend the model for exog/missing scenarios.

### Step 5 — Run tests, expect pass

```
uv run pytest tests/test_models/test_loader_v2.py -v
uv run pytest -m "not integration"
```

All must pass.

### Step 6 — Commit

```
git add src/mlops_agents/models/loader.py src/mlops_agents/models/registry.yaml tests/test_models/test_loader_v2.py
git commit -m "feat(models): add supports_exogenous + supports_missing to ModelSpec; summary/details dicts"
```

---

## Task 0.2 — DatasetProfile numeric target stats

**Files:**
- Modify: `src/mlops_agents/training/profiler.py`
- Create: `tests/test_training/test_profiler_target_stats.py`

### Step 1 — Failing test

```python
import pandas as pd
from pathlib import Path
from mlops_agents.training.profiler import build_dataset_profile

def test_regression_profile_has_numeric_target_stats(tmp_path):
    csv = tmp_path / "r.csv"
    pd.DataFrame({"x": range(10), "target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}).to_csv(csv, index=False)
    profile = build_dataset_profile(csv, {"problem_type": "regression", "target_column": "target"})
    assert profile.target_mean == 5.5
    assert profile.target_min == 1.0
    assert profile.target_max == 10.0
    assert profile.target_std is not None and profile.target_std > 0

def test_classification_profile_has_none_target_stats(tmp_path):
    csv = tmp_path / "c.csv"
    pd.DataFrame({"x": range(6), "target": ["a", "b", "a", "b", "a", "b"]}).to_csv(csv, index=False)
    profile = build_dataset_profile(csv, {"problem_type": "classification", "target_column": "target"})
    assert profile.target_mean is None
    assert profile.target_std is None
    assert profile.target_min is None
    assert profile.target_max is None

def test_forecasting_profile_has_numeric_target_stats(tmp_path):
    csv = tmp_path / "f.csv"
    pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=10, freq="D"),
                  "y": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]}).to_csv(csv, index=False)
    profile = build_dataset_profile(csv, {"problem_type": "forecasting", "target_column": "y", "datetime_column": "ds"})
    assert profile.target_mean == 55.0
    assert profile.target_min == 10.0
    assert profile.target_max == 100.0
```

Run: `uv run pytest tests/test_training/test_profiler_target_stats.py -v` → fail.

### Step 2 — Add fields to DatasetProfile + compute in profiler

In `src/mlops_agents/training/profiler.py`:

1. Add to the `DatasetProfile` class (Pydantic model):
   ```python
   target_mean: float | None = None
   target_std: float | None = None
   target_min: float | None = None
   target_max: float | None = None
   ```

2. In the profiler function (where `build_dataset_profile` constructs the profile), after determining `problem_type` and reading the target column:
   ```python
   target_mean = target_std = target_min = target_max = None
   if problem_type in ("regression", "forecasting"):
       target_col = task_meta.get("target_column")
       if target_col and target_col in df.columns:
           target_series = pd.to_numeric(df[target_col], errors="coerce").dropna()
           if len(target_series) > 0:
               target_mean = float(target_series.mean())
               target_std = float(target_series.std()) if len(target_series) > 1 else 0.0
               target_min = float(target_series.min())
               target_max = float(target_series.max())
   ```

3. Pass into `DatasetProfile(...)`:
   ```python
   target_mean=target_mean,
   target_std=target_std,
   target_min=target_min,
   target_max=target_max,
   ```

### Step 3 — Run tests, expect pass

```
uv run pytest tests/test_training/test_profiler_target_stats.py -v
uv run pytest -m "not integration"
```

### Step 4 — Commit

```
git add src/mlops_agents/training/profiler.py tests/test_training/test_profiler_target_stats.py
git commit -m "feat(profiler): add numeric target stats (mean/std/min/max) for numeric problems"
```

---

## Task 0.3 — ExperienceRecord target stats (Optional)

**Files:**
- Modify: `src/mlops_agents/experience/schema.py`
- Create: `tests/test_experience/test_record_target_stats.py`

### Step 1 — Failing test

```python
from mlops_agents.experience.schema import ExperienceRecord

def test_record_accepts_target_stats():
    rec = ExperienceRecord(
        task_id="t1",
        problem_type="regression",
        dataset_summary="test",
        selected_solution={"model_key": "lr", "validation_score": 0.5},
        models_tested=[],
        metric_to_optimize="rmse",
        target_mean=5.5,
        target_std=2.1,
        target_min=1.0,
        target_max=10.0,
    )
    assert rec.target_mean == 5.5

def test_record_legacy_compat_target_stats_default_none():
    rec = ExperienceRecord(
        task_id="t2",
        problem_type="classification",
        dataset_summary="test",
        selected_solution={"model_key": "rf", "validation_score": 0.9},
        models_tested=[],
        metric_to_optimize="accuracy",
    )
    assert rec.target_mean is None
    assert rec.target_std is None
    assert rec.target_min is None
    assert rec.target_max is None
```

Run: `uv run pytest tests/test_experience/test_record_target_stats.py -v` → fail.

### Step 2 — Add fields to ExperienceRecord

In `src/mlops_agents/experience/schema.py`:

```python
class ExperienceRecord(BaseModel):
    # ... existing fields ...
    target_mean: float | None = None
    target_std: float | None = None
    target_min: float | None = None
    target_max: float | None = None
```

Field signatures match what `DatasetProfile` exposes — the seed/recorder code will copy them through in a later commit. Existing pickled/JSON records without these fields stay valid because defaults are `None`.

### Step 3 — Run tests + commit

```
uv run pytest tests/test_experience/test_record_target_stats.py -v
git add src/mlops_agents/experience/schema.py tests/test_experience/test_record_target_stats.py
git commit -m "feat(experience): add Optional target stats to ExperienceRecord (legacy-safe)"
```

---

## Task 0.4 — Node taxonomy module

**Files:**
- Create: `src/mlops_agents/agents/taxonomy.py`
- Create: `tests/test_agents/test_taxonomy.py`

### Step 1 — Failing test

```python
from mlops_agents.agents.taxonomy import NODE_CATEGORIES, is_agent, is_llm_node, is_deterministic

def test_categories_are_disjoint_and_cover_all_nodes():
    agents = set(NODE_CATEGORIES["agents"])
    llm = set(NODE_CATEGORIES["llm_nodes"])
    det = set(NODE_CATEGORIES["deterministic"])
    assert agents.isdisjoint(llm)
    assert agents.isdisjoint(det)
    assert llm.isdisjoint(det)

def test_planner_is_agent_post_refactor():
    assert is_agent("planner")
    assert not is_llm_node("planner")

def test_report_writer_is_llm_node():
    assert is_llm_node("report_writer")

def test_executor_is_deterministic():
    assert is_deterministic("executor")

def test_unknown_node_is_none_of_the_above():
    assert not is_agent("foo")
    assert not is_llm_node("foo")
    assert not is_deterministic("foo")
```

Run: `uv run pytest tests/test_agents/test_taxonomy.py -v` → fail (module missing).

### Step 2 — Implement

`src/mlops_agents/agents/taxonomy.py`:

```python
"""Single source of truth for node categorization. Imported by api/services/pipeline.py
for the run_info SSE event and anywhere else that needs to classify a node by behavior."""
from __future__ import annotations

NODE_CATEGORIES: dict[str, list[str]] = {
    "agents":        ["data_validator", "planner"],
    "llm_nodes":     ["report_writer"],
    "deterministic": ["controller", "executor", "evaluation", "deployer"],
}


def is_agent(name: str) -> bool:
    return name in NODE_CATEGORIES["agents"]


def is_llm_node(name: str) -> bool:
    return name in NODE_CATEGORIES["llm_nodes"]


def is_deterministic(name: str) -> bool:
    return name in NODE_CATEGORIES["deterministic"]
```

### Step 3 — Run + commit

```
uv run pytest tests/test_agents/test_taxonomy.py -v
git add src/mlops_agents/agents/taxonomy.py tests/test_agents/test_taxonomy.py
git commit -m "feat(agents): add taxonomy.py — single source of truth for NODE_CATEGORIES"
```

---

# Phase 1 — Contract changes

## Task 1.1 — CandidateSpec + RejectedModelSpec field additions

**Files:**
- Modify: `src/mlops_agents/contracts/training.py` (or wherever `CandidateSpec` lives — check first)
- Modify: `src/mlops_agents/contracts/planner.py`
- Create: `tests/test_contracts/test_planner_schemas.py`

### Step 1 — Failing test

```python
import pytest
from pydantic import ValidationError
from mlops_agents.contracts.planner import (
    EvidenceReference, CandidateSpec, RejectedModelSpec,
)
from mlops_agents.contracts.training import TrainingPlan


def _ref(source="registry", source_id="lr"):
    return EvidenceReference(source=source, source_id=source_id, relevance_note="x")


def _minimal_training_plan():
    """Reusable helper — every test in this file that needs a TrainingPlan should call
    this rather than inline the construction. If TrainingPlan's signature differs from
    what's shown here (field names, required fields, forecasting_settings shape), this
    is the ONE place to fix it — adjust to match the real schema in src/mlops_agents/contracts/training.py."""
    return TrainingPlan(
        candidates=[CandidateSpec(model_key="lr", priority=1, reason="ok",
                                    evidence_refs=[_ref()])],
        models_not_recommended=[],
        trial_budget=10,
    )

def test_candidate_requires_priority_ge_1():
    with pytest.raises(ValidationError):
        CandidateSpec(model_key="lr", priority=0, reason="x", evidence_refs=[_ref()], risks=[])

def test_candidate_requires_non_empty_evidence_refs():
    with pytest.raises(ValidationError):
        CandidateSpec(model_key="lr", priority=1, reason="x", evidence_refs=[], risks=[])

def test_candidate_requires_non_empty_reason():
    with pytest.raises(ValidationError):
        CandidateSpec(model_key="lr", priority=1, reason="", evidence_refs=[_ref()], risks=[])

def test_candidate_valid_minimal():
    spec = CandidateSpec(model_key="lr", priority=1, reason="ok", evidence_refs=[_ref()])
    assert spec.risks == []  # default empty list

def test_rejected_requires_non_empty_evidence_refs():
    with pytest.raises(ValidationError):
        RejectedModelSpec(model_key="lr", reason="too complex", evidence_refs=[])

def test_rejected_accepts_optional_reconsider_if():
    spec = RejectedModelSpec(
        model_key="lr", reason="too complex",
        evidence_refs=[_ref()],
        reconsider_if="more data becomes available",
    )
    assert spec.reconsider_if == "more data becomes available"
```

Run: `uv run pytest tests/test_contracts/test_planner_schemas.py -v` → fail.

### Step 2 — Modify the schemas

In `src/mlops_agents/contracts/planner.py` (or training.py — wherever the canonical classes are):

```python
from pydantic import BaseModel, Field
from typing import Literal


class EvidenceReference(BaseModel):
    source: Literal["dataset_profile", "task_metadata", "registry", "experience", "rule"]
    source_id: str | None = None
    relevance_note: str | None = None


class CandidateSpec(BaseModel):
    model_key: str
    priority: int = Field(ge=1)
    reason: str = Field(min_length=1)
    evidence_refs: list[EvidenceReference] = Field(min_length=1)
    risks: list[str] = Field(default_factory=list)


class RejectedModelSpec(BaseModel):
    model_key: str
    reason: str = Field(min_length=1)
    evidence_refs: list[EvidenceReference] = Field(min_length=1)
    reconsider_if: str | None = None
```

If `CandidateSpec` already exists with other fields (e.g., `search_space_key`, `default_params`, `hyperparams_hint`), KEEP them as-is — only ADD `priority`, `reason`, `evidence_refs`, `risks`. Existing seed data + tests must continue to work; downstream consumers can read or ignore the new fields.

### Step 3 — Run tests, expect pass

```
uv run pytest tests/test_contracts/test_planner_schemas.py -v
uv run pytest -m "not integration"
```

Existing planner/training tests should still pass because the additions are non-breaking.

### Step 4 — Commit

```
git add src/mlops_agents/contracts/planner.py src/mlops_agents/contracts/training.py tests/test_contracts/test_planner_schemas.py
git commit -m "feat(contracts): CandidateSpec/RejectedModelSpec gain priority+reason+evidence_refs+risks/reconsider_if"
```

---

## Task 1.2 — DecisionBasis + EvidenceConflict + ExperienceSummary extension + PlannerOutput extension

**Files:**
- Modify: `src/mlops_agents/contracts/planner.py`
- Extend: `tests/test_contracts/test_planner_schemas.py`

### Step 1 — Add failing tests (append to file)

```python
from mlops_agents.contracts.planner import (
    DecisionBasis, EvidenceConflict, ExperienceSummary, PlannerOutput,
)

def test_decision_basis_requires_primary_evidence():
    with pytest.raises(ValidationError):
        DecisionBasis(primary_evidence=[], secondary_evidence=[], final_strategy="x")

def test_decision_basis_requires_non_empty_final_strategy():
    with pytest.raises(ValidationError):
        DecisionBasis(primary_evidence=[_ref()], secondary_evidence=[], final_strategy="")

def test_evidence_conflict_requires_resolution():
    with pytest.raises(ValidationError):
        EvidenceConflict(
            summary="x", affected_models=["lr"],
            conflicting_evidence_refs=[_ref()], resolution="",
        )

def test_experience_summary_has_relevance_tier_buckets_scale_note():
    es = ExperienceSummary(
        experience_id="e1",
        similarity_score=0.72,
        relevance_tier="high",
        matched_buckets=["forecasting", "weekly"],
        mismatched_buckets=["target_scale"],
        target_scale_note="candidate target std ≈10× experience",
        dataset_summary="s",
        models_trained=["ets"],
        best_model="ets",
        validation_score=0.5,
        metric_name="rmse",
        candidate_results=[],
    )
    assert es.relevance_tier == "high"

def test_planner_output_has_decision_basis_and_evidence_conflicts():
    out = PlannerOutput(
        planning_analysis="ok",
        decision_basis=DecisionBasis(
            primary_evidence=[_ref()],
            secondary_evidence=[],
            final_strategy="prioritize simple models",
        ),
        evidence_used=[],
        evidence_conflicts=[],
        risks_or_warnings=[],
        plan=_minimal_training_plan(),
    )
    assert out.decision_basis.final_strategy == "prioritize simple models"
    assert out.evidence_conflicts == []
```

Uses the `_minimal_training_plan()` helper defined at the top of the test file in Task 1.1.

### Step 2 — Add models to `contracts/planner.py`

```python
class DecisionBasis(BaseModel):
    primary_evidence: list[EvidenceReference] = Field(min_length=1)
    secondary_evidence: list[EvidenceReference] = Field(default_factory=list)
    final_strategy: str = Field(min_length=1)


class EvidenceConflict(BaseModel):
    summary: str = Field(min_length=1)
    affected_models: list[str] = Field(min_length=1)
    conflicting_evidence_refs: list[EvidenceReference] = Field(min_length=1)
    resolution: str = Field(min_length=1)


# ExperienceSummary — extend existing class
class ExperienceSummary(BaseModel):
    # ... existing fields ...
    # NOTE: temporary default "low" makes this commit non-breaking. Phase 2 Task 2.3
    # updates _to_experience_summary to always pass relevance_tier explicitly; at that
    # point this default should be REMOVED (Task 2.3 includes that step).
    relevance_tier: Literal["high", "medium", "low"] = "low"
    matched_buckets: list[str] = Field(default_factory=list)
    mismatched_buckets: list[str] = Field(default_factory=list)
    target_scale_note: str | None = None


# PlannerOutput — extend
class PlannerOutput(BaseModel):
    planning_analysis: str
    decision_basis: DecisionBasis                                    # NEW required
    evidence_used: list[EvidenceReference] = Field(default_factory=list)
    evidence_conflicts: list[EvidenceConflict] = Field(default_factory=list)
    risks_or_warnings: list[str] = Field(default_factory=list)
    plan: TrainingPlan
```

`relevance_tier` ships with a temporary `= "low"` default so the test suite stays green between Task 1.2 and Task 2.3. **Task 2.3 removes the default**, after every caller (`_to_experience_summary`) populates it explicitly. Do NOT leave the default permanently — it would hide silent miscategorization.

### Step 3 — Run + commit

```
uv run pytest tests/test_contracts/test_planner_schemas.py -v
git add src/mlops_agents/contracts/planner.py tests/test_contracts/test_planner_schemas.py
git commit -m "feat(contracts): add DecisionBasis, EvidenceConflict; extend ExperienceSummary + PlannerOutput"
```

Existing tests that construct `ExperienceSummary` without `relevance_tier` will now fail. They'll be fixed in Phase 2 (Task 2.3) when retrieval starts producing the field. **Until then, the full test suite has expected failures.** Note this explicitly in the commit message:

```
git commit -m "feat(contracts): add DecisionBasis, EvidenceConflict; extend ExperienceSummary + PlannerOutput"
```

---

## Task 1.3 — PlannerValidationContext model

**Files:**
- Modify: `src/mlops_agents/contracts/planner.py`
- Extend: `tests/test_contracts/test_planner_schemas.py`

### Step 1 — Failing test

```python
from mlops_agents.contracts.planner import PlannerValidationContext
from mlops_agents.models.loader import get_models_for

def test_planner_validation_context_minimal():
    specs = get_models_for("forecasting")
    ctx = PlannerValidationContext(
        problem_type="forecasting",
        task_metadata={},
        available_model_keys=[s.model_key for s in specs],
        available_model_specs=specs,
        similar_experiences=[],
        matched_rules=[],
        rules_by_id={},
    )
    assert ctx.problem_type == "forecasting"
    assert len(ctx.available_model_keys) > 0
```

### Step 2 — Implement

In `src/mlops_agents/contracts/planner.py`:

```python
from mlops_agents.models.loader import ModelSpec  # at top

class PlannerValidationContext(BaseModel):
    """Deterministic ground-truth context — independent of agent behavior."""
    problem_type: str
    task_metadata: dict
    available_model_keys: list[str]
    available_model_specs: list[ModelSpec]
    similar_experiences: list[ExperienceSummary]
    matched_rules: list[dict]
    rules_by_id: dict[str, dict]

    model_config = {"arbitrary_types_allowed": True}  # for ModelSpec
```

### Step 3 — Run + commit

```
uv run pytest tests/test_contracts/test_planner_schemas.py -v
git add src/mlops_agents/contracts/planner.py tests/test_contracts/test_planner_schemas.py
git commit -m "feat(contracts): add PlannerValidationContext model"
```

---

# Phase 2 — Retrieval upgrade

## Task 2.1 — `derive_relevance_tier` helper

**Files:**
- Modify: `src/mlops_agents/experience/retrieval.py`
- Create: `tests/test_experience/test_retrieval_helpers.py`

### Step 1 — Failing test

```python
from mlops_agents.experience.retrieval import derive_relevance_tier

def test_high_tier_ge_0_7():
    assert derive_relevance_tier(0.7) == "high"
    assert derive_relevance_tier(0.95) == "high"

def test_medium_tier_range():
    assert derive_relevance_tier(0.4) == "medium"
    assert derive_relevance_tier(0.69) == "medium"

def test_low_tier_below_0_4():
    assert derive_relevance_tier(0.0) == "low"
    assert derive_relevance_tier(0.39) == "low"
```

### Step 2 — Implement

In `src/mlops_agents/experience/retrieval.py`:

```python
from typing import Literal

def derive_relevance_tier(similarity_score: float) -> Literal["high", "medium", "low"]:
    """Map a similarity score to a coarse relevance tier for UI display.

    Thresholds match spec: high >= 0.7, medium 0.4-0.7, low < 0.4.
    """
    if similarity_score >= 0.7:
        return "high"
    if similarity_score >= 0.4:
        return "medium"
    return "low"
```

### Step 3 — Run + commit

```
uv run pytest tests/test_experience/test_retrieval_helpers.py -v
git add src/mlops_agents/experience/retrieval.py tests/test_experience/test_retrieval_helpers.py
git commit -m "feat(retrieval): add derive_relevance_tier helper (thresholds 0.7/0.4)"
```

---

## Task 2.2 — `compare_target_scales` helper

**Files:**
- Modify: `src/mlops_agents/experience/retrieval.py`
- Extend: `tests/test_experience/test_retrieval_helpers.py`

### Step 1 — Failing test (append)

```python
from mlops_agents.experience.retrieval import compare_target_scales

def test_similar_scales_returns_none():
    assert compare_target_scales(
        profile_target_std=2.0, experience_target_std=2.3,
    ) is None

def test_one_order_of_magnitude_returns_note():
    note = compare_target_scales(profile_target_std=25.0, experience_target_std=2.0)
    assert note is not None
    assert "10×" in note or "10x" in note.lower() or "order" in note.lower()

def test_missing_either_side_returns_none():
    assert compare_target_scales(profile_target_std=None, experience_target_std=2.0) is None
    assert compare_target_scales(profile_target_std=2.0, experience_target_std=None) is None
    assert compare_target_scales(profile_target_std=None, experience_target_std=None) is None

def test_zero_target_std_returns_none():
    assert compare_target_scales(profile_target_std=0.0, experience_target_std=2.0) is None
```

### Step 2 — Implement

In `src/mlops_agents/experience/retrieval.py`:

```python
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
```

### Step 3 — Run + commit

```
uv run pytest tests/test_experience/test_retrieval_helpers.py -v
git add src/mlops_agents/experience/retrieval.py tests/test_experience/test_retrieval_helpers.py
git commit -m "feat(retrieval): add compare_target_scales helper (~10× threshold)"
```

---

## Task 2.3 — Extend `RetrievalView` + `find_similar_impl` with buckets/tiers/scale-note + populate ExperienceSummary

**Files:**
- Modify: `src/mlops_agents/experience/schema.py` (extend `RetrievalView`)
- Modify: `src/mlops_agents/experience/retrieval.py` (extend `_build_view` and `find_similar_impl`)
- Modify: `src/mlops_agents/agents/planner.py` (update `_to_experience_summary` to pass new fields)
- Create: `tests/test_experience/test_retrieval_v2.py`

### Step 1 — Failing test

```python
import pandas as pd
import pytest
from pathlib import Path
from mlops_agents.experience.pool import ExperiencePool
from mlops_agents.experience.schema import ExperienceRecord

@pytest.fixture
def seeded_pool(tmp_path):
    db = tmp_path / "exp.db"
    pool = ExperiencePool(db)
    # Seed one matching + one mismatching experience
    pool.add(ExperienceRecord(
        task_id="match",
        problem_type="forecasting",
        dataset_summary="weekly",
        selected_solution={"model_key": "ets", "validation_score": 1.0},
        models_tested=[],
        metric_to_optimize="rmse",
        target_mean=10.0, target_std=2.0, target_min=5.0, target_max=15.0,
        profile={"history_length_bucket": "short", "frequency_bucket": "weekly"},
    ))
    pool.add(ExperienceRecord(
        task_id="mismatch",
        problem_type="forecasting",
        dataset_summary="daily different scale",
        selected_solution={"model_key": "naive", "validation_score": 50.0},
        models_tested=[],
        metric_to_optimize="rmse",
        target_mean=1000.0, target_std=200.0, target_min=500.0, target_max=1500.0,
        profile={"history_length_bucket": "long", "frequency_bucket": "daily"},
    ))
    return pool

def test_retrieval_view_has_matched_and_mismatched_buckets(seeded_pool):
    profile = {"history_length_bucket": "short", "frequency_bucket": "weekly",
               "target_std": 2.0, "target_mean": 10.0}
    views = seeded_pool.find_similar(profile, "forecasting", k=2)
    match_view = next(v for v in views if v.task_id == "match")
    assert "history_length_bucket" in match_view.matched_buckets or "frequency_bucket" in match_view.matched_buckets

def test_retrieval_view_target_scale_note_present_for_mismatch(seeded_pool):
    profile = {"history_length_bucket": "short", "frequency_bucket": "weekly",
               "target_std": 2.0, "target_mean": 10.0}
    views = seeded_pool.find_similar(profile, "forecasting", k=2)
    mismatch_view = next(v for v in views if v.task_id == "mismatch")
    assert mismatch_view.target_scale_note is not None
```

### Step 2 — Extend `RetrievalView` in `schema.py`

```python
class RetrievalView(BaseModel):
    # ... existing fields ...
    matched_buckets: list[str] = Field(default_factory=list)
    mismatched_buckets: list[str] = Field(default_factory=list)
    target_scale_note: str | None = None
```

### Step 3 — Extend `_build_view` in `retrieval.py`

```python
from mlops_agents.experience.retrieval import compare_target_scales

def _build_view(row, cand_rows, score: int, ratio: float, matched: list,
                profile: dict | None = None) -> RetrievalView | None:
    # ... existing logic ...
    
    # NEW: derive mismatched buckets — fields present in `profile` but not in `matched`
    profile_keys = set((profile or {}).keys())
    bucket_keys = {k for k in profile_keys if k.endswith("_bucket")}
    mismatched = sorted(bucket_keys - set(matched))
    
    # NEW: target_scale_note
    note = None
    if profile is not None:
        note = compare_target_scales(
            profile_target_std=profile.get("target_std"),
            experience_target_std=getattr(row, "target_std", None),
        )
    
    return RetrievalView(
        # ... existing kwargs ...
        matched_buckets=list(matched),     # already deterministic
        mismatched_buckets=mismatched,
        target_scale_note=note,
    )
```

Update the `_build_view` call site in `find_similar_impl` to pass `profile=profile`.

### Step 4 — Move `_to_experience_summary` to `experience/retrieval.py` as `to_experience_summary` (public)

The current `_to_experience_summary` lives in `src/mlops_agents/agents/planner.py`. Move it into `src/mlops_agents/experience/retrieval.py` (its inputs are retrieval types — that's its natural home) and rename to `to_experience_summary` (public). Also explicitly populate `relevance_tier` so the Phase 1 default can be removed:

```python
# in src/mlops_agents/experience/retrieval.py
from mlops_agents.contracts.planner import CandidateResultCompact, ExperienceSummary
from mlops_agents.experience.schema import RetrievalView


def to_experience_summary(view: RetrievalView) -> ExperienceSummary:
    sel_key = view.selected_solution.model_key
    scored = [c for c in view.models_tested if c.best_score is not None]
    failed = [c for c in view.models_tested if c.best_score is None]
    scored.sort(key=lambda c: (c.model_key != sel_key, -(c.best_score or 0.0)))
    compact = [
        CandidateResultCompact(model_key=c.model_key, rank=i + 1, metric_value=c.best_score)
        for i, c in enumerate(scored)
    ]
    for f in failed:
        compact.append(CandidateResultCompact(
            model_key=f.model_key, rank=len(compact) + 1, metric_value=None,
        ))
    return ExperienceSummary(
        experience_id=view.task_id,
        similarity_score=view.similarity_ratio,
        relevance_tier=derive_relevance_tier(view.similarity_ratio),
        matched_buckets=view.matched_buckets,
        mismatched_buckets=view.mismatched_buckets,
        target_scale_note=view.target_scale_note,
        dataset_summary=view.experience_summary or "",
        models_trained=[c.model_key for c in view.models_tested],
        best_model=sel_key,
        validation_score=view.selected_solution.validation_score,
        metric_name=view.metric_to_optimize,
        candidate_results=compact,
    )
```

Then DELETE `_to_experience_summary` from `agents/planner.py`. Now that every caller passes `relevance_tier`, REMOVE the temporary default from `ExperienceSummary` (Task 1.2):

```python
# in contracts/planner.py — final form
relevance_tier: Literal["high", "medium", "low"]  # no default
```

Grep for any remaining `_to_experience_summary` callers and update them to the public name + new module path.

### Step 5 — Run tests, expect pass

```
uv run pytest tests/test_experience/ -v
uv run pytest -m "not integration"
```

ExperienceSummary callers are now satisfied — Phase 1's expected failures resolve.

### Step 6 — Commit

```
git add src/mlops_agents/experience/schema.py src/mlops_agents/experience/retrieval.py src/mlops_agents/agents/planner.py tests/test_experience/test_retrieval_v2.py
git commit -m "feat(retrieval): RetrievalView + ExperienceSummary gain buckets/tier/scale_note"
```

---

# Phase 3 — Tool layer

## Task 3.1 — `ToolTrace` model + settings additions

**Files:**
- Create: `src/mlops_agents/planning/__init__.py` (empty)
- Create: `src/mlops_agents/planning/trace.py`
- Modify: `src/mlops_agents/config/settings.py`
- Create: `tests/test_planning/__init__.py` (empty)
- Create: `tests/test_planning/test_trace.py`

### Step 1 — Failing test

```python
from mlops_agents.planning.trace import ToolTrace

def test_tooltrace_defaults():
    t = ToolTrace()
    assert t.called_tools == []
    assert t.tool_call_count == 0
    assert t.raw_observations == []

def test_tooltrace_model_dump_roundtrip():
    t = ToolTrace()
    t.called_tools = ["a"]
    t.tool_call_count = 1
    t.raw_observations = [{"tool": "a"}]
    d = t.model_dump()
    assert d["tool_call_count"] == 1
    t2 = ToolTrace.model_validate(d)
    assert t2.called_tools == ["a"]
```

### Step 2 — Implement

`src/mlops_agents/planning/__init__.py`: empty (package marker).

`src/mlops_agents/planning/trace.py`:

```python
"""ToolTrace — records what the planner agent observed during a single run."""
from pydantic import BaseModel, Field


class ToolTrace(BaseModel):
    called_tools: list[str] = Field(default_factory=list)
    listed_model_keys: list[str] = Field(default_factory=list)
    retrieved_experience_ids: list[str] = Field(default_factory=list)
    retrieved_rule_ids: list[str] = Field(default_factory=list)
    inspected_model_keys: list[str] = Field(default_factory=list)
    inspect_model_details_count: int = 0  # CALL count — NOT len(inspected_model_keys),
                                          # because agents can call inspect_model_details("ets") 4x
                                          # and unique-set length stays at 1 (cap would never fire).
    tool_call_count: int = 0
    raw_observations: list[dict] = Field(default_factory=list)
```

Extend `src/mlops_agents/config/settings.py` `Settings` class with:

```python
    planner_max_iterations: int = 10
    planner_max_tool_calls: int = 6
    planner_max_inspect_calls: int = 3
    planner_max_retrieved: int = 20
    planner_timeout_seconds: int = 60  # RESERVED for future wall-clock enforcement; NOT enforced in v2
```

### Step 3 — Run + commit

```
uv run pytest tests/test_planning/test_trace.py -v
git add src/mlops_agents/planning/__init__.py src/mlops_agents/planning/trace.py src/mlops_agents/config/settings.py tests/test_planning/__init__.py tests/test_planning/test_trace.py
git commit -m "feat(planning): add ToolTrace model + planner_* settings"
```

---

## Task 3.2 — `build_planner_tools` factory + tool tests

**Files:**
- Create: `src/mlops_agents/planning/tools.py`
- Create: `tests/test_planning/test_tools.py`

### Step 1 — Failing test

```python
import pytest
from mlops_agents.planning.tools import build_planner_tools, _view_to_tool_dict
from mlops_agents.planning.trace import ToolTrace
from mlops_agents.config.settings import settings
import mlops_agents.experience.pool as pool_mod
from unittest.mock import patch, MagicMock


@pytest.fixture
def fresh_trace():
    return ToolTrace()

@pytest.fixture
def fake_pool():
    p = MagicMock()
    p.find_similar.return_value = []
    return p

def test_list_available_models_filters_by_problem_type(fresh_trace):
    tools = build_planner_tools({}, {}, "forecasting", fresh_trace)
    list_tool = next(t for t in tools if t.name == "list_available_models")
    result = list_tool.invoke({})
    assert isinstance(result, list)
    assert all(m["problem_type"] == "forecasting" for m in result)
    assert "list_available_models" in fresh_trace.called_tools
    assert len(fresh_trace.listed_model_keys) > 0
    assert fresh_trace.tool_call_count == 1

def test_inspect_model_details_returns_error_for_unknown_key(fresh_trace):
    tools = build_planner_tools({}, {}, "forecasting", fresh_trace)
    inspect_tool = next(t for t in tools if t.name == "inspect_model_details")
    result = inspect_tool.invoke({"model_key": "nonexistent_model"})
    assert "error" in result
    assert "nonexistent_model" in result["error"]

def test_per_tool_inspect_cap_counts_calls_not_unique_keys(fresh_trace, monkeypatch):
    """Cap is on call count, not unique keys — repeated inspects of the same model still
    burn the budget."""
    monkeypatch.setattr(settings, "planner_max_inspect_calls", 2)
    tools = build_planner_tools({}, {}, "forecasting", fresh_trace)
    inspect_tool = next(t for t in tools if t.name == "inspect_model_details")
    inspect_tool.invoke({"model_key": "ets"})  # call 1
    inspect_tool.invoke({"model_key": "ets"})  # call 2 — same key, still counts
    # Third call hits cap
    result = inspect_tool.invoke({"model_key": "ets"})
    assert "max inspect_model_details calls" in result["error"]
    assert fresh_trace.inspect_model_details_count == 2

def test_global_max_tool_calls_short_circuits(fresh_trace, monkeypatch):
    monkeypatch.setattr(settings, "planner_max_tool_calls", 2)
    tools = build_planner_tools({}, {}, "forecasting", fresh_trace)
    list_tool = next(t for t in tools if t.name == "list_available_models")
    rules_tool = next(t for t in tools if t.name == "retrieve_ml_knowledge")
    list_tool.invoke({})
    rules_tool.invoke({})
    # 3rd call should be rejected
    result = list_tool.invoke({})
    assert "max_tool_calls exceeded" in result["error"]

def test_retrieve_similar_experiences_clamps_top_k(fresh_trace, monkeypatch):
    """Patch at the USAGE site (planning.tools.ExperiencePool), not the source module.
    `from ... import ExperiencePool` rebinds the symbol into tools.py's namespace, so
    patching pool_mod has no effect on the already-imported reference."""
    monkeypatch.setattr(settings, "planner_max_retrieved", 5)
    captured = {}
    def fake_pool_factory(path):
        p = MagicMock()
        def fake_find(profile, problem_type, k):
            captured["k"] = k
            return []
        p.find_similar.side_effect = fake_find
        return p
    monkeypatch.setattr("mlops_agents.planning.tools.ExperiencePool", fake_pool_factory)
    tools = build_planner_tools({}, {}, "forecasting", fresh_trace)
    retrieve_tool = next(t for t in tools if t.name == "retrieve_similar_experiences")
    retrieve_tool.invoke({"top_k": 50})  # ask for way too many
    assert captured["k"] == 5  # clamped

def test_dedup_across_repeated_calls(fresh_trace):
    tools = build_planner_tools({}, {}, "forecasting", fresh_trace)
    list_tool = next(t for t in tools if t.name == "list_available_models")
    list_tool.invoke({})
    # Calling again would normally double-count — verify dedup
    assert fresh_trace.called_tools == ["list_available_models"]
    initial_models = set(fresh_trace.listed_model_keys)
    list_tool.invoke({})
    # Same models, still deduped
    assert set(fresh_trace.listed_model_keys) == initial_models
```

### Step 2 — Implement `src/mlops_agents/planning/tools.py`

```python
"""Planner agent tools — closure-bound deterministic wrappers around existing helpers.

Each tool records observations to a shared ToolTrace so validation can later
verify the agent only cited what it actually retrieved (hybrid validation A1)."""
from __future__ import annotations
from typing import Any

from langchain_core.tools import tool

from mlops_agents.config.settings import settings
from mlops_agents.experience.pool import ExperiencePool
from mlops_agents.experience.retrieval import derive_relevance_tier
from mlops_agents.experience.schema import RetrievalView
from mlops_agents.knowledge.reader import match_rules
from mlops_agents.models.loader import get_model, get_models_for
from mlops_agents.planning.trace import ToolTrace


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
) -> list:
    """Build closure-bound planner tools that record observations to the shared trace.

    `problem_type` is bound at closure time — the agent cannot override it.
    `dataset_profile` and `task_metadata` are also closed over (no per-call args)."""

    def _gate(tool_name: str | None = None) -> bool:
        """Return False if call should be rejected. Enforces global ceiling + per-tool inspect cap.
        Per-tool inspect cap is checked separately inside inspect_model_details (uses
        trace.inspect_model_details_count — call count, NOT len(inspected_model_keys))."""
        if trace.tool_call_count >= settings.planner_max_tool_calls:
            return False
        trace.tool_call_count += 1
        return True

    def _dedup(field: list[str], new_items: set[str]) -> list[str]:
        return sorted(set(field) | new_items)

    @tool
    def list_available_models() -> list[dict] | dict:
        """List all models in the registry for the current problem type. Returns one entry
        per model with headline fields (model_key, family, complexity_rank,
        supports_exogenous, supports_missing, use_when, avoid_when). Call this once at the
        start of planning. Models not in this list cannot be recommended."""
        if not _gate(): return _MAX_CALLS_ERR
        specs = get_models_for(problem_type)
        out = [s.summary_dict() for s in specs]
        trace.called_tools = _dedup(trace.called_tools, {"list_available_models"})
        trace.listed_model_keys = _dedup(trace.listed_model_keys, {s["model_key"] for s in out})
        trace.raw_observations.append({"tool": "list_available_models", "result_count": len(out)})
        return out

    @tool
    def retrieve_similar_experiences(top_k: int = 5) -> list[dict] | dict:
        """Retrieve the top-k most similar past training experiences for the current dataset.
        Similarity is deterministic (bucket-based, no embeddings). Each result includes
        experience_id, similarity_score, relevance_tier, matched_buckets, mismatched_buckets,
        target_scale_note, best_model, primary_metric, score, dataset_summary.
        Use these to inform candidate selection. Call this once unless you need a wider net.
        top_k is clamped to [1, planner_max_retrieved] so it never exceeds the
        deterministic validation context depth."""
        if not _gate(): return _MAX_CALLS_ERR
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
    def retrieve_ml_knowledge() -> list[dict] | dict:
        """Retrieve static ML rules that match the current dataset profile + task metadata.
        Each rule returns rule_id, prefer, avoid_or_deprioritize, recommend, summary.
        Call this once."""
        if not _gate(): return _MAX_CALLS_ERR
        # NOTE: if task_metadata keys collide with profile keys, task_metadata wins.
        rule_input = {**dataset_profile, **task_metadata, "problem_type": problem_type}
        matched = match_rules(rule_input)
        out = [{
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
    def inspect_model_details(model_key: str) -> dict:
        """Get full registry metadata for one model. Use sparingly — only when
        list_available_models doesn't give you enough info to decide. Hard cap of 3
        inspect CALLS per planner run (repeated calls on same key still burn budget).
        Returns {"error": ...} if model_key unknown."""
        # Per-tool cap check BEFORE incrementing budget
        if trace.inspect_model_details_count >= settings.planner_max_inspect_calls:
            return _MAX_INSPECT_ERR
        if not _gate("inspect_model_details"): return _MAX_CALLS_ERR
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
```

### Step 3 — Run tests, expect pass

```
uv run pytest tests/test_planning/test_tools.py -v
```

### Step 4 — Commit

```
git add src/mlops_agents/planning/tools.py tests/test_planning/test_tools.py
git commit -m "feat(planning): add build_planner_tools — 4 closure-bound deterministic tools with trace + budget caps"
```

---

# Phase 4 — Validation layer

## Task 4.1 — `build_planner_validation_context`

**Files:**
- Create: `src/mlops_agents/planning/context.py`
- Create: `tests/test_planning/test_context.py`

### Step 1 — Failing test

```python
import pytest
from mlops_agents.planning.context import build_planner_validation_context

def test_context_problem_type_propagated():
    ctx = build_planner_validation_context({}, {}, "forecasting")
    assert ctx.problem_type == "forecasting"
    assert len(ctx.available_model_keys) > 0
    assert len(ctx.available_model_specs) == len(ctx.available_model_keys)

def test_context_is_deterministic():
    ctx1 = build_planner_validation_context({"x": 1}, {"y": 2}, "regression")
    ctx2 = build_planner_validation_context({"x": 1}, {"y": 2}, "regression")
    assert ctx1.available_model_keys == ctx2.available_model_keys
    # rules_by_id deterministic too
    assert set(ctx1.rules_by_id.keys()) == set(ctx2.rules_by_id.keys())

def test_context_rules_by_id_lookup():
    ctx = build_planner_validation_context({"history_length_bucket": "short"},
                                            {"problem_type": "forecasting"},
                                            "forecasting")
    for rule_id, rule in ctx.rules_by_id.items():
        assert rule.get("rule_id") == rule_id
```

### Step 2 — Implement

`src/mlops_agents/planning/context.py`:

```python
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
```

### Step 3 — Run + commit

```
uv run pytest tests/test_planning/test_context.py -v
git add src/mlops_agents/planning/context.py tests/test_planning/test_context.py
git commit -m "feat(planning): add build_planner_validation_context — deterministic ground truth"
```

---

## Task 4.2 — Validation: `_check_plan_integrity` + `_check_plan_exhaustiveness` + `_check_evidence_references_hybrid`

**Files:**
- Create: `src/mlops_agents/planning/validation.py`
- Create: `tests/test_planning/test_validation.py`

### Step 1 — Failing test (comprehensive)

```python
import pytest
from mlops_agents.planning.validation import (
    PlannerValidationError,
    _check_plan_integrity,
    _check_plan_exhaustiveness,
    _check_evidence_references_hybrid,
    _collect_all_refs,
)
from mlops_agents.contracts.planner import (
    CandidateSpec, RejectedModelSpec, EvidenceReference,
    DecisionBasis, PlannerOutput, PlannerValidationContext,
)
from mlops_agents.contracts.training import TrainingPlan
from mlops_agents.planning.trace import ToolTrace


def _registry_ref(key): return EvidenceReference(source="registry", source_id=key)
def _profile_ref(): return EvidenceReference(source="dataset_profile", source_id=None)

def _candidate(key, priority=1):
    return CandidateSpec(model_key=key, priority=priority, reason="ok",
                          evidence_refs=[_registry_ref(key)], risks=[])

def _rejected(key):
    return RejectedModelSpec(model_key=key, reason="too complex",
                              evidence_refs=[_registry_ref(key)])

def _minimal_output(candidates=None, rejected=None):
    candidates = candidates or [_candidate("a")]
    rejected = rejected or []
    return PlannerOutput(
        planning_analysis="ok",
        decision_basis=DecisionBasis(
            primary_evidence=[_profile_ref()],
            secondary_evidence=[],
            final_strategy="strat",
        ),
        evidence_used=[],
        evidence_conflicts=[],
        risks_or_warnings=[],
        plan=TrainingPlan(candidates=candidates, models_not_recommended=rejected, trial_budget=10),
    )

def _minimal_ctx(available=None):
    available = available or ["a", "b"]
    return PlannerValidationContext(
        problem_type="classification",
        task_metadata={},
        available_model_keys=available,
        available_model_specs=[],
        similar_experiences=[],
        matched_rules=[],
        rules_by_id={},
    )

def _trace_with_required():
    t = ToolTrace()
    t.called_tools = ["list_available_models", "retrieve_similar_experiences", "retrieve_ml_knowledge"]
    t.tool_call_count = 3
    return t


def test_integrity_missing_required_tool_raises():
    out = _minimal_output()
    ctx = _minimal_ctx(["a"])
    trace = ToolTrace()
    trace.called_tools = ["list_available_models"]  # missing 2 required
    with pytest.raises(PlannerValidationError, match="required tools"):
        _check_plan_integrity(out, trace, ctx)

def test_integrity_duplicate_priorities_raises():
    out = _minimal_output(candidates=[_candidate("a", priority=1), _candidate("b", priority=1)])
    ctx = _minimal_ctx(["a", "b"])
    with pytest.raises(PlannerValidationError, match="unique"):
        _check_plan_integrity(out, ctx=ctx, trace=_trace_with_required())

def test_integrity_overlap_candidate_rejected_raises():
    out = _minimal_output(candidates=[_candidate("a")], rejected=[_rejected("a")])
    with pytest.raises(PlannerValidationError, match="overlap"):
        _check_plan_integrity(out, _trace_with_required(), _minimal_ctx(["a"]))

def test_integrity_missing_registry_self_cite_on_candidate_raises():
    cand = CandidateSpec(model_key="a", priority=1, reason="ok",
                          evidence_refs=[_profile_ref()], risks=[])
    out = _minimal_output(candidates=[cand])
    with pytest.raises(PlannerValidationError, match="registry self-citation"):
        _check_plan_integrity(out, _trace_with_required(), _minimal_ctx(["a"]))

def test_integrity_passes_on_clean_plan():
    out = _minimal_output()
    _check_plan_integrity(out, _trace_with_required(), _minimal_ctx(["a"]))  # no raise


def test_exhaustiveness_missing_model_raises():
    out = _minimal_output(candidates=[_candidate("a")])
    with pytest.raises(PlannerValidationError, match=r"\['b'\]"):
        _check_plan_exhaustiveness(out.plan, ["a", "b"])


def test_evidence_ref_registry_unknown_key_raises():
    cand = CandidateSpec(
        model_key="a", priority=1, reason="ok",
        evidence_refs=[_registry_ref("a"), _registry_ref("nonexistent")],
        risks=[],
    )
    out = _minimal_output(candidates=[cand])
    with pytest.raises(PlannerValidationError, match="nonexistent"):
        _check_evidence_references_hybrid(out, _minimal_ctx(["a"]), _trace_with_required())

def test_evidence_ref_experience_not_retrieved_raises():
    ref = EvidenceReference(source="experience", source_id="exp_999")
    out = _minimal_output()
    out.evidence_used = [ref]
    trace = _trace_with_required()
    # exp_999 not in trace.retrieved_experience_ids
    with pytest.raises(PlannerValidationError, match="exp_999"):
        _check_evidence_references_hybrid(out, _minimal_ctx(["a"]), trace)

def test_evidence_ref_experience_retrieved_passes():
    ref = EvidenceReference(source="experience", source_id="exp_001")
    out = _minimal_output()
    out.evidence_used = [ref]
    trace = _trace_with_required()
    trace.retrieved_experience_ids = ["exp_001"]
    _check_evidence_references_hybrid(out, _minimal_ctx(["a"]), trace)  # no raise

def test_collect_all_refs_includes_decision_basis_and_candidate_refs():
    out = _minimal_output()
    refs = _collect_all_refs(out)
    sources = {r.source for r in refs}
    assert "dataset_profile" in sources  # from decision_basis.primary_evidence
    assert "registry" in sources         # from candidate.evidence_refs
```

### Step 2 — Implement `src/mlops_agents/planning/validation.py`

```python
"""Validation chain for the Planner Agent's PlannerOutput.

Hybrid validation (A1):
- Hard invariants (registry, exhaustiveness, plan integrity) → deterministic context
- Citation honesty (experience + rule refs) → agent's observed ToolTrace
"""
from __future__ import annotations
from typing import Iterable

from mlops_agents.config.settings import settings
from mlops_agents.contracts.planner import (
    EvidenceReference, PlannerOutput, PlannerValidationContext,
)
from mlops_agents.contracts.training import TrainingPlan
from mlops_agents.planning.trace import ToolTrace


REQUIRED_TOOLS = {
    "list_available_models",
    "retrieve_similar_experiences",
    "retrieve_ml_knowledge",
}
ALLOWED_VAL_STRATEGIES = {"single_split", "rolling_window", "expanding_window"}
ALLOWED_EXOG_STRATEGIES = {"naive_carry", "ets", "auto_arima", "drop"}


class PlannerValidationError(Exception):
    """Raised when validation rejects a PlannerOutput. Caught by planner_node for retry."""


def _collect_all_refs(output: PlannerOutput) -> list[EvidenceReference]:
    """Union of every EvidenceReference appearing anywhere in PlannerOutput."""
    refs: list[EvidenceReference] = []
    refs.extend(output.evidence_used)
    refs.extend(output.decision_basis.primary_evidence)
    refs.extend(output.decision_basis.secondary_evidence)
    for c in output.plan.candidates:
        refs.extend(c.evidence_refs)
    for r in output.plan.models_not_recommended:
        refs.extend(r.evidence_refs)
    for conflict in output.evidence_conflicts:
        refs.extend(conflict.conflicting_evidence_refs)
    return refs


def _has_registry_self_ref(refs: Iterable[EvidenceReference], model_key: str) -> bool:
    return any(r.source == "registry" and r.source_id == model_key for r in refs)


def _check_plan_integrity(
    output: PlannerOutput,
    trace: ToolTrace,
    ctx: PlannerValidationContext,
) -> None:
    """All non-citation invariants. Raises PlannerValidationError on first failure."""
    # 1. Required tools
    missing = REQUIRED_TOOLS - set(trace.called_tools)
    if missing:
        raise PlannerValidationError(f"agent skipped required tools: {sorted(missing)}")

    # 2. Tool-call budget — global + per-tool inspect cap (defense-in-depth: tools
    # already reject calls past the cap, so this should never fire in practice).
    if trace.tool_call_count > settings.planner_max_tool_calls:
        raise PlannerValidationError(
            f"agent exceeded planner_max_tool_calls: {trace.tool_call_count} > "
            f"{settings.planner_max_tool_calls}"
        )
    if trace.inspect_model_details_count > settings.planner_max_inspect_calls:
        raise PlannerValidationError(
            f"agent exceeded planner_max_inspect_calls: "
            f"{trace.inspect_model_details_count} > {settings.planner_max_inspect_calls}"
        )

    # 3. Priority uniqueness + monotonicity
    priorities = [c.priority for c in output.plan.candidates]
    if any(p < 1 for p in priorities):
        raise PlannerValidationError(f"candidate priorities must be >= 1, got {priorities}")
    if len(set(priorities)) != len(priorities):
        raise PlannerValidationError(f"candidate priorities must be unique, got {priorities}")

    # 4. No candidate↔rejected overlap
    cand_set = {c.model_key for c in output.plan.candidates}
    rej_set = {r.model_key for r in output.plan.models_not_recommended}
    overlap = cand_set & rej_set
    if overlap:
        raise PlannerValidationError(
            f"models overlap candidates and rejected: {sorted(overlap)}"
        )

    # 5. Forecasting-specific settings
    if ctx.problem_type == "forecasting":
        fc = getattr(output.plan, "forecasting_settings", None)
        if fc is None:
            raise PlannerValidationError("forecasting plan missing forecasting_settings")
        if getattr(fc, "validation_strategy", None) not in ALLOWED_VAL_STRATEGIES:
            raise PlannerValidationError(
                f"invalid validation_strategy: {getattr(fc, 'validation_strategy', None)!r}. "
                f"Allowed: {sorted(ALLOWED_VAL_STRATEGIES)}"
            )
        exog_strats = getattr(fc, "exog_strategies", None) or {}
        known_future = set(ctx.task_metadata.get("known_future_columns", []))
        for col, strat in exog_strats.items():
            if strat not in ALLOWED_EXOG_STRATEGIES:
                raise PlannerValidationError(
                    f"invalid exog strategy {strat!r} for column {col!r}. "
                    f"Allowed: {sorted(ALLOWED_EXOG_STRATEGIES)}"
                )
            if col in known_future:
                raise PlannerValidationError(
                    f"known_future column {col!r} cannot appear in per-column "
                    f"unknown-future strategies (no 'drop' loophole)"
                )

    # 6. Per-candidate registry self-citation
    for c in output.plan.candidates:
        if not _has_registry_self_ref(c.evidence_refs, c.model_key):
            raise PlannerValidationError(
                f"candidate {c.model_key!r} missing registry self-citation "
                f"(evidence_refs must include source=registry, source_id={c.model_key!r})"
            )
    for r in output.plan.models_not_recommended:
        if not _has_registry_self_ref(r.evidence_refs, r.model_key):
            raise PlannerValidationError(
                f"rejected model {r.model_key!r} missing registry self-citation"
            )


def _check_plan_exhaustiveness(plan: TrainingPlan, available_model_keys: list[str]) -> None:
    accounted = (
        {c.model_key for c in plan.candidates}
        | {r.model_key for r in plan.models_not_recommended}
    )
    missing = set(available_model_keys) - accounted
    if missing:
        raise PlannerValidationError(
            f"models not classified as either candidates or rejected: {sorted(missing)}. "
            f"Every available model must be explicitly included."
        )


def _check_evidence_references_hybrid(
    output: PlannerOutput,
    ctx: PlannerValidationContext,
    trace: ToolTrace,
) -> None:
    """Hybrid: registry refs validated against deterministic context;
    experience/rule refs validated against agent's observed ToolTrace."""
    for ref in _collect_all_refs(output):
        if ref.source in ("dataset_profile", "task_metadata"):
            if ref.source_id is not None:
                raise PlannerValidationError(
                    f"{ref.source} ref must have source_id=None, got {ref.source_id!r}"
                )
        elif ref.source == "registry":
            if not ref.source_id:
                raise PlannerValidationError("registry ref requires non-empty source_id (model_key)")
            if ref.source_id not in ctx.available_model_keys:
                raise PlannerValidationError(
                    f"registry ref {ref.source_id!r} not in deterministic registry"
                )
        elif ref.source == "experience":
            if not ref.source_id:
                raise PlannerValidationError("experience ref requires non-empty source_id")
            if ref.source_id not in trace.retrieved_experience_ids:
                raise PlannerValidationError(
                    f"experience ref {ref.source_id!r} was never retrieved by the agent"
                )
        elif ref.source == "rule":
            if not ref.source_id:
                raise PlannerValidationError("rule ref requires non-empty source_id")
            if ref.source_id not in trace.retrieved_rule_ids:
                raise PlannerValidationError(
                    f"rule ref {ref.source_id!r} was never retrieved by the agent"
                )
```

### Step 3 — Run + commit

```
uv run pytest tests/test_planning/test_validation.py -v
git add src/mlops_agents/planning/validation.py tests/test_planning/test_validation.py
git commit -m "feat(planning): add validation (integrity, exhaustiveness, hybrid evidence refs)"
```

---

## Task 4.3 — Conflict detection + `_check_conflict_resolution_present_if_flagged`

**Files:**
- Extend: `src/mlops_agents/planning/validation.py`
- Create: `tests/test_planning/test_conflict_detection.py`

### Step 1 — Failing test

```python
import pytest
from mlops_agents.planning.validation import (
    _detect_conflicts, detect_soft_conflicts,
    _check_conflict_resolution_present_if_flagged,
    PlannerValidationError,
)
from mlops_agents.contracts.planner import (
    CandidateSpec, RejectedModelSpec, EvidenceReference, DecisionBasis,
    PlannerOutput, PlannerValidationContext, EvidenceConflict, ExperienceSummary,
)
from mlops_agents.contracts.training import TrainingPlan
from mlops_agents.planning.trace import ToolTrace


def _es(eid, best_model="extra_trees"):
    return ExperienceSummary(
        experience_id=eid, similarity_score=0.7, relevance_tier="high",
        matched_buckets=[], mismatched_buckets=[], target_scale_note=None,
        dataset_summary="", models_trained=[best_model], best_model=best_model,
        validation_score=0.5, metric_name="rmse", candidate_results=[],
    )

def _registry_ref(key): return EvidenceReference(source="registry", source_id=key)
def _exp_ref(eid): return EvidenceReference(source="experience", source_id=eid)

def _output(candidates, rejected=None, evidence_used=None, conflicts=None):
    return PlannerOutput(
        planning_analysis="ok",
        decision_basis=DecisionBasis(
            primary_evidence=[EvidenceReference(source="dataset_profile", source_id=None)],
            secondary_evidence=[], final_strategy="s",
        ),
        evidence_used=evidence_used or [],
        evidence_conflicts=conflicts or [],
        risks_or_warnings=[],
        plan=TrainingPlan(candidates=candidates, models_not_recommended=rejected or [], trial_budget=10),
    )


def test_detect_hard_conflict_cited_experience_winner_not_selected():
    cand = CandidateSpec(model_key="ets", priority=1, reason="ok",
                          evidence_refs=[_registry_ref("ets")])
    out = _output(candidates=[cand], evidence_used=[_exp_ref("e1")])
    ctx = PlannerValidationContext(
        problem_type="forecasting", task_metadata={},
        available_model_keys=["ets", "extra_trees"], available_model_specs=[],
        similar_experiences=[_es("e1", best_model="extra_trees")],
        matched_rules=[], rules_by_id={},
    )
    trace = ToolTrace(retrieved_experience_ids=["e1"])
    conflicts = _detect_conflicts(ctx, trace, out.plan, out)
    assert any(c["type"] == "cited_experience_winner_not_selected" for c in conflicts)

def test_soft_conflict_only_when_retrieved_but_not_cited():
    cand = CandidateSpec(model_key="ets", priority=1, reason="ok",
                          evidence_refs=[_registry_ref("ets")])
    out = _output(candidates=[cand])  # no cited experiences
    ctx = PlannerValidationContext(
        problem_type="forecasting", task_metadata={},
        available_model_keys=["ets", "extra_trees"], available_model_specs=[],
        similar_experiences=[_es("e1", best_model="extra_trees")],
        matched_rules=[], rules_by_id={},
    )
    trace = ToolTrace(retrieved_experience_ids=["e1"])
    soft = detect_soft_conflicts(ctx, trace, out.plan, out)
    assert any(c["type"] == "retrieved_experience_winner_not_selected" for c in soft)
    # And no hard conflict
    hard = _detect_conflicts(ctx, trace, out.plan, out)
    assert not any(c["type"] == "cited_experience_winner_not_selected" for c in hard)

def test_hard_conflict_resolution_required():
    cand = CandidateSpec(model_key="ets", priority=1, reason="ok",
                          evidence_refs=[_registry_ref("ets")])
    out = _output(candidates=[cand], evidence_used=[_exp_ref("e1")])
    ctx = PlannerValidationContext(
        problem_type="forecasting", task_metadata={},
        available_model_keys=["ets", "extra_trees"], available_model_specs=[],
        similar_experiences=[_es("e1", best_model="extra_trees")],
        matched_rules=[], rules_by_id={},
    )
    trace = ToolTrace(retrieved_experience_ids=["e1"])
    # Empty evidence_conflicts despite flagged hard conflict → raise
    with pytest.raises(PlannerValidationError, match="evidence_conflicts is empty"):
        _check_conflict_resolution_present_if_flagged(out, ctx, trace)

def test_hard_conflict_with_resolution_passes():
    cand = CandidateSpec(model_key="ets", priority=1, reason="ok",
                          evidence_refs=[_registry_ref("ets")])
    resolved = EvidenceConflict(
        summary="extra_trees won in cited experience but rule prefers statistical models",
        affected_models=["extra_trees"],
        conflicting_evidence_refs=[_exp_ref("e1")],
        resolution="short history; statistical baseline safer",
    )
    out = _output(candidates=[cand], evidence_used=[_exp_ref("e1")], conflicts=[resolved])
    ctx = PlannerValidationContext(
        problem_type="forecasting", task_metadata={},
        available_model_keys=["ets", "extra_trees"], available_model_specs=[],
        similar_experiences=[_es("e1", best_model="extra_trees")],
        matched_rules=[], rules_by_id={},
    )
    trace = ToolTrace(retrieved_experience_ids=["e1"])
    _check_conflict_resolution_present_if_flagged(out, ctx, trace)  # no raise
```

### Step 2 — Extend `src/mlops_agents/planning/validation.py`

Append:

```python
def _detect_conflicts(
    ctx: PlannerValidationContext,
    trace: ToolTrace,
    plan: TrainingPlan,
    output: PlannerOutput,
) -> list[dict]:
    """Deterministic HARD conflict detection. Returns list of flagged conflicts."""
    hard: list[dict] = []
    candidate_keys = {c.model_key for c in plan.candidates}
    rejected_keys = {r.model_key for r in plan.models_not_recommended}

    cited_experience_ids = {
        ref.source_id for ref in _collect_all_refs(output)
        if ref.source == "experience" and ref.source_id
    }
    cited_winners = {
        e.best_model for e in ctx.similar_experiences
        if e.experience_id in cited_experience_ids and e.best_model
    }
    omitted_cited = cited_winners - candidate_keys
    if omitted_cited:
        hard.append({
            "type": "cited_experience_winner_not_selected",
            "models": sorted(omitted_cited),
            "severity": "hard",
        })

    cited_rule_ids = {
        ref.source_id for ref in _collect_all_refs(output)
        if ref.source == "rule" and ref.source_id
    }
    for rid in cited_rule_ids:
        rule = ctx.rules_by_id.get(rid)
        if not rule:
            continue
        avoid_in_cands = set(rule.get("avoid_or_deprioritize", []) or []) & candidate_keys
        prefer_in_rej = set(rule.get("prefer", []) or []) & rejected_keys
        if avoid_in_cands:
            hard.append({
                "type": "cited_rule_avoid_violated", "rule_id": rid,
                "models": sorted(avoid_in_cands), "severity": "hard",
            })
        if prefer_in_rej:
            hard.append({
                "type": "cited_rule_prefer_rejected", "rule_id": rid,
                "models": sorted(prefer_in_rej), "severity": "hard",
            })
    return hard


def detect_soft_conflicts(
    ctx: PlannerValidationContext,
    trace: ToolTrace,
    plan: TrainingPlan,
    output: PlannerOutput,
) -> list[dict]:
    """Non-blocking conflicts surfaced as info in the UI. Excludes anything already in hard."""
    soft: list[dict] = []
    candidate_keys = {c.model_key for c in plan.candidates}

    retrieved_winners = {
        e.best_model for e in ctx.similar_experiences
        if e.experience_id in trace.retrieved_experience_ids and e.best_model
    }
    cited_experience_ids = {
        ref.source_id for ref in _collect_all_refs(output)
        if ref.source == "experience" and ref.source_id
    }
    cited_winners = {
        e.best_model for e in ctx.similar_experiences
        if e.experience_id in cited_experience_ids and e.best_model
    }
    soft_omitted = (retrieved_winners - cited_winners) - candidate_keys
    if soft_omitted:
        soft.append({
            "type": "retrieved_experience_winner_not_selected",
            "models": sorted(soft_omitted),
            "summary": (
                f"{len(soft_omitted)} model(s) won in retrieved experiences but were not "
                f"cited or selected: {sorted(soft_omitted)}."
            ),
        })
    return soft


def _check_conflict_resolution_present_if_flagged(
    output: PlannerOutput,
    ctx: PlannerValidationContext,
    trace: ToolTrace,
) -> None:
    flagged = _detect_conflicts(ctx, trace, output.plan, output)
    if flagged and not output.evidence_conflicts:
        raise PlannerValidationError(
            f"deterministic conflict detector flagged {len(flagged)} conflict(s) but "
            f"evidence_conflicts is empty. Flagged: {flagged}"
        )
    for c in output.evidence_conflicts:
        if not c.resolution.strip():
            raise PlannerValidationError(
                f"evidence_conflict for {c.affected_models} has empty resolution"
            )
```

### Step 3 — Run + commit

```
uv run pytest tests/test_planning/test_conflict_detection.py -v
git add src/mlops_agents/planning/validation.py tests/test_planning/test_conflict_detection.py
git commit -m "feat(planning): add deterministic conflict detection (hard + soft) + resolution check"
```

---

# Phase 5 — Agent + node integration

## Task 5.1 — `build_planner_agent` factory

**Files:**
- Create: `src/mlops_agents/planning/agent.py`
- Create: `tests/test_planning/test_agent.py`

### Step 1 — Failing test (mock LLM)

```python
from unittest.mock import patch, MagicMock
from mlops_agents.planning.agent import build_planner_agent
from mlops_agents.planning.trace import ToolTrace
from mlops_agents.planning.tools import build_planner_tools

def test_build_planner_agent_returns_compiled_graph():
    trace = ToolTrace()
    tools = build_planner_tools({}, {}, "forecasting", trace)
    # No real LLM call — just verify the agent builds without exception
    with patch("mlops_agents.utils.llm.get_llm") as mock_get_llm:
        mock_get_llm.return_value = MagicMock()
        agent = build_planner_agent(tools)
        assert agent is not None
        # CompiledStateGraph has an invoke method
        assert hasattr(agent, "invoke")
```

### Step 2 — Implement

`src/mlops_agents/planning/agent.py`:

```python
"""build_planner_agent — wraps langchain's create_agent with response_format=PlannerOutput."""
from langchain.agents import create_agent

from mlops_agents.contracts.planner import PlannerOutput
from mlops_agents.prompts import get_prompt
from mlops_agents.utils.llm import get_llm


def build_planner_agent(tools: list):
    """Build the planner ReAct agent. Tools must be closure-bound by build_planner_tools(...)
    in the caller so the agent never sees raw profile/task_metadata in tool args."""
    return create_agent(
        model=get_llm("planner", max_tokens=16000),
        tools=tools,
        system_prompt=get_prompt("planner").template,
        response_format=PlannerOutput,
        name="planner",
    )
```

### Step 3 — Run + commit

```
uv run pytest tests/test_planning/test_agent.py -v
git add src/mlops_agents/planning/agent.py tests/test_planning/test_agent.py
git commit -m "feat(planning): add build_planner_agent — create_agent with response_format=PlannerOutput"
```

---

## Task 5.2 — `planner_node` with retry loop + state update

**Files:**
- Create: `src/mlops_agents/planning/node.py`
- Create: `src/mlops_agents/planning/prompts.py`
- Create: `tests/test_planning/test_node.py`

### Step 1 — Failing test (mock agent + LLM)

```python
import json
import pytest
from unittest.mock import patch, MagicMock
from mlops_agents.planning.node import planner_node
from mlops_agents.contracts.planner import (
    CandidateSpec, RejectedModelSpec, EvidenceReference, DecisionBasis, PlannerOutput,
)
from mlops_agents.contracts.training import TrainingPlan

def _make_output_for(problem_type="forecasting"):
    refs = [EvidenceReference(source="registry", source_id="ets")]
    return PlannerOutput(
        planning_analysis="ok",
        decision_basis=DecisionBasis(
            primary_evidence=[EvidenceReference(source="dataset_profile", source_id=None)],
            secondary_evidence=[], final_strategy="prefer ets",
        ),
        evidence_used=[], evidence_conflicts=[], risks_or_warnings=[],
        plan=TrainingPlan(
            candidates=[CandidateSpec(model_key="ets", priority=1, reason="ok", evidence_refs=refs)],
            models_not_recommended=[
                # need to cover every available model — use registry-aware fixture in real test
            ],
            trial_budget=10,
        ),
    )

@patch("mlops_agents.planning.node._check_conflict_resolution_present_if_flagged")
@patch("mlops_agents.planning.node._check_evidence_references_hybrid")
@patch("mlops_agents.planning.node._check_plan_exhaustiveness")
@patch("mlops_agents.planning.node._check_plan_integrity")
@patch("mlops_agents.planning.node.build_planner_validation_context")
@patch("mlops_agents.planning.node.build_planner_agent")
@patch("mlops_agents.planning.node.build_planner_tools")
@patch("mlops_agents.planning.node.build_dataset_profile")
def test_planner_node_happy_path(
    mock_profile, mock_build_tools, mock_build_agent, mock_build_ctx,
    mock_integrity, mock_exhaust, mock_evidence, mock_conflict, tmp_path,
):
    """Validation has its own dedicated tests — patch the checks here so we test ONLY
    the node's orchestration (build context → run agent → assemble Command). This is
    cleaner than rebinding ToolTrace in the node module, which is brittle and obscures intent."""
    from mlops_agents.training.profiler import DatasetProfile
    mock_profile.return_value = DatasetProfile.model_validate({})  # minimal; adjust if required fields exist

    mock_ctx = MagicMock()
    mock_ctx.problem_type = "forecasting"
    mock_ctx.task_metadata = {}
    mock_ctx.available_model_keys = ["ets"]
    mock_ctx.similar_experiences = []
    mock_ctx.matched_rules = []
    mock_ctx.rules_by_id = {}
    mock_build_ctx.return_value = mock_ctx
    mock_build_tools.return_value = []

    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {
        "structured_response": _make_output_for("forecasting"),
        "messages": [],
    }
    mock_build_agent.return_value = fake_agent

    csv = tmp_path / "p.csv"
    csv.write_text("ds,y\n2024-01-01,1\n2024-01-02,2\n")
    state = {
        "processed_dataset_path": str(csv),
        "problem_type": "forecasting",
        "task_metadata": {"target_column": "y", "datetime_column": "ds"},
    }
    result = planner_node(state)

    assert result.goto == "workflow_controller"
    assert "training_plan" in result.update
    assert "planner_tool_trace" in result.update
    # All four validation checks were invoked exactly once
    mock_integrity.assert_called_once()
    mock_exhaust.assert_called_once()
    mock_evidence.assert_called_once()
    mock_conflict.assert_called_once()
```

### Step 2 — Implement

`src/mlops_agents/planning/prompts.py`:

```python
"""Message builders for the planner agent."""
from typing import Any
import json

from langchain_core.messages import HumanMessage, SystemMessage


def format_planner_inputs(
    dataset_profile: dict[str, Any],
    task_metadata: dict[str, Any],
    problem_type: str,
) -> str:
    """Compact human-readable summary of inputs to seed the agent's reasoning."""
    return (
        f"problem_type: {problem_type}\n\n"
        f"task_metadata:\n{json.dumps(task_metadata, indent=2, default=str)}\n\n"
        f"dataset_profile:\n{json.dumps(dataset_profile, indent=2, default=str)}\n\n"
        f"Use the tools to retrieve evidence, then produce the PlannerOutput."
    )


def build_retry_message(last_error: str) -> HumanMessage:
    return HumanMessage(
        content=(
            f"Your previous PlannerOutput was rejected by validation: {last_error}\n\n"
            f"Produce a corrected PlannerOutput. You may call tools again as needed; "
            f"the retry uses a fresh tool trace."
        )
    )
```

`src/mlops_agents/planning/node.py`:

```python
"""planner_node — entry point, retry orchestration, validation."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command

from mlops_agents.config.settings import settings
from mlops_agents.planning.agent import build_planner_agent
from mlops_agents.planning.context import build_planner_validation_context
from mlops_agents.planning.prompts import build_retry_message, format_planner_inputs
from mlops_agents.planning.tools import build_planner_tools
from mlops_agents.planning.trace import ToolTrace
from mlops_agents.planning.validation import (
    PlannerValidationError, _check_evidence_references_hybrid, _check_plan_exhaustiveness,
    _check_plan_integrity, _check_conflict_resolution_present_if_flagged,
    _detect_conflicts, detect_soft_conflicts,
)
from mlops_agents.prompts import get_prompt
from mlops_agents.state.agent_state import AgentState
from mlops_agents.training.profiler import build_dataset_profile
from mlops_agents.utils.logging import get_logger

logger = get_logger(__name__)


class PlannerError(Exception):
    """Raised when the planner agent fails after the retry attempt."""


def planner_node(state: AgentState) -> Command[Literal["workflow_controller"]]:
    """Entry: build profile + validation context once, run agent up to 2 attempts."""
    processed_path = Path(state["processed_dataset_path"])
    problem_type: str = state.get("problem_type", "")
    task_meta: dict[str, Any] = state.get("task_metadata") or {}

    profile = build_dataset_profile(
        processed_path, {**task_meta, "problem_type": problem_type}
    ).model_dump()
    profile = {k: v for k, v in profile.items() if v is not None}

    validation_ctx = build_planner_validation_context(profile, task_meta, problem_type)
    system_prompt = get_prompt("planner").template

    output = None
    trace = ToolTrace()  # placeholder; replaced inside loop
    last_error = ""
    retry_used = False

    for attempt in range(2):
        trace = ToolTrace()
        tools = build_planner_tools(profile, task_meta, problem_type, trace)
        agent = build_planner_agent(tools)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=format_planner_inputs(profile, task_meta, problem_type)),
        ]
        if attempt == 1:
            messages.append(build_retry_message(last_error))
            retry_used = True

        try:
            result = agent.invoke(
                {"messages": messages},
                config={"recursion_limit": settings.planner_max_iterations},
            )
            output = result.get("structured_response")
            if output is None:
                raise PlannerValidationError(
                    "agent returned no structured_response — model failed to produce PlannerOutput"
                )

            _check_plan_integrity(output, trace, validation_ctx)
            _check_plan_exhaustiveness(output.plan, validation_ctx.available_model_keys)
            _check_evidence_references_hybrid(output, validation_ctx, trace)
            _check_conflict_resolution_present_if_flagged(output, validation_ctx, trace)
            break  # success
        except (PlannerValidationError, ValueError) as exc:
            last_error = str(exc)
            logger.warning(f"[planner] attempt {attempt + 1} failed: {last_error}")
            if attempt == 1:
                raise PlannerError(f"Planner failed after retry: {last_error}") from exc

    # Sort candidates by priority (deterministic order for executor)
    sorted_candidates = sorted(output.plan.candidates, key=lambda c: c.priority)
    output.plan.candidates = sorted_candidates

    soft = detect_soft_conflicts(validation_ctx, trace, output.plan, output)
    cited_experience_ids = sorted({
        r.source_id for r in _collect_refs_for_record(output)
        if r.source == "experience" and r.source_id
    })
    cited_rule_ids = sorted({
        r.source_id for r in _collect_refs_for_record(output)
        if r.source == "rule" and r.source_id
    })

    planner_status = "retry_ok" if retry_used else "ok"
    record = _build_planner_output_record(
        output, trace, validation_ctx, soft,
        cited_experience_ids, cited_rule_ids, planner_status, last_error,
    )

    logger.info(
        f"[planner] status={planner_status} candidates={len(output.plan.candidates)} "
        f"rejected={len(output.plan.models_not_recommended)} tool_calls={trace.tool_call_count}"
    )

    return Command(
        goto="workflow_controller",
        update={
            "planner_analysis": output.planning_analysis,
            "planner_evidence_used": [e.model_dump() for e in output.evidence_used],
            "planner_warnings": output.risks_or_warnings,
            "planner_status": planner_status,
            "planner_retry_used": retry_used,
            "training_plan": output.plan.model_dump(),
            "planner_tool_trace": trace.model_dump(),
            "planner_validation_context": _audit_subset(validation_ctx),
            "_planner_output_record": record,
        },
    )


# Helpers — placed below so test file imports cleanly

def _collect_refs_for_record(output):
    from mlops_agents.planning.validation import _collect_all_refs
    return _collect_all_refs(output)


def _audit_subset(ctx) -> dict:
    """Compact, JSON-serializable subset of validation context for state/audit."""
    return {
        "problem_type": ctx.problem_type,
        "available_model_keys": list(ctx.available_model_keys),
        "matched_rule_ids": [r["rule_id"] for r in ctx.matched_rules],
        "similar_experience_ids": [e.experience_id for e in ctx.similar_experiences],
    }


def _build_planner_output_record(
    output, trace, validation_ctx, soft_conflicts,
    cited_experience_ids, cited_rule_ids, status: str, last_error: str,
) -> dict:
    """Compose the rich record consumed by the SSE pipeline & frontend PlannerPanel."""
    return {
        "planner_status": status,
        "retry_used": status == "retry_ok",
        "planning_analysis": output.planning_analysis,
        "decision_basis": output.decision_basis.model_dump(),
        "evidence_used": [e.model_dump() for e in output.evidence_used],
        "evidence_conflicts": [c.model_dump() for c in output.evidence_conflicts],
        "soft_conflicts": soft_conflicts,
        "risks_or_warnings": output.risks_or_warnings,
        "validation_errors": [last_error] if status == "retry_ok" else [],
        "plan_summary": {
            "candidate_rationales": [c.model_dump() for c in output.plan.candidates],
            "rejected_model_rationales": [r.model_dump() for r in output.plan.models_not_recommended],
            "candidate_models": [c.model_key for c in output.plan.candidates],
            "models_not_recommended": [r.model_key for r in output.plan.models_not_recommended],
        },
        "cited_experience_ids": cited_experience_ids,
        "cited_rule_ids": cited_rule_ids,
        "tool_trace": trace.model_dump(),
        "retrieved_experiences": [e.model_dump() for e in validation_ctx.similar_experiences],
        "matched_rules": validation_ctx.matched_rules,
        "prompt_version": "model_planner_v2",
    }
```

### Step 3 — Run + commit

```
uv run pytest tests/test_planning/test_node.py -v
git add src/mlops_agents/planning/node.py src/mlops_agents/planning/prompts.py tests/test_planning/test_node.py
git commit -m "feat(planning): add planner_node with retry loop, validation, and rich record"
```

---

## Task 5.3 — Shim + registry rewire + graph import update + prompt rewrite

**Files:**
- Modify: `src/mlops_agents/agents/planner.py` (shrink to shim)
- Modify: `src/mlops_agents/agents/registry.py`
- Modify: `src/mlops_agents/graphs/mlops_graph.py`
- Rewrite: `src/mlops_agents/prompts/planner.yaml`

### Step 1 — Replace `src/mlops_agents/agents/planner.py` body

After Tasks 2.3 / 5.x landed, the legacy file still has `build_planner_context`, `_to_experience_summary`, `_check_*` helpers, and `planner_node`. Shrink it to a thin shim that re-exports the new node, keeping `_to_experience_summary` (since Task 4.1's `context.py` imports it):

```python
"""DEPRECATED: legacy import path. Real implementation lives in mlops_agents.planning.*"""
from mlops_agents.planning.node import planner_node, PlannerError  # noqa: F401
```

That's the entire file. All helpers (`_to_experience_summary`, `_check_*`, `build_planner_context`, the old `build_planner_agent`) are deleted — `_to_experience_summary` moved to `experience/retrieval.py` in Task 2.3, the rest moved to `planning/*` modules.

### Step 2 — Update `agents/registry.py`

Locate the planner entry. It currently calls `build_planner_agent()` from the old module. Today the planner is registered as a one-shot LLM-with-structured-output; we're moving to a stateful agent that requires per-call tools.

The cleanest change: have the registry NOT register the planner as a pre-built agent. The planner is now stateful (needs profile + task_meta + trace per run). The graph node builds it on demand. Update registry:

```python
# in get_agent(name)
if name == "planner":
    raise ValueError(
        "Planner is now built per-run inside planner_node via build_planner_agent(tools). "
        "It is not a pre-built cached agent. Use planning.node.planner_node instead."
    )
```

### Step 3 — Update `graphs/mlops_graph.py`

Change the planner import to the new module path. Search for `from mlops_agents.agents.planner import planner_node` (or wherever it's currently imported) and replace with:

```python
from mlops_agents.planning.node import planner_node, PlannerError
```

If the file also imports `build_planner_context` or `_to_experience_summary`, drop those imports — they're internal to planning now.

### Step 4 — Rewrite `src/mlops_agents/prompts/planner.yaml`

Replace the entire file with:

```yaml
name: planner
description: Model Planning Agent prompt (v2 — tool-using)
template: |
  You are the Model Planner Agent.

  Your job: produce a validated TrainingPlan.

  ## You have four tools
  - list_available_models() — start here. Lists the registry universe for this problem_type.
  - retrieve_similar_experiences(top_k=5) — past runs by deterministic similarity to this dataset.
  - retrieve_ml_knowledge() — static ML rules matching this dataset profile.
  - inspect_model_details(model_key) — deep info on one model. Use sparingly; only when needed.

  ## You must
  - Call list_available_models() at least once.
  - Call retrieve_similar_experiences() at least once.
  - Call retrieve_ml_knowledge() at least once.
  - Recommend only models returned by list_available_models() for this problem_type.
  - Classify EVERY available model as either a candidate or a rejected model.
  - For every candidate: provide reason, evidence_refs, risks. evidence_refs MUST include
    at least one registry reference where source_id == the candidate's model_key.
  - For every rejected model: provide reason and evidence_refs (same registry rule). Optionally
    add reconsider_if when the rejection is conditional on circumstances that could change.
  - If a model is selected mainly because of a rule or registry prior (no supporting experience),
    state that explicitly in reason or risks. Do not imply empirical support that doesn't exist.
  - Cite only experiences and rules you actually retrieved this run.
  - Provide decision_basis with primary_evidence and secondary_evidence as lists of
    EvidenceReference objects (not free-text labels), plus a final_strategy narrative.
  - If you observe a conflict between retrieved experiences, retrieved rules, registry guidance,
    or your selected/rejected models, fill evidence_conflicts with at least one entry.
    A deterministic post-check will also flag conflicts; if it does and you did not provide
    a resolution, the planner run will be rejected and retried.

  ## You must NOT
  - Recommend a model_key not in the registry.
  - Cite an experience or rule you didn't retrieve via tools.
  - Call training, evaluation, MLflow, or deployment tools — you don't have any.
  - Exceed ~5 tool calls. Default sequence: list_available_models, retrieve_similar_experiences,
    retrieve_ml_knowledge, optional inspect_model_details (1–2), then produce output.

  ## Forecasting-specific guidance
  - If history_length is very_short or short: prefer single_split validation; include
    statistical baselines (naive, seasonal_naive, ets, auto_arima where registered);
    avoid high-complexity supervised models unless similar experiences strongly support them.
  - If expected_drift is high and history is sufficient: prefer rolling_window.
  - For unknown_future exogenous columns: choose extension strategies from
    {naive_carry, ets, auto_arima, drop}.
  - known_future variables must not appear in per-column unknown-future overrides at all.

  ## Output
  Produce a PlannerOutput. The schema is enforced — fill every required field.
  Conservative beats optimistic when evidence is weak or conflicting.
```

### Step 5 — Run full backend tests + commit

```
uv run pytest -m "not integration"
git add src/mlops_agents/agents/planner.py src/mlops_agents/agents/registry.py src/mlops_agents/graphs/mlops_graph.py src/mlops_agents/prompts/planner.yaml
git commit -m "refactor(planner): wire new planning module into graph; rewrite prompt for tool-using agent"
```

---

# Phase 6 — Backend SSE & state

## Task 6.1 — `agent_state.py` field additions

**Files:**
- Modify: `src/mlops_agents/state/agent_state.py`

### Step 1 — Add fields to `AgentState` TypedDict

```python
class AgentState(TypedDict, total=False):
    # ... existing fields ...
    planner_tool_trace: dict
    planner_validation_context: dict
```

### Step 2 — Commit (no behavior change)

```
git add src/mlops_agents/state/agent_state.py
git commit -m "feat(state): add planner_tool_trace + planner_validation_context state fields"
```

---

## Task 6.2 — `pipeline.py` `_planner_output_record` enrichment

**Files:**
- Modify: `api/services/pipeline.py`

### Step 1 — Update `_build_planner_ctx_event`

Locate the function `_build_planner_ctx_event(rec)` (around line 20 of `api/services/pipeline.py`). It currently transforms the legacy record into `PlannerContextData`. Extend it to forward the new v2 fields:

```python
def _build_planner_ctx_event(rec: dict[str, Any]) -> dict[str, Any]:
    """Transform _planner_output_record into the shape PlannerContextData expects."""
    # ... existing evidence_used / retrieved_experiences / matched_rules transforms ...

    # NEW v2 fields — forwarded verbatim from planner_node's record
    decision_basis = rec.get("decision_basis", {}) or {
        "primary_evidence": [], "secondary_evidence": [], "final_strategy": "",
    }
    evidence_conflicts = rec.get("evidence_conflicts", []) or []
    soft_conflicts = rec.get("soft_conflicts", []) or []
    cited_experience_ids = rec.get("cited_experience_ids", []) or []
    cited_rule_ids = rec.get("cited_rule_ids", []) or []

    plan_summary_raw = rec.get("plan_summary", {}) or {}
    plan_summary = {
        "candidate_models": plan_summary_raw.get("candidate_models", []),
        "models_not_recommended": plan_summary_raw.get("models_not_recommended", []),
        "candidate_rationales": plan_summary_raw.get("candidate_rationales", []),
        "rejected_model_rationales": plan_summary_raw.get("rejected_model_rationales", []),
    }

    planner_status = rec.get("planner_status", "ok")

    return {
        "retrieved_experiences": retrieved_experiences,  # existing
        "matched_rules": matched_rules,                  # existing
        "evidence_used": evidence_used,                  # existing
        "planning_analysis": rec.get("planning_analysis", ""),
        "plan_summary": plan_summary,
        "warnings": rec.get("risks_or_warnings", []),
        # NEW v2 fields
        "decision_basis": decision_basis,
        "evidence_conflicts": evidence_conflicts,
        "soft_conflicts": soft_conflicts,
        "cited_experience_ids": cited_experience_ids,
        "cited_rule_ids": cited_rule_ids,
        "planner_status": planner_status,
    }
```

### Step 2 — Run + commit

```
uv run pytest -m "not integration"
git add api/services/pipeline.py
git commit -m "feat(api): forward planner v2 fields (decision_basis, evidence_conflicts, soft_conflicts, etc.) in planner_context event"
```

---

## Task 6.3 — `run_info` event gets `node_categories`

**Files:**
- Modify: `api/services/pipeline.py`

### Step 1 — Update `info_event` in `pipeline_task`

Locate the `info_event` dict construction (around line 127). Import the taxonomy and extend the payload:

```python
from mlops_agents.agents.taxonomy import NODE_CATEGORIES

# ... existing problem_type parsing ...

info_event: dict = {
    "type": "run_info",
    "agent": "system",
    "timestamp_ms": time.time() * 1000,
    "data": {
        "models": {
            "data_validator": settings.openai_model_data_validator,
            "planner":        settings.openai_model_planner,
            "report_writer":  settings.openai_model_report_writer,
        },
        "problem_type": pt,
        "node_categories": {
            "agents":        NODE_CATEGORIES["agents"],
            "llm_nodes":     NODE_CATEGORIES["llm_nodes"],
            "deterministic": NODE_CATEGORIES["deterministic"],
        },
    },
}
```

### Step 2 — Commit

```
git add api/services/pipeline.py
git commit -m "feat(api): include node_categories (agents/llm/deterministic) in run_info event"
```

---

# Phase 7 — Frontend wiring (after prior refactor is tagged)

## Task 7.1 — `types/api.ts` extensions

**Files:**
- Modify: `frontend/types/api.ts`

### Step 1 — Add types

```ts
// Add to PipelineEventType union — no change needed (planner_context exists)

export interface EvidenceReference {
  source: 'dataset_profile' | 'task_metadata' | 'registry' | 'experience' | 'rule'
  source_id?: string | null
  relevance_note?: string
}

export interface CandidateRationale {
  model_key: string
  priority: number
  reason: string
  evidence_refs: EvidenceReference[]
  risks: string[]
}

export interface RejectedModelRationale {
  model_key: string
  reason: string
  evidence_refs: EvidenceReference[]
  reconsider_if?: string | null
}

export interface DecisionBasis {
  primary_evidence: EvidenceReference[]
  secondary_evidence: EvidenceReference[]
  final_strategy: string
}

export interface EvidenceConflict {
  summary: string
  affected_models: string[]
  conflicting_evidence_refs: EvidenceReference[]
  resolution: string
}

export interface SoftConflict {
  type: string
  models: string[]
  summary: string
}
```

Extend the existing `ExperienceSummary` interface (in same file):

```ts
export interface ExperienceSummary {
  experience_id: string
  similarity_score: number
  relevance_tier: 'high' | 'medium' | 'low'
  matched_buckets: string[]
  mismatched_buckets: string[]
  target_scale_note: string | null
  dataset_name: string
  problem_type: string
  best_model: string
  validation_score: number
  metric_name?: string
}
```

Extend `PlannerContextData`:

```ts
export interface PlannerContextData {
  retrieved_experiences: ExperienceSummary[]
  matched_rules: MatchedRule[]
  evidence_used: EvidenceReference[]
  planning_analysis: string
  plan_summary: {
    candidate_rationales: CandidateRationale[]
    rejected_model_rationales: RejectedModelRationale[]
    candidate_models: string[]                  // legacy
    models_not_recommended: string[]            // legacy
  }
  warnings: string[]
  decision_basis: DecisionBasis
  evidence_conflicts: EvidenceConflict[]
  soft_conflicts: SoftConflict[]
  cited_experience_ids: string[]
  cited_rule_ids: string[]
  planner_status: 'ok' | 'retry_ok' | 'failed'
}
```

### Step 2 — Build + commit

```
cd frontend && npm run build
git add frontend/types/api.ts
git commit -m "feat(types): add Planner v2 contracts (DecisionBasis, EvidenceConflict, SoftConflict, etc.)"
```

---

## Task 7.2 — `PipelineStepper` badge flip

**Files:**
- Modify: `frontend/components/pipeline/PipelineStepper.tsx`

### Step 1 — Change one line

Locate the `STAGES` array, change the `model_planning` entry:

```tsx
{ key: 'model_planning', label: 'Model Planning', type: 'agent' },
```

(was `type: 'llm'`)

### Step 2 — Build + commit

```
cd frontend && npm test -- PipelineStepper
git add frontend/components/pipeline/PipelineStepper.tsx
git commit -m "feat(stepper): model_planning is now an Agent (planner v2 is tool-using)"
```

---

## Task 7.3 — `RunHeader` taxonomy props + `pipeline/page.tsx` wiring

**Files:**
- Modify: `frontend/components/pipeline/RunHeader.tsx`
- Modify: `frontend/components/pipeline/__tests__/RunHeader.test.tsx`
- Modify: `frontend/app/pipeline/page.tsx`

### Step 1 — Update `RunHeader` props

In `frontend/components/pipeline/RunHeader.tsx`:

```tsx
interface RunHeaderProps {
  runId: string
  problemType: string
  stageLabel: string
  startedMs: number
  runOutcome: 'running' | 'complete' | 'failed' | 'candidate_rejected'
  attemptCount: number
  agents: string[]
  llmNodes: string[]
  deterministic: string[]
}

export function RunHeader({
  runId, problemType, stageLabel, startedMs, runOutcome, attemptCount,
  agents, llmNodes, deterministic,
}: RunHeaderProps) {
  // ... existing elapsed timer logic ...
  return (
    <header className="sticky top-0 z-10 rounded-md border border-[var(--color-border)] bg-[var(--color-surface)] px-4 py-3">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
        {/* existing run id + problem type + stage + elapsed + status pill */}
      </div>
      <div className="mt-1 flex flex-wrap gap-x-4 text-[11px] text-[var(--color-fg-subtle)]">
        <span>Agents: {agents.join(' · ') || '—'}</span>
        <span>LLM: {llmNodes.join(' · ') || '—'}</span>
        <span>Deterministic: {deterministic.join(' · ') || '—'}</span>
      </div>
    </header>
  )
}
```

### Step 2 — Update test

`frontend/__tests__/components/pipeline/RunHeader.test.tsx`: every test that constructed `RunHeader` with `llmModels=[...]` needs updating to pass `agents=[...]`, `llmNodes=[...]`, `deterministic=[...]`. Update each fixture.

### Step 3 — Wire in `pipeline/page.tsx`

Replace the `llmModels` derivation with three derivations from `run_info.data.node_categories`:

```tsx
const nodeCategories = useMemo(() => {
  const info = events.find((e) => e.type === 'run_info')
  const cats = (info?.data as { node_categories?: { agents: string[]; llm_nodes: string[]; deterministic: string[] } } | undefined)?.node_categories
  if (cats) return cats
  // Legacy fallback for old events without node_categories
  const models = (info?.data as { models?: Record<string, string> } | undefined)?.models ?? {}
  return {
    agents: ['data_validator', 'planner'].filter((n) => n in models),
    llm_nodes: ['report_writer'].filter((n) => n in models),
    deterministic: ['controller', 'executor', 'evaluation', 'deployer'],
  }
}, [events])
```

Pass to `<RunHeader>`:
```tsx
<RunHeader
  // ... existing props ...
  agents={nodeCategories.agents}
  llmNodes={nodeCategories.llm_nodes}
  deterministic={nodeCategories.deterministic}
/>
```

### Step 4 — Build + tests + commit

```
cd frontend && npm test -- RunHeader
cd frontend && npm run build
git add frontend/components/pipeline/RunHeader.tsx frontend/__tests__/components/pipeline/RunHeader.test.tsx frontend/app/pipeline/page.tsx
git commit -m "feat(header): RunHeader taxonomy props — Agents / LLM / Deterministic three-row"
```

---

# Phase 8 — Planner tab redesign

**Before starting:** verify the existing `Card` primitive path. The new components import `{ Card } from '@/components/ui/Card'`. If the project uses a different name/path (e.g., inline card wrappers, `<section className="rounded-md border…">`), adjust ALL Phase 8 imports to use the existing convention rather than creating a new primitive. Run `Grep -r "components/ui/Card"` to confirm before writing the new files.

## Task 8.1 — Extract `PlannerPanel` from `ResultsDashboard`

**Files:**
- Create: `frontend/components/pipeline/PlannerPanel.tsx`
- Modify: `frontend/components/pipeline/ResultsDashboard.tsx`

### Step 1 — Move existing `PlannerPanel` function out of `ResultsDashboard.tsx`

In `ResultsDashboard.tsx`, find the inline `PlannerPanel` function (and its `PlannerErrorBoundary` if present). Move both to a new file:

`frontend/components/pipeline/PlannerPanel.tsx`:

```tsx
'use client'
import { Component, type ReactNode, type ErrorInfo } from 'react'
import type { PlannerContextData } from '@/types/api'

// PASTE the existing PlannerPanel + PlannerErrorBoundary code here verbatim.
// (We'll redesign the contents in subsequent tasks; this task only extracts.)

export class PlannerErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  // ... existing ...
}

export function PlannerPanel({ ctx, running }: { ctx: PlannerContextData | null; running: boolean }) {
  // ... existing ...
}
```

In `ResultsDashboard.tsx`, add `import { PlannerPanel, PlannerErrorBoundary } from './PlannerPanel'` and delete the now-extracted in-file definitions.

### Step 2 — Build + commit

```
cd frontend && npm run build && npm test -- PlannerPanel
git add frontend/components/pipeline/PlannerPanel.tsx frontend/components/pipeline/ResultsDashboard.tsx
git commit -m "refactor(planner): extract PlannerPanel + PlannerErrorBoundary into own file"
```

---

## Task 8.2 — Sub-components: PlannerSummaryHeader + DecisionBasisCard + ConflictPanel

**Files:**
- Create: `frontend/components/pipeline/planner/PlannerSummaryHeader.tsx`
- Create: `frontend/components/pipeline/planner/DecisionBasisCard.tsx`
- Create: `frontend/components/pipeline/planner/ConflictPanel.tsx`
- Create: `frontend/__tests__/components/pipeline/planner/ConflictPanel.test.tsx`

### Step 1 — `PlannerSummaryHeader.tsx`

```tsx
'use client'
import type { CandidateRationale, RejectedModelRationale } from '@/types/api'

interface Props {
  candidates: CandidateRationale[]
  rejected: RejectedModelRationale[]
  status: 'ok' | 'retry_ok' | 'failed'
}

const STATUS_STYLES: Record<string, string> = {
  ok:        'bg-emerald-50 text-emerald-700 ring-emerald-200',
  retry_ok:  'bg-amber-50 text-amber-700 ring-amber-200',
  failed:    'bg-red-50 text-red-700 ring-red-200',
}

export function PlannerSummaryHeader({ candidates, rejected, status }: Props) {
  return (
    <div className="flex flex-wrap items-center gap-2 text-xs">
      <span className="font-semibold text-zinc-700">Selected ({candidates.length}):</span>
      {candidates.map((c) => (
        <span key={c.model_key} className="rounded-full bg-violet-50 px-2 py-0.5 text-violet-700">
          #{c.priority} {c.model_key}
        </span>
      ))}
      <span className="ml-3 font-semibold text-zinc-700">Rejected ({rejected.length}):</span>
      {rejected.map((r) => (
        <span key={r.model_key} className="rounded-full bg-red-50 px-2 py-0.5 text-red-600 line-through">
          {r.model_key}
        </span>
      ))}
      <span className={`ml-auto inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-medium ring-1 ring-inset ${STATUS_STYLES[status] ?? STATUS_STYLES.ok}`}>
        {status.replace('_', ' ')}
      </span>
    </div>
  )
}
```

### Step 2 — `DecisionBasisCard.tsx`

```tsx
'use client'
import type { DecisionBasis, EvidenceReference } from '@/types/api'
import { Card } from '@/components/ui/Card'

function EvidenceChip({ ref }: { ref: EvidenceReference }) {
  const label = ref.source_id ? `${ref.source}:${ref.source_id}` : ref.source
  return (
    <span className="rounded bg-zinc-100 px-2 py-0.5 text-[11px] font-mono text-zinc-700">
      {label}
    </span>
  )
}

export function DecisionBasisCard({ basis }: { basis: DecisionBasis }) {
  return (
    <Card title="Decision basis">
      <p className="mb-1 text-xs font-medium text-zinc-500">Primary evidence</p>
      <div className="mb-3 flex flex-wrap gap-1">
        {basis.primary_evidence.map((e, i) => <EvidenceChip key={i} ref={e} />)}
      </div>
      {basis.secondary_evidence.length > 0 && (
        <>
          <p className="mb-1 text-xs font-medium text-zinc-500">Secondary evidence</p>
          <div className="mb-3 flex flex-wrap gap-1">
            {basis.secondary_evidence.map((e, i) => <EvidenceChip key={i} ref={e} />)}
          </div>
        </>
      )}
      <p className="text-xs leading-relaxed text-zinc-700">{basis.final_strategy}</p>
    </Card>
  )
}
```

### Step 3 — `ConflictPanel.tsx`

```tsx
'use client'
import type { EvidenceConflict, SoftConflict } from '@/types/api'
import { Card } from '@/components/ui/Card'

interface Props {
  hard: EvidenceConflict[]
  soft: SoftConflict[]
}

export function ConflictPanel({ hard, soft }: Props) {
  if (hard.length === 0 && soft.length === 0) return null
  return (
    <Card title={hard.length > 0 ? '⚠ Evidence conflict' : 'ℹ Retrieved but not cited'}
          className={hard.length > 0 ? 'border-amber-300' : ''}>
      {hard.length > 0 && (
        <div className="space-y-2">
          {hard.map((c, i) => (
            <div key={i} className="rounded border border-amber-200 bg-amber-50 p-2 text-xs">
              <p className="font-semibold text-amber-900">{c.summary}</p>
              <p className="mt-0.5 text-amber-800">Affected: {c.affected_models.join(', ')}</p>
              <p className="mt-1 italic text-amber-700">Resolution: {c.resolution}</p>
            </div>
          ))}
        </div>
      )}
      {soft.length > 0 && (
        <div className={`text-xs text-zinc-600 ${hard.length > 0 ? 'mt-3 border-t border-zinc-200 pt-2' : ''}`}>
          {soft.map((s, i) => (
            <p key={i} className="italic">{s.summary}</p>
          ))}
        </div>
      )}
    </Card>
  )
}
```

### Step 4 — Test

`ConflictPanel.test.tsx`:

```tsx
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { ConflictPanel } from '@/components/pipeline/planner/ConflictPanel'

describe('<ConflictPanel>', () => {
  it('renders nothing when both hard and soft are empty', () => {
    const { container } = render(<ConflictPanel hard={[]} soft={[]} />)
    expect(container.firstChild).toBeNull()
  })
  it('renders hard conflicts with amber border', () => {
    render(<ConflictPanel hard={[{
      summary: 'extra_trees won but not selected',
      affected_models: ['extra_trees'],
      conflicting_evidence_refs: [],
      resolution: 'short history; conservative choice',
    }]} soft={[]} />)
    expect(screen.getByText(/extra_trees won/)).toBeInTheDocument()
    expect(screen.getByText(/Resolution: short history/)).toBeInTheDocument()
  })
  it('renders soft conflicts as info', () => {
    render(<ConflictPanel hard={[]} soft={[{
      type: 'retrieved_experience_winner_not_selected',
      models: ['extra_trees'],
      summary: '1 model won in retrieved experiences but was not cited or selected',
    }]} />)
    expect(screen.getByText(/1 model won/)).toBeInTheDocument()
  })
})
```

### Step 5 — Run + commit

```
cd frontend && npm test -- ConflictPanel
git add frontend/components/pipeline/planner frontend/__tests__/components/pipeline/planner
git commit -m "feat(planner): add PlannerSummaryHeader, DecisionBasisCard, ConflictPanel sub-components"
```

---

## Task 8.3 — CandidateCard + RejectedModelCard + their list wrappers

**Files:**
- Create: `frontend/components/pipeline/planner/CandidateCard.tsx`
- Create: `frontend/components/pipeline/planner/RejectedModelCard.tsx`

### Step 1 — `CandidateCard.tsx`

```tsx
'use client'
import { useState } from 'react'
import type { CandidateRationale, EvidenceReference } from '@/types/api'

function EvidenceChip({ ref, cited }: { ref: EvidenceReference; cited?: boolean }) {
  const label = ref.source_id ? `${ref.source}:${ref.source_id}` : ref.source
  return (
    <span className={`rounded px-2 py-0.5 text-[11px] font-mono ${cited ? 'bg-violet-100 text-violet-700' : 'bg-zinc-100 text-zinc-700'}`}>
      {label}{cited && ' (cited)'}
    </span>
  )
}

export function CandidateCard({ candidate }: { candidate: CandidateRationale }) {
  const [expanded, setExpanded] = useState(false)
  return (
    <div className="rounded border border-zinc-200 bg-white p-3 text-xs">
      <button type="button" onClick={() => setExpanded((e) => !e)} className="flex w-full items-center gap-2 text-left">
        <span className="rounded-full bg-violet-100 px-2 py-0.5 text-[11px] font-semibold text-violet-700">#{candidate.priority}</span>
        <span className="font-mono font-semibold text-zinc-900">{candidate.model_key}</span>
        <span className="ml-auto text-zinc-400">{expanded ? '▾' : '▸'}</span>
      </button>
      <p className="mt-1 text-zinc-700">{candidate.reason}</p>
      {expanded && (
        <>
          <p className="mt-2 text-[11px] font-medium text-zinc-500">Evidence</p>
          <div className="mt-1 flex flex-wrap gap-1">
            {candidate.evidence_refs.map((r, i) => <EvidenceChip key={i} ref={r} />)}
          </div>
          {candidate.risks.length > 0 && (
            <>
              <p className="mt-2 text-[11px] font-medium text-zinc-500">Risks</p>
              <ul className="ml-4 list-disc text-zinc-600">
                {candidate.risks.map((r, i) => <li key={i}>{r}</li>)}
              </ul>
            </>
          )}
        </>
      )}
    </div>
  )
}
```

### Step 2 — `RejectedModelCard.tsx`

```tsx
'use client'
import { useState } from 'react'
import type { RejectedModelRationale } from '@/types/api'

export function RejectedModelCard({ rejected }: { rejected: RejectedModelRationale }) {
  const [expanded, setExpanded] = useState(false)
  return (
    <div className="rounded border border-zinc-200 bg-white p-2 text-xs">
      <button type="button" onClick={() => setExpanded((e) => !e)} className="flex w-full items-center gap-2 text-left">
        <span className="font-mono font-semibold text-red-600 line-through">{rejected.model_key}</span>
        <span className="ml-auto text-zinc-400">{expanded ? '▾' : '▸'}</span>
      </button>
      {expanded && (
        <>
          <p className="mt-1 text-zinc-700">{rejected.reason}</p>
          {rejected.reconsider_if && (
            <p className="mt-1 italic text-zinc-500">Reconsider if: {rejected.reconsider_if}</p>
          )}
          <div className="mt-2 flex flex-wrap gap-1">
            {rejected.evidence_refs.map((r, i) => (
              <span key={i} className="rounded bg-zinc-100 px-2 py-0.5 text-[11px] font-mono text-zinc-700">
                {r.source_id ? `${r.source}:${r.source_id}` : r.source}
              </span>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
```

### Step 3 — Commit (no separate tests; covered by PlannerPanel integration tests)

```
git add frontend/components/pipeline/planner/CandidateCard.tsx frontend/components/pipeline/planner/RejectedModelCard.tsx
git commit -m "feat(planner): add CandidateCard + RejectedModelCard expandable cards"
```

---

## Task 8.4 — EvidenceQualityCard + ExperienceCard

**Files:**
- Create: `frontend/components/pipeline/planner/EvidenceQualityCard.tsx`
- Create: `frontend/components/pipeline/planner/ExperienceCard.tsx`

### Step 1 — `EvidenceQualityCard.tsx`

```tsx
'use client'
import type { ExperienceSummary } from '@/types/api'
import { Card } from '@/components/ui/Card'

interface Props {
  experiences: ExperienceSummary[]
  citedIds: string[]
}

export function EvidenceQualityCard({ experiences, citedIds }: Props) {
  const citedSet = new Set(citedIds)
  const tiers = { high: 0, medium: 0, low: 0 }
  let scaleMismatchCount = 0
  for (const e of experiences) {
    tiers[e.relevance_tier]++
    if (e.target_scale_note) scaleMismatchCount++
  }
  return (
    <Card title="Evidence quality">
      <p className="text-xs text-zinc-700">
        Available experiences: {experiences.length}   ·   Cited: {citedSet.size}
      </p>
      <p className="mt-1 text-xs text-zinc-700">
        Relevance:   high: {tiers.high}   ·   medium: {tiers.medium}   ·   low: {tiers.low}
      </p>
      {scaleMismatchCount > 0 && (
        <p className="mt-2 text-[11px] italic text-amber-700">
          ⚠ {scaleMismatchCount} experience(s) have target-scale mismatches — raw metric values
          may not be directly comparable.
        </p>
      )}
    </Card>
  )
}
```

### Step 2 — `ExperienceCard.tsx`

```tsx
'use client'
import type { ExperienceSummary } from '@/types/api'

const TIER_STYLES = {
  high:   'bg-emerald-50 text-emerald-700 ring-emerald-200',
  medium: 'bg-amber-50 text-amber-700 ring-amber-200',
  low:    'bg-zinc-100 text-zinc-600 ring-zinc-200',
}

export function ExperienceCard({ exp, cited }: { exp: ExperienceSummary; cited: boolean }) {
  return (
    <div className={`rounded border px-3 py-2 text-xs ${cited ? 'border-violet-200 bg-violet-50' : 'border-zinc-200 bg-white'}`}>
      <div className="flex items-center gap-2">
        <span className={`rounded-full px-2 py-0.5 text-[11px] font-medium ring-1 ring-inset ${TIER_STYLES[exp.relevance_tier]}`}>
          {exp.relevance_tier} · similarity {exp.similarity_score.toFixed(2)}
        </span>
        {cited && (
          <span className="rounded-full bg-violet-100 px-2 py-0.5 text-[11px] text-violet-700">cited</span>
        )}
      </div>
      <div className="mt-1 text-zinc-700">
        Best model: <span className="font-mono font-semibold">{exp.best_model}</span>
        {' · '}
        <span className="uppercase">{exp.metric_name ?? 'metric'}</span>: <span className="font-mono">{(exp.validation_score ?? 0).toFixed(4)}</span>
      </div>
      {exp.target_scale_note && (
        <p className="mt-1 italic text-amber-700">⚠ {exp.target_scale_note}</p>
      )}
      {exp.matched_buckets.length > 0 && (
        <p className="mt-1 text-[11px] text-zinc-500">Matched: {exp.matched_buckets.join(', ')}</p>
      )}
      {exp.mismatched_buckets.length > 0 && (
        <p className="text-[11px] text-zinc-500">Mismatched: {exp.mismatched_buckets.join(', ')}</p>
      )}
    </div>
  )
}
```

### Step 3 — Commit

```
git add frontend/components/pipeline/planner/EvidenceQualityCard.tsx frontend/components/pipeline/planner/ExperienceCard.tsx
git commit -m "feat(planner): add EvidenceQualityCard + ExperienceCard (tier-aware experience rendering)"
```

---

## Task 8.5 — Compose new `PlannerPanel` (replaces extracted v1)

**Files:**
- Rewrite: `frontend/components/pipeline/PlannerPanel.tsx`
- Modify: `frontend/__tests__/components/pipeline/PlannerPanel.test.tsx` (create or extend)

### Step 1 — Rewrite

```tsx
'use client'
import { Component, type ReactNode, type ErrorInfo } from 'react'
import type { PlannerContextData } from '@/types/api'
import { Card } from '@/components/ui/Card'
import { PlannerSummaryHeader } from './planner/PlannerSummaryHeader'
import { DecisionBasisCard } from './planner/DecisionBasisCard'
import { ConflictPanel } from './planner/ConflictPanel'
import { CandidateCard } from './planner/CandidateCard'
import { RejectedModelCard } from './planner/RejectedModelCard'
import { EvidenceQualityCard } from './planner/EvidenceQualityCard'
import { ExperienceCard } from './planner/ExperienceCard'

export class PlannerErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  constructor(props: { children: ReactNode }) { super(props); this.state = { error: null } }
  static getDerivedStateFromError(error: Error) { return { error } }
  componentDidCatch(error: Error, info: ErrorInfo) { console.error('[PlannerPanel] render error:', error, info) }
  render() {
    if (this.state.error) {
      return (
        <div className="rounded border border-red-200 bg-red-50 p-4 text-xs text-red-700">
          <p className="font-semibold">Planner panel failed to render</p>
          <p className="mt-1 font-mono opacity-70">{this.state.error.message}</p>
        </div>
      )
    }
    return this.props.children
  }
}

interface Props {
  ctx: PlannerContextData | null
  running: boolean
}

export function PlannerPanel({ ctx, running }: Props) {
  if (!ctx && running) {
    return <p className="text-xs text-zinc-400">Planner is running…</p>
  }
  if (!ctx) {
    return <p className="text-xs text-zinc-400">Planner has not run yet.</p>
  }

  const candidates = ctx.plan_summary.candidate_rationales ?? []
  const rejected = ctx.plan_summary.rejected_model_rationales ?? []
  const sortedCandidates = [...candidates].sort((a, b) => a.priority - b.priority)
  const sortedExperiences = [...ctx.retrieved_experiences].sort((a, b) => b.similarity_score - a.similarity_score)
  const citedSet = new Set(ctx.cited_experience_ids ?? [])
  const citedRules = new Set(ctx.cited_rule_ids ?? [])

  return (
    <div className="space-y-4">
      <PlannerSummaryHeader candidates={sortedCandidates} rejected={rejected} status={ctx.planner_status} />
      {ctx.decision_basis && <DecisionBasisCard basis={ctx.decision_basis} />}
      <ConflictPanel hard={ctx.evidence_conflicts ?? []} soft={ctx.soft_conflicts ?? []} />

      {sortedCandidates.length > 0 && (
        <Card title={`Candidate rationale (${sortedCandidates.length})`}>
          <div className="space-y-2">
            {sortedCandidates.map((c) => <CandidateCard key={c.model_key} candidate={c} />)}
          </div>
        </Card>
      )}

      {rejected.length > 0 && (
        <Card title={`Rejected models (${rejected.length})`}>
          <div className="space-y-1">
            {rejected.map((r) => <RejectedModelCard key={r.model_key} rejected={r} />)}
          </div>
        </Card>
      )}

      <EvidenceQualityCard experiences={ctx.retrieved_experiences} citedIds={ctx.cited_experience_ids ?? []} />

      {sortedExperiences.length > 0 && (
        <Card title={`Similar past runs (${sortedExperiences.length})`}>
          <div className="space-y-2">
            {sortedExperiences.map((e) => (
              <ExperienceCard key={e.experience_id} exp={e} cited={citedSet.has(e.experience_id)} />
            ))}
          </div>
        </Card>
      )}

      {ctx.matched_rules.length > 0 && (
        <Card title={`ML rules matched (${ctx.matched_rules.length})`}>
          {/* keep existing rule rendering — unchanged from v1 */}
          <div className="space-y-1.5">
            {ctx.matched_rules.map((rule) => {
              const cited = citedRules.has(rule.rule_id)
              return (
                <div key={rule.rule_id} className={`rounded border px-3 py-2 text-xs ${cited ? 'border-violet-200 bg-violet-50' : 'border-zinc-200 bg-white'}`}>
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-zinc-500">{rule.rule_id}</span>
                    {cited && <span className="ml-auto rounded-full bg-violet-100 px-2 py-0.5 text-violet-700">cited</span>}
                  </div>
                  <p className="mt-0.5 text-zinc-700">{rule.summary}</p>
                  {rule.prefer && rule.prefer.length > 0 && (
                    <p className="mt-0.5 text-emerald-700">↑ prefer: {rule.prefer.join(', ')}</p>
                  )}
                  {rule.avoid_or_deprioritize && rule.avoid_or_deprioritize.length > 0 && (
                    <p className="mt-0.5 text-red-600">↓ avoid: {rule.avoid_or_deprioritize.join(', ')}</p>
                  )}
                </div>
              )
            })}
          </div>
        </Card>
      )}

      {ctx.warnings.length > 0 && (
        <Card title="Planner warnings">
          <ul className="space-y-0.5 text-xs">
            {ctx.warnings.map((w, i) => (
              <li key={i} className="text-amber-700">⚠ {w}</li>
            ))}
          </ul>
        </Card>
      )}

      <details className="rounded border border-[var(--color-border)] bg-white p-3">
        <summary className="cursor-pointer text-xs text-zinc-500">View full planning analysis</summary>
        <pre className="mt-2 whitespace-pre-wrap text-xs text-zinc-700">{ctx.planning_analysis}</pre>
      </details>
    </div>
  )
}
```

### Step 2 — Test

`frontend/__tests__/components/pipeline/PlannerPanel.test.tsx`:

```tsx
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { PlannerPanel } from '@/components/pipeline/PlannerPanel'
import type { PlannerContextData } from '@/types/api'

const minimalCtx: PlannerContextData = {
  retrieved_experiences: [],
  matched_rules: [],
  evidence_used: [],
  planning_analysis: 'long analysis text here',
  plan_summary: {
    candidate_rationales: [{
      model_key: 'ets', priority: 1, reason: 'short history',
      evidence_refs: [{ source: 'registry', source_id: 'ets' }],
      risks: ['risk1'],
    }],
    rejected_model_rationales: [],
    candidate_models: ['ets'],
    models_not_recommended: [],
  },
  warnings: [],
  decision_basis: {
    primary_evidence: [{ source: 'dataset_profile' }],
    secondary_evidence: [],
    final_strategy: 'prefer statistical',
  },
  evidence_conflicts: [],
  soft_conflicts: [],
  cited_experience_ids: [],
  cited_rule_ids: [],
  planner_status: 'ok',
}

describe('<PlannerPanel>', () => {
  it('renders placeholder when ctx is null and not running', () => {
    render(<PlannerPanel ctx={null} running={false} />)
    expect(screen.getByText(/Planner has not run yet/)).toBeInTheDocument()
  })
  it('renders all sections with minimal valid ctx', () => {
    render(<PlannerPanel ctx={minimalCtx} running={false} />)
    expect(screen.getByText(/Selected \(1\)/)).toBeInTheDocument()
    expect(screen.getByText(/Decision basis/)).toBeInTheDocument()
    expect(screen.getByText(/Candidate rationale \(1\)/)).toBeInTheDocument()
    expect(screen.getByText(/Evidence quality/)).toBeInTheDocument()
    expect(screen.getByText(/View full planning analysis/)).toBeInTheDocument()
  })
  it('renders ConflictPanel only when conflicts present', () => {
    const ctxWithConflict = {
      ...minimalCtx,
      evidence_conflicts: [{
        summary: 'a', affected_models: ['ets'],
        conflicting_evidence_refs: [], resolution: 'r',
      }],
    }
    render(<PlannerPanel ctx={ctxWithConflict} running={false} />)
    expect(screen.getByText(/Evidence conflict/)).toBeInTheDocument()
  })
})
```

### Step 3 — Run + commit

```
cd frontend && npm test -- PlannerPanel
cd frontend && npm run build
git add frontend/components/pipeline/PlannerPanel.tsx frontend/__tests__/components/pipeline/PlannerPanel.test.tsx
git commit -m "feat(planner): redesigned PlannerPanel — rationale cards, decision basis, evidence quality, conflict surfacing"
```

---

# Phase 9 — Final integration + smoke

## Task 9.1 — Run full test suites green

- [ ] **Step 1: Backend**

```
uv run pytest -m "not integration"
```

Expected: all pass (the pre-existing `use-approve.test.tsx` failure remains in the frontend per prior refactor notes; backend should be 100%).

- [ ] **Step 2: Frontend**

```
cd frontend && npm test
```

Expected: previous baseline + new Planner v2 tests all pass.

- [ ] **Step 3: Build**

```
cd frontend && npm run build
```

Expected: clean build.

## Task 9.2 — Container smoke

- [ ] **Step 1: Rebuild + run**

```
docker compose down -v && docker compose up --build
```

- [ ] **Step 2: End-to-end pipeline**

Submit a real pipeline. Open http://localhost:3000/pipeline. Verify:

1. **PipelineStepper** shows `Model Planning` with the **Agent** badge (violet "Agent" pill, not "LLM").
2. **RunHeader** taxonomy line shows three categories:
   - `Agents: data_validator · planner`
   - `LLM: report_writer`
   - `Deterministic: controller · executor · evaluation · deployer`
3. **EventLog Timeline** shows `Planner selected K candidates` line.
4. **EventLog Tool Details** shows `Planner` subsection with `list_available_models · 1 call`, `retrieve_similar_experiences · 1 call`, `retrieve_ml_knowledge · 1 call`, and possibly `inspect_model_details · N calls`. Not `structured-output call · 1 call`.
5. **ResultsDashboard → Planner tab**:
   - `Selected (K)` + `Rejected (M)` chip row with priorities
   - `Decision basis` card with primary/secondary evidence chips + strategy
   - `ConflictPanel` only if conflicts (try forcing one by handcrafting a planner output that omits a winning model)
   - `Candidate rationale (K)` cards — click to expand, see evidence_refs chips + risks
   - `Rejected models (M)` collapsed list — click to expand, see reason + reconsider_if
   - `Evidence quality` card with tier counts
   - `Similar past runs (N)` cards with tier badges + cited markers + scale notes when present
   - `ML rules matched`, `Planner warnings`, `View full planning analysis` audit drawer at bottom
6. **Run completes end-to-end** — deployment still works (regression check from prior slices stays green).

If anything fails: surface specifically, dispatch fix, re-smoke.

---

## Closing notes for the engineer

- **TDD discipline**: every task above pairs a test with implementation. Don't skip the failing-test step — it's how you know the test exercises the behavior.
- **Phase order matters**: Phase 0 (foundations) → Phase 1 (contracts) → Phase 2 (retrieval) → Phase 3 (tools) → Phase 4 (validation) → Phase 5 (agent + node) → Phase 6 (backend SSE) → Phase 7 (frontend wiring — only after prior refactor's `refactor-frontend-mlops-v1` tag exists) → Phase 8 (UI redesign) → Phase 9 (smoke).
- **No expected-failure window between phases** — Phase 1.2 ships `relevance_tier` with a temporary default, removed in Phase 2.3 after every caller populates it. Test suite stays green throughout.
- **Phase 5 Task 5.3 deletes `build_planner_context`, `_to_experience_summary`, the old `_check_*` helpers, and the old `build_planner_agent`**. `_to_experience_summary` moves to `experience/retrieval.py` as public `to_experience_summary`; the rest move to `planning/*`. Grep for callers before deleting.
- **Frontend Phase 7+8 wait for the prior refactor tag**. Phases 0–6 (backend) can start immediately and ship without touching the UI; the frontend then layers on top.
- **Backward compatibility**: existing `ExperienceRecord`s without numeric target stats render with no target_scale_note. Existing planner records without `decision_basis` will fail to render the panel — but those are from before this refactor anyway; legacy SSE replays of old records will show the empty-state placeholder.
