# SP5 — Model Planning Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the deterministic `default_training_plan` fallback with an LLM-based Model Planning Agent that synthesizes evidence from the experience pool, ML rules, and the model registry to produce an auditable `TrainingPlan`.

**Architecture:** A new `planner` graph node (LLM structured-output, not ReAct) assembles a `PlannerContext` deterministically, calls the LLM once, validates the output through a 4-stage chain, and writes the `TrainingPlan` + `planning_analysis` to graph state. The existing `trainer` node is renamed to `executor`. The supervisor routes `data_validator → planner → executor → evaluator → deployer`.

**Tech Stack:** Python 3.12, Pydantic v2, LangChain `with_structured_output`, LangGraph `Command`, SQLite migrations, pytest with `unittest.mock`.

---

## File map

| Action | File |
|---|---|
| **New** | `src/mlops_agents/contracts/planner.py` |
| **New** | `src/mlops_agents/agents/planner.py` |
| **New** | `src/mlops_agents/prompts/planner.yaml` |
| **New** | `src/mlops_agents/experience/migrations/004_add_planner_output.sql` |
| **Modify** | `src/mlops_agents/state/agent_state.py` |
| **Modify** | `src/mlops_agents/config/settings.py` |
| **Modify** | `src/mlops_agents/utils/llm.py` |
| **Modify** | `src/mlops_agents/training/executor.py` |
| **Modify** | `src/mlops_agents/experience/schema.py` |
| **Modify** | `src/mlops_agents/experience/pool.py` |
| **Modify** | `src/mlops_agents/graphs/mlops_graph.py` |
| **Modify** | `src/mlops_agents/agents/supervisor.py` |
| **Modify** | `src/mlops_agents/prompts/supervisor.yaml` |
| **New** | `tests/test_contracts/test_planner_contracts.py` |
| **New** | `tests/test_agents/test_planner_context.py` |
| **New** | `tests/test_agents/test_planner_evidence.py` |
| **New** | `tests/test_agents/test_planner_node.py` |
| **Modify** | `tests/test_experience/test_migrations.py` |
| **New** | `tests/test_agents/test_planner_integration.py` |

---

## Task 1: Planner contracts (`contracts/planner.py`)

**Files:**
- Create: `src/mlops_agents/contracts/planner.py`
- Test: `tests/test_contracts/test_planner_contracts.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_contracts/test_planner_contracts.py
import pytest
from pydantic import ValidationError
from mlops_agents.contracts.planner import (
    EvidenceReference,
    PlannerOutput,
    PlannerContext,
    ExperienceSummary,
    CandidateResultCompact,
)
from mlops_agents.contracts.training import (
    TrainingPlan, TrainingPlanCandidate, RejectedModel, TrialBudget
)


def _minimal_plan(problem_type: str = "regression") -> TrainingPlan:
    # Only include models valid for regression; _check_plan_integrity enforces keys.
    from mlops_agents.models.loader import get_models_for
    models = get_models_for(problem_type)
    candidates = [TrainingPlanCandidate(priority=i+1, model_key=m.model_key, reason="test")
                  for i, m in enumerate(models)]
    return TrainingPlan(problem_type=problem_type, candidates=candidates)


def test_evidence_reference_valid_experience():
    ref = EvidenceReference(source="experience", source_id="task_001", summary="used LightGBM")
    assert ref.source == "experience"
    assert ref.source_id == "task_001"


def test_evidence_reference_dataset_profile_null_source_id():
    ref = EvidenceReference(source="dataset_profile", source_id=None, summary="medium dataset")
    assert ref.source_id is None


def test_evidence_reference_task_metadata_null_source_id():
    ref = EvidenceReference(source="task_metadata", source_id=None, summary="forecasting task")
    assert ref.source_id is None


def test_planner_output_requires_plan():
    with pytest.raises(ValidationError):
        PlannerOutput(planning_analysis="ok")  # missing plan


def test_planner_output_valid():
    plan = _minimal_plan("regression")
    out = PlannerOutput(planning_analysis="analysis text", plan=plan)
    assert out.plan.problem_type == "regression"
    assert out.evidence_used == []
    assert out.risks_or_warnings == []


def test_planner_context_empty_experiences():
    ctx = PlannerContext(
        current_dataset_profile={"problem_type": "regression", "n_rows": "medium"},
        task_metadata={"target_column": "y"},
        available_models=["ridge", "lightgbm_regressor"],
        similar_experiences=[],
        matched_rules=[],
    )
    assert ctx.similar_experiences == []


def test_candidate_result_compact():
    c = CandidateResultCompact(model_key="lightgbm_regressor", rank=1, metric_value=0.42)
    assert c.rank == 1


def test_experience_summary():
    es = ExperienceSummary(
        experience_id="task_001",
        similarity_score=0.84,
        dataset_summary="medium regression dataset",
        models_trained=["ridge", "lightgbm_regressor"],
        best_model="lightgbm_regressor",
        validation_score=0.12,
        notes="boosting worked well",
    )
    assert es.best_model == "lightgbm_regressor"
    assert es.candidate_results == []
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_contracts/test_planner_contracts.py -v
```

Expected: `ModuleNotFoundError: No module named 'mlops_agents.contracts.planner'`

- [ ] **Step 3: Create `contracts/planner.py`**

```python
# src/mlops_agents/contracts/planner.py
"""Pydantic contracts for the Model Planning Agent (SP5)."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field
from mlops_agents.contracts.training import TrainingPlan

PlannerStatus = Literal["ok", "retry_ok", "failed"]


class EvidenceReference(BaseModel):
    source: Literal[
        "dataset_profile",
        "task_metadata",
        "experience",
        "rule",
        "registry",
    ]
    source_id: str | None = None
    summary: str


class PlannerOutput(BaseModel):
    planning_analysis: str = Field(
        description=(
            "Detailed public analysis generated by the LLM. "
            "This is an explanation artifact — not guaranteed access to "
            "the model's internal reasoning process."
        )
    )
    evidence_used: list[EvidenceReference] = Field(default_factory=list)
    risks_or_warnings: list[str] = Field(default_factory=list)
    plan: TrainingPlan


class CandidateResultCompact(BaseModel):
    model_key: str
    rank: int
    metric_value: float | None = None


class ExperienceSummary(BaseModel):
    experience_id: str
    similarity_score: float
    dataset_summary: str
    models_trained: list[str]
    best_model: str
    validation_score: float
    candidate_results: list[CandidateResultCompact] = Field(default_factory=list)
    notes: str = ""


class PlannerContext(BaseModel):
    current_dataset_profile: dict
    task_metadata: dict
    available_models: list[str]
    similar_experiences: list[ExperienceSummary]
    matched_rules: list[dict]
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/test_contracts/test_planner_contracts.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```
git add src/mlops_agents/contracts/planner.py tests/test_contracts/test_planner_contracts.py
git commit -m "feat(sp5): add planner contracts — EvidenceReference, PlannerOutput, PlannerContext"
```

---

## Task 2: System prompt (`prompts/planner.yaml`)

**Files:**
- Create: `src/mlops_agents/prompts/planner.yaml`

- [ ] **Step 1: Create the prompt file**

```yaml
# src/mlops_agents/prompts/planner.yaml
_type: "prompt"
input_variables: []
template: |
  You are the Model Planning Agent for an MLOps pipeline.
  Your job is to synthesize evidence and produce an auditable planning
  analysis and a structured TrainingPlan. You do not train models.

  WHAT YOU RECEIVE
  ----------------
  A JSON object (PlannerContext) with these fields:
  - current_dataset_profile: bucketed characteristics of the dataset
  - task_metadata: target column, horizon, frequency (if forecasting)
  - available_models: the ONLY model keys you may use
  - similar_experiences: top-k past runs on similar datasets, each with
    candidate_results showing ranked performance of all tried models
  - matched_rules: expert ML rules that apply to this profile, each with
    prefer, avoid_or_deprioritize, recommend, and summary fields

  HOW TO USE EVIDENCE
  --------------------
  1. Start from current_dataset_profile + task_metadata (ground truth)
  2. Use similar_experiences as empirical evidence — what worked, what failed,
     how models ranked, under which validation setup
  3. Use matched_rules as expert guidance — when rules and experiences conflict,
     resolve via the current dataset profile as tiebreaker
  4. Use available_models as the hard constraint — never invent model keys

  HARD CONSTRAINTS
  ----------------
  - Every model_key in candidates and models_not_recommended must come from available_models
  - Every available model must appear in either candidates or models_not_recommended (no omissions)
  - Every source_id you cite in evidence_used must exist in the context you received
  - For dataset_profile and task_metadata sources, leave source_id as null
  - For forecasting tasks, forecasting_settings is required; only use validation strategies:
      single_split, rolling_window, expanding_window
  - For tabular classification/regression, leave forecasting_settings as null
  - Do not invent concrete hyperparameter search spaces; you may provide budget hints
    via requested_trials on candidates if empirical evidence justifies it
  - The real search spaces live in the model registry and Optuna; you supply intent

  OUTPUT FORMAT
  -------------
  Produce a PlannerOutput with:
    planning_analysis: detailed explanation of your evidence synthesis and decisions
    evidence_used: list of {source, source_id, summary} — cite what influenced each decision
    risks_or_warnings: list of strings — flag conflicts, uncertainty, or weak evidence
    plan: a valid TrainingPlan where every available model is in candidates or models_not_recommended
```

- [ ] **Step 2: Verify the prompt loads without error**

```
uv run python -c "from mlops_agents.prompts import get_prompt; p = get_prompt('planner'); print(p.template[:60])"
```

Expected: prints the first 60 chars of the template (no exception).

- [ ] **Step 3: Commit**

```
git add src/mlops_agents/prompts/planner.yaml
git commit -m "feat(sp5): add planner system prompt"
```

---

## Task 3: State fields + settings + LLM factory

**Files:**
- Modify: `src/mlops_agents/state/agent_state.py`
- Modify: `src/mlops_agents/config/settings.py`
- Modify: `src/mlops_agents/utils/llm.py`

- [ ] **Step 1: Add 5 new fields to `AgentState`**

In `src/mlops_agents/state/agent_state.py`, add inside the `AgentState` TypedDict after the `schema_json` field:

```python
    # SP5 planner outputs
    planner_analysis: str           # LLM-generated planning explanation artifact
    planner_evidence_used: list     # list of EvidenceReference dicts
    planner_warnings: list          # list of warning strings
    planner_status: str             # "ok" | "retry_ok" | "failed"
    planner_retry_used: bool        # True if second attempt was needed
```

- [ ] **Step 2: Add `openai_model_planner` to settings**

In `src/mlops_agents/config/settings.py`, add after `openai_model_trainer`:

```python
    openai_model_planner: str = "gpt-5-mini"
```

- [ ] **Step 3: Add `"planner"` key to `get_llm` model map**

In `src/mlops_agents/utils/llm.py`, update the `model_map` dict:

```python
    model_map = {
        "data_validator": settings.openai_model_data_validator,
        "trainer":        settings.openai_model_trainer,
        "planner":        settings.openai_model_planner,
        "evaluator":      settings.openai_model_evaluator,
        "deployer":       settings.openai_model_deployer,
    }
```

- [ ] **Step 4: Run existing tests to verify no regressions**

```
uv run pytest -m "not integration" -q
```

Expected: all tests pass (same count as before).

- [ ] **Step 5: Commit**

```
git add src/mlops_agents/state/agent_state.py src/mlops_agents/config/settings.py src/mlops_agents/utils/llm.py
git commit -m "feat(sp5): add planner state fields, settings, and LLM key"
```

---

## Task 4: Context builder and validators (`agents/planner.py` — deterministic parts)

**Files:**
- Create: `src/mlops_agents/agents/planner.py` (deterministic helpers only — `planner_node` added in Task 5)
- Test: `tests/test_agents/test_planner_context.py`
- Test: `tests/test_agents/test_planner_evidence.py`

- [ ] **Step 1: Write failing context-builder tests**

```python
# tests/test_agents/test_planner_context.py
import pytest
from pathlib import Path
from mlops_agents.agents.planner import build_planner_context, PlannerError
from mlops_agents.experience.pool import ExperiencePool
from mlops_agents.models.loader import get_models_for


@pytest.fixture()
def empty_pool(tmp_path: Path) -> ExperiencePool:
    return ExperiencePool(tmp_path / "test.db")


def _regression_profile() -> dict:
    return {
        "schema_version": 1, "problem_type": "regression",
        "n_rows": "medium", "n_features": "medium",
        "missing_rate": "none", "n_categorical_features": "none",
        "n_numerical_features": "many",
        "target_distribution": "near_normal",
    }


def _classification_profile() -> dict:
    return {
        "schema_version": 1, "problem_type": "classification",
        "n_rows": "medium", "n_features": "medium",
        "missing_rate": "none", "n_categorical_features": "none",
        "n_numerical_features": "many",
        "n_classes": "binary", "class_balance": "balanced",
    }


def _forecasting_profile() -> dict:
    return {
        "schema_version": 1, "problem_type": "forecasting",
        "n_rows": "medium", "n_features": "small",
        "missing_rate": "none", "n_categorical_features": "none",
        "n_numerical_features": "few",
        "n_series": "single", "history_length": "long",
        "horizon_difficulty": "short", "seasonality_detected": False,
    }


def test_context_empty_pool_returns_no_experiences(empty_pool):
    ctx = build_planner_context(_regression_profile(), {"target_column": "y"}, "regression", empty_pool)
    assert ctx.similar_experiences == []
    assert len(ctx.available_models) > 0


def test_context_available_models_regression_only(empty_pool):
    ctx = build_planner_context(_regression_profile(), {}, "regression", empty_pool)
    registry_keys = {m.model_key for m in get_models_for("regression")}
    assert set(ctx.available_models) == registry_keys


def test_context_available_models_classification_only(empty_pool):
    ctx = build_planner_context(_classification_profile(), {}, "classification", empty_pool)
    registry_keys = {m.model_key for m in get_models_for("classification")}
    assert set(ctx.available_models) == registry_keys


def test_context_available_models_forecasting_only(empty_pool):
    ctx = build_planner_context(_forecasting_profile(), {}, "forecasting", empty_pool)
    registry_keys = {m.model_key for m in get_models_for("forecasting")}
    assert set(ctx.available_models) == registry_keys


def test_context_matched_rules_have_rule_id_and_summary(empty_pool):
    ctx = build_planner_context(_regression_profile(), {}, "regression", empty_pool)
    for r in ctx.matched_rules:
        assert "rule_id" in r
        assert "summary" in r
```

- [ ] **Step 2: Write failing evidence-validator tests**

```python
# tests/test_agents/test_planner_evidence.py
import pytest
from mlops_agents.agents.planner import (
    PlannerError,
    _check_evidence_references,
    _check_plan_exhaustiveness,
)
from mlops_agents.contracts.planner import (
    EvidenceReference,
    PlannerContext,
    ExperienceSummary,
)
from mlops_agents.contracts.training import (
    TrainingPlan, TrainingPlanCandidate, RejectedModel
)


def _minimal_ctx() -> PlannerContext:
    return PlannerContext(
        current_dataset_profile={},
        task_metadata={},
        available_models=["ridge", "lightgbm_regressor"],
        similar_experiences=[
            ExperienceSummary(
                experience_id="task_001",
                similarity_score=0.8,
                dataset_summary="medium regression",
                models_trained=["ridge", "lightgbm_regressor"],
                best_model="lightgbm_regressor",
                validation_score=0.12,
            )
        ],
        matched_rules=[{"rule_id": "rule_001", "summary": "prefer boosting"}],
    )


def test_valid_experience_reference_passes():
    ctx = _minimal_ctx()
    refs = [EvidenceReference(source="experience", source_id="task_001", summary="used")]
    _check_evidence_references(refs, ctx)  # must not raise


def test_valid_rule_reference_passes():
    ctx = _minimal_ctx()
    refs = [EvidenceReference(source="rule", source_id="rule_001", summary="applied")]
    _check_evidence_references(refs, ctx)


def test_valid_registry_reference_passes():
    ctx = _minimal_ctx()
    refs = [EvidenceReference(source="registry", source_id="ridge", summary="baseline")]
    _check_evidence_references(refs, ctx)


def test_dataset_profile_null_source_id_passes():
    ctx = _minimal_ctx()
    refs = [EvidenceReference(source="dataset_profile", source_id=None, summary="medium")]
    _check_evidence_references(refs, ctx)


def test_task_metadata_null_source_id_passes():
    ctx = _minimal_ctx()
    refs = [EvidenceReference(source="task_metadata", source_id=None, summary="forecasting")]
    _check_evidence_references(refs, ctx)


def test_dataset_profile_non_null_source_id_raises():
    ctx = _minimal_ctx()
    refs = [EvidenceReference(source="dataset_profile", source_id="something", summary="x")]
    with pytest.raises(PlannerError, match="source_id=None"):
        _check_evidence_references(refs, ctx)


def test_task_metadata_non_null_source_id_raises():
    ctx = _minimal_ctx()
    refs = [EvidenceReference(source="task_metadata", source_id="something", summary="x")]
    with pytest.raises(PlannerError, match="source_id=None"):
        _check_evidence_references(refs, ctx)


def test_experience_unknown_source_id_raises():
    ctx = _minimal_ctx()
    refs = [EvidenceReference(source="experience", source_id="fake_999", summary="x")]
    with pytest.raises(PlannerError, match="fake_999"):
        _check_evidence_references(refs, ctx)


def test_rule_unknown_source_id_raises():
    ctx = _minimal_ctx()
    refs = [EvidenceReference(source="rule", source_id="fake_rule", summary="x")]
    with pytest.raises(PlannerError, match="fake_rule"):
        _check_evidence_references(refs, ctx)


def test_registry_unknown_source_id_raises():
    ctx = _minimal_ctx()
    refs = [EvidenceReference(source="registry", source_id="fake_model", summary="x")]
    with pytest.raises(PlannerError, match="fake_model"):
        _check_evidence_references(refs, ctx)


def test_exhaustiveness_passes_when_all_accounted():
    plan = TrainingPlan(
        problem_type="regression",
        candidates=[TrainingPlanCandidate(priority=1, model_key="ridge", reason="baseline")],
        models_not_recommended=[RejectedModel(model_key="lightgbm_regressor", reason="too slow")],
    )
    _check_plan_exhaustiveness(plan, ["ridge", "lightgbm_regressor"])  # must not raise


def test_exhaustiveness_raises_when_model_missing():
    plan = TrainingPlan(
        problem_type="regression",
        candidates=[TrainingPlanCandidate(priority=1, model_key="ridge", reason="baseline")],
        models_not_recommended=[],
    )
    with pytest.raises(PlannerError, match="lightgbm_regressor"):
        _check_plan_exhaustiveness(plan, ["ridge", "lightgbm_regressor"])
```

- [ ] **Step 3: Run both test files to verify they fail**

```
uv run pytest tests/test_agents/test_planner_context.py tests/test_agents/test_planner_evidence.py -v
```

Expected: `ModuleNotFoundError: No module named 'mlops_agents.agents.planner'`

- [ ] **Step 4: Create `agents/planner.py` with deterministic helpers**

```python
# src/mlops_agents/agents/planner.py
"""Model Planning Agent — context builder, validators, and planner node (SP5)."""
from __future__ import annotations
from typing import Any
from mlops_agents.contracts.planner import (
    CandidateResultCompact,
    EvidenceReference,
    ExperienceSummary,
    PlannerContext,
    PlannerOutput,
)
from mlops_agents.contracts.training import TrainingPlan
from mlops_agents.experience.pool import ExperiencePool
from mlops_agents.experience.schema import RetrievalView
from mlops_agents.knowledge.reader import match_rules
from mlops_agents.models.loader import get_models_for
from mlops_agents.utils.logging import get_logger

logger = get_logger(__name__)


class PlannerError(Exception):
    """Raised when the planner fails validation after all retry attempts."""


def _to_experience_summary(view: RetrievalView) -> ExperienceSummary:
    """Convert a RetrievalView to the compact ExperienceSummary sent to the LLM."""
    sel_key = view.selected_solution.model_key
    scored = [c for c in view.models_tested if c.best_score is not None]
    # Populated from CandidateResult rows ordered by rank according to metric direction.
    # Champion is always rank 1; remaining candidates sorted per metric direction.
    direction = getattr(view, "metric_direction", "maximize")
    if direction == "minimize":
        scored.sort(key=lambda c: (c.model_key != sel_key, c.best_score or float("inf")))
    else:
        scored.sort(key=lambda c: (c.model_key != sel_key, -(c.best_score or 0.0)))
    compact = [
        CandidateResultCompact(model_key=c.model_key, rank=i + 1, metric_value=c.best_score)
        for i, c in enumerate(scored)
    ]
    failed = [c for c in view.models_tested if c.best_score is None]
    for f in failed:
        compact.append(CandidateResultCompact(
            model_key=f.model_key, rank=len(compact) + 1, metric_value=None
        ))
    return ExperienceSummary(
        experience_id=view.task_id,
        similarity_score=view.similarity_ratio,
        dataset_summary=view.experience_summary or "",
        models_trained=[c.model_key for c in view.models_tested],
        best_model=sel_key,
        validation_score=view.selected_solution.validation_score,
        candidate_results=compact,
    )


def build_planner_context(
    profile: dict[str, Any],
    task_metadata: dict[str, Any],
    problem_type: str,
    pool: ExperiencePool,
    k: int = 5,
) -> PlannerContext:
    """Assemble PlannerContext deterministically — no LLM calls."""
    available_models = [m.model_key for m in get_models_for(problem_type)]
    views = pool.find_similar(profile, problem_type, k)
    similar_experiences = [_to_experience_summary(v) for v in views]
    rule_input = {**profile, **task_metadata, "problem_type": problem_type}
    matched = match_rules(rule_input)
    matched_rules_dicts = [
        {
            "rule_id": r.rule_id,
            "prefer": r.prefer,
            "avoid_or_deprioritize": r.avoid_or_deprioritize,
            "recommend": r.recommend,
            "summary": r.reason,
        }
        for r in matched
    ]
    return PlannerContext(
        current_dataset_profile=profile,
        task_metadata=task_metadata,
        available_models=available_models,
        similar_experiences=similar_experiences,
        matched_rules=matched_rules_dicts,
    )


def _check_evidence_references(
    refs: list[EvidenceReference], ctx: PlannerContext
) -> None:
    """Verify every EvidenceReference source_id exists in the context. Raises PlannerError."""
    exp_ids = {e.experience_id for e in ctx.similar_experiences}
    rule_ids = {r["rule_id"] for r in ctx.matched_rules}
    model_keys = set(ctx.available_models)

    for ref in refs:
        if ref.source in ("dataset_profile", "task_metadata"):
            if ref.source_id is not None:
                raise PlannerError(
                    f"{ref.source} reference must have source_id=None, got {ref.source_id!r}"
                )
        elif ref.source == "experience":
            if ref.source_id not in exp_ids:
                raise PlannerError(
                    f"experience source_id {ref.source_id!r} not in context"
                )
        elif ref.source == "rule":
            if ref.source_id not in rule_ids:
                raise PlannerError(
                    f"rule source_id {ref.source_id!r} not in context"
                )
        elif ref.source == "registry":
            if ref.source_id not in model_keys:
                raise PlannerError(
                    f"registry source_id {ref.source_id!r} not in available_models"
                )


def _check_plan_exhaustiveness(
    plan: TrainingPlan, available_models: list[str]
) -> None:
    """Every model in available_models must appear in candidates or models_not_recommended."""
    accounted = (
        {c.model_key for c in plan.candidates}
        | {r.model_key for r in plan.models_not_recommended}
    )
    missing = set(available_models) - accounted
    if missing:
        raise PlannerError(
            f"These models are neither in candidates nor models_not_recommended: "
            f"{sorted(missing)}. Every available model must be explicitly included or rejected."
        )
```

- [ ] **Step 5: Run both test files to verify they pass**

```
uv run pytest tests/test_agents/test_planner_context.py tests/test_agents/test_planner_evidence.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```
git add src/mlops_agents/agents/planner.py tests/test_agents/test_planner_context.py tests/test_agents/test_planner_evidence.py
git commit -m "feat(sp5): add planner context builder and evidence validators"
```

---

## Task 5: `planner_node` — LLM call + retry loop

**Files:**
- Modify: `src/mlops_agents/agents/planner.py` (append `planner_node`)
- Test: `tests/test_agents/test_planner_node.py`

- [ ] **Step 1: Write failing node tests**

```python
# tests/test_agents/test_planner_node.py
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest
from pydantic import ValidationError
from mlops_agents.agents.planner import PlannerError, planner_node
from mlops_agents.contracts.planner import PlannerOutput
from mlops_agents.contracts.training import (
    TrainingPlan, TrainingPlanCandidate, RejectedModel
)
from mlops_agents.models.loader import get_models_for


def _full_regression_plan() -> TrainingPlan:
    """Plan that includes ALL regression models (passes exhaustiveness check)."""
    models = get_models_for("regression")
    candidates = [
        TrainingPlanCandidate(priority=i + 1, model_key=m.model_key, reason="test")
        for i, m in enumerate(models)
    ]
    return TrainingPlan(problem_type="regression", candidates=candidates)


def _valid_planner_output() -> PlannerOutput:
    return PlannerOutput(
        planning_analysis="Selected all regression models for test.",
        plan=_full_regression_plan(),
    )


def _minimal_state(tmp_path: Path) -> dict:
    import pandas as pd
    csv = tmp_path / "data.csv"
    pd.DataFrame({"f1": range(200), "target": range(200)}).to_csv(csv, index=False)
    return {
        "processed_dataset_path": str(csv),
        "problem_type": "regression",
        "task_metadata": {"target_column": "target"},
        "messages": [],
        "planner_analysis": "",
        "planner_evidence_used": [],
        "planner_warnings": [],
        "planner_status": "",
        "planner_retry_used": False,
        "training_plan": None,
        "error_message": "",
    }


def test_planner_node_happy_path(tmp_path):
    state = _minimal_state(tmp_path)
    mock_output = _valid_planner_output()

    with patch("mlops_agents.agents.planner.get_llm") as mock_get_llm, \
         patch("mlops_agents.agents.planner.ExperiencePool") as mock_pool_cls:
        mock_pool_cls.return_value.find_similar.return_value = []
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = mock_output
        mock_get_llm.return_value = mock_llm

        cmd = planner_node(state)

    update = cmd.update
    assert update["planner_status"] == "ok"
    assert update["planner_retry_used"] is False
    assert update["planner_analysis"] == "Selected all regression models for test."
    assert update["training_plan"] is not None


def test_planner_node_retry_on_first_failure(tmp_path):
    state = _minimal_state(tmp_path)
    mock_output = _valid_planner_output()

    with patch("mlops_agents.agents.planner.get_llm") as mock_get_llm, \
         patch("mlops_agents.agents.planner.ExperiencePool") as mock_pool_cls:
        mock_pool_cls.return_value.find_similar.return_value = []
        mock_llm = MagicMock()
        invoke_mock = mock_llm.with_structured_output.return_value.invoke
        invoke_mock.side_effect = [PlannerError("bad evidence"), mock_output]
        mock_get_llm.return_value = mock_llm

        cmd = planner_node(state)

    update = cmd.update
    assert update["planner_status"] == "retry_ok"
    assert update["planner_retry_used"] is True
    assert invoke_mock.call_count == 2


def test_planner_node_raises_after_two_failures(tmp_path):
    state = _minimal_state(tmp_path)

    with patch("mlops_agents.agents.planner.get_llm") as mock_get_llm, \
         patch("mlops_agents.agents.planner.ExperiencePool") as mock_pool_cls:
        mock_pool_cls.return_value.find_similar.return_value = []
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.side_effect = PlannerError("bad")
        mock_get_llm.return_value = mock_llm

        with pytest.raises(PlannerError, match="Planner failed after retry"):
            planner_node(state)


def test_planner_node_check_integrity_failure_triggers_retry(tmp_path):
    """_check_plan_integrity failure (ValueError) also triggers retry."""
    state = _minimal_state(tmp_path)
    models = get_models_for("regression")
    # Plan missing models_not_recommended for some models (exhaustiveness fails)
    bad_plan = TrainingPlan(
        problem_type="regression",
        candidates=[TrainingPlanCandidate(priority=1, model_key=models[0].model_key, reason="x")],
        models_not_recommended=[],
    )
    bad_output = PlannerOutput(planning_analysis="bad", plan=bad_plan)
    good_output = _valid_planner_output()

    with patch("mlops_agents.agents.planner.get_llm") as mock_get_llm, \
         patch("mlops_agents.agents.planner.ExperiencePool") as mock_pool_cls:
        mock_pool_cls.return_value.find_similar.return_value = []
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.side_effect = [bad_output, good_output]
        mock_get_llm.return_value = mock_llm

        cmd = planner_node(state)

    assert cmd.update["planner_status"] == "retry_ok"
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_agents/test_planner_node.py -v
```

Expected: `ImportError: cannot import name 'planner_node'`

- [ ] **Step 3: Append `planner_node` to `agents/planner.py`**

Add these imports at the top of `src/mlops_agents/agents/planner.py` (after existing imports):

```python
import json
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command
from pydantic import ValidationError

from mlops_agents.config.settings import settings
from mlops_agents.prompts import get_prompt
from mlops_agents.state.agent_state import AgentState
from mlops_agents.training.profiler import build_dataset_profile
from mlops_agents.utils.llm import get_llm
```

Append the `planner_node` function at the bottom of `agents/planner.py`:

```python
_planner_prompt = get_prompt("planner").template


def planner_node(state: AgentState) -> Command[Literal["supervisor"]]:
    """Model Planning Agent node — assembles context, calls LLM, validates plan."""
    processed_path = Path(state["processed_dataset_path"])
    problem_type: str = state.get("problem_type", "")
    task_meta: dict = state.get("task_metadata") or {}

    # Reuse dataset_profile from state (produced by data_validator) to avoid recomputing.
    # Fall back to building it from the CSV only if the validator didn't store it.
    profile_raw = state.get("dataset_profile") or state.get("schema_json")
    if profile_raw:
        profile_dict = (
            profile_raw if isinstance(profile_raw, dict) else json.loads(profile_raw)
        )
    else:
        profile_dict = build_dataset_profile(processed_path, task_meta).model_dump()

    pool = ExperiencePool(settings.experience_db_path)
    ctx = build_planner_context(profile_dict, task_meta, problem_type, pool)

    llm = get_llm("planner").with_structured_output(PlannerOutput)
    last_error = ""
    output: PlannerOutput | None = None

    for attempt in range(2):
        try:
            messages: list = [SystemMessage(content=_planner_prompt),
                               HumanMessage(content=ctx.model_dump_json(indent=2))]
            if attempt == 1:
                messages.append(HumanMessage(
                    content=f"Your previous plan was rejected: {last_error}. "
                            "Please produce a corrected PlannerOutput."
                ))
            output = llm.invoke(messages)
            # Stage 3: evidence references
            _check_evidence_references(output.evidence_used, ctx)
            # Stage 4: exhaustiveness
            _check_plan_exhaustiveness(output.plan, ctx.available_models)
            break
        except (ValidationError, PlannerError, ValueError) as exc:
            last_error = str(exc)
            logger.warning(f"[planner] attempt {attempt + 1} failed: {last_error}")
            if attempt == 1:
                raise PlannerError(f"Planner failed after retry: {last_error}") from exc

    retry_used = attempt == 1
    assert output is not None

    planner_output_record = {
        "planner_status": "retry_ok" if retry_used else "ok",
        "retry_used": retry_used,
        "planning_analysis": output.planning_analysis,
        "evidence_used": [e.model_dump() for e in output.evidence_used],
        "risks_or_warnings": output.risks_or_warnings,
        "validation_errors": [last_error] if retry_used else [],
        "plan_summary": {
            "candidate_models": [c.model_key for c in output.plan.candidates],
            "models_not_recommended": [r.model_key for r in output.plan.models_not_recommended],
        },
        "prompt_version": "model_planner_v1",
    }

    logger.info(f"[planner] status={'retry_ok' if retry_used else 'ok'} "
                f"candidates={len(output.plan.candidates)} "
                f"rejected={len(output.plan.models_not_recommended)}")

    return Command(
        goto="supervisor",
        update={
            "planner_analysis": output.planning_analysis,
            "planner_evidence_used": [e.model_dump() for e in output.evidence_used],
            "planner_warnings": output.risks_or_warnings,
            "planner_status": "retry_ok" if retry_used else "ok",
            "planner_retry_used": retry_used,
            "training_plan": output.plan.model_dump(),
            "_planner_output_record": planner_output_record,
        },
    )
```

> **Note:** `_planner_output_record` is a private state key used by `executor_node` to write the planner output into the `ExperienceRecord`. Add it to `AgentState` as `_planner_output_record: dict | None` (with leading underscore to signal internal use).

- [ ] **Step 4: Add `_planner_output_record` to `AgentState`**

In `src/mlops_agents/state/agent_state.py`, add after the `planner_retry_used` field:

```python
    _planner_output_record: dict | None  # internal: written by planner, read by executor
```

- [ ] **Step 5: Run tests to verify they pass**

```
uv run pytest tests/test_agents/test_planner_node.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 6: Run full test suite**

```
uv run pytest -m "not integration" -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```
git add src/mlops_agents/agents/planner.py src/mlops_agents/state/agent_state.py tests/test_agents/test_planner_node.py
git commit -m "feat(sp5): add planner_node with LLM call, 4-stage validation, and retry"
```

---

## Task 6: Migration 004 + ExperienceRecord + pool

**Files:**
- Create: `src/mlops_agents/experience/migrations/004_add_planner_output.sql`
- Modify: `src/mlops_agents/experience/schema.py`
- Modify: `src/mlops_agents/experience/pool.py`
- Modify: `tests/test_experience/test_migrations.py`

- [ ] **Step 1: Update migration tests**

In `tests/test_experience/test_migrations.py`, change every assertion `version == 3` to `version == 4`:

```python
def test_migrations_are_idempotent(tmp_path):
    db = tmp_path / "test.db"
    apply_pending_migrations(db)
    apply_pending_migrations(db)
    conn = sqlite3.connect(db)
    version = conn.execute("SELECT MAX(version) FROM _schema_version").fetchone()[0]
    assert version == 4   # changed from 3
    conn.close()


def test_migration_sets_schema_version(tmp_path):
    db = tmp_path / "test.db"
    apply_pending_migrations(db)
    conn = sqlite3.connect(db)
    row = conn.execute("SELECT MAX(version) FROM _schema_version").fetchone()
    assert row[0] == 4   # changed from 3
    conn.close()
```

Also add a new test:

```python
def test_migration_004_adds_planner_output_column(tmp_path):
    db = tmp_path / "test.db"
    apply_pending_migrations(db)
    conn = sqlite3.connect(db)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(experiences)").fetchall()}
    assert "planner_output_json" in cols
    conn.close()
```

- [ ] **Step 2: Run migration tests to verify they fail**

```
uv run pytest tests/test_experience/test_migrations.py -v
```

Expected: the two version assertions fail (expected 4, got 3). `test_migration_004_adds_planner_output_column` also fails.

- [ ] **Step 3: Create migration SQL**

```sql
-- src/mlops_agents/experience/migrations/004_add_planner_output.sql
ALTER TABLE experiences ADD COLUMN planner_output_json TEXT;

CREATE TABLE IF NOT EXISTS _schema_version (version INTEGER PRIMARY KEY);
INSERT OR IGNORE INTO _schema_version(version) VALUES (4);
```

- [ ] **Step 4: Run migration tests to verify they pass**

```
uv run pytest tests/test_experience/test_migrations.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Add `planner_output` field to `ExperienceRecord`**

In `src/mlops_agents/experience/schema.py`, add after `expected_drift`:

```python
    planner_output: dict | None = None
```

- [ ] **Step 6: Update `pool.py` INSERT and `get()`**

In `src/mlops_agents/experience/pool.py`:

**In `insert_from_record`**, change the INSERT SQL to include `planner_output_json`:

```python
            conn.execute(
                """INSERT OR REPLACE INTO experiences
                (task_id, problem_type, dataset_name, dataset_profile_json,
                 training_plan_json, selected_model_key, metric_to_optimize,
                 metric_direction, validation_score, validation_std,
                 experience_summary, mlflow_parent_run_id, created_at,
                 validation_strategy_json, exog_availability_json,
                 exog_strategies_json, per_fold_metrics_json, exog_fit_failures_json,
                 expected_drift, planner_output_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (record.task_id, record.problem_type, record.dataset_name,
                 json.dumps(record.dataset_profile), json.dumps(record.training_plan_input),
                 sol.model_key if sol else None, record.metric_to_optimize,
                 record.metric_direction,
                 sol.validation_score if sol else None, sol.validation_std if sol else None,
                 record.experience_summary, record.mlflow.get("parent_run_id"), created_at,
                 _opt_json(record.validation_strategy),
                 _opt_json(record.exog_availability),
                 _opt_json(record.exog_strategies),
                 _opt_json(record.per_fold_metrics),
                 _opt_json(record.exog_fit_failures),
                 record.expected_drift,
                 _opt_json(record.planner_output)),
            )
```

**In `get()`**, add `planner_output` to the returned `ExperienceRecord`:

```python
        return ExperienceRecord(
            ...
            expected_drift=row["expected_drift"],
            planner_output=_opt_load(row["planner_output_json"]),
        )
```

- [ ] **Step 7: Run full test suite**

```
uv run pytest -m "not integration" -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```
git add src/mlops_agents/experience/migrations/004_add_planner_output.sql src/mlops_agents/experience/schema.py src/mlops_agents/experience/pool.py tests/test_experience/test_migrations.py
git commit -m "feat(sp5): migration 004, planner_output field in ExperienceRecord and pool"
```

---

## Task 7: Pass `planner_output` through executor

**Files:**
- Modify: `src/mlops_agents/training/executor.py`

- [ ] **Step 1: Add `planner_output` parameter to `run_training_plan`**

In `src/mlops_agents/training/executor.py`, update the `run_training_plan` signature:

```python
def run_training_plan(
    plan: TrainingPlan,
    processed_dataset_path: Path,
    target_column: str,
    task_metadata: dict[str, Any],
    output_dir: Path,
    mlflow_experiment: str,
    random_state: int = 42,
    planner_output: dict | None = None,
) -> TrainingResult:
```

- [ ] **Step 2: Add `planner_output` to the record dict**

In the same function, find the `record: dict[str, Any] = {` block (around line 808) and add `"planner_output": planner_output` alongside the other fields:

```python
        record: dict[str, Any] = {
            "task_id": task_id,
            "problem_type": plan.problem_type,
            "dataset_name": processed_dataset_path.stem,
            "dataset_profile": profile.model_dump(),
            "training_plan_input": plan.model_dump(),
            "split_artifacts": {
                "train_pool_path": str(train_pool_path),
                "test_path": str(test_path),
                "split_metadata_path": str(split_meta_path),
            },
            "mlflow": {
                "experiment_name": mlflow_experiment,
                "parent_run_id": parent_run_id,
            },
            "metric_to_optimize": metric,
            "metric_direction": direction,
            "candidate_selection_policy": {
                "primary": "best_validation_score",
                "tie_breaker_chain": ["complexity_rank", "priority"],
                "tie_tolerance_relative": settings.tie_tolerance_relative,
            },
            "models_tested": [
                {k: v for k, v in r.items() if k != "traceback"}
                for r in candidate_results
            ],
            "selected_solution": {
                "model_key": champion["model_key"],
                "hyperparameters": champion["best_params"],
                "validation_strategy": val_strategy,
                "main_metric": metric,
                "validation_score": champion["best_score"],
                "validation_std": champion.get("best_score_std", 0.0),
                "complexity_rank": champion["complexity_rank"],
            },
            "experience_summary": "",
            "planner_output": planner_output,
            **forecasting_extras,
        }
```

- [ ] **Step 3: Run full test suite**

```
uv run pytest -m "not integration" -q
```

Expected: all tests pass (the `planner_output=None` default preserves backward compat).

- [ ] **Step 4: Commit**

```
git add src/mlops_agents/training/executor.py
git commit -m "feat(sp5): pass planner_output through run_training_plan into experience record"
```

---

## Task 8: Graph + supervisor updates

**Files:**
- Modify: `src/mlops_agents/graphs/mlops_graph.py`
- Modify: `src/mlops_agents/agents/supervisor.py`
- Modify: `src/mlops_agents/prompts/supervisor.yaml`

- [ ] **Step 1: Update `mlops_graph.py`**

**a) Add import at the top:**

```python
from mlops_agents.agents.planner import PlannerError, planner_node
```

**b) Rename `trainer_node` → `executor_node`** throughout `mlops_graph.py` (function definition and `builder.add_node` call):

- Line with `def trainer_node(state: AgentState)` → `def executor_node(state: AgentState)`
- Line with `builder.add_node("trainer", trainer_node)` → `builder.add_node("executor", executor_node)`

**c) In `executor_node`, replace `default_training_plan` with the planner-generated plan from state:**

This is the core SP5 change. The executor must stop generating its own default plan. Replace the existing plan-building block (which calls `default_training_plan`) with:

```python
    raw_plan = state.get("training_plan")
    if raw_plan is None:
        raise RuntimeError(
            "executor_node expected a planner-generated training_plan, but none was found. "
            "Ensure the planner node ran successfully before executor."
        )
    plan = TrainingPlan.model_validate(raw_plan)

    planner_out = state.get("_planner_output_record")
    result = run_training_plan(
        plan=plan,
        processed_dataset_path=processed_path,
        target_column=task_meta.get("target_column", "target"),
        task_metadata=task_meta,
        output_dir=_Path("data/processed"),
        mlflow_experiment=settings.mlflow_experiment_name,
        planner_output=planner_out,
    )
```

Also add the `TrainingPlan` import at the top of `mlops_graph.py` if not already present:

```python
from mlops_agents.contracts.training import TrainingPlan
```

> **Why:** Without this guard, executor_node would silently fall back to `default_training_plan` if `state["training_plan"]` was missing — defeating the entire SP5 purpose. The `RuntimeError` makes the missing-planner-output a loud failure instead of a silent regression.

**d) Add `planner_node` to the graph builder and add error handling:**

Replace the `_build_graph` function with:

```python
def _build_graph(checkpointer=None) -> StateGraph:
    builder = StateGraph(AgentState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("data_validator", data_validator_node)
    builder.add_node("planner", _planner_node_with_error_handling)
    builder.add_node("executor", executor_node)
    builder.add_node("evaluator", evaluator_node)
    builder.add_node("deployer", deployer_node)
    builder.add_edge(START, "supervisor")
    return builder.compile(checkpointer=checkpointer)
```

**e) Add the error-handling wrapper just before `_build_graph`:**

```python
def _planner_node_with_error_handling(
    state: AgentState,
) -> Command[Literal["supervisor"]]:
    """Wrap planner_node to catch PlannerError and route to END gracefully."""
    try:
        return planner_node(state)
    except PlannerError as exc:
        logger.error(f"[planner] failed after retry: {exc}")
        return Command(
            goto="supervisor",
            update={
                "planner_status": "failed",
                "planner_retry_used": True,
                "error_message": f"Model planner failed: {exc}",
                "messages": [HumanMessage(content=f"Planner failed: {exc}", name="planner")],
            },
        )
```

- [ ] **Step 2: Update `supervisor.py` routing Literal**

In `src/mlops_agents/agents/supervisor.py`, change:

```python
) -> Command[Literal["data_validator", "trainer", "evaluator", "deployer", "__end__"]]:
```

to:

```python
) -> Command[Literal["data_validator", "planner", "executor", "evaluator", "deployer", "__end__"]]:
```

Also add `planner_status` to the state snapshot dict:

```python
    snapshot_data = {
        "problem_type": state.get("problem_type", ""),
        "validation_passed": state.get("validation_passed") if dv_has_run else None,
        "planner_status": state.get("planner_status", ""),
        "evaluation_passed": state.get("evaluation_passed"),
        "deployment_decision": state.get("deployment_decision", "pending"),
        "error_message": state.get("error_message", ""),
        "training_run_id": state.get("training_run_id", ""),
    }
```

- [ ] **Step 3: Update `supervisor.yaml`**

Replace the contents of `src/mlops_agents/prompts/supervisor.yaml` with:

```yaml
_type: "prompt"
input_variables: []
template: |
  You are the MLOps Pipeline Supervisor coordinating a team of 5 specialist nodes.

  Your team:
  - data_validator: Validates dataset schema, checks for missing values, and detects data drift using Evidently AI.
  - planner: LLM-based Model Planning Agent that synthesizes evidence from the experience pool, ML rules, and the model registry to produce a structured TrainingPlan.
  - executor: Deterministic training executor — runs Optuna hyperparameter tuning on the TrainingPlan candidates, selects the champion, and logs everything to MLflow.
  - evaluator: Evaluates the trained champion model on test data, compares against the production baseline, and recommends promotion or rejection.
  - deployer: Registers the best model in the MLflow Model Registry and requests human approval before promotion to production.

  PIPELINE RULES (follow these strictly):
  1. Always start with data_validator — never skip data validation.
  2. Only proceed to planner if validation_passed=True.
  3. Only proceed to executor after planner completes (planner_status is "ok" or "retry_ok").
  4. Only proceed to evaluator after executor completes (training_run_id is set in state).
  5. Only proceed to deployer if the evaluator recommends 'promote'.
  6. Check the "Pipeline state:" message (always the last message) for routing signals:
     - If error_message is non-empty → always select FINISH, no exceptions.
     - validation_passed is null before data_validator has run — only treat it as a
       signal after data_validator has been invoked. If validation_passed=False (not null)
       → select FINISH, never retry data_validator.
     - planner_status="failed" → select FINISH immediately.
     - Use deployment_decision, evaluation_passed, and training_run_id to confirm
       pipeline stage — do not infer these from narrative summaries alone.
  7. Select FINISH when the full pipeline completes successfully.
  8. Do not route to the same agent twice in a row unless recovering from a transient error.

  Always include a brief reasoning for your routing decision.
```

- [ ] **Step 4: Run full test suite**

```
uv run pytest -m "not integration" -q
```

Expected: all tests pass.

- [ ] **Step 5: Verify graph compiles with correct nodes**

```
uv run python -c "
from mlops_agents.graphs.mlops_graph import graph
nodes = set(graph.nodes.keys())
assert 'planner' in nodes, f'planner missing from {nodes}'
assert 'executor' in nodes, f'executor missing from {nodes}'
assert 'trainer' not in nodes, f'trainer should be renamed to executor'
print('Graph OK:', sorted(nodes))
"
```

Expected: prints `Graph OK: ['__end__', '__start__', 'data_validator', 'deployer', 'evaluator', 'executor', 'planner', 'supervisor']`

- [ ] **Step 6: Commit**

```
git add src/mlops_agents/graphs/mlops_graph.py src/mlops_agents/agents/supervisor.py src/mlops_agents/prompts/supervisor.yaml
git commit -m "feat(sp5): wire planner node into graph, rename trainer→executor, update supervisor routing"
```

---

## Task 9: Integration test (optional — requires LLM key)

**Files:**
- Create: `tests/test_agents/test_planner_integration.py`

- [ ] **Step 1: Create the integration test**

```python
# tests/test_agents/test_planner_integration.py
"""Integration test for the full planner flow — requires GITHUB_TOKEN + --llm flag."""
import pytest
from pathlib import Path
import pandas as pd
from mlops_agents.agents.planner import build_planner_context, PlannerError
from mlops_agents.contracts.planner import PlannerOutput
from mlops_agents.contracts.training import TrainingPlan
from mlops_agents.experience.pool import ExperiencePool
from mlops_agents.models.loader import get_models_for
from mlops_agents.prompts import get_prompt
from langchain_core.messages import HumanMessage, SystemMessage
from mlops_agents.agents.planner import (
    _check_evidence_references,
    _check_plan_exhaustiveness,
)
from mlops_agents.utils.llm import get_llm


@pytest.mark.integration
@pytest.mark.llm
def test_planner_produces_valid_plan_for_regression(tmp_path: Path):
    """Real LLM call — produces a valid plan for a medium regression dataset."""
    pool = ExperiencePool(tmp_path / "test.db")

    profile = {
        "schema_version": 1, "problem_type": "regression",
        "n_rows": "small", "n_features": "small",
        "missing_rate": "none", "n_categorical_features": "none",
        "n_numerical_features": "few",
        "target_distribution": "near_normal",
    }
    task_metadata = {"target_column": "target"}

    ctx = build_planner_context(profile, task_metadata, "regression", pool)
    assert len(ctx.available_models) > 0

    llm = get_llm("planner").with_structured_output(PlannerOutput)
    prompt = get_prompt("planner").template
    output: PlannerOutput = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=ctx.model_dump_json(indent=2)),
    ])

    # Stage 3: evidence references
    _check_evidence_references(output.evidence_used, ctx)
    # Stage 4: exhaustiveness
    _check_plan_exhaustiveness(output.plan, ctx.available_models)

    # All candidate model_keys must be in available_models
    for cand in output.plan.candidates:
        assert cand.model_key in ctx.available_models, f"{cand.model_key} not in available_models"

    # All rejected models must have non-empty reason
    for rej in output.plan.models_not_recommended:
        assert rej.reason.strip(), f"{rej.model_key} has empty reason"

    # Every available model accounted for
    accounted = (
        {c.model_key for c in output.plan.candidates}
        | {r.model_key for r in output.plan.models_not_recommended}
    )
    assert accounted == set(ctx.available_models)

    # planning_analysis is non-empty
    assert len(output.planning_analysis) > 50
```

- [ ] **Step 2: Verify integration test is skipped in normal test run**

```
uv run pytest tests/test_agents/test_planner_integration.py -v -m "not integration"
```

Expected: `0 passed, 1 skipped` (or deselected).

- [ ] **Step 3: Commit**

```
git add tests/test_agents/test_planner_integration.py
git commit -m "test(sp5): add integration test for planner LLM call"
```

---

## Task 10: Final verification

- [ ] **Step 1: Run full unit test suite**

```
uv run pytest -m "not integration" -q
```

Expected: all tests pass (count ≥ 326 + new SP5 tests).

- [ ] **Step 2: Lint check**

```
uv run ruff check src/mlops_agents/contracts/planner.py src/mlops_agents/agents/planner.py src/mlops_agents/graphs/mlops_graph.py
```

Expected: no errors.

- [ ] **Step 3: Verify graph topology**

```
uv run python -c "
from mlops_agents.graphs.mlops_graph import graph
nodes = set(graph.nodes.keys())
print('Nodes:', sorted(nodes))
assert {'planner', 'executor', 'data_validator', 'evaluator', 'deployer', 'supervisor'}.issubset(nodes)
assert 'trainer' not in nodes
print('OK')
"
```

Expected: prints `OK`.
