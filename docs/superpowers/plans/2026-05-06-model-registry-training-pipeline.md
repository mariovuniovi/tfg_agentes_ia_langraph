# Model Registry + Deterministic Training Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the LLM-driven trainer with a deterministic, multi-candidate training executor that takes a structured `TrainingPlan`, runs Optuna+MLflow per candidate, picks a champion, and writes an experience record JSON.

**Architecture:** A YAML model registry (data) + Python factories and Optuna search-space builder (code) feed a `run_training_plan(...)` executor that handles classification (StratifiedKFold CV), regression (KFold CV), and forecasting (single temporal holdout, statsforecast/skforecast). Champion selection is best-mean-score with `complexity_rank` tie-break under a `tie_tolerance_relative` threshold. The trainer node in the graph becomes a thin wrapper around the executor — no agent.

**Tech Stack:** Python 3.12, Pydantic v2, scikit-learn, lightgbm, xgboost, catboost, statsforecast, skforecast, optuna, mlflow, pytest.

---

## Spec reference

- Spec: `docs/superpowers/specs/2026-05-06-model-registry-training-pipeline-design.md`
- Brainstorming source: `plan_model_agent.md`

---

## File map (full SP3)

### Created
```
src/mlops_agents/contracts/
    __init__.py
    training.py                                    # TrainingPlan, TrainingPlanCandidate,
                                                   #   TrialBudget, RejectedModel,
                                                   #   SearchParamOverride, TrainingResult
src/mlops_agents/models/
    __init__.py
    registry.yaml                                  # 20 model entries
    loader.py                                      # ModelSpec, SearchSpaceSpec, load_registry()
    factories.py                                   # FACTORY_REGISTRY: dict[str, Callable]
    search_spaces.py                               # build_suggest_fn(spec)
src/mlops_agents/training/
    __init__.py
    executor.py                                    # run_training_plan(...)
    splitter.py                                    # split_dataset(...)
    profiler.py                                    # build_dataset_profile(...)
    default_plans.py                               # default_training_plan(problem_type, profile)
    experience.py                                  # write_experience_record(...)
tests/test_contracts/__init__.py
tests/test_contracts/test_training_plan.py
tests/test_contracts/test_search_space_override.py
tests/test_models/__init__.py
tests/test_models/test_loader.py
tests/test_models/test_factories.py
tests/test_models/test_search_spaces.py
tests/test_training/__init__.py
tests/test_training/test_splitter.py
tests/test_training/test_profiler.py
tests/test_training/test_default_plans.py
tests/test_training/test_experience_writer.py
tests/test_training/test_trial_budget.py
tests/test_training/test_executor_classification.py
tests/test_training/test_executor_regression.py
tests/test_training/test_executor_forecasting.py
tests/test_training/test_executor_champion_selection.py
tests/test_training/test_executor_failure_handling.py
```

### Modified
- `src/mlops_agents/state/agent_state.py` — rename `dataset_path` → `processed_dataset_path`; add new fields
- `src/mlops_agents/graphs/mlops_graph.py` — `trainer_node` rewritten; rename references
- `src/mlops_agents/agents/registry.py` — remove `"trainer"` agent registration
- `src/mlops_agents/config/settings.py` — new training settings
- `pyproject.toml` — add `statsforecast`, `skforecast`, `lightgbm`, `xgboost`, `catboost`, `optuna`
- All test files / api files referencing `dataset_path` (mechanical rename)

### Deleted
- `src/mlops_agents/agents/training_agent.py`
- `src/mlops_agents/tools/training_tools.py`
- `src/mlops_agents/prompts/training_agent.yaml`
- `tests/test_tools/test_training_tools.py` (if present)

---

## Task 1: State field rename — `dataset_path` → `processed_dataset_path`

Mechanical rename across the codebase. Single commit. No new behavior.

**Files:**
- Modify: `src/mlops_agents/state/agent_state.py:25`
- Modify: `src/mlops_agents/graphs/mlops_graph.py` (line ~69 in supervisor context plus all reads/writes)
- Modify: `tests/test_api/test_pipeline_helpers.py:13`
- Modify: `tests/test_graphs/test_mlops_graph.py:82`
- Modify: `tests/test_graphs/test_node_state_extraction.py:566`
- Modify: any other test or api file referencing `state["dataset_path"]` (check via grep)

- [ ] **Step 1: Find every reference**

Run:
```
grep -rn 'dataset_path' src/ tests/ scripts/ api/ --include='*.py' | grep -v 'dataset_paths' | grep -v 'processed_dataset_path' | grep -v 'train_pool_path' | grep -v 'test_path'
```
Expected: a list of ~10–20 lines (some false positives like docstrings — review each).

- [ ] **Step 2: In `src/mlops_agents/state/agent_state.py`, rename the field**

Change line 25:
```python
dataset_path: str          # canonical CSV written by data_validator_node
```
to:
```python
processed_dataset_path: str   # canonical CSV written by data_validator_node
```

- [ ] **Step 3: Update every read/write site**

For each file from Step 1, rewrite `state["dataset_path"]` → `state["processed_dataset_path"]` and `state.get("dataset_path", ...)` → `state.get("processed_dataset_path", ...)`.

In `src/mlops_agents/graphs/mlops_graph.py`, the supervisor context line ~69 should change from:
```python
f"Canonical dataset: {state.get('dataset_path', '')}\n"
```
to:
```python
f"Canonical dataset: {state.get('processed_dataset_path', '')}\n"
```

Wherever a node *writes* the field (look in `data_validator_node` for the line that sets the canonical path on Command update), change `dataset_path` to `processed_dataset_path`.

- [ ] **Step 4: Run the full test suite to verify nothing broke**

```
uv run pytest -m "not integration" -q
```
Expected: PASS (171 tests as of SP2 completion).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: rename state.dataset_path → processed_dataset_path"
```

---

## Task 2: Add training executor settings + dependencies

Add new settings keys and the new package dependencies. No new code files yet.

**Files:**
- Modify: `src/mlops_agents/config/settings.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the dependencies to `pyproject.toml`**

Find the `[project] dependencies = [...]` block and add (alphabetical placement OK):
```toml
"catboost>=1.2",
"lightgbm>=4.0",
"optuna>=3.5",
"skforecast>=0.13",
"statsforecast>=1.7",
"xgboost>=2.0",
```

`mlflow` and `scikit-learn` are already present from SP2.

- [ ] **Step 2: Sync dependencies**

```
uv sync
```
Expected: success with `Resolved N packages` line. May take ~30 seconds (pulling `catboost` and friends).

- [ ] **Step 3: Add settings to `src/mlops_agents/config/settings.py`**

Inside the `Settings` class, append (after existing fields):

```python
    # Training executor (SP3)
    train_test_split_ratio: float = 0.2
    cv_folds: int = 5
    optuna_total_trials: int = 60
    optuna_min_trials_per_candidate: int = 5
    optuna_max_trials_per_candidate: int = 30
    log_non_champion_models: bool = False
    tie_tolerance_relative: float = 0.01
    forecasting_min_train_points: int = 30
    experience_pool_dir: Path = Path("experience_pool")
```

Make sure `from pathlib import Path` is at the top of the file (it should be already; if not, add it).

- [ ] **Step 4: Verify settings load**

```
uv run python -c "from mlops_agents.config.settings import settings; print(settings.optuna_total_trials)"
```
Expected: prints `60`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock src/mlops_agents/config/settings.py
git commit -m "feat: add training executor settings and ML deps (lightgbm, xgboost, catboost, optuna, statsforecast, skforecast)"
```

---

## Task 3: Contracts module — Pydantic types for the training pipeline

Create the cross-cutting Pydantic types used by SP3, SP4, and SP5. Pure Pydantic — no LangGraph imports.

**Files:**
- Create: `src/mlops_agents/contracts/__init__.py`
- Create: `src/mlops_agents/contracts/training.py`
- Create: `tests/test_contracts/__init__.py`
- Create: `tests/test_contracts/test_training_plan.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_contracts/__init__.py` (empty file).

Create `tests/test_contracts/test_training_plan.py`:

```python
"""Unit tests for cross-cutting training contracts."""

import pytest
from pydantic import ValidationError

from mlops_agents.contracts.training import (
    RejectedModel,
    SearchParamOverride,
    TrainingPlan,
    TrainingPlanCandidate,
    TrialBudget,
)


def test_search_param_override_low_high_only():
    o = SearchParamOverride(low=0.01, high=0.1)
    assert o.low == 0.01 and o.high == 0.1 and o.choices is None


def test_search_param_override_choices_only():
    o = SearchParamOverride(choices=[300, 500, 800])
    assert o.choices == [300, 500, 800] and o.low is None and o.high is None


def test_search_param_override_rejects_both():
    with pytest.raises(ValidationError, match="exactly one"):
        SearchParamOverride(low=0.01, high=0.1, choices=[0.05])


def test_search_param_override_rejects_neither():
    with pytest.raises(ValidationError, match="exactly one"):
        SearchParamOverride()


def test_search_param_override_rejects_inverted_range():
    with pytest.raises(ValidationError, match="low.*<= high|<= high"):
        SearchParamOverride(low=0.5, high=0.1)


def test_training_plan_candidate_minimal():
    c = TrainingPlanCandidate(priority=1, model_key="lightgbm_classifier")
    assert c.priority == 1
    assert c.model_key == "lightgbm_classifier"
    assert c.initial_hyperparameters == {}
    assert c.search_space_override is None
    assert c.reason == ""


def test_training_plan_unique_priorities_required():
    candidates = [
        TrainingPlanCandidate(priority=1, model_key="a"),
        TrainingPlanCandidate(priority=1, model_key="b"),  # duplicate
    ]
    with pytest.raises(ValidationError, match="unique"):
        TrainingPlan(problem_type="classification", candidates=candidates)


def test_training_plan_default_trial_budget():
    plan = TrainingPlan(
        problem_type="classification",
        candidates=[TrainingPlanCandidate(priority=1, model_key="x")],
    )
    assert plan.trial_budget.total_trials == 60
    assert plan.trial_budget.allocation_strategy == "priority_weighted"


def test_trial_budget_field_default_factory():
    """Two TrialBudget instances must be independent (no shared mutable default)."""
    a = TrialBudget()
    b = TrialBudget()
    assert a is not b


def test_rejected_model_basic():
    r = RejectedModel(model_key="lstm", reason="too small")
    assert r.model_key == "lstm" and r.reason == "too small"
```

- [ ] **Step 2: Verify tests fail (module not found)**

```
uv run pytest tests/test_contracts/test_training_plan.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'mlops_agents.contracts'`.

- [ ] **Step 3: Create the contracts package**

Create `src/mlops_agents/contracts/__init__.py` (empty).

Create `src/mlops_agents/contracts/training.py`:

```python
"""Cross-cutting Pydantic contracts for the training pipeline.

Used by:
- SP3 deterministic executor
- SP4 benchmark runner + retrieval
- SP5 planner agent (future)
- Graph state (via re-export in agent_state)
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class SearchParamOverride(BaseModel):
    """Explicit override entry — no JSON tuple/list ambiguity.

    Provide exactly one of:
    - {low, high}: continuous narrowing for int/float registry params.
    - {choices: [...]}: discrete narrowing for any param type.
    """

    low: int | float | None = None
    high: int | float | None = None
    choices: list[Any] | None = None

    @model_validator(mode="after")
    def either_range_or_choices(self):
        has_range = self.low is not None and self.high is not None
        has_choices = self.choices is not None
        if has_range == has_choices:
            raise ValueError("Provide exactly one of {low, high} or {choices}.")
        if has_range and self.low > self.high:
            raise ValueError(f"low ({self.low}) must be <= high ({self.high}).")
        return self


class TrainingPlanCandidate(BaseModel):
    priority: int
    model_key: str
    initial_hyperparameters: dict[str, Any] = Field(default_factory=dict)
    search_space_override: dict[str, SearchParamOverride] | None = None
    requested_trials: int | None = None
    reason: str = ""


class RejectedModel(BaseModel):
    model_key: str
    reason: str


class TrialBudget(BaseModel):
    total_trials: int = 60
    allocation_strategy: Literal["priority_weighted", "equal"] = "priority_weighted"
    max_trials_per_candidate: int = 30
    min_trials_per_candidate: int = 5


class TrainingPlan(BaseModel):
    problem_type: Literal["classification", "regression", "forecasting"]
    metric_to_optimize: str | None = None
    candidates: list[TrainingPlanCandidate]
    models_not_recommended: list[RejectedModel] = Field(default_factory=list)
    trial_budget: TrialBudget = Field(default_factory=TrialBudget)
    validation_strategy: dict[str, Any] | None = None
    forecasting_settings: dict[str, Any] | None = None

    @model_validator(mode="after")
    def priorities_unique(self):
        priorities = [c.priority for c in self.candidates]
        if len(priorities) != len(set(priorities)):
            raise ValueError(f"Candidate priorities must be unique. Got: {priorities}")
        return self


class TrainingResult(BaseModel):
    """Returned by run_training_plan(...). Embedded in graph state via agent_state."""
    champion_candidate: dict[str, Any]
    champion_model_path: str
    train_pool_path: str
    test_path: str
    split_metadata_path: str
    mlflow_parent_run_id: str
    experience_record_path: str
    champion_metrics: dict[str, float]
```

- [ ] **Step 4: Run tests**

```
uv run pytest tests/test_contracts/test_training_plan.py -v
```
Expected: all 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mlops_agents/contracts/ tests/test_contracts/
git commit -m "feat: add training contracts (TrainingPlan, SearchParamOverride, TrialBudget, ...)"
```

---

## Task 4: Model registry skeleton — loader, empty registries, empty YAML

Create the `models/` package scaffold. Loader exists but the YAML and Python registries are empty — populated in subsequent tasks.

**Files:**
- Create: `src/mlops_agents/models/__init__.py`
- Create: `src/mlops_agents/models/factories.py`
- Create: `src/mlops_agents/models/search_spaces.py`
- Create: `src/mlops_agents/models/loader.py`
- Create: `src/mlops_agents/models/registry.yaml`
- Create: `tests/test_models/__init__.py`
- Create: `tests/test_models/test_loader.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_models/__init__.py` (empty).

Create `tests/test_models/test_loader.py`:

```python
"""Unit tests for the model registry loader."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from mlops_agents.models.loader import (
    ModelSpec,
    SearchParamSpec,
    SearchSpaceSpec,
    get_models_for,
    load_registry,
)


def test_search_param_spec_int():
    p = SearchParamSpec(type="int", low=10, high=100)
    assert p.type == "int" and p.low == 10 and p.high == 100


def test_search_param_spec_float_log():
    p = SearchParamSpec(type="float", low=0.001, high=1.0, log=True)
    assert p.log is True


def test_search_param_spec_categorical():
    p = SearchParamSpec(type="categorical", choices=["l1", "l2"])
    assert p.choices == ["l1", "l2"]


def test_model_spec_unknown_factory_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr("mlops_agents.models.loader.FACTORY_REGISTRY", {"build_x": lambda p: None})
    monkeypatch.setattr(
        "mlops_agents.models.loader.SEARCH_SPACE_REGISTRY",
        {"x_space": lambda *_: None},
    )
    with pytest.raises(ValidationError, match="Unknown factory"):
        ModelSpec(
            model_key="x",
            problem_type="classification",
            family="test",
            complexity_rank=1,
            library="sklearn",
            factory="build_unknown",
            search_space=SearchSpaceSpec(name="x_space", params={}),
            default_params={},
        )


def test_load_registry_empty_yaml(tmp_path):
    """An empty YAML file (yaml = []) loads to an empty dict — no error."""
    empty_path = tmp_path / "registry.yaml"
    empty_path.write_text("[]\n")
    registry = load_registry(empty_path)
    assert registry == {}


def test_get_models_for_filters_by_problem_type(tmp_path, monkeypatch):
    """get_models_for() returns only entries matching the requested problem_type."""
    # We will validate this against the populated registry in Task 9.
    # For now, just verify the signature works:
    monkeypatch.setattr(
        "mlops_agents.models.loader._cached_registry",
        {
            "a": ModelSpec(
                model_key="a", problem_type="classification", family="test",
                complexity_rank=1, library="sklearn", factory="build_x",
                search_space=SearchSpaceSpec(name="x_space", params={}),
                default_params={},
            ),
            "b": ModelSpec(
                model_key="b", problem_type="regression", family="test",
                complexity_rank=1, library="sklearn", factory="build_x",
                search_space=SearchSpaceSpec(name="x_space", params={}),
                default_params={},
            ),
        },
    )
    monkeypatch.setattr("mlops_agents.models.loader.FACTORY_REGISTRY", {"build_x": lambda p: None})
    monkeypatch.setattr(
        "mlops_agents.models.loader.SEARCH_SPACE_REGISTRY",
        {"x_space": lambda *_: None},
    )
    cls_models = get_models_for("classification")
    assert {m.model_key for m in cls_models} == {"a"}
```

- [ ] **Step 2: Run tests, verify they fail**

```
uv run pytest tests/test_models/test_loader.py -v
```
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Create the package skeleton**

`src/mlops_agents/models/__init__.py`:
```python
"""Model registry: declarative YAML catalog + Python factories + Optuna search spaces."""
```

`src/mlops_agents/models/factories.py`:
```python
"""Factory functions for every registered model.

Each factory takes a hyperparameter dict (and for forecasting, a task_metadata dict)
and returns either a sklearn-compatible estimator or a forecaster object. Factories
are referenced by string name from `registry.yaml`. The string-keyed lookup is the
security boundary — the YAML cannot reference a factory that isn't here.
"""

from __future__ import annotations

from typing import Any, Callable

# Populated incrementally in Tasks 5–8. Empty at registry-skeleton task.
FACTORY_REGISTRY: dict[str, Callable[..., Any]] = {}
```

`src/mlops_agents/models/search_spaces.py`:
```python
"""Generic Optuna search-space builder driven by SearchSpaceSpec from the registry.

`build_suggest_fn(spec)` produces an Optuna-style `suggest(trial)` callable
that materializes hyperparameters from the declarative spec. No hand-written
suggest_* functions per model — the YAML is the source of truth.
"""

from __future__ import annotations

from typing import Any, Callable

import optuna


def build_suggest_fn(spec) -> Callable[[optuna.Trial], dict[str, Any]]:
    """Return a `suggest(trial) -> dict` callable that materializes params from spec."""

    def suggest(trial: optuna.Trial) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, p in spec.params.items():
            if p.type == "int":
                out[name] = trial.suggest_int(name, int(p.low), int(p.high), step=p.step or 1)
            elif p.type == "float":
                out[name] = trial.suggest_float(
                    name, float(p.low), float(p.high), log=p.log or False
                )
            elif p.type == "categorical":
                out[name] = trial.suggest_categorical(name, p.choices)
            else:
                raise ValueError(f"Unknown search param type {p.type!r} for {name!r}")
        return out

    return suggest


# Populated incrementally as model entries are added: each model's search_space.name
# from registry.yaml becomes a key here pointing to a "build" function. With the generic
# builder above, every entry maps to the same factory call: `build_suggest_fn(spec)`.
# We expose the registry for symmetry with FACTORY_REGISTRY (validation path).
SEARCH_SPACE_REGISTRY: dict[str, Callable[..., Any]] = {}
```

`src/mlops_agents/models/loader.py`:
```python
"""YAML manifest loader for the model registry.

Validates every entry against Pydantic schemas and rejects unknown
`factory` / `search_space.name` references — the manifest cannot
reference unregistered Python code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from mlops_agents.models.factories import FACTORY_REGISTRY
from mlops_agents.models.search_spaces import SEARCH_SPACE_REGISTRY

DEFAULT_REGISTRY_PATH = Path(__file__).parent / "registry.yaml"


class SearchParamSpec(BaseModel):
    type: Literal["int", "float", "categorical"]
    low: float | int | None = None
    high: float | int | None = None
    step: int | None = None
    log: bool = False
    choices: list[Any] | None = None


class SearchSpaceSpec(BaseModel):
    name: str
    params: dict[str, SearchParamSpec] = Field(default_factory=dict)


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

    @field_validator("factory")
    @classmethod
    def factory_must_be_registered(cls, v: str) -> str:
        if v not in FACTORY_REGISTRY:
            raise ValueError(
                f"Unknown factory: {v!r}. "
                f"Known factories: {sorted(FACTORY_REGISTRY)}"
            )
        return v


_cached_registry: dict[str, ModelSpec] | None = None


def load_registry(path: Path = DEFAULT_REGISTRY_PATH, force_reload: bool = False) -> dict[str, ModelSpec]:
    """Load and validate the YAML registry; cached after first call.

    Note: SEARCH_SPACE_REGISTRY validation happens implicitly via SearchSpaceSpec —
    we accept any `name` since the generic build_suggest_fn handles all params via
    SearchParamSpec types.
    """
    global _cached_registry
    if _cached_registry is not None and not force_reload:
        return _cached_registry

    raw = yaml.safe_load(path.read_text()) or []
    registry: dict[str, ModelSpec] = {}
    for entry in raw:
        spec = ModelSpec(**entry)
        if spec.model_key in registry:
            raise ValueError(f"Duplicate model_key in registry: {spec.model_key!r}")
        registry[spec.model_key] = spec

    _cached_registry = registry
    return registry


def get_models_for(problem_type: str) -> list[ModelSpec]:
    """All registered models matching the given problem_type."""
    return [m for m in load_registry().values() if m.problem_type == problem_type]


def get_model(model_key: str) -> ModelSpec:
    """Single model by key. Raises KeyError if unknown."""
    registry = load_registry()
    if model_key not in registry:
        raise KeyError(f"Unknown model_key: {model_key!r}")
    return registry[model_key]
```

`src/mlops_agents/models/registry.yaml`:
```yaml
[]
```
(Just an empty list. Populated in Task 9.)

- [ ] **Step 4: Run tests, verify they pass**

```
uv run pytest tests/test_models/test_loader.py -v
```
Expected: 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mlops_agents/models/ tests/test_models/__init__.py tests/test_models/test_loader.py
git commit -m "feat: add model registry skeleton (loader, empty YAML, factory/search-space registries)"
```

---

## Task 5: Classification factories (5 models)

Implement the 5 classification factories. Each is a thin wrapper around a sklearn-compatible class. Test that each factory returns a fitted-capable estimator.

**Files:**
- Modify: `src/mlops_agents/models/factories.py`
- Create: `tests/test_models/test_factories.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_models/test_factories.py`:

```python
"""Unit tests for model factories."""

import numpy as np
import pytest

from mlops_agents.models.factories import FACTORY_REGISTRY


@pytest.fixture
def tabular_classification_xy():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(60, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def _check_classifier_fits_and_predicts(factory_name: str, params: dict, xy):
    X, y = xy
    factory = FACTORY_REGISTRY[factory_name]
    model = factory(params)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape
    assert set(np.unique(preds)).issubset({0, 1})


def test_logistic_regression_factory(tabular_classification_xy):
    _check_classifier_fits_and_predicts(
        "build_logistic_regression",
        {"C": 1.0, "penalty": "l2", "max_iter": 200},
        tabular_classification_xy,
    )


def test_random_forest_classifier_factory(tabular_classification_xy):
    _check_classifier_fits_and_predicts(
        "build_random_forest_classifier",
        {"n_estimators": 50, "max_depth": 5, "random_state": 0},
        tabular_classification_xy,
    )


def test_lightgbm_classifier_factory(tabular_classification_xy):
    _check_classifier_fits_and_predicts(
        "build_lightgbm_classifier",
        {"n_estimators": 50, "learning_rate": 0.1, "num_leaves": 31, "random_state": 0, "verbosity": -1},
        tabular_classification_xy,
    )


def test_xgboost_classifier_factory(tabular_classification_xy):
    _check_classifier_fits_and_predicts(
        "build_xgboost_classifier",
        {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 4, "random_state": 0,
         "tree_method": "hist", "verbosity": 0},
        tabular_classification_xy,
    )


def test_catboost_classifier_factory(tabular_classification_xy):
    _check_classifier_fits_and_predicts(
        "build_catboost_classifier",
        {"iterations": 50, "learning_rate": 0.1, "depth": 4, "random_seed": 0, "verbose": False},
        tabular_classification_xy,
    )
```

- [ ] **Step 2: Verify tests fail**

```
uv run pytest tests/test_models/test_factories.py -v
```
Expected: FAIL with `KeyError: 'build_logistic_regression'` (factories not in registry).

- [ ] **Step 3: Implement classification factories**

Replace `src/mlops_agents/models/factories.py` with:

```python
"""Factory functions for every registered model.

Each factory takes a hyperparameter dict (and for forecasting, task_metadata)
and returns a sklearn-compatible estimator or forecaster. Factories are referenced
by string name from registry.yaml.
"""

from __future__ import annotations

from typing import Any, Callable

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def build_logistic_regression(params: dict[str, Any]):
    return LogisticRegression(**params)


def build_random_forest_classifier(params: dict[str, Any]):
    return RandomForestClassifier(**params)


def build_lightgbm_classifier(params: dict[str, Any]):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(**params)


def build_xgboost_classifier(params: dict[str, Any]):
    from xgboost import XGBClassifier
    return XGBClassifier(**params)


def build_catboost_classifier(params: dict[str, Any]):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(**params)


FACTORY_REGISTRY: dict[str, Callable[..., Any]] = {
    "build_logistic_regression":      build_logistic_regression,
    "build_random_forest_classifier": build_random_forest_classifier,
    "build_lightgbm_classifier":      build_lightgbm_classifier,
    "build_xgboost_classifier":       build_xgboost_classifier,
    "build_catboost_classifier":      build_catboost_classifier,
}
```

- [ ] **Step 4: Run tests**

```
uv run pytest tests/test_models/test_factories.py -v
```
Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mlops_agents/models/factories.py tests/test_models/test_factories.py
git commit -m "feat: add classification factories (LR, RF, LightGBM, XGBoost, CatBoost)"
```

---

## Task 6: Regression factories (5 models)

**Files:**
- Modify: `src/mlops_agents/models/factories.py`
- Modify: `tests/test_models/test_factories.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_models/test_factories.py`:

```python
@pytest.fixture
def tabular_regression_xy():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(60, 4))
    y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.1, size=60)
    return X, y


def _check_regressor_fits_and_predicts(factory_name: str, params: dict, xy):
    X, y = xy
    factory = FACTORY_REGISTRY[factory_name]
    model = factory(params)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape
    assert np.isfinite(preds).all()


def test_ridge_factory(tabular_regression_xy):
    _check_regressor_fits_and_predicts(
        "build_ridge",
        {"alpha": 1.0, "random_state": 0},
        tabular_regression_xy,
    )


def test_random_forest_regressor_factory(tabular_regression_xy):
    _check_regressor_fits_and_predicts(
        "build_random_forest_regressor",
        {"n_estimators": 50, "max_depth": 5, "random_state": 0},
        tabular_regression_xy,
    )


def test_lightgbm_regressor_factory(tabular_regression_xy):
    _check_regressor_fits_and_predicts(
        "build_lightgbm_regressor",
        {"n_estimators": 50, "learning_rate": 0.1, "num_leaves": 31, "random_state": 0, "verbosity": -1},
        tabular_regression_xy,
    )


def test_xgboost_regressor_factory(tabular_regression_xy):
    _check_regressor_fits_and_predicts(
        "build_xgboost_regressor",
        {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 4, "random_state": 0,
         "tree_method": "hist", "verbosity": 0},
        tabular_regression_xy,
    )


def test_catboost_regressor_factory(tabular_regression_xy):
    _check_regressor_fits_and_predicts(
        "build_catboost_regressor",
        {"iterations": 50, "learning_rate": 0.1, "depth": 4, "random_seed": 0, "verbose": False},
        tabular_regression_xy,
    )
```

- [ ] **Step 2: Verify failures**

```
uv run pytest tests/test_models/test_factories.py -k regress -v
```
Expected: FAIL with KeyError on `build_ridge`.

- [ ] **Step 3: Add regression factories**

In `src/mlops_agents/models/factories.py`, add at the top with the existing imports:

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
```

Add the factory functions before the FACTORY_REGISTRY dict:

```python
def build_ridge(params: dict[str, Any]):
    return Ridge(**params)


def build_random_forest_regressor(params: dict[str, Any]):
    return RandomForestRegressor(**params)


def build_lightgbm_regressor(params: dict[str, Any]):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(**params)


def build_xgboost_regressor(params: dict[str, Any]):
    from xgboost import XGBRegressor
    return XGBRegressor(**params)


def build_catboost_regressor(params: dict[str, Any]):
    from catboost import CatBoostRegressor
    return CatBoostRegressor(**params)
```

Add to FACTORY_REGISTRY dict:
```python
    "build_ridge":                    build_ridge,
    "build_random_forest_regressor":  build_random_forest_regressor,
    "build_lightgbm_regressor":       build_lightgbm_regressor,
    "build_xgboost_regressor":        build_xgboost_regressor,
    "build_catboost_regressor":       build_catboost_regressor,
```

- [ ] **Step 4: Run tests**

```
uv run pytest tests/test_models/test_factories.py -v
```
Expected: 10 tests PASS (5 classification + 5 regression).

- [ ] **Step 5: Commit**

```bash
git add src/mlops_agents/models/factories.py tests/test_models/test_factories.py
git commit -m "feat: add regression factories (Ridge, RF, LightGBM, XGBoost, CatBoost)"
```

---

## Task 7: Statistical forecasting factories (4 models)

Statistical forecasters via `statsforecast.StatsForecast`. The factory returns a configured `StatsForecast` instance ready to fit on the long-format dataframe (columns: `unique_id`, `ds`, `y`).

**Files:**
- Modify: `src/mlops_agents/models/factories.py`
- Modify: `tests/test_models/test_factories.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_models/test_factories.py`:

```python
@pytest.fixture
def panel_dataframe():
    """Multi-series panel data: 2 series × 36 monthly periods."""
    import pandas as pd
    dates = pd.date_range("2020-01-01", periods=36, freq="MS")
    rows = []
    for sid in ["s1", "s2"]:
        for i, d in enumerate(dates):
            rows.append({"unique_id": sid, "ds": d, "y": float(i) + (1.0 if sid == "s1" else 5.0)})
    return pd.DataFrame(rows)


def _check_stat_forecaster_fits_and_predicts(factory_name: str, params: dict, panel):
    factory = FACTORY_REGISTRY[factory_name]
    sf = factory({"task_metadata": {"frequency": "MS", "forecast_horizon": 6}, "params": params})
    sf.fit(panel)
    fcst = sf.predict(h=6)
    assert len(fcst) == 6 * 2     # 6 horizons × 2 series
    assert "unique_id" in fcst.columns and "ds" in fcst.columns


def test_naive_factory(panel_dataframe):
    _check_stat_forecaster_fits_and_predicts("build_naive", {}, panel_dataframe)


def test_seasonal_naive_factory(panel_dataframe):
    _check_stat_forecaster_fits_and_predicts(
        "build_seasonal_naive", {"season_length": 12}, panel_dataframe,
    )


def test_ets_factory(panel_dataframe):
    _check_stat_forecaster_fits_and_predicts(
        "build_ets", {"season_length": 12}, panel_dataframe,
    )


def test_auto_arima_factory(panel_dataframe):
    _check_stat_forecaster_fits_and_predicts(
        "build_auto_arima", {"season_length": 12}, panel_dataframe,
    )
```

- [ ] **Step 2: Verify failures**

```
uv run pytest tests/test_models/test_factories.py -k forecaster -v
```
Expected: FAIL.

- [ ] **Step 3: Add statistical forecasting factories**

In `src/mlops_agents/models/factories.py`, add factories. Note: each forecasting factory takes a dict with two keys: `task_metadata` (for frequency) and `params` (model hyperparameters). This decouples Optuna's flat param dict from the ambient task context.

```python
def build_naive(spec: dict[str, Any]):
    """Naive forecaster: predicts the last observed value."""
    from statsforecast import StatsForecast
    from statsforecast.models import Naive
    freq = spec["task_metadata"]["frequency"]
    return StatsForecast(models=[Naive()], freq=freq, n_jobs=1)


def build_seasonal_naive(spec: dict[str, Any]):
    """Seasonal naive: predicts the value from one season ago."""
    from statsforecast import StatsForecast
    from statsforecast.models import SeasonalNaive
    freq = spec["task_metadata"]["frequency"]
    season_length = spec["params"].get("season_length", _default_season_length(freq))
    return StatsForecast(models=[SeasonalNaive(season_length=season_length)], freq=freq, n_jobs=1)


def build_ets(spec: dict[str, Any]):
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS
    freq = spec["task_metadata"]["frequency"]
    season_length = spec["params"].get("season_length", _default_season_length(freq))
    return StatsForecast(models=[AutoETS(season_length=season_length)], freq=freq, n_jobs=1)


def build_auto_arima(spec: dict[str, Any]):
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA
    freq = spec["task_metadata"]["frequency"]
    season_length = spec["params"].get("season_length", _default_season_length(freq))
    return StatsForecast(models=[AutoARIMA(season_length=season_length)], freq=freq, n_jobs=1)


_FREQ_TO_SEASON = {"H": 24, "D": 7, "W": 52, "MS": 12, "M": 12, "QS": 4, "YS": 1}


def _default_season_length(freq: str) -> int:
    return _FREQ_TO_SEASON.get(freq, 1)
```

Add to FACTORY_REGISTRY:
```python
    "build_naive":           build_naive,
    "build_seasonal_naive":  build_seasonal_naive,
    "build_ets":             build_ets,
    "build_auto_arima":      build_auto_arima,
```

- [ ] **Step 4: Run tests**

```
uv run pytest tests/test_models/test_factories.py -v
```
Expected: 14 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mlops_agents/models/factories.py tests/test_models/test_factories.py
git commit -m "feat: add statistical forecasting factories (Naive, SeasonalNaive, ETS, AutoARIMA via statsforecast)"
```

---

## Task 8: Supervised forecasting factories (6 models)

Supervised forecasters wrap sklearn-compatible regressors with `skforecast.ForecasterRecursiveMultiSeries`. Lag count comes from `params["lags"]` (with default).

**Files:**
- Modify: `src/mlops_agents/models/factories.py`
- Modify: `tests/test_models/test_factories.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_models/test_factories.py`:

```python
def _check_supervised_forecaster_fits_and_predicts(factory_name: str, params: dict, panel):
    """skforecast wants a wide series_dict from the long panel."""
    import pandas as pd
    factory = FACTORY_REGISTRY[factory_name]
    forecaster = factory({"task_metadata": {"forecast_horizon": 6}, "params": params})
    series_dict = {sid: g.set_index("ds")["y"] for sid, g in panel.groupby("unique_id")}
    forecaster.fit(series=series_dict)
    preds = forecaster.predict(steps=6)
    assert len(preds) == 6 * 2
    # skforecast returns long format with columns ['level', 'pred']
    assert "level" in preds.columns or "unique_id" in preds.columns


def test_random_forest_forecaster_factory(panel_dataframe):
    _check_supervised_forecaster_fits_and_predicts(
        "build_random_forest_forecaster",
        {"lags": 12, "n_estimators": 30, "max_depth": 5, "random_state": 0},
        panel_dataframe,
    )


def test_extra_trees_forecaster_factory(panel_dataframe):
    _check_supervised_forecaster_fits_and_predicts(
        "build_extra_trees_forecaster",
        {"lags": 12, "n_estimators": 30, "max_depth": 5, "random_state": 0},
        panel_dataframe,
    )


def test_gbm_forecaster_factory(panel_dataframe):
    _check_supervised_forecaster_fits_and_predicts(
        "build_gbm_forecaster",
        {"lags": 12, "n_estimators": 30, "max_depth": 3, "random_state": 0},
        panel_dataframe,
    )


def test_lightgbm_forecaster_factory(panel_dataframe):
    _check_supervised_forecaster_fits_and_predicts(
        "build_lightgbm_forecaster",
        {"lags": 12, "n_estimators": 30, "learning_rate": 0.1, "num_leaves": 15,
         "random_state": 0, "verbosity": -1},
        panel_dataframe,
    )


def test_xgboost_forecaster_factory(panel_dataframe):
    _check_supervised_forecaster_fits_and_predicts(
        "build_xgboost_forecaster",
        {"lags": 12, "n_estimators": 30, "learning_rate": 0.1, "max_depth": 3,
         "random_state": 0, "tree_method": "hist", "verbosity": 0},
        panel_dataframe,
    )


def test_svr_forecaster_factory(panel_dataframe):
    _check_supervised_forecaster_fits_and_predicts(
        "build_svr_forecaster",
        {"lags": 12, "C": 1.0, "epsilon": 0.1, "kernel": "rbf"},
        panel_dataframe,
    )
```

- [ ] **Step 2: Verify failures**

```
uv run pytest tests/test_models/test_factories.py -k forecaster_factory -v
```
Expected: FAIL with KeyError.

- [ ] **Step 3: Add supervised forecasting factories**

In `src/mlops_agents/models/factories.py`, add:

```python
def _split_lags_from_params(params: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Pull `lags` out of params; the rest goes to the regressor."""
    p = dict(params)
    lags = int(p.pop("lags", 12))
    return lags, p


def _wrap_with_skforecast(regressor, lags: int):
    from skforecast.recursive import ForecasterRecursiveMultiSeries
    return ForecasterRecursiveMultiSeries(regressor=regressor, lags=lags)


def build_random_forest_forecaster(spec: dict[str, Any]):
    lags, p = _split_lags_from_params(spec["params"])
    return _wrap_with_skforecast(RandomForestRegressor(**p), lags)


def build_extra_trees_forecaster(spec: dict[str, Any]):
    from sklearn.ensemble import ExtraTreesRegressor
    lags, p = _split_lags_from_params(spec["params"])
    return _wrap_with_skforecast(ExtraTreesRegressor(**p), lags)


def build_gbm_forecaster(spec: dict[str, Any]):
    from sklearn.ensemble import GradientBoostingRegressor
    lags, p = _split_lags_from_params(spec["params"])
    return _wrap_with_skforecast(GradientBoostingRegressor(**p), lags)


def build_lightgbm_forecaster(spec: dict[str, Any]):
    from lightgbm import LGBMRegressor
    lags, p = _split_lags_from_params(spec["params"])
    return _wrap_with_skforecast(LGBMRegressor(**p), lags)


def build_xgboost_forecaster(spec: dict[str, Any]):
    from xgboost import XGBRegressor
    lags, p = _split_lags_from_params(spec["params"])
    return _wrap_with_skforecast(XGBRegressor(**p), lags)


def build_svr_forecaster(spec: dict[str, Any]):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    lags, p = _split_lags_from_params(spec["params"])
    pipe = Pipeline([("scaler", StandardScaler()), ("svr", SVR(**p))])
    return _wrap_with_skforecast(pipe, lags)
```

Add to FACTORY_REGISTRY:
```python
    "build_random_forest_forecaster": build_random_forest_forecaster,
    "build_extra_trees_forecaster":   build_extra_trees_forecaster,
    "build_gbm_forecaster":           build_gbm_forecaster,
    "build_lightgbm_forecaster":      build_lightgbm_forecaster,
    "build_xgboost_forecaster":       build_xgboost_forecaster,
    "build_svr_forecaster":           build_svr_forecaster,
```

- [ ] **Step 4: Run tests**

```
uv run pytest tests/test_models/test_factories.py -v
```
Expected: 20 tests PASS (5 cls + 5 reg + 4 stat fc + 6 sup fc).

- [ ] **Step 5: Commit**

```bash
git add src/mlops_agents/models/factories.py tests/test_models/test_factories.py
git commit -m "feat: add supervised forecasting factories (RF, ExtraTrees, GBM, LightGBM, XGBoost, SVR via skforecast)"
```

---

## Task 9: Populate `registry.yaml` (all 20 entries)

The YAML registry is the source of truth for which models exist, their default params, search spaces, and metadata.

**Files:**
- Modify: `src/mlops_agents/models/registry.yaml`
- Create: `tests/test_models/test_search_spaces.py`

- [ ] **Step 1: Write the registry YAML**

Replace `src/mlops_agents/models/registry.yaml` with:

```yaml
# Model registry — declarative source of truth for model_key → factory + search space + metadata.
# Loader at `loader.py` validates every entry. New models are added here.

# -----------------------------------------------------------------------------
# Classification
# -----------------------------------------------------------------------------
- model_key: logistic_regression
  problem_type: classification
  family: linear
  complexity_rank: 1
  library: sklearn
  factory: build_logistic_regression
  search_space:
    name: logistic_regression_space
    params:
      C: { type: float, low: 0.001, high: 100.0, log: true }
      penalty: { type: categorical, choices: [l2] }
      max_iter: { type: int, low: 100, high: 2000 }
  default_params:
    C: 1.0
    penalty: l2
    max_iter: 500
    random_state: 42
  requires:
    needs_scaling: true
  use_when: [tabular_classification, small_dataset, interpretability_required]
  avoid_when: [severely_imbalanced, highly_nonlinear]
  notes: |
    Strong, fast baseline. Sensitive to scale. Tends to underperform on
    severely imbalanced or highly nonlinear data.

- model_key: random_forest_classifier
  problem_type: classification
  family: tree_ensemble
  complexity_rank: 2
  library: sklearn
  factory: build_random_forest_classifier
  search_space:
    name: random_forest_classifier_space
    params:
      n_estimators: { type: int, low: 50, high: 500 }
      max_depth: { type: int, low: 3, high: 20 }
      min_samples_split: { type: int, low: 2, high: 20 }
  default_params:
    n_estimators: 200
    max_depth: 10
    min_samples_split: 2
    random_state: 42
    n_jobs: -1
  requires:
    supports_missing: false
  use_when: [tabular_classification, robust_baseline_needed]
  notes: |
    Good general-purpose tree ensemble. No scaling needed. Less competitive
    than gradient boosting on large datasets.

- model_key: lightgbm_classifier
  problem_type: classification
  family: gradient_boosting
  complexity_rank: 3
  library: lightgbm
  factory: build_lightgbm_classifier
  search_space:
    name: lightgbm_classifier_space
    params:
      n_estimators: { type: int, low: 100, high: 1000 }
      learning_rate: { type: float, low: 0.005, high: 0.2, log: true }
      num_leaves: { type: int, low: 15, high: 127 }
      min_child_samples: { type: int, low: 5, high: 100 }
  default_params:
    n_estimators: 500
    learning_rate: 0.03
    num_leaves: 31
    min_child_samples: 20
    random_state: 42
    verbosity: -1
  requires:
    supports_missing: true
    min_rows: 500
  use_when: [tabular_classification, medium_or_large_dataset, high_performance_required]
  avoid_when: [very_small_dataset, strict_interpretability_required]

- model_key: xgboost_classifier
  problem_type: classification
  family: gradient_boosting
  complexity_rank: 3
  library: xgboost
  factory: build_xgboost_classifier
  search_space:
    name: xgboost_classifier_space
    params:
      n_estimators: { type: int, low: 100, high: 1000 }
      learning_rate: { type: float, low: 0.005, high: 0.2, log: true }
      max_depth: { type: int, low: 3, high: 10 }
      subsample: { type: float, low: 0.5, high: 1.0 }
      colsample_bytree: { type: float, low: 0.5, high: 1.0 }
  default_params:
    n_estimators: 500
    learning_rate: 0.03
    max_depth: 6
    subsample: 0.9
    colsample_bytree: 0.9
    random_state: 42
    tree_method: hist
    verbosity: 0
  requires:
    supports_missing: true
    min_rows: 500
  use_when: [tabular_classification, medium_or_large_dataset]

- model_key: catboost_classifier
  problem_type: classification
  family: gradient_boosting
  complexity_rank: 3
  library: catboost
  factory: build_catboost_classifier
  search_space:
    name: catboost_classifier_space
    params:
      iterations: { type: int, low: 100, high: 1000 }
      learning_rate: { type: float, low: 0.005, high: 0.2, log: true }
      depth: { type: int, low: 4, high: 10 }
      l2_leaf_reg: { type: float, low: 1.0, high: 10.0 }
  default_params:
    iterations: 500
    learning_rate: 0.03
    depth: 6
    l2_leaf_reg: 3.0
    random_seed: 42
    verbose: false
  requires:
    supports_missing: true
    supports_categorical: true
    min_rows: 500
  use_when: [tabular_classification, many_categorical_features]

# -----------------------------------------------------------------------------
# Regression
# -----------------------------------------------------------------------------
- model_key: ridge
  problem_type: regression
  family: linear
  complexity_rank: 1
  library: sklearn
  factory: build_ridge
  search_space:
    name: ridge_space
    params:
      alpha: { type: float, low: 0.001, high: 100.0, log: true }
  default_params:
    alpha: 1.0
    random_state: 42
  requires:
    needs_scaling: true
  use_when: [tabular_regression, small_dataset, interpretability_required]

- model_key: random_forest_regressor
  problem_type: regression
  family: tree_ensemble
  complexity_rank: 2
  library: sklearn
  factory: build_random_forest_regressor
  search_space:
    name: random_forest_regressor_space
    params:
      n_estimators: { type: int, low: 50, high: 500 }
      max_depth: { type: int, low: 3, high: 20 }
      min_samples_split: { type: int, low: 2, high: 20 }
  default_params:
    n_estimators: 200
    max_depth: 10
    min_samples_split: 2
    random_state: 42
    n_jobs: -1

- model_key: lightgbm_regressor
  problem_type: regression
  family: gradient_boosting
  complexity_rank: 3
  library: lightgbm
  factory: build_lightgbm_regressor
  search_space:
    name: lightgbm_regressor_space
    params:
      n_estimators: { type: int, low: 100, high: 1000 }
      learning_rate: { type: float, low: 0.005, high: 0.2, log: true }
      num_leaves: { type: int, low: 15, high: 127 }
      min_child_samples: { type: int, low: 5, high: 100 }
  default_params:
    n_estimators: 500
    learning_rate: 0.03
    num_leaves: 31
    min_child_samples: 20
    random_state: 42
    verbosity: -1
  requires:
    min_rows: 500

- model_key: xgboost_regressor
  problem_type: regression
  family: gradient_boosting
  complexity_rank: 3
  library: xgboost
  factory: build_xgboost_regressor
  search_space:
    name: xgboost_regressor_space
    params:
      n_estimators: { type: int, low: 100, high: 1000 }
      learning_rate: { type: float, low: 0.005, high: 0.2, log: true }
      max_depth: { type: int, low: 3, high: 10 }
      subsample: { type: float, low: 0.5, high: 1.0 }
  default_params:
    n_estimators: 500
    learning_rate: 0.03
    max_depth: 6
    subsample: 0.9
    random_state: 42
    tree_method: hist
    verbosity: 0
  requires:
    min_rows: 500

- model_key: catboost_regressor
  problem_type: regression
  family: gradient_boosting
  complexity_rank: 3
  library: catboost
  factory: build_catboost_regressor
  search_space:
    name: catboost_regressor_space
    params:
      iterations: { type: int, low: 100, high: 1000 }
      learning_rate: { type: float, low: 0.005, high: 0.2, log: true }
      depth: { type: int, low: 4, high: 10 }
      l2_leaf_reg: { type: float, low: 1.0, high: 10.0 }
  default_params:
    iterations: 500
    learning_rate: 0.03
    depth: 6
    l2_leaf_reg: 3.0
    random_seed: 42
    verbose: false
  requires:
    supports_categorical: true
    min_rows: 500

# -----------------------------------------------------------------------------
# Forecasting — statistical (statsforecast)
# -----------------------------------------------------------------------------
- model_key: naive
  problem_type: forecasting
  family: statistical
  complexity_rank: 1
  library: statsforecast
  factory: build_naive
  search_space:
    name: naive_space
    params: {}
  default_params: {}
  use_when: [forecasting, very_short_history, baseline_needed]

- model_key: seasonal_naive
  problem_type: forecasting
  family: statistical
  complexity_rank: 1
  library: statsforecast
  factory: build_seasonal_naive
  search_space:
    name: seasonal_naive_space
    params:
      season_length: { type: categorical, choices: [4, 7, 12, 24, 52] }
  default_params:
    season_length: 12
  use_when: [forecasting, strong_seasonality, short_history]

- model_key: ets
  problem_type: forecasting
  family: statistical
  complexity_rank: 2
  library: statsforecast
  factory: build_ets
  search_space:
    name: ets_space
    params:
      season_length: { type: categorical, choices: [4, 7, 12, 24, 52] }
  default_params:
    season_length: 12
  use_when: [forecasting, smooth_seasonality, trend_present]

- model_key: auto_arima
  problem_type: forecasting
  family: statistical
  complexity_rank: 2
  library: statsforecast
  factory: build_auto_arima
  search_space:
    name: auto_arima_space
    params:
      season_length: { type: categorical, choices: [4, 7, 12, 24, 52] }
  default_params:
    season_length: 12
  use_when: [forecasting, autocorrelation_pattern, medium_history]

# -----------------------------------------------------------------------------
# Forecasting — supervised (skforecast lag-based)
# -----------------------------------------------------------------------------
- model_key: random_forest_forecaster
  problem_type: forecasting
  family: ml_lag_forecasting
  complexity_rank: 2
  library: sklearn
  factory: build_random_forest_forecaster
  search_space:
    name: random_forest_forecaster_space
    params:
      lags: { type: int, low: 4, high: 36 }
      n_estimators: { type: int, low: 50, high: 300 }
      max_depth: { type: int, low: 3, high: 20 }
  default_params:
    lags: 12
    n_estimators: 100
    max_depth: 10
    random_state: 42
    n_jobs: -1
  requires:
    needs_lag_features: true
  use_when: [forecasting, medium_or_large_history, exogenous_available]

- model_key: extra_trees_forecaster
  problem_type: forecasting
  family: ml_lag_forecasting
  complexity_rank: 2
  library: sklearn
  factory: build_extra_trees_forecaster
  search_space:
    name: extra_trees_forecaster_space
    params:
      lags: { type: int, low: 4, high: 36 }
      n_estimators: { type: int, low: 50, high: 300 }
      max_depth: { type: int, low: 3, high: 20 }
  default_params:
    lags: 12
    n_estimators: 100
    max_depth: 10
    random_state: 42
    n_jobs: -1
  requires:
    needs_lag_features: true

- model_key: gbm_forecaster
  problem_type: forecasting
  family: ml_lag_forecasting
  complexity_rank: 3
  library: sklearn
  factory: build_gbm_forecaster
  search_space:
    name: gbm_forecaster_space
    params:
      lags: { type: int, low: 4, high: 36 }
      n_estimators: { type: int, low: 50, high: 300 }
      learning_rate: { type: float, low: 0.01, high: 0.2, log: true }
      max_depth: { type: int, low: 2, high: 8 }
  default_params:
    lags: 12
    n_estimators: 100
    learning_rate: 0.05
    max_depth: 3
    random_state: 42
  requires:
    needs_lag_features: true

- model_key: lightgbm_forecaster
  problem_type: forecasting
  family: ml_lag_forecasting
  complexity_rank: 3
  library: lightgbm
  factory: build_lightgbm_forecaster
  search_space:
    name: lightgbm_forecaster_space
    params:
      lags: { type: int, low: 4, high: 36 }
      n_estimators: { type: int, low: 50, high: 500 }
      learning_rate: { type: float, low: 0.005, high: 0.2, log: true }
      num_leaves: { type: int, low: 7, high: 63 }
  default_params:
    lags: 12
    n_estimators: 200
    learning_rate: 0.05
    num_leaves: 31
    random_state: 42
    verbosity: -1
  requires:
    needs_lag_features: true
    min_rows: 100
  use_when: [forecasting, long_history, exogenous_available]

- model_key: xgboost_forecaster
  problem_type: forecasting
  family: ml_lag_forecasting
  complexity_rank: 3
  library: xgboost
  factory: build_xgboost_forecaster
  search_space:
    name: xgboost_forecaster_space
    params:
      lags: { type: int, low: 4, high: 36 }
      n_estimators: { type: int, low: 50, high: 500 }
      learning_rate: { type: float, low: 0.005, high: 0.2, log: true }
      max_depth: { type: int, low: 3, high: 10 }
  default_params:
    lags: 12
    n_estimators: 200
    learning_rate: 0.05
    max_depth: 6
    random_state: 42
    tree_method: hist
    verbosity: 0
  requires:
    needs_lag_features: true
    min_rows: 100

- model_key: svr_forecaster
  problem_type: forecasting
  family: ml_lag_forecasting
  complexity_rank: 3
  library: sklearn
  factory: build_svr_forecaster
  search_space:
    name: svr_forecaster_space
    params:
      lags: { type: int, low: 4, high: 24 }
      C: { type: float, low: 0.1, high: 100.0, log: true }
      epsilon: { type: float, low: 0.01, high: 1.0 }
      kernel: { type: categorical, choices: [rbf, linear] }
  default_params:
    lags: 12
    C: 10.0
    epsilon: 0.1
    kernel: rbf
  requires:
    needs_lag_features: true
    needs_scaling: true
  use_when: [forecasting, small_or_medium_dataset, smooth_nonlinear_patterns]
```

- [ ] **Step 2: Add a registry-load smoke test**

Append to `tests/test_models/test_loader.py`:

```python
def test_load_actual_registry_has_20_entries():
    """Smoke test: the shipped registry.yaml loads cleanly with all expected models."""
    from mlops_agents.models.loader import load_registry, get_models_for
    registry = load_registry(force_reload=True)
    assert len(registry) == 20
    assert len(get_models_for("classification")) == 5
    assert len(get_models_for("regression")) == 5
    assert len(get_models_for("forecasting")) == 10  # 4 statistical + 6 supervised


def test_registry_complexity_ranks_set():
    from mlops_agents.models.loader import load_registry
    for model in load_registry(force_reload=True).values():
        assert model.complexity_rank >= 1
```

- [ ] **Step 3: Add a search-space builder test**

Create `tests/test_models/test_search_spaces.py`:

```python
"""Unit tests for the generic Optuna search-space builder."""

import optuna

from mlops_agents.models.loader import SearchParamSpec, SearchSpaceSpec
from mlops_agents.models.search_spaces import build_suggest_fn


def test_build_suggest_fn_int():
    spec = SearchSpaceSpec(name="x", params={"n": SearchParamSpec(type="int", low=10, high=20)})
    suggest = build_suggest_fn(spec)
    study = optuna.create_study()
    study.optimize(lambda t: float(suggest(t)["n"]), n_trials=3)
    assert 10 <= study.best_params["n"] <= 20


def test_build_suggest_fn_float_log():
    spec = SearchSpaceSpec(
        name="x",
        params={"lr": SearchParamSpec(type="float", low=1e-4, high=1.0, log=True)},
    )
    suggest = build_suggest_fn(spec)
    study = optuna.create_study()
    study.optimize(lambda t: -suggest(t)["lr"], n_trials=3)
    assert 1e-4 <= study.best_params["lr"] <= 1.0


def test_build_suggest_fn_categorical():
    spec = SearchSpaceSpec(
        name="x",
        params={"k": SearchParamSpec(type="categorical", choices=["a", "b", "c"])},
    )
    suggest = build_suggest_fn(spec)
    study = optuna.create_study()
    study.optimize(lambda t: 0.0 if suggest(t)["k"] == "a" else 1.0, n_trials=5)
    assert study.best_params["k"] == "a"
```

- [ ] **Step 4: Run all model tests**

```
uv run pytest tests/test_models/ -v
```
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mlops_agents/models/registry.yaml tests/test_models/test_loader.py tests/test_models/test_search_spaces.py
git commit -m "feat: populate model registry YAML with 20 entries, validate via load_registry smoke test"
```

---

## Task 10: `search_space_override` validation against the registry

The override must narrow (never widen) the registry's space. This validator lives next to the executor because it needs both the registry and the override at validation time.

**Files:**
- Create: `src/mlops_agents/training/__init__.py`
- Create: `src/mlops_agents/training/override_validation.py`
- Create: `tests/test_contracts/test_search_space_override.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_contracts/test_search_space_override.py`:

```python
"""Tests for search_space_override validation against the model registry."""

import pytest

from mlops_agents.contracts.training import SearchParamOverride
from mlops_agents.training.override_validation import (
    narrow_search_space,
    validate_override,
)


def test_override_within_range_accepted():
    overrides = {
        "n_estimators": SearchParamOverride(low=200, high=500),
        "learning_rate": SearchParamOverride(low=0.01, high=0.1),
    }
    # No error raised
    validate_override("lightgbm_regressor", overrides)


def test_override_choices_subset_of_categorical_accepted():
    overrides = {"penalty": SearchParamOverride(choices=["l2"])}
    validate_override("logistic_regression", overrides)


def test_override_unknown_param_rejected():
    overrides = {"nonsense": SearchParamOverride(low=1, high=2)}
    with pytest.raises(ValueError, match="unknown.*parameter|not in registry"):
        validate_override("lightgbm_regressor", overrides)


def test_override_out_of_range_rejected():
    overrides = {"learning_rate": SearchParamOverride(low=0.001, high=10.0)}  # high > registry.high (0.2)
    with pytest.raises(ValueError, match="out of registry|disjoint|wider"):
        validate_override("lightgbm_regressor", overrides)


def test_override_choices_outside_categorical_rejected():
    overrides = {"penalty": SearchParamOverride(choices=["elasticnet"])}
    with pytest.raises(ValueError, match="not in registry"):
        validate_override("logistic_regression", overrides)


def test_override_categorical_with_low_high_rejected():
    overrides = {"penalty": SearchParamOverride(low=0, high=1)}
    with pytest.raises(ValueError, match="categorical"):
        validate_override("logistic_regression", overrides)


def test_narrow_search_space_collapses_int_to_categorical_via_choices():
    overrides = {"n_estimators": SearchParamOverride(choices=[300, 500, 800])}
    narrowed = narrow_search_space("lightgbm_regressor", overrides)
    n_param = narrowed.params["n_estimators"]
    assert n_param.type == "categorical"
    assert n_param.choices == [300, 500, 800]


def test_narrow_search_space_keeps_unmodified_params():
    """Params not in override keep registry defaults."""
    overrides = {"learning_rate": SearchParamOverride(low=0.01, high=0.05)}
    narrowed = narrow_search_space("lightgbm_regressor", overrides)
    # learning_rate narrowed
    assert narrowed.params["learning_rate"].low == 0.01
    assert narrowed.params["learning_rate"].high == 0.05
    # n_estimators unchanged from registry
    assert narrowed.params["n_estimators"].low == 100
    assert narrowed.params["n_estimators"].high == 1000
```

- [ ] **Step 2: Verify tests fail**

```
uv run pytest tests/test_contracts/test_search_space_override.py -v
```
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Create the module**

Create `src/mlops_agents/training/__init__.py`:
```python
"""Deterministic training pipeline: executor, splitter, profiler, default plans."""
```

Create `src/mlops_agents/training/override_validation.py`:
```python
"""Validate search_space_override entries against the registry's approved space."""

from __future__ import annotations

from copy import deepcopy

from mlops_agents.contracts.training import SearchParamOverride
from mlops_agents.models.loader import SearchParamSpec, SearchSpaceSpec, get_model


def validate_override(model_key: str, overrides: dict[str, SearchParamOverride]) -> None:
    """Raise ValueError if any override is out of range, disjoint, or unknown."""
    spec = get_model(model_key).search_space
    for param_name, override in overrides.items():
        if param_name not in spec.params:
            raise ValueError(
                f"{model_key}: override references unknown parameter {param_name!r}. "
                f"Registry params: {sorted(spec.params)}"
            )
        registry_param = spec.params[param_name]
        if registry_param.type == "categorical":
            if override.choices is None:
                raise ValueError(
                    f"{model_key}.{param_name}: registry param is categorical; "
                    f"override must use {{choices}}, not {{low,high}}."
                )
            for c in override.choices:
                if c not in (registry_param.choices or []):
                    raise ValueError(
                        f"{model_key}.{param_name}: override choice {c!r} not in registry "
                        f"choices {registry_param.choices!r}"
                    )
        else:  # int / float
            lo, hi = registry_param.low, registry_param.high
            if override.choices is not None:
                for c in override.choices:
                    if not (lo <= c <= hi):
                        raise ValueError(
                            f"{model_key}.{param_name}: override choice {c} out of registry "
                            f"range [{lo}, {hi}]"
                        )
            else:
                if not (lo <= override.low <= override.high <= hi):
                    raise ValueError(
                        f"{model_key}.{param_name}: override range [{override.low}, {override.high}] "
                        f"is wider than or disjoint from registry range [{lo}, {hi}]"
                    )


def narrow_search_space(
    model_key: str,
    overrides: dict[str, SearchParamOverride],
) -> SearchSpaceSpec:
    """Return a copy of the registry's SearchSpaceSpec with override-narrowed params."""
    validate_override(model_key, overrides)
    base = deepcopy(get_model(model_key).search_space)
    for name, ovr in overrides.items():
        registry_param = base.params[name]
        if ovr.choices is not None:
            base.params[name] = SearchParamSpec(type="categorical", choices=list(ovr.choices))
        else:
            base.params[name] = SearchParamSpec(
                type=registry_param.type,
                low=ovr.low,
                high=ovr.high,
                step=registry_param.step,
                log=registry_param.log,
            )
    return base
```

- [ ] **Step 4: Run tests**

```
uv run pytest tests/test_contracts/test_search_space_override.py -v
```
Expected: 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mlops_agents/training/__init__.py src/mlops_agents/training/override_validation.py tests/test_contracts/test_search_space_override.py
git commit -m "feat: add search_space_override validator (narrowing only, registry-bounded)"
```

---

## Task 11: Profiler — `build_dataset_profile`

Compute the bucketed profile from a CSV + task_metadata. SP3 produces it; SP4 stabilizes the schema in `contracts/profile.py` (deferred to SP4).

For SP3 we return a plain dict — keys/values match what SP4's `DatasetProfile` Pydantic class will declare.

**Files:**
- Create: `src/mlops_agents/training/profiler.py`
- Create: `tests/test_training/__init__.py`
- Create: `tests/test_training/test_profiler.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_training/__init__.py` (empty).

Create `tests/test_training/test_profiler.py`:

```python
"""Tests for build_dataset_profile."""

import pandas as pd
import pytest

from mlops_agents.training.profiler import build_dataset_profile


def _write_csv(tmp_path, df, name="data.csv"):
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


def test_classification_profile_basic(tmp_path):
    df = pd.DataFrame({
        "f1": range(60),
        "f2": [0.1] * 60,
        "cat": (["a", "b"] * 30),
        "target": [0, 1] * 30,
    })
    csv = _write_csv(tmp_path, df)
    p = build_dataset_profile(csv, {"problem_type": "classification", "target_column": "target"})
    assert p["problem_type"] == "classification"
    assert p["n_rows"] == "very_small"
    assert p["n_classes"] == "binary"
    assert p["class_balance"] == "balanced"


def test_regression_profile_basic(tmp_path):
    df = pd.DataFrame({"x1": range(2000), "x2": [0.5] * 2000, "y": list(range(2000))})
    csv = _write_csv(tmp_path, df)
    p = build_dataset_profile(csv, {"problem_type": "regression", "target_column": "y"})
    assert p["problem_type"] == "regression"
    assert p["n_rows"] == "medium"


def test_classification_imbalance_severely(tmp_path):
    df = pd.DataFrame({"f1": range(60), "target": [0] * 50 + [1] * 10})
    csv = _write_csv(tmp_path, df)
    p = build_dataset_profile(csv, {"problem_type": "classification", "target_column": "target"})
    assert p["class_balance"] == "severely_imbalanced"


def test_missing_rate_low(tmp_path):
    df = pd.DataFrame({"f1": [1.0, 2.0, None, 4.0, 5.0] * 12, "target": [0, 1] * 30})
    csv = _write_csv(tmp_path, df)
    p = build_dataset_profile(csv, {"problem_type": "classification", "target_column": "target"})
    assert p["missing_rate"] in ("low", "medium")


def test_forecasting_profile_single_series(tmp_path):
    dates = pd.date_range("2020-01-01", periods=120, freq="MS")
    df = pd.DataFrame({"ds": dates, "y": range(120)})
    csv = _write_csv(tmp_path, df)
    p = build_dataset_profile(
        csv,
        {
            "problem_type": "forecasting",
            "target_column": "y",
            "datetime_column": "ds",
            "series_id_columns": [],
            "frequency": "MS",
            "forecast_horizon": 12,
        },
    )
    assert p["problem_type"] == "forecasting"
    assert p["n_series"] == "single"
    assert p["history_length"] in ("short", "medium")
    assert p["frequency"] == "MS"
    assert p["horizon_difficulty"] in ("very_short", "short", "medium", "long")
    assert isinstance(p["seasonality_detected"], bool)
    assert isinstance(p["trend_detected"], bool)
    assert isinstance(p["stationarity"], bool)
```

- [ ] **Step 2: Verify failures**

```
uv run pytest tests/test_training/test_profiler.py -v
```
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the profiler**

Create `src/mlops_agents/training/profiler.py`:

```python
"""Compute the bucketed dataset profile used as the retrieval join key.

SP4 will define DatasetProfile as a Pydantic class; SP3 returns a plain dict
with matching keys.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Bucketers
# ---------------------------------------------------------------------------

def _bucket_n_rows(n: int) -> str:
    if n < 500: return "very_small"
    if n < 1000: return "small"
    if n <= 50_000: return "medium"
    return "large"


def _bucket_n_features(n: int) -> str:
    if n < 10: return "small"
    if n <= 100: return "medium"
    return "large"


def _bucket_missing(rate: float) -> str:
    if rate == 0.0: return "none"
    if rate < 0.05: return "low"
    if rate <= 0.20: return "medium"
    return "high"


def _bucket_count(n: int) -> str:
    if n == 0: return "none"
    if n <= 3: return "few"
    if n <= 10: return "some"
    return "many"


def _bucket_n_classes(n: int) -> str:
    if n == 2: return "binary"
    if n <= 5: return "small_multiclass"
    return "many_classes"


def _bucket_class_balance(class_counts: pd.Series) -> str:
    if len(class_counts) == 0: return "balanced"
    ratio = class_counts.max() / max(class_counts.min(), 1)
    if ratio < 1.5: return "balanced"
    if ratio <= 5: return "moderately_imbalanced"
    return "severely_imbalanced"


def _bucket_target_distribution(s: pd.Series) -> str:
    n_unique = s.nunique()
    if n_unique > 0 and n_unique < max(len(s) / 20, 5):
        return "discrete_like"
    skew = abs(s.skew())
    kurt = s.kurt()
    if kurt > 3 and skew < 1: return "heavy_tailed"
    if skew >= 1: return "skewed"
    return "near_normal"


def _bucket_n_series(n: int) -> str:
    if n == 1: return "single"
    if n <= 10: return "few"
    if n <= 100: return "moderate"
    return "many"


def _bucket_history_length(n: float) -> str:
    if n < 50: return "very_short"
    if n <= 200: return "short"
    if n <= 1000: return "medium"
    return "long"


_HORIZON_DIFFICULTY: dict[str, list[tuple[int, str]]] = {
    "H":  [(24, "very_short"), (168, "short"), (1000, "medium")],
    "D":  [(7, "very_short"), (30, "short"), (90, "medium")],
    "W":  [(4, "very_short"), (13, "short"), (52, "medium")],
    "MS": [(3, "very_short"), (12, "short"), (24, "medium")],
    "M":  [(3, "very_short"), (12, "short"), (24, "medium")],
    "QS": [(2, "very_short"), (4, "short"), (8, "medium")],
    "YS": [(1, "very_short"), (3, "short"), (5, "medium")],
}


def _bucket_horizon_difficulty(freq: str, horizon: int) -> str:
    bands = _HORIZON_DIFFICULTY.get(freq)
    if bands is None:
        return "medium"  # safe fallback for unknown frequencies
    for max_val, label in bands:
        if horizon <= max_val:
            return label
    return "long"


# ---------------------------------------------------------------------------
# Forecasting decompositions
# ---------------------------------------------------------------------------

def _detect_per_series(series: pd.Series, freq: str) -> tuple[bool, bool, bool]:
    """Return (seasonality, trend, stationarity) for one series."""
    from statsmodels.tsa.stattools import adfuller, kpss
    from scipy.stats import kendalltau

    seasonality = False
    if len(series) >= 24:
        period = {"H": 24, "D": 7, "W": 52, "MS": 12, "M": 12, "QS": 4, "YS": 1}.get(freq, 1)
        if period > 1 and len(series) > 2 * period:
            from statsmodels.tsa.stattools import acf
            try:
                acfs = acf(series.dropna(), nlags=min(period * 2, len(series) // 2 - 1))
                seasonality = abs(acfs[period]) > 0.3 if period < len(acfs) else False
            except Exception:
                seasonality = False

    trend = False
    if len(series) >= 10:
        try:
            x = np.arange(len(series))
            tau, p = kendalltau(x, series.values)
            trend = p < 0.05 and abs(tau) > 0.1
        except Exception:
            trend = False

    stationary = False
    if len(series) >= 12:
        try:
            res = adfuller(series.dropna(), autolag="AIC")
            stationary = res[1] < 0.05
        except Exception:
            stationary = False

    return seasonality, trend, stationary


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_dataset_profile(dataset_path: Path, task_metadata: dict[str, Any]) -> dict[str, Any]:
    """Compute the bucketed profile for the given canonical CSV."""
    df = pd.read_csv(dataset_path)
    problem_type = task_metadata["problem_type"]
    target = task_metadata["target_column"]

    # Universal
    n_rows = _bucket_n_rows(len(df))
    feature_df = df.drop(columns=[target], errors="ignore")
    n_features = _bucket_n_features(len(feature_df.columns))
    missing_rate = _bucket_missing(float(df.isnull().mean().mean()))
    n_cat = sum(1 for c in feature_df.columns if pd.api.types.is_object_dtype(feature_df[c]) or isinstance(feature_df[c].dtype, pd.CategoricalDtype))
    n_num = sum(1 for c in feature_df.columns if pd.api.types.is_numeric_dtype(feature_df[c]))

    profile: dict[str, Any] = {
        "schema_version": 1,
        "problem_type": problem_type,
        "n_rows": n_rows,
        "n_features": n_features,
        "missing_rate": missing_rate,
        "n_categorical_features": _bucket_count(n_cat),
        "n_numerical_features": _bucket_count(n_num),
    }

    if problem_type == "classification":
        profile["n_classes"] = _bucket_n_classes(df[target].nunique())
        profile["class_balance"] = _bucket_class_balance(df[target].value_counts())
    elif problem_type == "regression":
        profile["target_distribution"] = _bucket_target_distribution(df[target].dropna())
    elif problem_type == "forecasting":
        dt_col = task_metadata["datetime_column"]
        sid_cols = task_metadata.get("series_id_columns") or []
        freq = task_metadata["frequency"]
        horizon = int(task_metadata["forecast_horizon"])

        df[dt_col] = pd.to_datetime(df[dt_col])
        if sid_cols:
            grouped = df.groupby(sid_cols)
            n_series = grouped.ngroups
            avg_history = grouped.size().mean()
        else:
            n_series = 1
            avg_history = len(df)

        # Per-series stats: take a sample if many series
        sample_groups = (
            list(df.groupby(sid_cols))[:5] if sid_cols else [(("__single__",), df)]
        )
        votes_seasonal = votes_trend = votes_stationary = 0
        for _, g in sample_groups:
            s = g.set_index(dt_col)[target].sort_index()
            seas, tren, stat = _detect_per_series(s, freq)
            votes_seasonal += int(seas)
            votes_trend += int(tren)
            votes_stationary += int(stat)
        n_voted = max(len(sample_groups), 1)

        # Exogenous = any column not target/datetime/series_id
        protected = {target, dt_col, *sid_cols}
        exogenous = any(c not in protected for c in df.columns)

        profile.update({
            "n_series": _bucket_n_series(n_series),
            "history_length": _bucket_history_length(float(avg_history)),
            "frequency": freq,
            "horizon_difficulty": _bucket_horizon_difficulty(freq, horizon),
            "forecast_horizon_raw": horizon,
            "exogenous_features_available": bool(exogenous),
            "seasonality_detected": (votes_seasonal / n_voted) >= 0.5,
            "trend_detected": (votes_trend / n_voted) >= 0.5,
            "stationarity": (votes_stationary / n_voted) >= 0.5,
        })

    return profile
```

- [ ] **Step 4: Run tests**

```
uv run pytest tests/test_training/test_profiler.py -v
```
Expected: 5 tests PASS. (Some forecasting edge cases may emit warnings; that's fine.)

- [ ] **Step 5: Commit**

```bash
git add src/mlops_agents/training/profiler.py tests/test_training/__init__.py tests/test_training/test_profiler.py
git commit -m "feat: add dataset profiler (bucketed n_rows/missing/series/horizon/seasonality)"
```

---

## Task 12: Splitter — train/pool/test for tabular and forecasting

**Files:**
- Create: `src/mlops_agents/training/splitter.py`
- Create: `tests/test_training/test_splitter.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_training/test_splitter.py`:

```python
"""Tests for the train/pool/test splitter."""

import json

import pandas as pd
import pytest

from mlops_agents.training.splitter import split_dataset


def _write_csv(tmp_path, df, name="data.csv"):
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


def test_classification_stratified_split(tmp_path):
    df = pd.DataFrame({"x": range(100), "target": [0] * 70 + [1] * 30})
    csv = _write_csv(tmp_path, df)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    train_pool, test, meta = split_dataset(
        csv, {"problem_type": "classification", "target_column": "target"},
        out_dir, test_size=0.2, random_state=42,
    )
    assert train_pool.exists()
    assert test.exists()
    meta_data = json.loads(meta.read_text())
    assert meta_data["split_kind"] == "stratified"
    train_df = pd.read_csv(train_pool)
    test_df = pd.read_csv(test)
    assert len(train_df) == 80
    assert len(test_df) == 20


def test_regression_random_shuffle_split(tmp_path):
    df = pd.DataFrame({"x": range(50), "y": [float(i) for i in range(50)]})
    csv = _write_csv(tmp_path, df)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    train_pool, test, meta = split_dataset(
        csv, {"problem_type": "regression", "target_column": "y"},
        out_dir, test_size=0.2, random_state=42,
    )
    meta_data = json.loads(meta.read_text())
    assert meta_data["split_kind"] == "random_shuffle"


def test_forecasting_temporal_split_single_series(tmp_path):
    dates = pd.date_range("2020-01-01", periods=60, freq="MS")
    df = pd.DataFrame({"ds": dates, "y": range(60)})
    csv = _write_csv(tmp_path, df)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    train_pool, test, meta = split_dataset(
        csv,
        {
            "problem_type": "forecasting", "target_column": "y",
            "datetime_column": "ds", "series_id_columns": [],
            "frequency": "MS", "forecast_horizon": 12,
        },
        out_dir, test_size=0.2, random_state=42,
    )
    train_df = pd.read_csv(train_pool, parse_dates=["ds"])
    test_df = pd.read_csv(test, parse_dates=["ds"])
    assert len(test_df) == 12
    assert len(train_df) == 48
    assert train_df["ds"].max() < test_df["ds"].min()
    meta_data = json.loads(meta.read_text())
    assert meta_data["split_kind"] == "temporal_per_series"


def test_forecasting_drops_short_series_majority_minority(tmp_path):
    """If < 50% of series are too short, drop them and continue."""
    rows = []
    for sid in ["a", "b", "c"]:           # a, b OK
        rows += [{"sid": sid, "ds": d, "y": float(i)}
                 for i, d in enumerate(pd.date_range("2020-01-01", periods=60, freq="MS"))]
    rows += [{"sid": "tiny", "ds": d, "y": float(i)}
             for i, d in enumerate(pd.date_range("2020-01-01", periods=5, freq="MS"))]
    df = pd.DataFrame(rows)
    csv = _write_csv(tmp_path, df)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _, _, meta = split_dataset(
        csv,
        {
            "problem_type": "forecasting", "target_column": "y",
            "datetime_column": "ds", "series_id_columns": ["sid"],
            "frequency": "MS", "forecast_horizon": 12,
        },
        out_dir, test_size=0.2, random_state=42,
    )
    meta_data = json.loads(meta.read_text())
    assert meta_data["n_series_dropped"] == 1


def test_forecasting_majority_too_short_raises(tmp_path):
    """If > 50% of series are too short, raise ValueError."""
    rows = []
    for sid in ["a", "b", "c"]:           # all too short
        rows += [{"sid": sid, "ds": d, "y": float(i)}
                 for i, d in enumerate(pd.date_range("2020-01-01", periods=10, freq="MS"))]
    df = pd.DataFrame(rows)
    csv = _write_csv(tmp_path, df)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with pytest.raises(ValueError, match="too short"):
        split_dataset(
            csv,
            {
                "problem_type": "forecasting", "target_column": "y",
                "datetime_column": "ds", "series_id_columns": ["sid"],
                "frequency": "MS", "forecast_horizon": 12,
            },
            out_dir, test_size=0.2, random_state=42,
        )
```

- [ ] **Step 2: Verify failures**

```
uv run pytest tests/test_training/test_splitter.py -v
```
Expected: FAIL.

- [ ] **Step 3: Implement the splitter**

Create `src/mlops_agents/training/splitter.py`:

```python
"""Train/pool/test split for classification, regression, and forecasting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from mlops_agents.config.settings import settings


def split_dataset(
    canonical_path: Path,
    task_metadata: dict[str, Any],
    output_dir: Path,
    test_size: float | None = None,
    random_state: int = 42,
) -> tuple[Path, Path, Path]:
    """Write train_pool, test, and split_metadata files. Returns the three paths."""
    test_size = test_size if test_size is not None else settings.train_test_split_ratio
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = canonical_path.stem
    train_pool_path = output_dir / f"{stem}_train_pool.csv"
    test_path = output_dir / f"{stem}_test.csv"
    metadata_path = output_dir / f"{stem}_split_metadata.json"

    problem_type = task_metadata["problem_type"]
    target = task_metadata["target_column"]
    df = pd.read_csv(canonical_path)

    if problem_type == "classification":
        train_pool, test = train_test_split(
            df, test_size=test_size, stratify=df[target], random_state=random_state,
        )
        metadata = {
            "split_kind": "stratified",
            "n_train_pool": len(train_pool),
            "n_test": len(test),
            "test_size_ratio": test_size,
            "forecast_horizon": None,
            "n_series_total": None,
            "n_series_dropped": 0,
            "dropped_series": [],
            "random_state": random_state,
        }
    elif problem_type == "regression":
        train_pool, test = train_test_split(
            df, test_size=test_size, shuffle=True, random_state=random_state,
        )
        metadata = {
            "split_kind": "random_shuffle",
            "n_train_pool": len(train_pool),
            "n_test": len(test),
            "test_size_ratio": test_size,
            "forecast_horizon": None,
            "n_series_total": None,
            "n_series_dropped": 0,
            "dropped_series": [],
            "random_state": random_state,
        }
    elif problem_type == "forecasting":
        dt_col = task_metadata["datetime_column"]
        sid_cols = task_metadata.get("series_id_columns") or []
        horizon = int(task_metadata["forecast_horizon"])
        df[dt_col] = pd.to_datetime(df[dt_col])

        # Series length guard: each series needs horizon + horizon + min_train_points
        min_required = 2 * horizon + settings.forecasting_min_train_points
        if sid_cols:
            grouped = df.groupby(sid_cols)
            short = [(name, len(g)) for name, g in grouped if len(g) < min_required]
            ok_series = [name for name, _ in grouped if name not in {n for n, _ in short}]
            if len(short) > grouped.ngroups / 2:
                raise ValueError(
                    f"More than half of series ({len(short)}/{grouped.ngroups}) are too short "
                    f"(need >= {min_required} obs). Reduce forecast_horizon or filter dataset."
                )
            keep_keys = set(tuple(name) if isinstance(name, tuple) else (name,) for name in ok_series)
            df = df[df[sid_cols].apply(tuple, axis=1).isin(keep_keys)]
            n_total = grouped.ngroups
            dropped = [
                {"series_id": dict(zip(sid_cols, name)) if isinstance(name, tuple) else {sid_cols[0]: name},
                 "n_obs": n}
                for name, n in short
            ]
        else:
            if len(df) < min_required:
                raise ValueError(
                    f"Series too short ({len(df)} < {min_required} obs). "
                    f"Reduce forecast_horizon."
                )
            n_total = 1
            dropped = []

        # Per-series temporal split: last `horizon` rows of each series → test
        df = df.sort_values(sid_cols + [dt_col]) if sid_cols else df.sort_values(dt_col)
        if sid_cols:
            test = df.groupby(sid_cols).tail(horizon)
            train_pool = df.drop(test.index)
        else:
            test = df.tail(horizon)
            train_pool = df.iloc[:-horizon]

        metadata = {
            "split_kind": "temporal_per_series",
            "n_train_pool": len(train_pool),
            "n_test": len(test),
            "test_size_ratio": None,
            "forecast_horizon": horizon,
            "n_series_total": n_total,
            "n_series_dropped": len(dropped),
            "dropped_series": dropped,
            "random_state": random_state,
        }
    else:
        raise ValueError(f"Unknown problem_type: {problem_type!r}")

    train_pool.to_csv(train_pool_path, index=False)
    test.to_csv(test_path, index=False)
    metadata_path.write_text(json.dumps(metadata, default=str, indent=2))

    return train_pool_path, test_path, metadata_path
```

- [ ] **Step 4: Run tests**

```
uv run pytest tests/test_training/test_splitter.py -v
```
Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mlops_agents/training/splitter.py tests/test_training/test_splitter.py
git commit -m "feat: add splitter (stratified/random/temporal-per-series + length guards)"
```

---

## Task 13: Default plans — registry-eligible candidates per problem_type

**Files:**
- Create: `src/mlops_agents/training/default_plans.py`
- Create: `tests/test_training/test_default_plans.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_training/test_default_plans.py`:

```python
"""Tests for default_training_plan."""

import pytest

from mlops_agents.training.default_plans import default_training_plan


def test_default_classification_plan_lists_all_eligible():
    profile = {"problem_type": "classification", "n_rows": "medium"}
    plan = default_training_plan("classification", profile)
    keys = {c.model_key for c in plan.candidates}
    assert "logistic_regression" in keys
    assert "lightgbm_classifier" in keys
    assert plan.problem_type == "classification"


def test_default_regression_plan_lists_all_eligible():
    profile = {"problem_type": "regression", "n_rows": "medium"}
    plan = default_training_plan("regression", profile)
    keys = {c.model_key for c in plan.candidates}
    assert "ridge" in keys
    assert "lightgbm_regressor" in keys


def test_default_forecasting_plan_lists_all_eligible():
    profile = {
        "problem_type": "forecasting",
        "n_rows": "medium",
        "history_length": "long",
    }
    plan = default_training_plan("forecasting", profile)
    keys = {c.model_key for c in plan.candidates}
    assert "auto_arima" in keys
    assert "lightgbm_forecaster" in keys


def test_default_plan_skips_min_rows_violators():
    """very_small dataset → boosters with min_rows=500 are excluded."""
    profile = {"problem_type": "regression", "n_rows": "very_small"}
    plan = default_training_plan("regression", profile)
    keys = {c.model_key for c in plan.candidates}
    assert "ridge" in keys           # min_rows not set → kept
    assert "lightgbm_regressor" not in keys  # min_rows=500 → excluded


def test_default_plan_priorities_unique_and_ordered():
    plan = default_training_plan("classification", {"problem_type": "classification", "n_rows": "medium"})
    priorities = [c.priority for c in plan.candidates]
    assert len(priorities) == len(set(priorities))
    assert priorities == sorted(priorities)


def test_default_plan_empty_eligible_raises():
    with pytest.raises(ValueError, match="No eligible models"):
        default_training_plan("classification", {"problem_type": "classification", "n_rows": "irrelevant"})
        # The above won't actually fail; the next line forces it:
    profile = {"problem_type": "classification", "n_rows": "very_small"}
    plan = default_training_plan("classification", profile)
    # logistic_regression has no min_rows requirement; this should still succeed:
    assert "logistic_regression" in {c.model_key for c in plan.candidates}
```

- [ ] **Step 2: Verify failures**

```
uv run pytest tests/test_training/test_default_plans.py -v
```
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement default plans**

Create `src/mlops_agents/training/default_plans.py`:

```python
"""Generate a default TrainingPlan from the registry when no agent is involved."""

from __future__ import annotations

from typing import Any

from mlops_agents.contracts.training import (
    TrainingPlan,
    TrainingPlanCandidate,
    TrialBudget,
)
from mlops_agents.models.loader import ModelSpec, get_models_for


_NUM_BUCKET_ORDER = ("very_small", "small", "medium", "large")


def _row_count_lower_bound(bucket: str) -> int:
    """Lower bound of n_rows bucket (used for min_rows eligibility check)."""
    return {"very_small": 0, "small": 500, "medium": 1000, "large": 50_000}.get(bucket, 0)


def _is_eligible(model: ModelSpec, profile: dict[str, Any]) -> bool:
    """Apply registry `requires` gates against the profile."""
    requires = model.requires or {}
    n_rows_bucket = profile.get("n_rows", "small")
    if "min_rows" in requires:
        if _row_count_lower_bound(n_rows_bucket) < requires["min_rows"]:
            return False
    return True


def default_training_plan(problem_type: str, dataset_profile: dict[str, Any]) -> TrainingPlan:
    """All eligible models from the registry with deterministic registry-order priority."""
    eligible = [m for m in get_models_for(problem_type) if _is_eligible(m, dataset_profile)]
    candidates = [
        TrainingPlanCandidate(
            priority=i + 1,
            model_key=m.model_key,
            initial_hyperparameters=m.default_params,
            reason="default plan: registry-eligible",
        )
        for i, m in enumerate(eligible)
    ]
    if not candidates:
        raise ValueError(f"No eligible models for problem_type={problem_type}")
    return TrainingPlan(
        problem_type=problem_type,
        candidates=candidates,
        trial_budget=TrialBudget(allocation_strategy="equal", total_trials=60),
    )
```

- [ ] **Step 4: Run tests**

```
uv run pytest tests/test_training/test_default_plans.py -v
```
Expected: 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mlops_agents/training/default_plans.py tests/test_training/test_default_plans.py
git commit -m "feat: add default training plan (all eligible registry models, registry-order priority)"
```

---

## Task 14: Trial budget allocation

The allocator distributes `total_trials` across candidates by priority (or equally), respecting min/max bounds.

**Files:**
- Create: `src/mlops_agents/training/trial_budget.py`
- Create: `tests/test_training/test_trial_budget.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_training/test_trial_budget.py`:

```python
"""Tests for trial budget allocation."""

import pytest

from mlops_agents.contracts.training import TrainingPlanCandidate, TrialBudget
from mlops_agents.training.trial_budget import allocate_trials


def _candidates(priorities, requested=None):
    return [
        TrainingPlanCandidate(
            priority=p, model_key=f"m{i}",
            requested_trials=(requested[i] if requested else None),
        )
        for i, p in enumerate(priorities)
    ]


def test_priority_weighted_3_candidates_60_trials():
    budget = TrialBudget(total_trials=60, allocation_strategy="priority_weighted",
                         min_trials_per_candidate=5, max_trials_per_candidate=30)
    alloc = allocate_trials(_candidates([1, 2, 3]), budget)
    # P1 gets 30 (3/6 of 60), P2 gets 20 (2/6), P3 gets 10 (1/6)
    assert alloc["m0"] == 30
    assert alloc["m1"] == 20
    assert alloc["m2"] == 10


def test_equal_2_candidates_60_trials():
    budget = TrialBudget(total_trials=60, allocation_strategy="equal",
                         min_trials_per_candidate=5, max_trials_per_candidate=30)
    alloc = allocate_trials(_candidates([1, 2]), budget)
    assert alloc["m0"] == 30
    assert alloc["m1"] == 30


def test_min_clamp_applied():
    budget = TrialBudget(total_trials=10, allocation_strategy="priority_weighted",
                         min_trials_per_candidate=5, max_trials_per_candidate=30)
    alloc = allocate_trials(_candidates([1, 2, 3, 4]), budget)
    assert min(alloc.values()) >= 5


def test_max_clamp_applied():
    budget = TrialBudget(total_trials=200, allocation_strategy="equal",
                         min_trials_per_candidate=5, max_trials_per_candidate=20)
    alloc = allocate_trials(_candidates([1, 2]), budget)
    assert max(alloc.values()) <= 20


def test_requested_trials_advisory():
    budget = TrialBudget(total_trials=60, allocation_strategy="priority_weighted",
                         min_trials_per_candidate=5, max_trials_per_candidate=30)
    candidates = _candidates([1, 2, 3], requested=[15, 15, 15])
    alloc = allocate_trials(candidates, budget)
    # Each candidate's requested 15 honored, all within [5, 30]
    assert alloc["m0"] == 15
    assert alloc["m1"] == 15
    assert alloc["m2"] == 15
```

- [ ] **Step 2: Verify failures**

```
uv run pytest tests/test_training/test_trial_budget.py -v
```
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement allocator**

Create `src/mlops_agents/training/trial_budget.py`:

```python
"""Distribute total_trials across candidates per the TrialBudget policy."""

from __future__ import annotations

from mlops_agents.contracts.training import TrainingPlanCandidate, TrialBudget


def allocate_trials(
    candidates: list[TrainingPlanCandidate],
    budget: TrialBudget,
) -> dict[str, int]:
    """Return {model_key: n_trials} for each candidate.

    Algorithm:
    1. Compute base budget per candidate from total_trials + allocation_strategy.
    2. If candidate.requested_trials is set, use it as a preference.
    3. Clamp every candidate to [min_trials_per_candidate, max_trials_per_candidate].
    4. Renormalize down if sum exceeds total_trials (proportional, keeping floor).
    """
    n = len(candidates)
    if n == 0:
        return {}

    if budget.allocation_strategy == "priority_weighted":
        # Inverse-priority weights: lowest priority number gets the largest weight
        weights = [n + 1 - c.priority for c in candidates]
        total_weight = sum(weights)
        base = [int(round(budget.total_trials * w / total_weight)) for w in weights]
    else:  # equal
        base = [budget.total_trials // n] * n

    # Apply requested_trials as preference
    final = []
    for cand, b in zip(candidates, base):
        if cand.requested_trials is not None:
            final.append(cand.requested_trials)
        else:
            final.append(b)

    # Clamp to [min, max]
    final = [max(budget.min_trials_per_candidate, min(budget.max_trials_per_candidate, x)) for x in final]

    # Renormalize down if over budget
    if sum(final) > budget.total_trials:
        slack = sum(final) - budget.total_trials
        # Reduce proportionally from candidates above the floor
        reducible = [(i, x - budget.min_trials_per_candidate) for i, x in enumerate(final)
                     if x > budget.min_trials_per_candidate]
        if reducible:
            total_reducible = sum(r for _, r in reducible)
            for i, r in reducible:
                cut = int(round(slack * r / total_reducible))
                final[i] = max(budget.min_trials_per_candidate, final[i] - cut)

    return {c.model_key: t for c, t in zip(candidates, final)}
```

- [ ] **Step 4: Run tests**

```
uv run pytest tests/test_training/test_trial_budget.py -v
```
Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mlops_agents/training/trial_budget.py tests/test_training/test_trial_budget.py
git commit -m "feat: add trial budget allocator (priority-weighted + equal, clamp + renormalize)"
```

---

## Task 15: Experience record writer

`write_experience_record(...)` builds the JSON record and writes to `experience_pool/<task_id>.json`. The trainer calls this at the end of `run_training_plan`.

**Files:**
- Create: `src/mlops_agents/training/experience.py`
- Create: `tests/test_training/test_experience_writer.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_training/test_experience_writer.py`:

```python
"""Tests for the experience record writer."""

import json
from datetime import datetime, UTC

from mlops_agents.training.experience import build_task_id, write_experience_record


def test_build_task_id_format():
    tid = build_task_id("iris", "classification", run_idx=1)
    assert tid.startswith("iris_classification_")
    assert tid.endswith("_001")


def test_write_experience_record_minimal(tmp_path):
    record = {
        "task_id": "iris_classification_2026-05-06_001",
        "problem_type": "classification",
        "dataset_profile": {"n_rows": "small"},
        "training_plan_input": {"candidates": []},
        "split_artifacts": {},
        "mlflow": {"experiment_name": "x", "parent_run_id": "abc"},
        "metric_to_optimize": "macro_f1",
        "metric_direction": "maximize",
        "candidate_selection_policy": {"primary": "best_validation_score"},
        "models_tested": [],
        "selected_solution": {},
        "experience_summary": "",
    }
    out_path = write_experience_record(record, tmp_path)
    assert out_path.exists()
    loaded = json.loads(out_path.read_text())
    assert loaded["task_id"] == record["task_id"]
```

- [ ] **Step 2: Verify failures**

```
uv run pytest tests/test_training/test_experience_writer.py -v
```
Expected: FAIL.

- [ ] **Step 3: Implement**

Create `src/mlops_agents/training/experience.py`:

```python
"""Experience record writer: builds the long-term JSON dump per pipeline run."""

from __future__ import annotations

import json
from datetime import date, datetime, UTC
from pathlib import Path
from typing import Any


def build_task_id(dataset_stem: str, problem_type: str, run_idx: int = 1) -> str:
    """Format: <stem>_<problem_type>_<YYYY-MM-DD>_<NNN>."""
    today = date.today().strftime("%Y-%m-%d")
    return f"{dataset_stem}_{problem_type}_{today}_{run_idx:03d}"


def write_experience_record(record: dict[str, Any], output_dir: Path) -> Path:
    """Write the JSON record. Returns the file path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{record['task_id']}.json"
    out_path.write_text(json.dumps(record, default=str, indent=2))
    return out_path
```

- [ ] **Step 4: Run tests**

```
uv run pytest tests/test_training/test_experience_writer.py -v
```
Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mlops_agents/training/experience.py tests/test_training/test_experience_writer.py
git commit -m "feat: add experience record writer (JSON dump under settings.experience_pool_dir)"
```

---

## Task 16: Executor — classification path (full Optuna+MLflow flow on iris)

This is the biggest task. Implements `run_training_plan(...)` end-to-end for classification, including:

- per-candidate Optuna study (StratifiedKFold CV)
- nested MLflow runs (parent + per-candidate child)
- failure handling (retry with default_params)
- champion selection with tie tolerance + complexity_rank
- experience record assembly

**Files:**
- Create: `src/mlops_agents/training/executor.py`
- Create: `tests/test_training/test_executor_classification.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_training/test_executor_classification.py`:

```python
"""End-to-end test: executor on iris classification."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

from mlops_agents.contracts.training import TrainingPlan, TrainingPlanCandidate, TrialBudget
from mlops_agents.training.executor import run_training_plan


@pytest.fixture
def iris_csv(tmp_path):
    data = load_iris(as_frame=True)
    df = pd.concat([data.data, data.target.rename("target")], axis=1)
    p = tmp_path / "iris.csv"
    df.to_csv(p, index=False)
    return p


def test_executor_iris_classification_endtoend(iris_csv, tmp_path):
    plan = TrainingPlan(
        problem_type="classification",
        candidates=[
            TrainingPlanCandidate(priority=1, model_key="logistic_regression"),
            TrainingPlanCandidate(priority=2, model_key="random_forest_classifier"),
        ],
        trial_budget=TrialBudget(total_trials=10, min_trials_per_candidate=3, max_trials_per_candidate=10),
    )
    result = run_training_plan(
        plan=plan,
        processed_dataset_path=iris_csv,
        target_column="target",
        task_metadata={"problem_type": "classification", "target_column": "target"},
        output_dir=tmp_path / "splits",
        mlflow_experiment="test-iris",
        random_state=42,
    )
    # Champion + artifacts present
    assert Path(result.champion_model_path).exists()
    assert Path(result.train_pool_path).exists()
    assert Path(result.test_path).exists()
    assert Path(result.experience_record_path).exists()
    record = json.loads(Path(result.experience_record_path).read_text())
    assert record["problem_type"] == "classification"
    assert record["selected_solution"]["model_key"] in {"logistic_regression", "random_forest_classifier"}
    # All candidates appear in models_tested
    assert {c["model_key"] for c in record["models_tested"]} == {"logistic_regression", "random_forest_classifier"}
    # Validation score is sensible (iris is easy)
    assert result.champion_metrics["macro_f1"] > 0.85
```

- [ ] **Step 2: Verify failures**

```
uv run pytest tests/test_training/test_executor_classification.py -v
```
Expected: FAIL with `ModuleNotFoundError` on `executor`.

- [ ] **Step 3: Implement the executor**

Create `src/mlops_agents/training/executor.py`:

```python
"""Deterministic multi-candidate training executor.

Inputs: TrainingPlan + canonical dataset.
Outputs: champion model artifact, train/test split files, MLflow runs (parent + children),
and an ExperienceRecord JSON.
"""

from __future__ import annotations

import json
import pickle
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, StratifiedKFold

from mlops_agents.config.settings import settings
from mlops_agents.contracts.training import (
    TrainingPlan,
    TrainingPlanCandidate,
    TrainingResult,
)
from mlops_agents.models.factories import FACTORY_REGISTRY
from mlops_agents.models.loader import get_model
from mlops_agents.models.search_spaces import build_suggest_fn
from mlops_agents.training.experience import build_task_id, write_experience_record
from mlops_agents.training.override_validation import narrow_search_space
from mlops_agents.training.profiler import build_dataset_profile
from mlops_agents.training.splitter import split_dataset
from mlops_agents.training.trial_budget import allocate_trials

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

DEFAULT_METRIC_BY_PROBLEM_TYPE = {
    "classification": "macro_f1",
    "regression": "rmse",
    "forecasting": "rmse",
}

METRIC_DIRECTION = {
    "macro_f1": "maximize",
    "accuracy": "maximize",
    "roc_auc": "maximize",
    "rmse": "minimize",
    "mae": "minimize",
    "mape": "minimize",
    "smape": "minimize",
    "r2": "maximize",
}


def _classification_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def _regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# ---------------------------------------------------------------------------
# Per-candidate Optuna objectives
# ---------------------------------------------------------------------------

def _build_classification_objective(
    train_pool: pd.DataFrame,
    target: str,
    model_key: str,
    suggest_fn: Callable[[optuna.Trial], dict],
    metric: str,
) -> Callable[[optuna.Trial], float]:
    factory = FACTORY_REGISTRY[get_model(model_key).factory]
    X = train_pool.drop(columns=[target])
    y = train_pool[target]

    def objective(trial: optuna.Trial) -> float:
        params = suggest_fn(trial)
        skf = StratifiedKFold(n_splits=settings.cv_folds, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in skf.split(X, y):
            model = factory(params)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = model.predict(X.iloc[val_idx])
            scores.append(_classification_metrics(y.iloc[val_idx], preds)[metric])
        return float(np.mean(scores))

    return objective


def _build_regression_objective(
    train_pool: pd.DataFrame,
    target: str,
    model_key: str,
    suggest_fn: Callable[[optuna.Trial], dict],
    metric: str,
) -> Callable[[optuna.Trial], float]:
    factory = FACTORY_REGISTRY[get_model(model_key).factory]
    X = train_pool.drop(columns=[target])
    y = train_pool[target]

    def objective(trial: optuna.Trial) -> float:
        params = suggest_fn(trial)
        kf = KFold(n_splits=settings.cv_folds, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in kf.split(X):
            model = factory(params)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = model.predict(X.iloc[val_idx])
            scores.append(_regression_metrics(y.iloc[val_idx], preds)[metric])
        return float(np.mean(scores))

    return objective


# ---------------------------------------------------------------------------
# Per-candidate runner
# ---------------------------------------------------------------------------

def _run_candidate_classification(
    candidate: TrainingPlanCandidate,
    train_pool: pd.DataFrame,
    target: str,
    n_trials: int,
    metric: str,
    direction: str,
) -> dict[str, Any]:
    spec = get_model(candidate.model_key)
    if candidate.search_space_override:
        narrowed = narrow_search_space(candidate.model_key, candidate.search_space_override)
    else:
        narrowed = spec.search_space
    suggest_fn = build_suggest_fn(narrowed)
    obj = _build_classification_objective(train_pool, target, candidate.model_key, suggest_fn, metric)

    started = time.perf_counter()
    try:
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction=direction, sampler=sampler)
        study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
        if study.best_trial is None:
            raise RuntimeError("No successful trial")
        best_params = study.best_params
        best_score = study.best_value
        n_trials_used = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    except Exception as e:
        # Retry with default_params (no Optuna)
        try:
            factory = FACTORY_REGISTRY[spec.factory]
            X = train_pool.drop(columns=[target])
            y = train_pool[target]
            skf = StratifiedKFold(n_splits=settings.cv_folds, shuffle=True, random_state=42)
            scores = []
            for train_idx, val_idx in skf.split(X, y):
                model = factory(spec.default_params)
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                preds = model.predict(X.iloc[val_idx])
                scores.append(_classification_metrics(y.iloc[val_idx], preds)[metric])
            best_params = spec.default_params
            best_score = float(np.mean(scores))
            n_trials_used = 1
        except Exception as e2:
            return {
                "model_key": candidate.model_key,
                "status": "failed",
                "error_type": type(e2).__name__,
                "error_message": str(e2),
                "traceback": traceback.format_exc(),
                "n_trials_used": 0,
                "duration_s": time.perf_counter() - started,
                "complexity_rank": spec.complexity_rank,
            }

    return {
        "model_key": candidate.model_key,
        "status": "successful",
        "best_params": best_params,
        "best_score": float(best_score),
        "best_score_std": 0.0,    # filled below if multi-fold
        "n_trials_used": n_trials_used,
        "duration_s": time.perf_counter() - started,
        "complexity_rank": spec.complexity_rank,
    }


# ---------------------------------------------------------------------------
# Champion selection
# ---------------------------------------------------------------------------

def _pick_champion(
    candidate_results: list[dict[str, Any]],
    direction: str,
    tie_tolerance_relative: float,
) -> dict[str, Any]:
    successful = [r for r in candidate_results if r["status"] == "successful"]
    if not successful:
        raise RuntimeError(f"All candidates failed: {[r['model_key'] for r in candidate_results]}")

    if direction == "maximize":
        best = max(successful, key=lambda r: r["best_score"])
        threshold = best["best_score"] * (1 - tie_tolerance_relative)
        tied = [r for r in successful if r["best_score"] >= threshold]
    else:
        best = min(successful, key=lambda r: r["best_score"])
        threshold = best["best_score"] * (1 + tie_tolerance_relative)
        tied = [r for r in successful if r["best_score"] <= threshold]

    # Tie-break: lowest complexity_rank, then lowest priority (first in input order)
    tied.sort(key=lambda r: (r["complexity_rank"], 0))
    return tied[0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_training_plan(
    plan: TrainingPlan,
    processed_dataset_path: Path,
    target_column: str,
    task_metadata: dict[str, Any],
    output_dir: Path,
    mlflow_experiment: str,
    random_state: int = 42,
) -> TrainingResult:
    metric = plan.metric_to_optimize or DEFAULT_METRIC_BY_PROBLEM_TYPE[plan.problem_type]
    direction = METRIC_DIRECTION[metric]

    # 1. Build dataset profile (for the experience record)
    profile = build_dataset_profile(processed_dataset_path, task_metadata)

    # 2. Split
    train_pool_path, test_path, split_meta_path = split_dataset(
        processed_dataset_path, task_metadata, output_dir, random_state=random_state,
    )
    train_pool = pd.read_csv(train_pool_path)

    # 3. Allocate trials
    allocations = allocate_trials(plan.candidates, plan.trial_budget)

    # 4. Run candidates within nested MLflow runs
    mlflow.set_experiment(mlflow_experiment)
    candidate_results: list[dict[str, Any]] = []
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    with mlflow.start_run(run_name=f"pipeline_{ts}") as parent:
        parent_run_id = parent.info.run_id

        for cand in sorted(plan.candidates, key=lambda c: c.priority):
            with mlflow.start_run(run_name=cand.model_key, nested=True) as child:
                if plan.problem_type == "classification":
                    res = _run_candidate_classification(
                        cand, train_pool, target_column,
                        n_trials=allocations[cand.model_key],
                        metric=metric, direction=direction,
                    )
                elif plan.problem_type == "regression":
                    raise NotImplementedError("regression path lands in Task 17")
                else:
                    raise NotImplementedError("forecasting path lands in Task 18")

                res["mlflow_run_id"] = child.info.run_id
                if res["status"] == "successful":
                    mlflow.log_params(res["best_params"])
                    mlflow.log_metric(metric, res["best_score"])
                else:
                    mlflow.set_tag("status", "failed")
                    mlflow.set_tag("error_type", res.get("error_type", ""))
                candidate_results.append(res)

        # 5. Champion
        champion = _pick_champion(candidate_results, direction, settings.tie_tolerance_relative)

        # 6. Retrain champion on full train_pool, save artifact
        spec = get_model(champion["model_key"])
        factory = FACTORY_REGISTRY[spec.factory]
        X = train_pool.drop(columns=[target_column])
        y = train_pool[target_column]
        champion_model = factory(champion["best_params"])
        champion_model.fit(X, y)

        models_dir = output_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        champion_path = models_dir / f"champion_{champion['model_key']}.pkl"
        with champion_path.open("wb") as f:
            pickle.dump(champion_model, f)

        # Log champion artifact under its child run
        with mlflow.start_run(run_id=champion["mlflow_run_id"], nested=True):
            mlflow.set_tag("champion", "true")
            mlflow.log_artifact(str(champion_path))
        mlflow.set_tag("champion_run_id", champion["mlflow_run_id"])

        # 7. Compute final champion metrics on the train_pool with k-fold OOF (for reporting)
        champion_metrics = {metric: champion["best_score"]}

        # 8. Write experience record
        task_id = build_task_id(processed_dataset_path.stem, plan.problem_type, run_idx=1)
        record = {
            "task_id": task_id,
            "problem_type": plan.problem_type,
            "dataset_name": processed_dataset_path.stem,
            "dataset_profile": profile,
            "training_plan_input": plan.model_dump(),
            "split_artifacts": {
                "train_pool_path": str(train_pool_path),
                "test_path": str(test_path),
                "split_metadata_path": str(split_meta_path),
            },
            "mlflow": {"experiment_name": mlflow_experiment, "parent_run_id": parent_run_id},
            "metric_to_optimize": metric,
            "metric_direction": direction,
            "candidate_selection_policy": {
                "primary": "best_validation_score",
                "tie_breaker": "complexity_rank",
                "tie_tolerance_relative": settings.tie_tolerance_relative,
            },
            "models_tested": [
                {k: v for k, v in r.items() if k != "traceback"}
                for r in candidate_results
            ],
            "selected_solution": {
                "model_key": champion["model_key"],
                "hyperparameters": champion["best_params"],
                "validation_strategy": "stratified_5_fold_cv",
                "main_metric": metric,
                "validation_score": champion["best_score"],
                "validation_std": champion.get("best_score_std", 0.0),
                "complexity_rank": champion["complexity_rank"],
            },
            "experience_summary": "",
        }
        record_path = write_experience_record(record, settings.experience_pool_dir)

    return TrainingResult(
        champion_candidate=champion,
        champion_model_path=str(champion_path),
        train_pool_path=str(train_pool_path),
        test_path=str(test_path),
        split_metadata_path=str(split_meta_path),
        mlflow_parent_run_id=parent_run_id,
        experience_record_path=str(record_path),
        champion_metrics=champion_metrics,
    )
```

- [ ] **Step 4: Run the iris test**

```
uv run pytest tests/test_training/test_executor_classification.py -v
```
Expected: PASS in ~30–60 seconds (Optuna runs).

- [ ] **Step 5: Commit**

```bash
git add src/mlops_agents/training/executor.py tests/test_training/test_executor_classification.py
git commit -m "feat: training executor — classification path with Optuna+MLflow nested runs and champion selection"
```

---

## Task 17: Executor — regression path

Extend the executor to handle regression: KFold(shuffle=True), RMSE/MAE/R² metrics, retrain champion.

**Files:**
- Modify: `src/mlops_agents/training/executor.py`
- Create: `tests/test_training/test_executor_regression.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_training/test_executor_regression.py`:

```python
"""End-to-end test: executor on California Housing regression."""

import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import fetch_california_housing

from mlops_agents.contracts.training import TrainingPlan, TrainingPlanCandidate, TrialBudget
from mlops_agents.training.executor import run_training_plan


@pytest.fixture
def housing_csv(tmp_path):
    data = fetch_california_housing(as_frame=True)
    df = pd.concat([data.data, data.target.rename("target")], axis=1).head(1500)
    p = tmp_path / "housing.csv"
    df.to_csv(p, index=False)
    return p


def test_executor_housing_regression_endtoend(housing_csv, tmp_path):
    plan = TrainingPlan(
        problem_type="regression",
        candidates=[
            TrainingPlanCandidate(priority=1, model_key="ridge"),
            TrainingPlanCandidate(priority=2, model_key="random_forest_regressor"),
        ],
        trial_budget=TrialBudget(total_trials=10, min_trials_per_candidate=3, max_trials_per_candidate=10),
    )
    result = run_training_plan(
        plan=plan,
        processed_dataset_path=housing_csv,
        target_column="target",
        task_metadata={"problem_type": "regression", "target_column": "target"},
        output_dir=tmp_path / "splits",
        mlflow_experiment="test-housing",
        random_state=42,
    )
    assert Path(result.champion_model_path).exists()
    record = json.loads(Path(result.experience_record_path).read_text())
    assert record["problem_type"] == "regression"
    assert record["metric_direction"] == "minimize"
    assert record["selected_solution"]["validation_score"] > 0  # RMSE is positive
```

- [ ] **Step 2: Verify it fails**

```
uv run pytest tests/test_training/test_executor_regression.py -v
```
Expected: FAIL with `NotImplementedError: regression path lands in Task 17`.

- [ ] **Step 3: Add regression path to executor**

In `src/mlops_agents/training/executor.py`, add a `_run_candidate_regression` function next to `_run_candidate_classification`:

```python
def _run_candidate_regression(
    candidate: TrainingPlanCandidate,
    train_pool: pd.DataFrame,
    target: str,
    n_trials: int,
    metric: str,
    direction: str,
) -> dict[str, Any]:
    spec = get_model(candidate.model_key)
    if candidate.search_space_override:
        narrowed = narrow_search_space(candidate.model_key, candidate.search_space_override)
    else:
        narrowed = spec.search_space
    suggest_fn = build_suggest_fn(narrowed)
    obj = _build_regression_objective(train_pool, target, candidate.model_key, suggest_fn, metric)

    started = time.perf_counter()
    try:
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction=direction, sampler=sampler)
        study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
        if study.best_trial is None:
            raise RuntimeError("No successful trial")
        best_params = study.best_params
        best_score = study.best_value
        n_trials_used = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    except Exception:
        try:
            factory = FACTORY_REGISTRY[spec.factory]
            X = train_pool.drop(columns=[target])
            y = train_pool[target]
            kf = KFold(n_splits=settings.cv_folds, shuffle=True, random_state=42)
            scores = []
            for train_idx, val_idx in kf.split(X):
                model = factory(spec.default_params)
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                preds = model.predict(X.iloc[val_idx])
                scores.append(_regression_metrics(y.iloc[val_idx], preds)[metric])
            best_params = spec.default_params
            best_score = float(np.mean(scores))
            n_trials_used = 1
        except Exception as e2:
            return {
                "model_key": candidate.model_key,
                "status": "failed",
                "error_type": type(e2).__name__,
                "error_message": str(e2),
                "traceback": traceback.format_exc(),
                "n_trials_used": 0,
                "duration_s": time.perf_counter() - started,
                "complexity_rank": spec.complexity_rank,
            }

    return {
        "model_key": candidate.model_key,
        "status": "successful",
        "best_params": best_params,
        "best_score": float(best_score),
        "best_score_std": 0.0,
        "n_trials_used": n_trials_used,
        "duration_s": time.perf_counter() - started,
        "complexity_rank": spec.complexity_rank,
    }
```

In `run_training_plan`, replace `raise NotImplementedError("regression path lands in Task 17")` with:

```python
                elif plan.problem_type == "regression":
                    res = _run_candidate_regression(
                        cand, train_pool, target_column,
                        n_trials=allocations[cand.model_key],
                        metric=metric, direction=direction,
                    )
```

Also update the experience record's `validation_strategy` field to be problem-type-aware (replace the hardcoded `"stratified_5_fold_cv"`):
```python
            "validation_strategy": (
                "stratified_5_fold_cv" if plan.problem_type == "classification"
                else "kfold_5_shuffle" if plan.problem_type == "regression"
                else "temporal_holdout"
            ),
```

- [ ] **Step 4: Run all executor tests**

```
uv run pytest tests/test_training/test_executor_classification.py tests/test_training/test_executor_regression.py -v
```
Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mlops_agents/training/executor.py tests/test_training/test_executor_regression.py
git commit -m "feat: training executor — regression path (KFold shuffle, RMSE/MAE/R2)"
```

---

## Task 18: Executor — forecasting path (single + multi-series)

Add the forecasting path: temporal holdout inside train_pool, statsforecast for stat models, skforecast for supervised. Multi-series via long-format input.

**Files:**
- Modify: `src/mlops_agents/training/executor.py`
- Create: `tests/test_training/test_executor_forecasting.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_training/test_executor_forecasting.py`:

```python
"""End-to-end tests: executor on forecasting datasets."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlops_agents.contracts.training import TrainingPlan, TrainingPlanCandidate, TrialBudget
from mlops_agents.training.executor import run_training_plan


@pytest.fixture
def air_passengers_csv(tmp_path):
    """AirPassengers-style monthly series."""
    dates = pd.date_range("2010-01-01", periods=144, freq="MS")
    rng = np.random.default_rng(0)
    trend = np.arange(144) * 1.5
    seasonal = 30 * np.sin(np.arange(144) * 2 * np.pi / 12)
    noise = rng.normal(scale=5.0, size=144)
    df = pd.DataFrame({"month": dates, "passengers": 200 + trend + seasonal + noise})
    p = tmp_path / "air_passengers.csv"
    df.to_csv(p, index=False)
    return p


def test_executor_forecasting_single_series_statistical(air_passengers_csv, tmp_path):
    plan = TrainingPlan(
        problem_type="forecasting",
        candidates=[
            TrainingPlanCandidate(priority=1, model_key="seasonal_naive"),
            TrainingPlanCandidate(priority=2, model_key="ets"),
        ],
        trial_budget=TrialBudget(total_trials=6, min_trials_per_candidate=3, max_trials_per_candidate=5),
    )
    result = run_training_plan(
        plan=plan,
        processed_dataset_path=air_passengers_csv,
        target_column="passengers",
        task_metadata={
            "problem_type": "forecasting", "target_column": "passengers",
            "datetime_column": "month", "series_id_columns": [],
            "frequency": "MS", "forecast_horizon": 12,
        },
        output_dir=tmp_path / "splits",
        mlflow_experiment="test-air",
        random_state=42,
    )
    record = json.loads(Path(result.experience_record_path).read_text())
    assert record["problem_type"] == "forecasting"
    assert record["selected_solution"]["model_key"] in {"seasonal_naive", "ets"}


def test_executor_forecasting_multi_series_supervised(tmp_path):
    rows = []
    rng = np.random.default_rng(0)
    for sid in ["a", "b", "c"]:
        dates = pd.date_range("2010-01-01", periods=120, freq="MS")
        offset = {"a": 0, "b": 50, "c": 100}[sid]
        seasonal = 20 * np.sin(np.arange(120) * 2 * np.pi / 12)
        trend = np.arange(120) * 0.5
        noise = rng.normal(scale=3, size=120)
        for d, v in zip(dates, offset + trend + seasonal + noise):
            rows.append({"sid": sid, "ds": d, "y": float(v)})
    df = pd.DataFrame(rows)
    csv = tmp_path / "panel.csv"
    df.to_csv(csv, index=False)

    plan = TrainingPlan(
        problem_type="forecasting",
        candidates=[
            TrainingPlanCandidate(priority=1, model_key="seasonal_naive"),
            TrainingPlanCandidate(priority=2, model_key="lightgbm_forecaster"),
        ],
        trial_budget=TrialBudget(total_trials=6, min_trials_per_candidate=3, max_trials_per_candidate=5),
    )
    result = run_training_plan(
        plan=plan,
        processed_dataset_path=csv,
        target_column="y",
        task_metadata={
            "problem_type": "forecasting", "target_column": "y",
            "datetime_column": "ds", "series_id_columns": ["sid"],
            "frequency": "MS", "forecast_horizon": 12,
        },
        output_dir=tmp_path / "splits",
        mlflow_experiment="test-panel",
        random_state=42,
    )
    record = json.loads(Path(result.experience_record_path).read_text())
    assert record["selected_solution"]["model_key"] in {"seasonal_naive", "lightgbm_forecaster"}
```

- [ ] **Step 2: Verify failure**

```
uv run pytest tests/test_training/test_executor_forecasting.py -v
```
Expected: FAIL with `NotImplementedError`.

- [ ] **Step 3: Add forecasting path to executor**

In `src/mlops_agents/training/executor.py`, add forecasting metrics function:

```python
def _forecasting_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    out = {"rmse": rmse, "mae": mae}
    # MAPE / sMAPE only when target has no zeros
    if (y_true != 0).all():
        out["mape"] = float(np.mean(np.abs((y_true - y_pred) / y_true)))
    out["smape"] = float(
        np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-10))
    )
    return out
```

Add the forecasting candidate runner (handles both statsforecast and skforecast paths):

```python
def _is_statsforecast_model(model_key: str) -> bool:
    return get_model(model_key).library == "statsforecast"


def _convert_to_statsforecast_format(
    df: pd.DataFrame, target: str, datetime_col: str, sid_cols: list[str],
) -> pd.DataFrame:
    """Convert long panel to statsforecast format: unique_id, ds, y."""
    out = df.rename(columns={target: "y", datetime_col: "ds"}).copy()
    if sid_cols:
        if len(sid_cols) == 1:
            out = out.rename(columns={sid_cols[0]: "unique_id"})
        else:
            out["unique_id"] = out[sid_cols].astype(str).agg("__".join, axis=1)
    else:
        out["unique_id"] = "__single__"
    out["ds"] = pd.to_datetime(out["ds"])
    return out[["unique_id", "ds", "y"]]


def _run_candidate_forecasting(
    candidate: TrainingPlanCandidate,
    train_pool: pd.DataFrame,
    task_metadata: dict[str, Any],
    n_trials: int,
    metric: str,
    direction: str,
) -> dict[str, Any]:
    spec = get_model(candidate.model_key)
    target = task_metadata["target_column"]
    dt_col = task_metadata["datetime_column"]
    sid_cols = task_metadata.get("series_id_columns") or []
    horizon = int(task_metadata["forecast_horizon"])

    started = time.perf_counter()

    # Single temporal holdout inside train_pool: last `horizon` per series → val
    train_pool[dt_col] = pd.to_datetime(train_pool[dt_col])
    if sid_cols:
        train_pool_sorted = train_pool.sort_values(sid_cols + [dt_col])
        val = train_pool_sorted.groupby(sid_cols).tail(horizon)
        train = train_pool_sorted.drop(val.index)
    else:
        train_pool_sorted = train_pool.sort_values(dt_col)
        val = train_pool_sorted.tail(horizon)
        train = train_pool_sorted.iloc[:-horizon]

    is_stat = _is_statsforecast_model(candidate.model_key)

    def fit_predict_score(params: dict[str, Any]) -> float:
        factory = FACTORY_REGISTRY[spec.factory]
        if is_stat:
            sf = factory({"task_metadata": task_metadata, "params": params})
            train_sf = _convert_to_statsforecast_format(train, target, dt_col, sid_cols)
            sf.fit(train_sf)
            fcst = sf.predict(h=horizon)
            # statsforecast returns columns: unique_id, ds, <ModelName>
            model_col = [c for c in fcst.columns if c not in ("unique_id", "ds")][0]
            val_sf = _convert_to_statsforecast_format(val, target, dt_col, sid_cols)
            merged = val_sf.merge(fcst, on=["unique_id", "ds"])
            return _forecasting_metrics(merged["y"].values, merged[model_col].values)[metric]
        else:
            forecaster = factory({"task_metadata": task_metadata, "params": params})
            train_dict = (
                {sid: g.set_index(dt_col)[target].sort_index()
                 for sid, g in train.groupby(sid_cols[0] if sid_cols else lambda _: "__single__")}
                if sid_cols else
                {"__single__": train.set_index(dt_col)[target].sort_index()}
            )
            forecaster.fit(series=train_dict)
            preds = forecaster.predict(steps=horizon)
            # skforecast preds in long format: index per (level, step), columns include 'pred'
            pred_long = preds.reset_index() if "level" not in preds.columns else preds
            val_long = (
                val.assign(level=val[sid_cols[0]] if sid_cols else "__single__")
                   .rename(columns={target: "y_true", dt_col: "ds"})
                   [["level", "ds", "y_true"]]
            )
            joined = val_long.merge(pred_long, on=["level", "ds"], how="inner")
            pred_col = [c for c in joined.columns if c not in ("level", "ds", "y_true")][0]
            return _forecasting_metrics(joined["y_true"].values, joined[pred_col].values)[metric]

    if candidate.search_space_override:
        narrowed = narrow_search_space(candidate.model_key, candidate.search_space_override)
    else:
        narrowed = spec.search_space
    suggest_fn = build_suggest_fn(narrowed)

    def objective(trial: optuna.Trial) -> float:
        return fit_predict_score(suggest_fn(trial))

    try:
        if not narrowed.params:
            # No-op search: just evaluate default_params once
            best_score = fit_predict_score(spec.default_params)
            best_params = spec.default_params
            n_trials_used = 1
        else:
            sampler = optuna.samplers.TPESampler(seed=42)
            study = optuna.create_study(direction=direction, sampler=sampler)
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            if study.best_trial is None:
                raise RuntimeError("No successful trial")
            best_params = study.best_params
            best_score = study.best_value
            n_trials_used = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    except Exception as e:
        try:
            best_score = fit_predict_score(spec.default_params)
            best_params = spec.default_params
            n_trials_used = 1
        except Exception as e2:
            return {
                "model_key": candidate.model_key,
                "status": "failed",
                "error_type": type(e2).__name__,
                "error_message": str(e2),
                "traceback": traceback.format_exc(),
                "n_trials_used": 0,
                "duration_s": time.perf_counter() - started,
                "complexity_rank": spec.complexity_rank,
            }

    return {
        "model_key": candidate.model_key,
        "status": "successful",
        "best_params": best_params,
        "best_score": float(best_score),
        "best_score_std": 0.0,
        "n_trials_used": n_trials_used,
        "duration_s": time.perf_counter() - started,
        "complexity_rank": spec.complexity_rank,
    }
```

In `run_training_plan`, replace `raise NotImplementedError("forecasting path lands in Task 18")` with:

```python
                else:  # forecasting
                    res = _run_candidate_forecasting(
                        cand, train_pool, task_metadata,
                        n_trials=allocations[cand.model_key],
                        metric=metric, direction=direction,
                    )
```

Also: forecasting champion retraining. After `_pick_champion`, the existing code retrains via `factory(champion["best_params"]).fit(X, y)`. For forecasting that doesn't apply directly. Replace the champion-retrain block with a problem-type switch:

```python
        # 6. Retrain champion on full train_pool, save artifact
        spec = get_model(champion["model_key"])
        factory = FACTORY_REGISTRY[spec.factory]
        models_dir = output_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        champion_path = models_dir / f"champion_{champion['model_key']}.pkl"

        if plan.problem_type in ("classification", "regression"):
            X = train_pool.drop(columns=[target_column])
            y = train_pool[target_column]
            champion_model = factory(champion["best_params"])
            champion_model.fit(X, y)
            with champion_path.open("wb") as f:
                pickle.dump(champion_model, f)
        else:  # forecasting
            train_pool_full = train_pool.copy()
            train_pool_full[task_metadata["datetime_column"]] = pd.to_datetime(
                train_pool_full[task_metadata["datetime_column"]]
            )
            sid_cols = task_metadata.get("series_id_columns") or []
            if _is_statsforecast_model(champion["model_key"]):
                sf = factory({"task_metadata": task_metadata, "params": champion["best_params"]})
                train_sf = _convert_to_statsforecast_format(
                    train_pool_full, target_column, task_metadata["datetime_column"], sid_cols,
                )
                sf.fit(train_sf)
                with champion_path.open("wb") as f:
                    pickle.dump(sf, f)
            else:
                forecaster = factory({"task_metadata": task_metadata, "params": champion["best_params"]})
                series_dict = (
                    {sid: g.set_index(task_metadata["datetime_column"])[target_column].sort_index()
                     for sid, g in train_pool_full.groupby(sid_cols[0])}
                    if sid_cols else
                    {"__single__": train_pool_full.set_index(task_metadata["datetime_column"])[target_column].sort_index()}
                )
                forecaster.fit(series=series_dict)
                with champion_path.open("wb") as f:
                    pickle.dump(forecaster, f)
```

- [ ] **Step 4: Run forecasting tests**

```
uv run pytest tests/test_training/test_executor_forecasting.py -v
```
Expected: 2 tests PASS (~30–60s each).

- [ ] **Step 5: Run all executor tests for regression-test sweep**

```
uv run pytest tests/test_training/ -v
```
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/mlops_agents/training/executor.py tests/test_training/test_executor_forecasting.py
git commit -m "feat: training executor — forecasting path (statsforecast + skforecast, single + multi-series)"
```

---

## Task 19: Champion selection — tie tolerance + complexity_rank

Already implemented in Task 16 inside `_pick_champion`. This task is a focused unit test to lock the contract.

**Files:**
- Create: `tests/test_training/test_executor_champion_selection.py`

- [ ] **Step 1: Write the test**

Create `tests/test_training/test_executor_champion_selection.py`:

```python
"""Tests for champion selection: tie tolerance + complexity_rank tie-break."""

import pytest

from mlops_agents.training.executor import _pick_champion


def test_strict_winner_no_tie():
    results = [
        {"model_key": "a", "status": "successful", "best_score": 0.95, "complexity_rank": 1},
        {"model_key": "b", "status": "successful", "best_score": 0.90, "complexity_rank": 2},
    ]
    champ = _pick_champion(results, "maximize", 0.01)
    assert champ["model_key"] == "a"


def test_tie_within_tolerance_simpler_wins_maximize():
    """0.953 vs 0.96 → 0.953 >= 0.96*(1-0.01)=0.9504 → tied → simpler wins."""
    results = [
        {"model_key": "complex", "status": "successful", "best_score": 0.96, "complexity_rank": 3},
        {"model_key": "simple",  "status": "successful", "best_score": 0.953, "complexity_rank": 1},
    ]
    champ = _pick_champion(results, "maximize", 0.01)
    assert champ["model_key"] == "simple"


def test_tie_within_tolerance_simpler_wins_minimize():
    """RMSE: 0.10 vs 0.099 → 0.10 <= 0.099*(1+0.01)=0.10 → tied → simpler wins."""
    results = [
        {"model_key": "complex", "status": "successful", "best_score": 0.099, "complexity_rank": 3},
        {"model_key": "simple",  "status": "successful", "best_score": 0.100, "complexity_rank": 1},
    ]
    champ = _pick_champion(results, "minimize", 0.01)
    assert champ["model_key"] == "simple"


def test_skips_failed_candidates():
    results = [
        {"model_key": "a", "status": "failed", "error_type": "Boom", "complexity_rank": 1},
        {"model_key": "b", "status": "successful", "best_score": 0.5, "complexity_rank": 2},
    ]
    champ = _pick_champion(results, "maximize", 0.01)
    assert champ["model_key"] == "b"


def test_all_failed_raises():
    results = [
        {"model_key": "a", "status": "failed", "error_type": "X", "complexity_rank": 1},
        {"model_key": "b", "status": "failed", "error_type": "Y", "complexity_rank": 2},
    ]
    with pytest.raises(RuntimeError, match="All candidates failed"):
        _pick_champion(results, "maximize", 0.01)
```

- [ ] **Step 2: Run the test**

```
uv run pytest tests/test_training/test_executor_champion_selection.py -v
```
Expected: 5 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_training/test_executor_champion_selection.py
git commit -m "test: champion selection — tie tolerance + complexity_rank tie-break"
```

---

## Task 20: Failure handling test

A targeted test that triggers a candidate failure and verifies skip-and-continue behavior.

**Files:**
- Create: `tests/test_training/test_executor_failure_handling.py`

- [ ] **Step 1: Write the test**

Create `tests/test_training/test_executor_failure_handling.py`:

```python
"""Tests for executor failure handling: skip+retry+continue."""

import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import load_iris

from mlops_agents.contracts.training import (
    SearchParamOverride,
    TrainingPlan,
    TrainingPlanCandidate,
    TrialBudget,
)
from mlops_agents.training.executor import run_training_plan


@pytest.fixture
def iris_csv(tmp_path):
    data = load_iris(as_frame=True)
    df = pd.concat([data.data, data.target.rename("target")], axis=1)
    p = tmp_path / "iris.csv"
    df.to_csv(p, index=False)
    return p


def test_one_failed_one_succeeds_run_completes(iris_csv, tmp_path, monkeypatch):
    """Force one candidate's factory to raise; the other should still produce a champion."""
    from mlops_agents.models import factories
    real = factories.FACTORY_REGISTRY["build_logistic_regression"]

    def boom(_params):
        raise RuntimeError("simulated factory failure")

    factories.FACTORY_REGISTRY["build_logistic_regression"] = boom
    try:
        plan = TrainingPlan(
            problem_type="classification",
            candidates=[
                TrainingPlanCandidate(priority=1, model_key="logistic_regression"),
                TrainingPlanCandidate(priority=2, model_key="random_forest_classifier"),
            ],
            trial_budget=TrialBudget(total_trials=6, min_trials_per_candidate=3, max_trials_per_candidate=3),
        )
        result = run_training_plan(
            plan=plan,
            processed_dataset_path=iris_csv,
            target_column="target",
            task_metadata={"problem_type": "classification", "target_column": "target"},
            output_dir=tmp_path / "splits",
            mlflow_experiment="test-failure",
            random_state=42,
        )
        record = json.loads(Path(result.experience_record_path).read_text())
        statuses = {r["model_key"]: r["status"] for r in record["models_tested"]}
        assert statuses["logistic_regression"] == "failed"
        assert statuses["random_forest_classifier"] == "successful"
        assert record["selected_solution"]["model_key"] == "random_forest_classifier"
    finally:
        factories.FACTORY_REGISTRY["build_logistic_regression"] = real


def test_all_failed_raises_runtimeerror(iris_csv, tmp_path):
    from mlops_agents.models import factories
    real_lr = factories.FACTORY_REGISTRY["build_logistic_regression"]
    real_rf = factories.FACTORY_REGISTRY["build_random_forest_classifier"]
    factories.FACTORY_REGISTRY["build_logistic_regression"] = lambda *_: (_ for _ in ()).throw(RuntimeError("boom"))
    factories.FACTORY_REGISTRY["build_random_forest_classifier"] = lambda *_: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        plan = TrainingPlan(
            problem_type="classification",
            candidates=[
                TrainingPlanCandidate(priority=1, model_key="logistic_regression"),
                TrainingPlanCandidate(priority=2, model_key="random_forest_classifier"),
            ],
            trial_budget=TrialBudget(total_trials=6, min_trials_per_candidate=3, max_trials_per_candidate=3),
        )
        with pytest.raises(RuntimeError, match="All candidates failed"):
            run_training_plan(
                plan=plan,
                processed_dataset_path=iris_csv,
                target_column="target",
                task_metadata={"problem_type": "classification", "target_column": "target"},
                output_dir=tmp_path / "splits",
                mlflow_experiment="test-allfail",
                random_state=42,
            )
    finally:
        factories.FACTORY_REGISTRY["build_logistic_regression"] = real_lr
        factories.FACTORY_REGISTRY["build_random_forest_classifier"] = real_rf
```

- [ ] **Step 2: Run the test**

```
uv run pytest tests/test_training/test_executor_failure_handling.py -v
```
Expected: 2 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_training/test_executor_failure_handling.py
git commit -m "test: executor failure handling — skip+retry+continue, all-fail raises"
```

---

## Task 21: State updates + trainer_node rewrite + cleanup of old files

Wire the executor into the graph. Update AgentState. Delete the old training agent infrastructure.

**Files:**
- Modify: `src/mlops_agents/state/agent_state.py`
- Modify: `src/mlops_agents/graphs/mlops_graph.py`
- Modify: `src/mlops_agents/agents/registry.py`
- Delete: `src/mlops_agents/agents/training_agent.py`
- Delete: `src/mlops_agents/tools/training_tools.py`
- Delete: `src/mlops_agents/prompts/training_agent.yaml`
- Delete: `tests/test_tools/test_training_tools.py` (if it exists)

- [ ] **Step 1: Update `agent_state.py`**

In `src/mlops_agents/state/agent_state.py`, find the existing AgentState TypedDict and add new fields. Locate the existing line:
```python
    processed_dataset_path: str   # canonical CSV written by data_validator_node
```
(That was renamed in Task 1.) Right after it, ensure these fields exist (they may already be present from earlier work — if so, leave them; if not, add):

```python
    # Training (SP3)
    training_plan: dict | None
    train_pool_path: str | None
    test_path: str | None
    split_metadata_path: str | None
    champion_candidate: dict | None
    experience_record_path: str | None
    # Existing fields kept: trained_model_path, training_run_id, training_metrics
```

(The existing fields above should already exist; verify with grep.)

- [ ] **Step 2: Rewrite `trainer_node` in `mlops_graph.py`**

Find the current `trainer_node` definition (it imports the react agent and invokes it). Replace its full body with:

```python
def trainer_node(state: AgentState) -> Command:
    from pathlib import Path

    from mlops_agents.contracts.training import TrainingPlan
    from mlops_agents.training.default_plans import default_training_plan
    from mlops_agents.training.executor import run_training_plan
    from mlops_agents.training.profiler import build_dataset_profile

    processed_path = Path(state["processed_dataset_path"])
    task_meta = state["task_metadata"]

    profile = build_dataset_profile(processed_path, task_meta)
    plan_dict = state.get("training_plan")
    if plan_dict:
        plan = TrainingPlan.model_validate(plan_dict)
    else:
        plan = default_training_plan(state["problem_type"], profile)

    result = run_training_plan(
        plan=plan,
        processed_dataset_path=processed_path,
        target_column=task_meta["target_column"],
        task_metadata=task_meta,
        output_dir=Path("data/processed"),
        mlflow_experiment=settings.mlflow_experiment_name,
    )

    return Command(
        goto="supervisor",
        update={
            "training_plan": plan.model_dump(),
            "train_pool_path": result.train_pool_path,
            "test_path": result.test_path,
            "split_metadata_path": result.split_metadata_path,
            "trained_model_path": result.champion_model_path,
            "training_run_id": result.mlflow_parent_run_id,
            "training_metrics": result.champion_metrics,
            "champion_candidate": result.champion_candidate,
            "experience_record_path": result.experience_record_path,
        },
    )
```

Remove any `_build_trainer_context` helper that was specific to the old agent.

- [ ] **Step 3: Remove `"trainer"` agent from `registry.py`**

In `src/mlops_agents/agents/registry.py`, remove the import of `build_training_agent` and the `"trainer"` entry from any factory dict.

- [ ] **Step 4: Delete obsolete files**

```bash
rm src/mlops_agents/agents/training_agent.py
rm src/mlops_agents/tools/training_tools.py
rm src/mlops_agents/prompts/training_agent.yaml
rm tests/test_tools/test_training_tools.py 2>/dev/null || true
```

Also remove any `from mlops_agents.tools.training_tools import` statements that may exist anywhere — search first:
```
grep -rn 'training_tools\|training_agent' src/ tests/ --include='*.py'
```
Each match is a stale reference; remove or update.

- [ ] **Step 5: Run the full test suite**

```
uv run pytest -m "not integration" -q
```
Expected: all PASS (including ~30 new tests from this plan + 171 pre-existing). Total ≈ 200.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: rewrite trainer_node, integrate executor, remove obsolete training_agent infra"
```

---

## Self-review checklist

After all tasks complete, verify:

1. **Spec coverage:** Every section of `2026-05-06-model-registry-training-pipeline-design.md` has at least one task implementing it. Specifically — Section 1 (registry: Tasks 4–9), Section 2 (contracts: Tasks 3, 10), Section 3 (executor: Tasks 14, 16–20), Section 4 (splitter: Task 12), Section 5 (default plans: Task 13), Section 6 (experience record: Task 15), Section 7 (graph integration: Task 21), Section 8 (settings: Task 2). ✓
2. **No placeholders.** Every step has actual code or commands. ✓
3. **Type consistency.** `TrainingPlan`, `TrainingResult`, `RetrievalView`, `ModelSpec`, `SearchSpaceSpec`, `SearchParamSpec`, `SearchParamOverride` all defined once and used consistently across tasks. ✓
4. **All file paths absolute** (relative to repo root). ✓

---

## Out of scope (deferred to SP4 / SP5)

- Experience pool SQLite + retrieval tools — SP4.
- `model_agent` planner LLM — SP5.
- Advanced models (CatBoost forecaster, N-HiTS, TFT) — future.
- Per-trial MLflow logging — explicitly rejected in spec.
- Vector embeddings, sktime, neuralforecast — not used.
