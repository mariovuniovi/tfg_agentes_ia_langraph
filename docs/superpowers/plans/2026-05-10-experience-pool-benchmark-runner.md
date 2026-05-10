# Experience Pool + Benchmark Runner + Retrieval Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the long-term memory infrastructure for the MLCopilot-style model_agent: SQLite experience pool, static ML knowledge YAML, weighted-overlap retrieval tools, and a benchmark runner that seeds the pool with 12+ records across all three problem types.

**Architecture:** SP3's `run_training_plan` already writes JSON experience records. SP4 adds: (1) a SQLite-backed `ExperiencePool` that indexes those records for fast retrieval, (2) a handcrafted `ml_rules.yaml` with conditional model-selection rules, (3) two LangChain `@tool` functions that query the pool and rules, and (4) an offline benchmark runner script that seeds the pool from public datasets using the SP3 executor directly (no LLM, no graph).

**Tech Stack:** Python 3.12, Pydantic v2, sqlite3 (stdlib), pyyaml (already installed), LangChain `@tool`, pytest. No new dependencies.

---

## Spec reference

`docs/superpowers/specs/2026-05-06-experience-pool-benchmark-runner-design.md`

---

## File map

### Created
```
src/mlops_agents/contracts/profile.py                # DatasetProfile Pydantic class
src/mlops_agents/experience/__init__.py
src/mlops_agents/experience/pool.py                  # ExperiencePool (sqlite3 wrapper)
src/mlops_agents/experience/schema.py                # ExperienceRecord, RetrievalView Pydantic
src/mlops_agents/experience/retrieval.py             # RETRIEVAL_WEIGHTS, build_retrieval_view
src/mlops_agents/experience/migrations/__init__.py
src/mlops_agents/experience/migrations/001_init.sql  # CREATE TABLE statements
src/mlops_agents/experience/migrations/_runner.py    # apply_pending_migrations()
src/mlops_agents/knowledge/__init__.py
src/mlops_agents/knowledge/ml_rules.yaml             # 13 curated rules
src/mlops_agents/knowledge/reader.py                 # MLRule, load_rules(), match_rules()
src/mlops_agents/tools/memory_tools.py               # @tool retrieve_similar_experiences, retrieve_ml_knowledge
data/benchmarks/air_passengers.csv
data/benchmarks/m4_monthly_sample.csv
data/benchmarks/electricity_demand_sample.csv
data/benchmarks/sales_sample.csv
data/benchmarks/weather_sample.csv
data/benchmarks/stock_sample.csv
scripts/_dataset_sources.py                          # fetch_dataset adapters (sklearn/openml/local)
scripts/benchmark_manifest.yaml                      # 18 dataset entries
scripts/run_benchmark.py                             # offline seeder
tests/test_contracts/__init__.py (exists)
tests/test_contracts/test_profile.py
tests/test_experience/__init__.py
tests/test_experience/test_migrations.py
tests/test_experience/test_pool.py
tests/test_experience/test_retrieval.py
tests/test_knowledge/__init__.py
tests/test_knowledge/test_reader.py
tests/test_knowledge/test_starter_rules.py
tests/test_tools/test_memory_tools.py
tests/test_scripts/__init__.py
tests/test_scripts/test_benchmark_runner.py
```

### Modified
- `src/mlops_agents/config/settings.py` — add 5 new fields
- `src/mlops_agents/tools/__init__.py` — export memory tools

---

## Task 1: Settings + DatasetProfile Pydantic class

**Files:**
- Modify: `src/mlops_agents/config/settings.py`
- Create: `src/mlops_agents/contracts/profile.py`
- Create: `tests/test_contracts/test_profile.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_contracts/test_profile.py`:

```python
"""Tests for DatasetProfile Pydantic schema."""
import pytest
from pydantic import ValidationError
from mlops_agents.contracts.profile import DatasetProfile


def test_classification_profile_valid():
    p = DatasetProfile(
        problem_type="classification",
        n_rows="small", n_features="small", missing_rate="none",
        n_categorical_features="none", n_numerical_features="few",
        n_classes="binary", class_balance="balanced",
    )
    assert p.problem_type == "classification"
    assert p.n_classes == "binary"


def test_regression_profile_valid():
    p = DatasetProfile(
        problem_type="regression",
        n_rows="medium", n_features="small", missing_rate="low",
        n_categorical_features="none", n_numerical_features="few",
        target_distribution="skewed",
    )
    assert p.target_distribution == "skewed"


def test_forecasting_profile_valid():
    p = DatasetProfile(
        problem_type="forecasting",
        n_rows="medium", n_features="small", missing_rate="none",
        n_categorical_features="none", n_numerical_features="few",
        n_series="single", history_length="medium", frequency="MS",
        horizon_difficulty="short", forecast_horizon_raw=12,
        exogenous_features_available=False,
        seasonality_detected=True, trend_detected=False, stationarity=False,
    )
    assert p.n_series == "single"
    assert p.seasonality_detected is True


def test_invalid_n_rows_bucket_rejected():
    with pytest.raises(ValidationError):
        DatasetProfile(
            problem_type="classification",
            n_rows="gigantic",   # invalid bucket
            n_features="small", missing_rate="none",
            n_categorical_features="none", n_numerical_features="few",
        )


def test_schema_version_default():
    p = DatasetProfile(
        problem_type="regression",
        n_rows="small", n_features="small", missing_rate="none",
        n_categorical_features="none", n_numerical_features="few",
    )
    assert p.schema_version == 1
```

- [ ] **Step 2: Run tests to verify failure**

```
uv run pytest tests/test_contracts/test_profile.py -v
```
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Create `src/mlops_agents/contracts/profile.py`**

```python
"""Pydantic schema for the dataset profile — the retrieval join key for SP4/SP5."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class DatasetProfile(BaseModel):
    schema_version: int = 1
    problem_type: Literal["classification", "regression", "forecasting"]
    # Universal
    n_rows: Literal["very_small", "small", "medium", "large"]
    n_features: Literal["small", "medium", "large"]
    missing_rate: Literal["none", "low", "medium", "high"]
    n_categorical_features: Literal["none", "few", "some", "many"]
    n_numerical_features: Literal["none", "few", "some", "many"]
    # Classification-only
    n_classes: Literal["binary", "small_multiclass", "many_classes"] | None = None
    class_balance: Literal["balanced", "moderately_imbalanced", "severely_imbalanced"] | None = None
    # Regression-only
    target_distribution: Literal["near_normal", "skewed", "heavy_tailed", "discrete_like"] | None = None
    # Forecasting-only
    n_series: Literal["single", "few", "moderate", "many"] | None = None
    history_length: Literal["very_short", "short", "medium", "long"] | None = None
    frequency: str | None = None
    horizon_difficulty: Literal["very_short", "short", "medium", "long"] | None = None
    forecast_horizon_raw: int | None = None
    exogenous_features_available: bool | None = None
    seasonality_detected: bool | None = None
    trend_detected: bool | None = None
    stationarity: bool | None = None
```

- [ ] **Step 4: Add SP4 settings to `src/mlops_agents/config/settings.py`**

Read the current file first, then append inside the `Settings` class after the existing SP3 fields:

```python
    # Experience pool (SP4)
    experience_db_path: Path = Path("storage/mlops_metadata.db")
    experience_audit_dir: Path = Path("experience_pool")
    data_benchmarks_dir: Path = Path("data/benchmarks")
    ml_rules_path: Path = Path("src/mlops_agents/knowledge/ml_rules.yaml")
    retrieval_default_k: int = 5
    retrieval_weights_override: dict = Field(default_factory=dict)
```

- [ ] **Step 5: Run tests**

```
uv run pytest tests/test_contracts/test_profile.py -v
```
Expected: 5 PASS.

- [ ] **Step 6: Verify settings load**

```
uv run python -c "from mlops_agents.config.settings import settings; print(settings.experience_db_path)"
```
Expected: prints `storage/mlops_metadata.db`.

- [ ] **Step 7: Commit**

```bash
git add src/mlops_agents/contracts/profile.py src/mlops_agents/config/settings.py tests/test_contracts/test_profile.py
git commit -m "feat: add DatasetProfile Pydantic class and SP4 settings"
```

---

## Task 2: SQLite migrations + pool infrastructure

**Files:**
- Create: `src/mlops_agents/experience/__init__.py`
- Create: `src/mlops_agents/experience/migrations/__init__.py`
- Create: `src/mlops_agents/experience/migrations/001_init.sql`
- Create: `src/mlops_agents/experience/migrations/_runner.py`
- Create: `tests/test_experience/__init__.py`
- Create: `tests/test_experience/test_migrations.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_experience/__init__.py` (empty).

Create `tests/test_experience/test_migrations.py`:

```python
"""Tests for SQLite migration runner."""
import sqlite3
from pathlib import Path
from mlops_agents.experience.migrations._runner import apply_pending_migrations


def test_migrations_create_three_tables(tmp_path):
    db = tmp_path / "test.db"
    apply_pending_migrations(db)
    conn = sqlite3.connect(db)
    tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert {"experiences", "candidate_results", "model_artifacts", "_schema_version"}.issubset(tables)
    conn.close()


def test_migrations_are_idempotent(tmp_path):
    db = tmp_path / "test.db"
    apply_pending_migrations(db)
    apply_pending_migrations(db)  # second call must not raise
    conn = sqlite3.connect(db)
    version = conn.execute("SELECT MAX(version) FROM _schema_version").fetchone()[0]
    assert version == 1
    conn.close()


def test_migration_sets_schema_version(tmp_path):
    db = tmp_path / "test.db"
    apply_pending_migrations(db)
    conn = sqlite3.connect(db)
    row = conn.execute("SELECT version FROM _schema_version").fetchone()
    assert row[0] == 1
    conn.close()
```

- [ ] **Step 2: Verify failure**

```
uv run pytest tests/test_experience/test_migrations.py -v
```
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Create `src/mlops_agents/experience/__init__.py`**

```python
"""Experience pool: SQLite storage + retrieval for ML experiment records."""
```

- [ ] **Step 4: Create `src/mlops_agents/experience/migrations/__init__.py`**

Empty file.

- [ ] **Step 5: Create `src/mlops_agents/experience/migrations/001_init.sql`**

```sql
CREATE TABLE IF NOT EXISTS _schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiences (
    task_id TEXT PRIMARY KEY,
    problem_type TEXT NOT NULL,
    dataset_name TEXT,
    dataset_profile_json TEXT NOT NULL,
    training_plan_json TEXT NOT NULL,
    selected_model_key TEXT,
    metric_to_optimize TEXT,
    metric_direction TEXT,
    validation_score REAL,
    validation_std REAL,
    experience_summary TEXT,
    experience_json_path TEXT,
    mlflow_parent_run_id TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_experiences_problem_type ON experiences(problem_type);
CREATE INDEX IF NOT EXISTS idx_experiences_created_at ON experiences(created_at);

CREATE TABLE IF NOT EXISTS candidate_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    model_key TEXT NOT NULL,
    status TEXT NOT NULL,
    best_params_json TEXT,
    best_score REAL,
    best_score_std REAL,
    n_trials_used INTEGER,
    duration_s REAL,
    complexity_rank INTEGER,
    mlflow_run_id TEXT,
    error_type TEXT,
    error_message TEXT,
    FOREIGN KEY (task_id) REFERENCES experiences(task_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_candidate_results_task_id ON candidate_results(task_id);

CREATE TABLE IF NOT EXISTS model_artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    model_key TEXT NOT NULL,
    mlflow_run_id TEXT,
    artifact_path TEXT,
    artifact_uri TEXT,
    model_uri TEXT,
    is_champion INTEGER NOT NULL,
    metric_name TEXT,
    metric_value REAL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (task_id) REFERENCES experiences(task_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_model_artifacts_task_id ON model_artifacts(task_id);

INSERT OR IGNORE INTO _schema_version (version, applied_at)
VALUES (1, datetime('now'));
```

- [ ] **Step 6: Create `src/mlops_agents/experience/migrations/_runner.py`**

```python
"""Apply pending SQLite migrations in version order."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from mlops_agents.utils.logging import get_logger

logger = get_logger(__name__)

_MIGRATIONS_DIR = Path(__file__).parent


def _read_current_version(conn: sqlite3.Connection) -> int:
    try:
        row = conn.execute("SELECT MAX(version) FROM _schema_version").fetchone()
        return row[0] or 0
    except sqlite3.OperationalError:
        return 0


def apply_pending_migrations(db_path: Path) -> None:
    """Apply numbered migrations <NNN>_*.sql in order if their version > current. Idempotent."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        current = _read_current_version(conn)
        sql_files = sorted(
            f for f in _MIGRATIONS_DIR.glob("*.sql")
            if f.stem[0].isdigit()
        )
        for mig in sql_files:
            version = int(mig.stem.split("_")[0])
            if version > current:
                logger.info(f"Applying migration {mig.name}")
                with conn:
                    conn.executescript(mig.read_text())
    finally:
        conn.close()
```

- [ ] **Step 7: Run tests**

```
uv run pytest tests/test_experience/test_migrations.py -v
```
Expected: 3 PASS.

- [ ] **Step 8: Commit**

```bash
git add src/mlops_agents/experience/ tests/test_experience/
git commit -m "feat: add SQLite migration runner for experience pool (3 tables + schema_version)"
```

---

## Task 3: ExperienceRecord schema + ExperiencePool (insert + get + count)

**Files:**
- Create: `src/mlops_agents/experience/schema.py`
- Create: `src/mlops_agents/experience/pool.py`
- Create: `tests/test_experience/test_pool.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_experience/test_pool.py`:

```python
"""Tests for ExperiencePool insert and query."""
import json
from pathlib import Path
import pytest
from mlops_agents.experience.pool import ExperiencePool
from mlops_agents.experience.schema import ExperienceRecord


def _sample_record(task_id: str = "iris_classification_2026-05-10_001") -> dict:
    return {
        "task_id": task_id,
        "problem_type": "classification",
        "dataset_name": "iris",
        "dataset_profile": {
            "schema_version": 1,
            "problem_type": "classification",
            "n_rows": "small",
            "n_features": "small",
            "missing_rate": "none",
            "n_categorical_features": "none",
            "n_numerical_features": "few",
            "n_classes": "small_multiclass",
            "class_balance": "balanced",
        },
        "training_plan_input": {"candidates": []},
        "split_artifacts": {
            "train_pool_path": "/tmp/iris_train_pool.csv",
            "test_path": "/tmp/iris_test.csv",
            "split_metadata_path": "/tmp/iris_split_metadata.json",
        },
        "mlflow": {"experiment_name": "test", "parent_run_id": "abc123"},
        "metric_to_optimize": "macro_f1",
        "metric_direction": "maximize",
        "candidate_selection_policy": {
            "primary": "best_validation_score",
            "tie_breaker": "complexity_rank",
            "tie_tolerance_relative": 0.01,
        },
        "models_tested": [
            {
                "model_key": "logistic_regression",
                "status": "successful",
                "best_params": {"C": 1.0},
                "best_score": 0.94,
                "best_score_std": 0.02,
                "n_trials_used": 10,
                "duration_s": 3.5,
                "complexity_rank": 1,
                "mlflow_run_id": "run_lr",
            },
            {
                "model_key": "random_forest_classifier",
                "status": "failed",
                "error_type": "ValueError",
                "error_message": "boom",
                "n_trials_used": 0,
                "duration_s": 0.1,
                "complexity_rank": 2,
                "mlflow_run_id": "run_rf",
            },
        ],
        "selected_solution": {
            "model_key": "logistic_regression",
            "hyperparameters": {"C": 1.0},
            "validation_strategy": "stratified_5_fold_cv",
            "main_metric": "macro_f1",
            "validation_score": 0.94,
            "validation_std": 0.02,
            "complexity_rank": 1,
        },
        "experience_summary": "",
    }


def test_insert_from_record_writes_experiences_row(tmp_path):
    pool = ExperiencePool(tmp_path / "test.db")
    record = ExperienceRecord.model_validate(_sample_record())
    pool.insert_from_record(record)
    assert pool.count() == 1


def test_insert_from_record_writes_candidate_rows(tmp_path):
    import sqlite3
    pool = ExperiencePool(tmp_path / "test.db")
    record = ExperienceRecord.model_validate(_sample_record())
    pool.insert_from_record(record)
    conn = sqlite3.connect(str(tmp_path / "test.db"))
    rows = conn.execute("SELECT * FROM candidate_results WHERE task_id = ?",
                        ("iris_classification_2026-05-10_001",)).fetchall()
    conn.close()
    assert len(rows) == 2


def test_insert_writes_audit_json(tmp_path):
    audit_dir = tmp_path / "pool"
    pool = ExperiencePool(tmp_path / "test.db", audit_dir=audit_dir)
    record = ExperienceRecord.model_validate(_sample_record())
    pool.insert_from_record(record)
    expected = audit_dir / "iris_classification_2026-05-10_001.json"
    assert expected.exists()
    data = json.loads(expected.read_text())
    assert data["task_id"] == "iris_classification_2026-05-10_001"


def test_count_by_problem_type(tmp_path):
    pool = ExperiencePool(tmp_path / "test.db")
    pool.insert_from_record(ExperienceRecord.model_validate(_sample_record("a_classification_x_001")))
    pool.insert_from_record(ExperienceRecord.model_validate(
        {**_sample_record("b_regression_x_001"), "problem_type": "regression"}
    ))
    assert pool.count("classification") == 1
    assert pool.count("regression") == 1
    assert pool.count() == 2


def test_get_retrieves_record(tmp_path):
    pool = ExperiencePool(tmp_path / "test.db")
    record = ExperienceRecord.model_validate(_sample_record())
    pool.insert_from_record(record)
    fetched = pool.get("iris_classification_2026-05-10_001")
    assert fetched.task_id == "iris_classification_2026-05-10_001"
    assert fetched.problem_type == "classification"
```

- [ ] **Step 2: Verify failure**

```
uv run pytest tests/test_experience/test_pool.py -v
```
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Create `src/mlops_agents/experience/schema.py`**

```python
"""Pydantic schemas for experience records and retrieval views."""
from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field


class CandidateResult(BaseModel):
    model_key: str
    status: Literal["successful", "failed"]
    best_params: dict[str, Any] | None = None
    best_score: float | None = None
    best_score_std: float | None = None
    n_trials_used: int | None = None
    duration_s: float | None = None
    complexity_rank: int | None = None
    mlflow_run_id: str | None = None
    error_type: str | None = None
    error_message: str | None = None


class SelectedSolution(BaseModel):
    model_key: str
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    validation_strategy: str | None = None
    main_metric: str | None = None
    validation_score: float | None = None
    validation_std: float | None = None
    complexity_rank: int | None = None


class ExperienceRecord(BaseModel):
    task_id: str
    problem_type: str
    dataset_name: str | None = None
    dataset_profile: dict[str, Any]
    training_plan_input: dict[str, Any] = Field(default_factory=dict)
    split_artifacts: dict[str, str] = Field(default_factory=dict)
    mlflow: dict[str, str] = Field(default_factory=dict)
    metric_to_optimize: str | None = None
    metric_direction: str | None = None
    candidate_selection_policy: dict[str, Any] = Field(default_factory=dict)
    models_tested: list[CandidateResult] = Field(default_factory=list)
    selected_solution: SelectedSolution | None = None
    experience_summary: str | None = None


class CandidateResultView(BaseModel):
    model_key: str
    status: Literal["successful", "failed"]
    best_score: float | None = None
    complexity_rank: int | None = None
    error_type: str | None = None


class SelectedSolutionView(BaseModel):
    model_key: str
    validation_score: float
    validation_std: float | None = None
    complexity_rank: int


class RetrievalView(BaseModel):
    task_id: str
    dataset_name: str | None
    dataset_profile: dict[str, Any]
    models_tested: list[CandidateResultView]
    selected_solution: SelectedSolutionView
    models_not_recommended: list[dict] = Field(default_factory=list)
    experience_summary: str | None
    similarity_score: int
    similarity_ratio: float
    matched_fields: list[str]
```

- [ ] **Step 4: Create `src/mlops_agents/experience/pool.py`**

```python
"""ExperiencePool — SQLite-backed read/write layer for experience records."""
from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mlops_agents.experience.migrations._runner import apply_pending_migrations
from mlops_agents.experience.schema import ExperienceRecord, RetrievalView
from mlops_agents.utils.logging import get_logger

logger = get_logger(__name__)


class ExperiencePool:
    def __init__(self, db_path: Path, audit_dir: Path | None = None):
        apply_pending_migrations(db_path)
        self._db_path = db_path
        self._audit_dir = audit_dir

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def insert_from_record(self, record: ExperienceRecord) -> None:
        """Insert into all 3 tables atomically, then write JSON audit copy."""
        created_at = datetime.now(UTC).isoformat()
        sol = record.selected_solution

        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO experiences
                (task_id, problem_type, dataset_name, dataset_profile_json,
                 training_plan_json, selected_model_key, metric_to_optimize,
                 metric_direction, validation_score, validation_std,
                 experience_summary, mlflow_parent_run_id, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    record.task_id,
                    record.problem_type,
                    record.dataset_name,
                    json.dumps(record.dataset_profile),
                    json.dumps(record.training_plan_input),
                    sol.model_key if sol else None,
                    record.metric_to_optimize,
                    record.metric_direction,
                    sol.validation_score if sol else None,
                    sol.validation_std if sol else None,
                    record.experience_summary,
                    record.mlflow.get("parent_run_id"),
                    created_at,
                ),
            )
            # Delete stale candidates before re-inserting (for OR REPLACE semantics)
            conn.execute("DELETE FROM candidate_results WHERE task_id = ?", (record.task_id,))
            for cand in record.models_tested:
                conn.execute(
                    """
                    INSERT INTO candidate_results
                    (task_id, model_key, status, best_params_json, best_score,
                     best_score_std, n_trials_used, duration_s, complexity_rank,
                     mlflow_run_id, error_type, error_message)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        record.task_id, cand.model_key, cand.status,
                        json.dumps(cand.best_params) if cand.best_params else None,
                        cand.best_score, cand.best_score_std,
                        cand.n_trials_used, cand.duration_s, cand.complexity_rank,
                        cand.mlflow_run_id, cand.error_type, cand.error_message,
                    ),
                )
            # model_artifacts — one row for the champion
            conn.execute("DELETE FROM model_artifacts WHERE task_id = ?", (record.task_id,))
            if sol:
                conn.execute(
                    """
                    INSERT INTO model_artifacts
                    (task_id, model_key, mlflow_run_id, is_champion, metric_name,
                     metric_value, created_at)
                    VALUES (?,?,?,1,?,?,?)
                    """,
                    (
                        record.task_id, sol.model_key,
                        record.mlflow.get("parent_run_id"),
                        record.metric_to_optimize, sol.validation_score, created_at,
                    ),
                )

        # JSON audit copy (best-effort — failure here does not roll back SQL)
        if self._audit_dir is not None:
            try:
                self._audit_dir.mkdir(parents=True, exist_ok=True)
                out = self._audit_dir / f"{record.task_id}.json"
                out.write_text(json.dumps(record.model_dump(), default=str, indent=2))
            except Exception as e:
                logger.warning(f"Failed to write audit JSON for {record.task_id}: {e}")

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, task_id: str) -> ExperienceRecord:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM experiences WHERE task_id = ?", (task_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"No experience found for task_id={task_id!r}")
            cand_rows = conn.execute(
                "SELECT * FROM candidate_results WHERE task_id = ?", (task_id,)
            ).fetchall()

        profile = json.loads(row["dataset_profile_json"])
        plan = json.loads(row["training_plan_json"])
        candidates = [
            {
                "model_key": r["model_key"], "status": r["status"],
                "best_params": json.loads(r["best_params_json"]) if r["best_params_json"] else None,
                "best_score": r["best_score"], "best_score_std": r["best_score_std"],
                "n_trials_used": r["n_trials_used"], "duration_s": r["duration_s"],
                "complexity_rank": r["complexity_rank"], "mlflow_run_id": r["mlflow_run_id"],
                "error_type": r["error_type"], "error_message": r["error_message"],
            }
            for r in cand_rows
        ]
        sol = None
        if row["selected_model_key"]:
            sol = {
                "model_key": row["selected_model_key"],
                "validation_score": row["validation_score"],
                "validation_std": row["validation_std"],
            }
        return ExperienceRecord(
            task_id=row["task_id"],
            problem_type=row["problem_type"],
            dataset_name=row["dataset_name"],
            dataset_profile=profile,
            training_plan_input=plan,
            mlflow={"parent_run_id": row["mlflow_parent_run_id"] or ""},
            metric_to_optimize=row["metric_to_optimize"],
            metric_direction=row["metric_direction"],
            models_tested=candidates,
            selected_solution=sol,
            experience_summary=row["experience_summary"],
        )

    def count(self, problem_type: str | None = None) -> int:
        with self._conn() as conn:
            if problem_type:
                return conn.execute(
                    "SELECT COUNT(*) FROM experiences WHERE problem_type = ?", (problem_type,)
                ).fetchone()[0]
            return conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]

    def find_similar(
        self,
        profile: dict[str, Any],
        problem_type: str,
        k: int = 5,
    ) -> list[RetrievalView]:
        """Weighted-overlap retrieval — implemented in Task 4."""
        from mlops_agents.experience.retrieval import find_similar_impl
        return find_similar_impl(self, profile, problem_type, k)
```

- [ ] **Step 5: Run tests**

```
uv run pytest tests/test_experience/test_pool.py -v
```
Expected: 5 PASS.

- [ ] **Step 6: Commit**

```bash
git add src/mlops_agents/experience/schema.py src/mlops_agents/experience/pool.py tests/test_experience/test_pool.py
git commit -m "feat: add ExperienceRecord schema + ExperiencePool (insert, get, count)"
```

---

## Task 4: Retrieval algorithm (find_similar + RetrievalView)

**Files:**
- Create: `src/mlops_agents/experience/retrieval.py`
- Create: `tests/test_experience/test_retrieval.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_experience/test_retrieval.py`:

```python
"""Tests for weighted-overlap experience retrieval."""
from mlops_agents.experience.pool import ExperiencePool
from mlops_agents.experience.schema import ExperienceRecord


def _make_cls_record(task_id: str, n_rows: str, score: float) -> dict:
    return {
        "task_id": task_id,
        "problem_type": "classification",
        "dataset_name": task_id,
        "dataset_profile": {
            "schema_version": 1, "problem_type": "classification",
            "n_rows": n_rows, "n_features": "small",
            "missing_rate": "none", "n_categorical_features": "none",
            "n_numerical_features": "few",
            "n_classes": "binary", "class_balance": "balanced",
        },
        "training_plan_input": {}, "split_artifacts": {},
        "mlflow": {}, "metric_to_optimize": "macro_f1",
        "metric_direction": "maximize",
        "candidate_selection_policy": {},
        "models_tested": [{
            "model_key": "logistic_regression", "status": "successful",
            "best_score": score, "complexity_rank": 1, "n_trials_used": 5,
            "duration_s": 1.0,
        }],
        "selected_solution": {
            "model_key": "logistic_regression", "validation_score": score,
            "complexity_rank": 1,
        },
    }


def test_find_similar_returns_closest_profile(tmp_path):
    pool = ExperiencePool(tmp_path / "db.db")
    pool.insert_from_record(ExperienceRecord.model_validate(_make_cls_record("a", "small", 0.90)))
    pool.insert_from_record(ExperienceRecord.model_validate(_make_cls_record("b", "medium", 0.85)))

    # Query with n_rows=small → "a" is more similar
    views = pool.find_similar(
        {"problem_type": "classification", "n_rows": "small", "n_features": "small",
         "missing_rate": "none", "n_categorical_features": "none",
         "n_numerical_features": "few", "n_classes": "binary", "class_balance": "balanced"},
        problem_type="classification", k=5,
    )
    assert views[0].task_id == "a"


def test_find_similar_hard_filters_by_problem_type(tmp_path):
    pool = ExperiencePool(tmp_path / "db.db")
    pool.insert_from_record(ExperienceRecord.model_validate(_make_cls_record("cls1", "small", 0.9)))
    pool.insert_from_record(ExperienceRecord.model_validate({
        **_make_cls_record("reg1", "small", 0.9),
        "problem_type": "regression",
    }))
    views = pool.find_similar(
        {"n_rows": "small"}, problem_type="classification", k=5
    )
    task_ids = {v.task_id for v in views}
    assert "cls1" in task_ids
    assert "reg1" not in task_ids


def test_find_similar_empty_pool_returns_empty(tmp_path):
    pool = ExperiencePool(tmp_path / "db.db")
    views = pool.find_similar({"n_rows": "small"}, "classification", k=5)
    assert views == []


def test_similarity_ratio_is_normalized(tmp_path):
    pool = ExperiencePool(tmp_path / "db.db")
    pool.insert_from_record(ExperienceRecord.model_validate(_make_cls_record("x", "small", 0.9)))
    views = pool.find_similar(
        {"problem_type": "classification", "n_rows": "small", "n_features": "small",
         "missing_rate": "none", "n_categorical_features": "none",
         "n_numerical_features": "few", "n_classes": "binary", "class_balance": "balanced"},
        "classification", k=1,
    )
    assert 0.0 <= views[0].similarity_ratio <= 1.0


def test_find_similar_returns_at_most_k(tmp_path):
    pool = ExperiencePool(tmp_path / "db.db")
    for i in range(5):
        pool.insert_from_record(ExperienceRecord.model_validate(_make_cls_record(f"r{i}", "small", 0.9)))
    views = pool.find_similar({"n_rows": "small"}, "classification", k=3)
    assert len(views) <= 3
```

- [ ] **Step 2: Verify failure**

```
uv run pytest tests/test_experience/test_retrieval.py -v
```
Expected: FAIL with `ModuleNotFoundError` for `retrieval`.

- [ ] **Step 3: Create `src/mlops_agents/experience/retrieval.py`**

```python
"""Weighted-overlap retrieval for experience records."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from mlops_agents.experience.schema import (
    CandidateResultView,
    RetrievalView,
    SelectedSolutionView,
)

if TYPE_CHECKING:
    from mlops_agents.experience.pool import ExperiencePool

RETRIEVAL_WEIGHTS: dict[str, int] = {
    # Structural (3) — decides which model family applies
    "n_rows": 3,
    "n_series": 3,
    "history_length": 3,
    "horizon_difficulty": 3,
    "seasonality_detected": 3,
    # Strong influence (2)
    "class_balance": 2,
    "n_classes": 2,
    "target_distribution": 2,
    "exogenous_features_available": 2,
    "frequency": 2,
    "trend_detected": 2,
    "stationarity": 2,
    # Weak / preprocessing-concern (1)
    "n_features": 1,
    "missing_rate": 1,
    "n_categorical_features": 1,
    "n_numerical_features": 1,
}

MAX_SCORE_BY_PROBLEM_TYPE: dict[str, int] = {
    "classification": 13,
    "regression": 11,
    "forecasting": 29,
}


def _parse_ts(iso: str | None) -> float:
    if not iso:
        return 0.0
    try:
        return datetime.fromisoformat(iso).timestamp()
    except ValueError:
        return 0.0


def _build_retrieval_view(
    row: Any,
    cand_rows: list[Any],
    similarity_score: int,
    similarity_ratio: float,
    matched_fields: list[str],
) -> RetrievalView:
    profile = json.loads(row["dataset_profile_json"])
    candidates = [
        CandidateResultView(
            model_key=r["model_key"],
            status=r["status"],
            best_score=r["best_score"],
            complexity_rank=r["complexity_rank"],
            error_type=r["error_type"],
        )
        for r in cand_rows
    ]
    sol = None
    if row["selected_model_key"] and row["validation_score"] is not None:
        sol = SelectedSolutionView(
            model_key=row["selected_model_key"],
            validation_score=row["validation_score"],
            validation_std=row["validation_std"],
            complexity_rank=next(
                (c.complexity_rank for c in candidates if c.model_key == row["selected_model_key"]),
                0,
            ) or 0,
        )
    if sol is None:
        return None  # skip records without a selected solution
    return RetrievalView(
        task_id=row["task_id"],
        dataset_name=row["dataset_name"],
        dataset_profile=profile,
        models_tested=candidates,
        selected_solution=sol,
        experience_summary=row["experience_summary"],
        similarity_score=similarity_score,
        similarity_ratio=similarity_ratio,
        matched_fields=matched_fields,
    )


def find_similar_impl(
    pool: "ExperiencePool",
    profile: dict[str, Any],
    problem_type: str,
    k: int,
) -> list[RetrievalView]:
    max_score = MAX_SCORE_BY_PROBLEM_TYPE.get(problem_type, 10)
    with pool._conn() as conn:
        rows = conn.execute(
            "SELECT * FROM experiences WHERE problem_type = ? ORDER BY created_at DESC",
            (problem_type,),
        ).fetchall()

    scored: list[tuple[int, float, Any]] = []
    for row in rows:
        candidate_profile = json.loads(row["dataset_profile_json"])
        score = 0
        matched = ["problem_type"]
        for field, weight in RETRIEVAL_WEIGHTS.items():
            pv = profile.get(field)
            cv = candidate_profile.get(field)
            if pv is not None and cv is not None and pv == cv:
                score += weight
                matched.append(field)
        ratio = round(score / max_score, 3)
        ts = _parse_ts(row["created_at"])
        scored.append((score, ts, ratio, matched, row))

    # Sort: score desc, then recency desc (ts desc)
    scored.sort(key=lambda x: (-x[0], -x[1]))

    views: list[RetrievalView] = []
    for score, ts, ratio, matched, row in scored[:k]:
        with pool._conn() as conn:
            cand_rows = conn.execute(
                "SELECT * FROM candidate_results WHERE task_id = ?", (row["task_id"],)
            ).fetchall()
        view = _build_retrieval_view(row, cand_rows, score, ratio, matched)
        if view is not None:
            views.append(view)
    return views
```

- [ ] **Step 4: Run tests**

```
uv run pytest tests/test_experience/test_retrieval.py -v
```
Expected: 5 PASS.

- [ ] **Step 5: Run full suite to check for regressions**

```
uv run pytest -m "not integration" -q
```
Expected: 253+ PASS (241 + 3 migrations + 5 pool + 5 retrieval + 5 profile = 259).

- [ ] **Step 6: Commit**

```bash
git add src/mlops_agents/experience/retrieval.py tests/test_experience/test_retrieval.py
git commit -m "feat: add weighted-overlap retrieval algorithm (find_similar, similarity_ratio)"
```

---

## Task 5: ML knowledge base — reader + 13 curated rules

**Files:**
- Create: `src/mlops_agents/knowledge/__init__.py`
- Create: `src/mlops_agents/knowledge/reader.py`
- Create: `src/mlops_agents/knowledge/ml_rules.yaml`
- Create: `tests/test_knowledge/__init__.py`
- Create: `tests/test_knowledge/test_reader.py`
- Create: `tests/test_knowledge/test_starter_rules.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_knowledge/__init__.py` (empty).

Create `tests/test_knowledge/test_reader.py`:

```python
"""Tests for ML rules reader."""
import pytest
from pydantic import ValidationError
from mlops_agents.knowledge.reader import MLRule, match_rules


def _cls_profile(n_rows: str = "medium", class_balance: str = "balanced") -> dict:
    return {
        "problem_type": "classification",
        "n_rows": n_rows, "n_features": "small",
        "missing_rate": "none", "n_categorical_features": "none",
        "n_numerical_features": "few",
        "n_classes": "binary", "class_balance": class_balance,
    }


def _fc_profile(history_length: str = "short", seasonality_detected: bool = False) -> dict:
    return {
        "problem_type": "forecasting",
        "n_rows": "medium", "n_features": "small",
        "missing_rate": "none", "n_categorical_features": "none",
        "n_numerical_features": "few",
        "n_series": "single", "history_length": history_length,
        "frequency": "MS", "horizon_difficulty": "short",
        "exogenous_features_available": False,
        "seasonality_detected": seasonality_detected,
        "trend_detected": False, "stationarity": True,
    }


def test_match_rules_returns_matching_rules():
    profile = _cls_profile(n_rows="very_small")
    rules = match_rules(profile)
    rule_ids = {r.rule_id for r in rules}
    assert "classification_very_small_prefers_simple_models" in rule_ids


def test_match_rules_does_not_return_non_matching():
    # medium dataset — very_small rule should NOT match
    profile = _cls_profile(n_rows="medium")
    rules = match_rules(profile)
    rule_ids = {r.rule_id for r in rules}
    assert "classification_very_small_prefers_simple_models" not in rule_ids


def test_match_rules_forecasting_short_history():
    profile = _fc_profile(history_length="short")
    rules = match_rules(profile)
    rule_ids = {r.rule_id for r in rules}
    assert "forecasting_short_history_prefers_statistical" in rule_ids


def test_match_rules_list_applies_when():
    """A rule with applies_when value as a list matches if profile value is in the list."""
    profile = _fc_profile(history_length="very_short")
    rules = match_rules(profile)
    rule_ids = {r.rule_id for r in rules}
    # very_short is in [very_short, short] → should match forecasting_short_history rule
    assert "forecasting_short_history_prefers_statistical" in rule_ids


def test_mlrule_rejects_unknown_profile_field():
    with pytest.raises(ValidationError, match="unknown profile field"):
        MLRule(
            rule_id="bad_rule",
            applies_when={"history_lenght": "short"},  # typo
            prefer=["naive"],
            reason="test",
        )


def test_mlrule_rejects_wrong_problem_type_model():
    """A forecasting rule cannot recommend logistic_regression (classification model)."""
    with pytest.raises(ValidationError, match="does not match"):
        MLRule(
            rule_id="bad_mix",
            applies_when={"problem_type": "forecasting"},
            prefer=["logistic_regression"],  # classification model!
            reason="test",
        )
```

Create `tests/test_knowledge/test_starter_rules.py`:

```python
"""Smoke test: the shipped ml_rules.yaml loads and validates cleanly."""
from mlops_agents.knowledge.reader import load_rules


def test_starter_rules_load_without_error():
    rules = load_rules()
    assert len(rules) >= 10  # at least 10 curated rules


def test_starter_rules_all_have_reason():
    for rule in load_rules():
        assert rule.reason.strip(), f"Rule {rule.rule_id} has empty reason"


def test_starter_rules_all_have_prefer_or_avoid():
    for rule in load_rules():
        assert rule.prefer or rule.avoid_or_deprioritize, \
            f"Rule {rule.rule_id} has neither prefer nor avoid_or_deprioritize"
```

- [ ] **Step 2: Verify failure**

```
uv run pytest tests/test_knowledge/ -v
```
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Create `src/mlops_agents/knowledge/__init__.py`**

```python
"""Static ML knowledge base: curated model-selection rules."""
```

- [ ] **Step 4: Create `src/mlops_agents/knowledge/reader.py`**

```python
"""MLRule loader and match_rules() for the static knowledge base."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

from mlops_agents.config.settings import settings


class MLRule(BaseModel):
    rule_id: str
    applies_when: dict[str, Any]
    prefer: list[str] = Field(default_factory=list)
    avoid_or_deprioritize: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    reason: str
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_applies_when_fields(self) -> "MLRule":
        from mlops_agents.contracts.profile import DatasetProfile
        valid = set(DatasetProfile.model_fields.keys())
        for field in self.applies_when:
            if field not in valid:
                raise ValueError(
                    f"Rule {self.rule_id}: applies_when references unknown profile field "
                    f"'{field}'. Valid fields: {sorted(valid)}"
                )
        return self

    @model_validator(mode="after")
    def validate_model_keys(self) -> "MLRule":
        from mlops_agents.models.loader import load_registry
        registry = load_registry()
        rule_pt = self.applies_when.get("problem_type")
        if isinstance(rule_pt, str):
            allowed = {rule_pt}
        elif isinstance(rule_pt, list):
            allowed = set(rule_pt)
        else:
            allowed = None
        for k in self.prefer + self.avoid_or_deprioritize:
            if k not in registry:
                raise ValueError(f"Rule {self.rule_id}: unknown model_key '{k}'")
            if allowed is not None and registry[k].problem_type not in allowed:
                raise ValueError(
                    f"Rule {self.rule_id}: model_key '{k}' "
                    f"(problem_type={registry[k].problem_type!r}) does not match "
                    f"applies_when.problem_type {sorted(allowed)}"
                )
        return self


@lru_cache(maxsize=1)
def load_rules(path: Path | None = None) -> list[MLRule]:
    """Load and validate ml_rules.yaml; cached after first call."""
    p = path or settings.ml_rules_path
    raw = yaml.safe_load(Path(p).read_text()) or []
    return [MLRule(**entry) for entry in raw]


def match_rules(profile: dict[str, Any]) -> list[MLRule]:
    """All rules whose applies_when conditions are fully satisfied by the profile."""
    matched = []
    for rule in load_rules():
        ok = True
        for field, expected in rule.applies_when.items():
            actual = profile.get(field)
            if actual is None:
                ok = False
                break
            if isinstance(expected, list):
                if actual not in expected:
                    ok = False
                    break
            else:
                if actual != expected:
                    ok = False
                    break
        if ok:
            matched.append(rule)
    return matched
```

- [ ] **Step 5: Create `src/mlops_agents/knowledge/ml_rules.yaml`**

```yaml
# Curated ML model-selection rules.
# Each rule applies when ALL conditions in applies_when are satisfied.
# prefer/avoid_or_deprioritize model_keys must exist in models/registry.yaml
# and must match the rule's problem_type.

- rule_id: classification_very_small_prefers_simple_models
  applies_when:
    problem_type: classification
    n_rows: very_small
  prefer: [logistic_regression, random_forest_classifier]
  avoid_or_deprioritize: [lightgbm_classifier, xgboost_classifier, catboost_classifier]
  reason: |
    With <500 rows, complex boosting models overfit easily. Regularized linear
    models and simple tree ensembles are safer baselines.
  tags: [classification, sample_size]

- rule_id: regression_very_small_prefers_simple_models
  applies_when:
    problem_type: regression
    n_rows: very_small
  prefer: [ridge, random_forest_regressor]
  avoid_or_deprioritize: [lightgbm_regressor, xgboost_regressor, catboost_regressor]
  reason: |
    With <500 rows, regularized linear models and simple tree baselines
    generalize better than high-capacity boosting models.
  tags: [regression, sample_size]

- rule_id: forecasting_short_history_prefers_statistical
  applies_when:
    problem_type: forecasting
    history_length: [very_short, short]
  prefer: [seasonal_naive, ets, auto_arima, naive]
  avoid_or_deprioritize: [svr_forecaster, lightgbm_forecaster, xgboost_forecaster,
                          random_forest_forecaster, extra_trees_forecaster, gbm_forecaster]
  reason: |
    With <200 obs/series, statistical models with strong structural priors
    outperform feature-heavy supervised forecasters.
  tags: [forecasting, sample_size]

- rule_id: forecasting_long_history_with_exogenous_prefers_supervised
  applies_when:
    problem_type: forecasting
    history_length: [medium, long]
    exogenous_features_available: true
  prefer: [lightgbm_forecaster, xgboost_forecaster, gbm_forecaster,
          random_forest_forecaster, extra_trees_forecaster]
  reason: |
    Supervised lag-based models exploit nonlinear lagged effects and external
    regressors when history is sufficient and exogenous variables are available.
  tags: [forecasting, exogenous]

- rule_id: forecasting_strong_seasonality_prefers_seasonal_models
  applies_when:
    problem_type: forecasting
    seasonality_detected: true
  prefer: [seasonal_naive, ets, auto_arima]
  reason: |
    Detected seasonality strongly favors models with explicit seasonal
    decomposition (ETS, seasonal ARIMA) over non-seasonal alternatives.
  tags: [forecasting, seasonality]

- rule_id: classification_severe_imbalance_prefers_tree_ensembles
  applies_when:
    problem_type: classification
    class_balance: severely_imbalanced
  prefer: [lightgbm_classifier, xgboost_classifier, catboost_classifier,
          random_forest_classifier]
  avoid_or_deprioritize: [logistic_regression]
  reason: |
    Severely imbalanced classes (>5x ratio) benefit from tree ensembles that
    handle class weights flexibly. Logistic regression often collapses to the
    majority class.
  tags: [classification, imbalance]

- rule_id: regression_skewed_target_prefers_tree_ensembles
  applies_when:
    problem_type: regression
    target_distribution: skewed
  prefer: [lightgbm_regressor, xgboost_regressor, random_forest_regressor]
  avoid_or_deprioritize: [ridge]
  reason: |
    Skewed target distributions violate the normality assumption behind ridge
    regression. Tree ensembles are more robust to non-normal targets.
  tags: [regression, target_distribution]

- rule_id: forecasting_non_stationary_prefers_differencing_models
  applies_when:
    problem_type: forecasting
    stationarity: false
    trend_detected: true
  prefer: [auto_arima, ets]
  reason: |
    Non-stationary series with trend are best handled by models that explicitly
    model trend components (AutoARIMA with differencing, ETS with trend).
  tags: [forecasting, stationarity]

- rule_id: forecasting_many_series_prefers_global_ml
  applies_when:
    problem_type: forecasting
    n_series: [moderate, many]
    history_length: [medium, long]
  prefer: [lightgbm_forecaster, xgboost_forecaster, gbm_forecaster]
  reason: |
    With many series, global ML models share signal across series and scale
    better than fitting one statistical model per series.
  tags: [forecasting, multi_series]

- rule_id: classification_large_dataset_prefers_boosting
  applies_when:
    problem_type: classification
    n_rows: large
  prefer: [lightgbm_classifier, xgboost_classifier, catboost_classifier]
  avoid_or_deprioritize: [logistic_regression]
  reason: |
    On large datasets (>50k rows), gradient boosting typically outperforms
    linear models and is competitive with deep learning for tabular data.
  tags: [classification, sample_size]

- rule_id: regression_large_dataset_prefers_boosting
  applies_when:
    problem_type: regression
    n_rows: large
  prefer: [lightgbm_regressor, xgboost_regressor, catboost_regressor]
  avoid_or_deprioritize: [ridge]
  reason: |
    On large regression datasets, gradient boosting handles nonlinear
    interactions better than linear models.
  tags: [regression, sample_size]

- rule_id: forecasting_long_horizon_prefers_decomposition
  applies_when:
    problem_type: forecasting
    horizon_difficulty: long
  prefer: [ets, auto_arima]
  avoid_or_deprioritize: [naive, svr_forecaster]
  reason: |
    Long forecast horizons amplify errors from simple extrapolation. Models with
    structural decomposition (ETS, ARIMA) produce more reliable multi-step forecasts.
  tags: [forecasting, horizon]

- rule_id: forecasting_single_series_no_exogenous_prefers_statistical
  applies_when:
    problem_type: forecasting
    n_series: single
    exogenous_features_available: false
  prefer: [ets, auto_arima, seasonal_naive, naive]
  reason: |
    Single univariate series without external regressors is the classic use case
    for statistical time series models. Supervised lag-based models add complexity
    without additional signal.
  tags: [forecasting, univariate]
```

- [ ] **Step 6: Run all knowledge tests**

```
uv run pytest tests/test_knowledge/ -v
```
Expected: 9 PASS (6 reader + 3 starter rules).

- [ ] **Step 7: Commit**

```bash
git add src/mlops_agents/knowledge/ tests/test_knowledge/
git commit -m "feat: add ML knowledge base — 13 curated rules + reader with typo/problem-type validation"
```

---

## Task 6: Memory retrieval @tools

**Files:**
- Create: `src/mlops_agents/tools/memory_tools.py`
- Modify: `src/mlops_agents/tools/__init__.py`
- Create: `tests/test_tools/test_memory_tools.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_tools/test_memory_tools.py`:

```python
"""Tests for memory retrieval tools."""
import json
from mlops_agents.tools.memory_tools import retrieve_similar_experiences, retrieve_ml_knowledge


def _cls_profile_json(n_rows: str = "medium") -> str:
    return json.dumps({
        "schema_version": 1, "problem_type": "classification",
        "n_rows": n_rows, "n_features": "small",
        "missing_rate": "none", "n_categorical_features": "none",
        "n_numerical_features": "few",
        "n_classes": "binary", "class_balance": "balanced",
    })


def test_retrieve_similar_experiences_returns_json(tmp_path, monkeypatch):
    monkeypatch.setattr("mlops_agents.tools.memory_tools.settings.experience_db_path", tmp_path / "db.db")
    result = retrieve_similar_experiences.invoke({
        "dataset_profile_json": _cls_profile_json(),
        "problem_type": "classification",
        "k": 5,
    })
    data = json.loads(result)
    assert isinstance(data, list)  # empty pool → empty list


def test_retrieve_similar_experiences_round_trip_pydantic(tmp_path, monkeypatch):
    """If pool has records, returned views pass round-trip Pydantic validation."""
    from mlops_agents.experience.pool import ExperiencePool
    from mlops_agents.experience.schema import ExperienceRecord
    db = tmp_path / "db.db"
    monkeypatch.setattr("mlops_agents.tools.memory_tools.settings.experience_db_path", db)
    pool = ExperiencePool(db)
    pool.insert_from_record(ExperienceRecord.model_validate({
        "task_id": "test_cls_001",
        "problem_type": "classification",
        "dataset_name": "test",
        "dataset_profile": {
            "schema_version": 1, "problem_type": "classification",
            "n_rows": "medium", "n_features": "small",
            "missing_rate": "none", "n_categorical_features": "none",
            "n_numerical_features": "few",
            "n_classes": "binary", "class_balance": "balanced",
        },
        "training_plan_input": {}, "split_artifacts": {}, "mlflow": {},
        "metric_to_optimize": "macro_f1", "metric_direction": "maximize",
        "candidate_selection_policy": {},
        "models_tested": [{"model_key": "logistic_regression", "status": "successful",
                           "best_score": 0.93, "complexity_rank": 1, "n_trials_used": 5, "duration_s": 1.0}],
        "selected_solution": {"model_key": "logistic_regression",
                              "validation_score": 0.93, "complexity_rank": 1},
    }))
    result = retrieve_similar_experiences.invoke({
        "dataset_profile_json": _cls_profile_json("medium"),
        "problem_type": "classification",
        "k": 5,
    })
    data = json.loads(result)
    assert len(data) == 1
    assert data[0]["task_id"] == "test_cls_001"


def test_retrieve_ml_knowledge_returns_json():
    result = retrieve_ml_knowledge.invoke({
        "dataset_profile_json": _cls_profile_json("very_small"),
        "problem_type": "classification",
    })
    data = json.loads(result)
    assert isinstance(data, list)
    rule_ids = [r["rule_id"] for r in data]
    assert "classification_very_small_prefers_simple_models" in rule_ids


def test_retrieve_ml_knowledge_empty_for_no_match():
    # A large regression dataset — not many_classes rule, no imbalance rule, etc.
    profile_json = json.dumps({
        "schema_version": 1, "problem_type": "regression",
        "n_rows": "medium", "n_features": "medium",
        "missing_rate": "none", "n_categorical_features": "none",
        "n_numerical_features": "many",
        "target_distribution": "near_normal",
    })
    result = retrieve_ml_knowledge.invoke({
        "dataset_profile_json": profile_json,
        "problem_type": "regression",
    })
    data = json.loads(result)
    # near_normal + medium → no matching rule triggers (regression_very_small and regression_large don't match)
    assert isinstance(data, list)
```

- [ ] **Step 2: Verify failure**

```
uv run pytest tests/test_tools/test_memory_tools.py -v
```
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Create `src/mlops_agents/tools/memory_tools.py`**

```python
"""LangChain @tool retrieval functions for experience pool and ML knowledge."""
from __future__ import annotations

import json

from langchain_core.tools import tool

from mlops_agents.config.settings import settings
from mlops_agents.experience.pool import ExperiencePool
from mlops_agents.knowledge.reader import match_rules


@tool
def retrieve_similar_experiences(
    dataset_profile_json: str,
    problem_type: str,
    k: int = 5,
) -> str:
    """Retrieve up to k past experience records with the most similar dataset_profile.

    Returns JSON list of RetrievalView objects ordered by similarity_score descending.
    Empty list if no experiences match the problem_type.

    Args:
        dataset_profile_json: JSON string of the DatasetProfile dict.
        problem_type: One of "classification", "regression", "forecasting".
        k: Maximum number of results to return.
    """
    profile = json.loads(dataset_profile_json)
    pool = ExperiencePool(settings.experience_db_path)
    views = pool.find_similar(profile, problem_type, k)
    return json.dumps([v.model_dump() for v in views], default=str)


@tool
def retrieve_ml_knowledge(
    dataset_profile_json: str,
    problem_type: str,
) -> str:
    """Retrieve all curated ML rules whose applies_when conditions are satisfied by the profile.

    Returns JSON list of MLRule objects in YAML file order (curated order is meaningful).

    Args:
        dataset_profile_json: JSON string of the DatasetProfile dict.
        problem_type: One of "classification", "regression", "forecasting".
    """
    profile = json.loads(dataset_profile_json)
    profile["problem_type"] = problem_type
    rules = match_rules(profile)
    return json.dumps([r.model_dump() for r in rules], default=str)
```

- [ ] **Step 4: Update `src/mlops_agents/tools/__init__.py`**

Append at the end (after the existing `__all__` list, inside `__all__`):

```python
from mlops_agents.tools.memory_tools import retrieve_similar_experiences, retrieve_ml_knowledge
```

And add both to `__all__`:
```python
    "retrieve_similar_experiences",
    "retrieve_ml_knowledge",
```

- [ ] **Step 5: Run tests**

```
uv run pytest tests/test_tools/test_memory_tools.py -v
```
Expected: 4 PASS.

- [ ] **Step 6: Run full unit suite**

```
uv run pytest -m "not integration" -q
```
Expected: ~273 PASS (no regressions).

- [ ] **Step 7: Commit**

```bash
git add src/mlops_agents/tools/memory_tools.py src/mlops_agents/tools/__init__.py tests/test_tools/test_memory_tools.py
git commit -m "feat: add retrieve_similar_experiences and retrieve_ml_knowledge @tool functions"
```

---

## Task 7: Benchmark forecasting datasets (data/benchmarks/)

SP4 needs 6 small forecasting CSVs bundled in the repo. This task generates them with synthetic data and the manifest for all 18 benchmark datasets.

**Files:**
- Create: `data/benchmarks/air_passengers.csv`
- Create: `data/benchmarks/m4_monthly_sample.csv`
- Create: `data/benchmarks/electricity_demand_sample.csv`
- Create: `data/benchmarks/sales_sample.csv`
- Create: `data/benchmarks/weather_sample.csv`
- Create: `data/benchmarks/stock_sample.csv`
- Create: `scripts/benchmark_manifest.yaml`

- [ ] **Step 1: Generate the 6 forecasting CSVs**

Run this Python snippet to generate them:

```python
import numpy as np
import pandas as pd
from pathlib import Path

Path("data/benchmarks").mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)

# 1. AirPassengers — univariate monthly, 144 obs, clear seasonality + trend
dates = pd.date_range("1949-01-01", periods=144, freq="MS")
trend = np.arange(144) * 1.8
seasonal = 60 * np.sin(np.arange(144) * 2 * np.pi / 12)
noise = rng.normal(scale=8, size=144)
pd.DataFrame({"month": dates, "passengers": np.round(112 + trend + seasonal + noise, 1)})\
  .to_csv("data/benchmarks/air_passengers.csv", index=False)

# 2. M4 monthly sample — 5 series × 120 obs
rows = []
for i, sid in enumerate(["M1", "M2", "M3", "M4", "M5"]):
    base = 200 + i * 50
    t = np.arange(120)
    y = base + t * 0.5 + 30 * np.sin(t * 2 * np.pi / 12) + rng.normal(scale=10, size=120)
    for d, v in zip(pd.date_range("2000-01-01", periods=120, freq="MS"), y):
        rows.append({"series_id": sid, "date": d, "value": round(v, 2)})
pd.DataFrame(rows).to_csv("data/benchmarks/m4_monthly_sample.csv", index=False)

# 3. Electricity demand — univariate daily, 730 obs (2 years)
dates_d = pd.date_range("2022-01-01", periods=730, freq="D")
trend_d = np.arange(730) * 0.05
weekly = 20 * np.sin(np.arange(730) * 2 * np.pi / 7)
yearly = 40 * np.sin(np.arange(730) * 2 * np.pi / 365)
noise_d = rng.normal(scale=5, size=730)
pd.DataFrame({"date": dates_d, "demand_gwh": np.round(400 + trend_d + weekly + yearly + noise_d, 2)})\
  .to_csv("data/benchmarks/electricity_demand_sample.csv", index=False)

# 4. Sales — 3 stores × 156 weekly obs
rows = []
for sid in ["Store_A", "Store_B", "Store_C"]:
    base = rng.integers(500, 1500)
    t = np.arange(156)
    y = base + t * 0.3 + 100 * np.sin(t * 2 * np.pi / 52) + rng.normal(scale=30, size=156)
    for d, v in zip(pd.date_range("2020-01-01", periods=156, freq="W"), y):
        rows.append({"store_id": sid, "week": d, "sales": max(0, round(v, 0))})
pd.DataFrame(rows).to_csv("data/benchmarks/sales_sample.csv", index=False)

# 5. Weather — univariate daily temperature, 365 obs
dates_w = pd.date_range("2023-01-01", periods=365, freq="D")
temp = 15 + 12 * np.sin(np.arange(365) * 2 * np.pi / 365 - np.pi / 2) + rng.normal(scale=3, size=365)
pd.DataFrame({"date": dates_w, "temp_c": np.round(temp, 1)})\
  .to_csv("data/benchmarks/weather_sample.csv", index=False)

# 6. Stock — 2 tickers × 252 daily obs
rows = []
for ticker in ["STOCK_A", "STOCK_B"]:
    price = 100.0
    prices = []
    for _ in range(252):
        price *= np.exp(rng.normal(0.0003, 0.015))
        prices.append(round(price, 2))
    for d, p in zip(pd.date_range("2023-01-01", periods=252, freq="B"), prices):
        rows.append({"ticker": ticker, "date": d, "close": p})
pd.DataFrame(rows).to_csv("data/benchmarks/stock_sample.csv", index=False)

print("Generated 6 benchmark CSVs")
```

Run it:
```
uv run python -c "exec(open('scripts/_generate_benchmarks.py').read())"
```
Or paste and run directly in Python.

Save the script as `scripts/_generate_benchmarks.py` first, then run:
```
uv run python scripts/_generate_benchmarks.py
```

- [ ] **Step 2: Verify the CSVs were created**

```
ls data/benchmarks/
```
Expected: 6 CSV files.

```
uv run python -c "import pandas as pd; print(pd.read_csv('data/benchmarks/air_passengers.csv').shape)"
```
Expected: `(144, 2)`.

- [ ] **Step 3: Create `scripts/benchmark_manifest.yaml`**

```yaml
# Benchmark manifest — 18 datasets across classification, regression, and forecasting.
# source: sklearn | openml | local
# sklearn: source_id is the function name in sklearn.datasets
# openml: source_id is the OpenML dataset integer ID
# local: source_id is the path to the CSV in data/benchmarks/

# --- Classification (7) ---

- dataset_id: iris
  source: sklearn
  source_id: load_iris
  problem_type: classification
  target_column: target

- dataset_id: wine
  source: sklearn
  source_id: load_wine
  problem_type: classification
  target_column: target

- dataset_id: breast_cancer
  source: sklearn
  source_id: load_breast_cancer
  problem_type: classification
  target_column: target

- dataset_id: titanic
  source: openml
  source_id: 40945
  problem_type: classification
  target_column: survived

- dataset_id: adult_income
  source: openml
  source_id: 1590
  problem_type: classification
  target_column: class

- dataset_id: bank_marketing
  source: openml
  source_id: 1461
  problem_type: classification
  target_column: y

- dataset_id: heart_disease
  source: openml
  source_id: 53
  problem_type: classification
  target_column: num

# --- Regression (5) ---

- dataset_id: california_housing
  source: sklearn
  source_id: fetch_california_housing
  problem_type: regression
  target_column: target

- dataset_id: diabetes
  source: sklearn
  source_id: load_diabetes
  problem_type: regression
  target_column: target

- dataset_id: bike_sharing
  source: openml
  source_id: 42712
  problem_type: regression
  target_column: cnt

- dataset_id: concrete_strength
  source: openml
  source_id: 4353
  problem_type: regression
  target_column: class

- dataset_id: energy_efficiency
  source: openml
  source_id: 41669
  problem_type: regression
  target_column: Y1

# --- Forecasting (6) ---

- dataset_id: air_passengers
  source: local
  source_id: data/benchmarks/air_passengers.csv
  problem_type: forecasting
  target_column: passengers
  datetime_column: month
  series_id_columns: []
  frequency: MS
  forecast_horizon: 12

- dataset_id: m4_monthly_sample
  source: local
  source_id: data/benchmarks/m4_monthly_sample.csv
  problem_type: forecasting
  target_column: value
  datetime_column: date
  series_id_columns: [series_id]
  frequency: MS
  forecast_horizon: 12

- dataset_id: electricity_demand
  source: local
  source_id: data/benchmarks/electricity_demand_sample.csv
  problem_type: forecasting
  target_column: demand_gwh
  datetime_column: date
  series_id_columns: []
  frequency: D
  forecast_horizon: 30

- dataset_id: sales_weekly
  source: local
  source_id: data/benchmarks/sales_sample.csv
  problem_type: forecasting
  target_column: sales
  datetime_column: week
  series_id_columns: [store_id]
  frequency: W
  forecast_horizon: 13

- dataset_id: weather_daily
  source: local
  source_id: data/benchmarks/weather_sample.csv
  problem_type: forecasting
  target_column: temp_c
  datetime_column: date
  series_id_columns: []
  frequency: D
  forecast_horizon: 30

- dataset_id: stock_daily
  source: local
  source_id: data/benchmarks/stock_sample.csv
  problem_type: forecasting
  target_column: close
  datetime_column: date
  series_id_columns: [ticker]
  frequency: B
  forecast_horizon: 21
```

- [ ] **Step 4: Commit**

```bash
git add data/benchmarks/ scripts/benchmark_manifest.yaml scripts/_generate_benchmarks.py
git commit -m "feat: add 6 synthetic benchmark CSVs and 18-dataset manifest for benchmark runner"
```

---

## Task 8: Benchmark runner + smoke test

**Files:**
- Create: `scripts/_dataset_sources.py`
- Create: `scripts/run_benchmark.py`
- Create: `tests/test_scripts/__init__.py`
- Create: `tests/test_scripts/test_benchmark_runner.py`

- [ ] **Step 1: Write failing smoke test**

Create `tests/test_scripts/__init__.py` (empty).

Create `tests/test_scripts/test_benchmark_runner.py`:

```python
"""Smoke test for the benchmark runner — iris classification only (fast)."""
import json
from pathlib import Path
import pytest
from mlops_agents.experience.pool import ExperiencePool


@pytest.mark.slow
def test_benchmark_runner_iris_smoke(tmp_path, monkeypatch):
    """Run the benchmark runner on iris only and verify the pool is populated."""
    import sys
    import yaml
    from pathlib import Path as _Path

    # Redirect storage and audit dirs to tmp_path
    monkeypatch.setattr("mlops_agents.config.settings.settings.experience_db_path",
                        tmp_path / "bench.db")
    monkeypatch.setattr("mlops_agents.config.settings.settings.experience_audit_dir",
                        tmp_path / "pool")

    # Write a minimal manifest with only iris
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.dump([{
        "dataset_id": "iris",
        "source": "sklearn",
        "source_id": "load_iris",
        "problem_type": "classification",
        "target_column": "target",
    }]))

    from scripts.run_benchmark import run_benchmark
    run_benchmark(
        manifest_path=manifest_path,
        db_path=tmp_path / "bench.db",
        audit_dir=tmp_path / "pool",
        splits_dir=tmp_path / "splits",
        staged_dir=tmp_path / "staged",
        n_trials_override=4,
    )

    pool = ExperiencePool(tmp_path / "bench.db")
    assert pool.count() >= 1
    assert pool.count("classification") >= 1
```

- [ ] **Step 2: Verify failure**

```
uv run pytest tests/test_scripts/test_benchmark_runner.py -v
```
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Create `scripts/_dataset_sources.py`**

```python
"""Dataset fetchers for the benchmark runner.

source: sklearn — use sklearn.datasets function by name
source: openml — use sklearn.datasets.fetch_openml by integer ID
source: local — read CSV from source_id path
"""
from __future__ import annotations

import pandas as pd


def fetch_dataset(entry: dict) -> pd.DataFrame:
    src = entry["source"]
    if src == "sklearn":
        from sklearn import datasets
        loader = getattr(datasets, entry["source_id"])
        bunch = loader()
        df = pd.DataFrame(bunch.data, columns=bunch.feature_names).copy()
        df["target"] = bunch.target
        return df
    if src == "openml":
        from sklearn.datasets import fetch_openml
        bunch = fetch_openml(data_id=int(entry["source_id"]), as_frame=True, parser="auto")
        return bunch.frame
    if src == "local":
        return pd.read_csv(entry["source_id"])
    raise ValueError(f"Unknown source: {src!r}. Valid: sklearn, openml, local")
```

- [ ] **Step 4: Create `scripts/run_benchmark.py`**

```python
"""Offline benchmark runner — seeds the experience pool from public datasets.

Usage:
    uv run python scripts/run_benchmark.py
    uv run python scripts/run_benchmark.py --manifest scripts/benchmark_manifest.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

# Make src/ importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mlops_agents.config.settings import settings
from mlops_agents.experience.pool import ExperiencePool
from mlops_agents.experience.schema import ExperienceRecord
from mlops_agents.training.default_plans import default_training_plan
from mlops_agents.training.executor import run_training_plan
from mlops_agents.training.profiler import build_dataset_profile
from mlops_agents.contracts.training import TrialBudget
from mlops_agents.utils.logging import get_logger
from scripts._dataset_sources import fetch_dataset

logger = get_logger(__name__)


def build_task_metadata(entry: dict) -> dict:
    meta = {
        "problem_type": entry["problem_type"],
        "target_column": entry["target_column"],
    }
    if entry["problem_type"] == "forecasting":
        meta.update({
            "datetime_column": entry["datetime_column"],
            "series_id_columns": entry.get("series_id_columns", []),
            "frequency": entry["frequency"],
            "forecast_horizon": entry["forecast_horizon"],
        })
    return meta


def stage_dataset(df, entry: dict, staged_dir: Path) -> Path:
    staged_dir.mkdir(parents=True, exist_ok=True)
    # Rename target column for sklearn bunches that name it differently
    target = entry["target_column"]
    if target not in df.columns and "target" in df.columns:
        df = df.rename(columns={"target": target})
    csv_path = staged_dir / f"{entry['dataset_id']}.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def run_benchmark(
    manifest_path: Path = Path("scripts/benchmark_manifest.yaml"),
    db_path: Path | None = None,
    audit_dir: Path | None = None,
    splits_dir: Path | None = None,
    staged_dir: Path | None = None,
    n_trials_override: int | None = None,
) -> tuple[int, int]:
    """Run the benchmark and return (n_success, n_fail)."""
    db_path = db_path or settings.experience_db_path
    audit_dir = audit_dir or settings.experience_audit_dir
    splits_dir = splits_dir or Path("data/benchmarks/_splits")
    staged_dir = staged_dir or Path("data/benchmarks")

    manifest = yaml.safe_load(manifest_path.read_text()) or []
    pool = ExperiencePool(db_path, audit_dir=audit_dir)
    n_success = n_fail = 0

    for entry in manifest:
        dataset_id = entry["dataset_id"]
        try:
            logger.info(f"[{dataset_id}] Fetching dataset...")
            df = fetch_dataset(entry)
            csv_path = stage_dataset(df, entry, staged_dir)

            task_meta = build_task_metadata(entry)
            profile = build_dataset_profile(csv_path, task_meta)
            plan = default_training_plan(entry["problem_type"], profile)

            # Override trial budget for benchmark speed
            if n_trials_override is not None:
                from mlops_agents.contracts.training import TrialBudget
                plan = plan.model_copy(update={
                    "trial_budget": TrialBudget(
                        total_trials=n_trials_override * len(plan.candidates),
                        allocation_strategy="equal",
                        min_trials_per_candidate=max(2, n_trials_override // 2),
                        max_trials_per_candidate=n_trials_override,
                    )
                })

            result = run_training_plan(
                plan=plan,
                processed_dataset_path=csv_path,
                target_column=entry["target_column"],
                task_metadata=task_meta,
                output_dir=splits_dir / dataset_id,
                mlflow_experiment="mlops-agents-benchmark",
            )

            record = ExperienceRecord.model_validate(
                json.loads(Path(result.experience_record_path).read_text())
            )
            pool.insert_from_record(record)
            n_success += 1
            logger.info(f"[{dataset_id}] ✓ champion={result.champion_candidate['model_key']}")

        except Exception as e:
            n_fail += 1
            logger.error(f"[{dataset_id}] FAILED: {e}")

    logger.info(f"Benchmark complete: {n_success} success, {n_fail} failed "
                f"({n_success + n_fail} total)")
    return n_success, n_fail


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the experience pool from public datasets.")
    parser.add_argument("--manifest", type=Path, default=Path("scripts/benchmark_manifest.yaml"))
    parser.add_argument("--trials", type=int, default=8,
                        help="Optuna trials per candidate (lower = faster, less optimal)")
    args = parser.parse_args()
    n_ok, n_fail = run_benchmark(manifest_path=args.manifest, n_trials_override=args.trials)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the smoke test (marked slow — runs Optuna)**

```
uv run pytest tests/test_scripts/test_benchmark_runner.py -v -m slow
```
Expected: 1 PASS (~30–60s for iris with 4 trials per candidate).

- [ ] **Step 6: Run the full benchmark (optional — takes ~20–60 min)**

This is not a CI test. Run manually once to populate the pool:

```
uv run python scripts/run_benchmark.py --trials 8
```

The script logs progress per dataset. Failures (e.g., OpenML network issues) do not abort the batch.

- [ ] **Step 7: Verify pool after full benchmark run**

```
uv run python -c "
from mlops_agents.experience.pool import ExperiencePool
from mlops_agents.config.settings import settings
pool = ExperiencePool(settings.experience_db_path)
print('Total records:', pool.count())
print('Classification:', pool.count('classification'))
print('Regression:', pool.count('regression'))
print('Forecasting:', pool.count('forecasting'))
"
```
Expected: ≥ 12 total, ≥ 4 per type.

- [ ] **Step 8: Run full unit suite (no slow tests)**

```
uv run pytest -m "not integration and not slow" -q
```
Expected: ~277+ PASS.

- [ ] **Step 9: Commit**

```bash
git add scripts/_dataset_sources.py scripts/run_benchmark.py scripts/_generate_benchmarks.py
git add tests/test_scripts/
git commit -m "feat: add benchmark runner — seeds experience pool from 18 public datasets"
```

---

## Self-review against acceptance criteria

| # | Criterion | Covered by |
|---|---|---|
| 1 | Migrations idempotent, creates 3 tables | `test_migrations.py` |
| 2 | `insert_from_record` atomic, writes JSON audit | `test_pool.py` |
| 3 | `match_rules` validates applies_when fields, model_keys, problem-type consistency | `test_reader.py` |
| 4 | Retrieval `@tools` return JSON, pass Pydantic | `test_memory_tools.py` |
| 5 | `find_similar` sorted by score, carries similarity_ratio, empty on no match | `test_retrieval.py` |
| 6 | Benchmark runner runs end-to-end, failures don't abort | `test_benchmark_runner.py` (smoke) |
| 7 | ≥ 12 records after full benchmark | `run_benchmark.py` + manual check |
| 8 | Starter rules load, pass all validators | `test_starter_rules.py` |
| 9 | No LLM calls in test suite | All tests use sqlite3 / pandas (no LLM imports) |
