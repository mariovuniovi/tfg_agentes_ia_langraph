# Experience Pool + Benchmark Runner + Retrieval Tools — Design Spec

**Sub-project:** 4 of the MLCopilot-style model_agent foundation
**Date:** 2026-05-06
**Branch:** feature/experience-pool (TBD)
**Depends on:** SP3 (Model Registry + Deterministic Training Pipeline)

---

## Goal

Build the long-term memory infrastructure that the SP5 `model_agent` will reason over. SP4 produces no agent — only deterministic infrastructure:

1. **Experience pool storage** — SQLite-backed read/write layer over the experience records SP3 produces.
2. **Static ML knowledge base** — handcrafted YAML of conditional rules (curated wisdom, version-controlled).
3. **Retrieval tools** — two LangChain `@tool` functions SP5 will call: `retrieve_similar_experiences` and `retrieve_ml_knowledge`.
4. **Benchmark runner** — offline script that runs SP3's executor over ~18 public datasets to seed the pool.

After SP4 is complete, the pool contains 12+ experience records across all three problem types, the static rule base is loaded with a curated starter set, and the retrieval tools work in isolation. SP5 then plugs in.

---

## Architecture

```
                ┌─────────────────────────────────┐
                │  scripts/run_benchmark.py        │
                │  (offline pool seeder)           │
                └────────────────┬─────────────────┘
                                 │ direct calls (no graph, no agent)
                                 ▼
                ┌─────────────────────────────────┐
                │  SP3 executor                    │
                │  run_training_plan(...)          │
                └────────────────┬─────────────────┘
                                 │ writes ExperienceRecord (JSON file)
                                 ▼
        ExperiencePool.insert_from_record()
                                 │ inserts into SQLite + keeps JSON audit copy
                                 ▼
storage/mlops_metadata.db  ◄── SP4 reads ──►  experience_pool/<task_id>.json
        ▲                       (3 tables)              (audit, optional)
        │
        │ queried via
        │
┌───────┴────────────────────────────────────────┐
│  retrieval tools (memory_tools.py)              │
│   - retrieve_similar_experiences(profile, k)    │
│   - retrieve_ml_knowledge(profile)              │
└─────────────────┬───────────────────────────────┘
                  │ also reads
                  ▼
src/mlops_agents/knowledge/ml_rules.yaml (static, version-controlled)
```

**Storage separation (recap from brainstorm):**

| File | Owner | Contents |
|---|---|---|
| `mlflow.db` | MLflow | runs, params, metrics, model_versions (untouched by SP4) |
| `mlruns/` | MLflow | model artifacts (untouched) |
| `storage/mlops_metadata.db` | SP4 | experiences, candidate_results, model_artifacts (3 tables) |
| `experience_pool/<task_id>.json` | SP4 | human-readable audit dumps (gitignored) |
| `src/mlops_agents/knowledge/ml_rules.yaml` | SP4 | curated rules, version-controlled |
| `data/benchmarks/` | SP4 | bundled CSVs for non-sklearn/non-OpenML datasets |

We **never** copy MLflow data into our SQLite. Every cross-reference uses `mlflow_run_id` strings as foreign-key-like pointers. The SP3 spec already specifies that artifacts live in MLflow; SP4 just stores the location and summary metrics for fast retrieval.

---

## Section 1 — Dataset profile schema

The `DatasetProfile` is the join key for retrieval. It's a Pydantic model with bucketed string/bool fields, computed by `profiler.build_dataset_profile(dataset_path, task_metadata)` (introduced in SP3, formalized here).

### 1.1 Universal fields (all problem types)

| Field | Type | Buckets / values |
|---|---|---|
| `problem_type` | enum (mandatory) | `classification`, `regression`, `forecasting` |
| `n_rows` | bucket | `very_small` (<500), `small` (500–999), `medium` (1,000–50,000), `large` (>50,000) |
| `n_features` | bucket | `small` (<10), `medium` (10–100), `large` (>100) |
| `missing_rate` | bucket | `none` (0%), `low` (<5%), `medium` (5–20%), `high` (>20%) — measured **before** imputation |
| `n_categorical_features` | bucket | `none`, `few` (1–3), `some` (4–10), `many` (>10) |
| `n_numerical_features` | bucket | `none`, `few`, `some`, `many` |

### 1.2 Classification-only

| Field | Buckets |
|---|---|
| `n_classes` | `binary` (2), `small_multiclass` (3–5), `many_classes` (>5) |
| `class_balance` | `balanced` (max/min < 1.5), `moderately_imbalanced` (1.5–5×), `severely_imbalanced` (>5×) |

### 1.3 Regression-only

| Field | Buckets |
|---|---|
| `target_distribution` | `near_normal`, `skewed`, `heavy_tailed`, `discrete_like` |

Computed via skewness threshold (|skew| > 1) and kurtosis threshold (kurt > 3); `discrete_like` if number of unique target values < `n_rows / 20`.

### 1.4 Forecasting-only

| Field | Type | Description / buckets |
|---|---|---|
| `n_series` | bucket | `single`, `few` (2–10), `moderate` (11–100), `many` (>100) |
| `history_length` | bucket (avg per series) | `very_short` (<50), `short` (50–200), `medium` (200–1000), `long` (>1000) |
| `frequency` | string passthrough | pandas offset alias (`H`, `D`, `W`, `MS`, `QS`, `YS`) |
| `horizon_difficulty` | bucket (lookup matrix) | see 1.5 |
| `exogenous_features_available` | bool | `true` if any non-target/non-datetime/non-series_id columns present |
| `seasonality_detected` | bool | majority-rule across series; per-series check via autocorrelation peak at expected lag |
| `trend_detected` | bool | majority-rule; per-series Mann-Kendall test |
| `stationarity` | bool | majority-rule; per-series ADF test (p < 0.05 → stationary) |

### 1.5 Forecast horizon difficulty matrix

`horizon_difficulty` is computed from `frequency` + raw `forecast_horizon` (an int from `task_metadata`). Raw horizon also stored as int in the profile for audit but not used for retrieval matching.

| frequency \ horizon | very_short | short | medium | long |
|---|---|---|---|---|
| H (hourly) | 1–24 | 25–168 | 169–1000 | > 1000 |
| D (daily) | 1–7 | 8–30 | 31–90 | > 90 |
| W (weekly) | 1–4 | 5–13 | 14–52 | > 52 |
| MS (monthly) | 1–3 | 4–12 | 13–24 | > 24 |
| QS (quarterly) | 1–2 | 3–4 | 5–8 | > 8 |
| YS (yearly) | 1 | 2–3 | 4–5 | > 5 |

So `30 daily steps` → `medium`; `30 monthly steps` → `long`. Difficulty is frequency-aware, capturing the user's intuition that long-period forecasting is harder.

### 1.6 Pydantic schema (`src/mlops_agents/contracts/profile.py`)

Lives in the contracts folder introduced by SP3 (cross-cutting type used by SP3, SP4, SP5):

```python
class DatasetProfile(BaseModel):
    schema_version: int = 1                            # for future profile-schema migrations
    problem_type: Literal["classification", "regression", "forecasting"]
    # Universal
    n_rows: Literal["very_small", "small", "medium", "large"]
    n_features: Literal["small", "medium", "large"]
    missing_rate: Literal["none", "low", "medium", "high"]
    n_categorical_features: Literal["none", "few", "some", "many"]
    n_numerical_features: Literal["none", "few", "some", "many"]
    # Classification
    n_classes: Literal["binary", "small_multiclass", "many_classes"] | None = None
    class_balance: Literal["balanced", "moderately_imbalanced", "severely_imbalanced"] | None = None
    # Regression
    target_distribution: Literal["near_normal", "skewed", "heavy_tailed", "discrete_like"] | None = None
    # Forecasting
    n_series: Literal["single", "few", "moderate", "many"] | None = None
    history_length: Literal["very_short", "short", "medium", "long"] | None = None
    frequency: str | None = None
    horizon_difficulty: Literal["very_short", "short", "medium", "long"] | None = None
    forecast_horizon_raw: int | None = None             # audit only, not used in matching
    exogenous_features_available: bool | None = None
    seasonality_detected: bool | None = None
    trend_detected: bool | None = None
    stationarity: bool | None = None
```

A `model_validator` enforces "fields populated iff problem_type matches" (forecasting fields must be set for forecasting profiles, etc.).

---

## Section 2 — Storage layer (SQLite + JSON audit)

### 2.1 New folder: `src/mlops_agents/experience/`

```
src/mlops_agents/experience/
├── __init__.py
├── pool.py                   # ExperiencePool class — sqlite3 wrapper
├── schema.py                 # ExperienceRecord, RetrievalView (Pydantic)
├── retrieval.py              # weighted matcher; the @tool functions
└── migrations/
    ├── 001_init.sql
    └── _runner.py            # apply pending migrations on startup
```

### 2.2 SQLite schema (`migrations/001_init.sql`)

```sql
CREATE TABLE IF NOT EXISTS _schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE experiences (
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
CREATE INDEX idx_experiences_problem_type ON experiences(problem_type);
CREATE INDEX idx_experiences_created_at ON experiences(created_at);

CREATE TABLE candidate_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    model_key TEXT NOT NULL,
    status TEXT NOT NULL,           -- 'successful' | 'failed'
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
CREATE INDEX idx_candidate_results_task_id ON candidate_results(task_id);

CREATE TABLE model_artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    model_key TEXT NOT NULL,
    mlflow_run_id TEXT,
    artifact_path TEXT,
    artifact_uri TEXT,
    is_champion INTEGER NOT NULL,
    metric_name TEXT,
    metric_value REAL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (task_id) REFERENCES experiences(task_id) ON DELETE CASCADE
);
CREATE INDEX idx_model_artifacts_task_id ON model_artifacts(task_id);

INSERT INTO _schema_version (version, applied_at) VALUES (1, datetime('now'));
```

### 2.3 Migration runner (`migrations/_runner.py`)

```python
def apply_pending_migrations(db_path: Path) -> None:
    """Apply numbered migrations <NNN>_*.sql in order if their version > current."""
    conn = sqlite3.connect(db_path)
    current = _read_current_version(conn)  # 0 if _schema_version table missing
    pending = sorted(_list_migration_files() if v > current)
    for mig in pending:
        with conn:
            conn.executescript(mig.read_text())
        logger.info(f"Applied migration {mig.name}")
    conn.close()
```

Called on `ExperiencePool` initialization. Idempotent (CREATE TABLE IF NOT EXISTS in `001_init.sql`).

### 2.4 `ExperiencePool` API

```python
class ExperiencePool:
    def __init__(self, db_path: Path, audit_dir: Path | None = None):
        apply_pending_migrations(db_path)
        self._conn = sqlite3.connect(db_path)
        self._audit_dir = audit_dir

    def insert_from_record(self, record: ExperienceRecord) -> None:
        """Insert into all 3 tables atomically, then write JSON audit copy."""

    def get(self, task_id: str) -> ExperienceRecord:
        ...

    def find_similar(
        self,
        profile: DatasetProfile,
        k: int = 5,
        weights: dict[str, int] | None = None,
    ) -> list[RetrievalView]:
        """Weighted-overlap retrieval. See Section 4."""

    def count(self, problem_type: str | None = None) -> int: ...
```

`insert_from_record` uses a single SQL transaction (`BEGIN/COMMIT`) so all 3 inserts succeed or none do. The JSON audit dump is written *after* commit (best-effort: a failed JSON write does not roll back the SQL insert; just logs a warning).

---

## Section 3 — Static ML knowledge base

### 3.1 New folder: `src/mlops_agents/knowledge/`

```
src/mlops_agents/knowledge/
├── __init__.py
├── ml_rules.yaml             # the curated rules
└── reader.py                 # MLRule loader, match_rules()
```

Lives **in the package** (versioned, ships with installs), not in `storage/`.

### 3.2 `MLRule` Pydantic schema

```python
class MLRule(BaseModel):
    rule_id: str
    applies_when: dict[str, str | list[str] | bool]
    prefer: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    reason: str
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_model_keys(self):
        from mlops_agents.models.loader import load_registry
        registry = load_registry()
        for k in self.prefer + self.avoid:
            if k not in registry:
                raise ValueError(f"Rule {self.rule_id}: unknown model_key '{k}'")
        return self
```

Mutable defaults all use `Field(default_factory=list)` (Pydantic best practice).

### 3.3 Matching algorithm (`reader.match_rules`)

For each rule, iterate over `applies_when` entries. Every entry must hold:

- `applies_when[field]` is a scalar (string/bool) → `profile[field] == value`.
- `applies_when[field]` is a list → `profile[field] in list` (in-set semantics).
- `field` not in profile → rule fails (fail-closed; the rule wasn't designed for this profile shape).

If all entries match, the rule applies. Returned in YAML file order (curated order is meaningful).

### 3.4 Starter rule set (`ml_rules.yaml`)

Drafted ~13 rules covering the patterns from `plan_model_agent.md` plus broadly useful heuristics. Examples (truncated for brevity):

```yaml
- rule_id: very_small_dataset_prefers_simple_models
  applies_when:
    n_rows: very_small
  prefer: [logistic_regression, ridge, naive, seasonal_naive]
  avoid: [lightgbm_classifier, xgboost_classifier, catboost_classifier,
          lightgbm_regressor, xgboost_regressor, catboost_regressor,
          lightgbm_forecaster, xgboost_forecaster]
  reason: |
    With <500 rows, complex tree ensembles overfit and regularized linear
    or simple statistical baselines generalize better.
  tags: [sample_size]

- rule_id: forecasting_short_history_prefers_statistical
  applies_when:
    problem_type: forecasting
    history_length: [very_short, short]
  prefer: [seasonal_naive, ets, auto_arima, naive]
  avoid: [svr_forecaster, lightgbm_forecaster, xgboost_forecaster,
          random_forest_forecaster, extra_trees_forecaster, gbm_forecaster]
  reason: |
    With short history (<200 obs/series), statistical models with strong
    structural priors outperform feature-heavy supervised models.
  tags: [forecasting, sample_size]

- rule_id: forecasting_long_history_with_exogenous_prefers_supervised
  applies_when:
    problem_type: forecasting
    history_length: [medium, long]
    exogenous_features_available: true
  prefer: [lightgbm_forecaster, xgboost_forecaster, gbm_forecaster,
          random_forest_forecaster, extra_trees_forecaster]
  reason: |
    Supervised lag-based models exploit nonlinear lagged effects and
    external regressors when history is sufficient.
  tags: [forecasting, exogenous]

- rule_id: forecasting_strong_seasonality_prefers_seasonal_models
  applies_when:
    problem_type: forecasting
    seasonality_detected: true
  prefer: [seasonal_naive, ets, auto_arima]
  reason: |
    Detected seasonality strongly favors models with explicit seasonal
    decomposition (ETS, seasonal ARIMA) over plain Naive or non-seasonal
    supervised forecasters.
  tags: [forecasting, seasonality]

- rule_id: classification_severe_imbalance_prefers_tree_ensembles
  applies_when:
    problem_type: classification
    class_balance: severely_imbalanced
  prefer: [lightgbm_classifier, xgboost_classifier, catboost_classifier,
          random_forest_classifier]
  avoid: [logistic_regression]
  reason: |
    Severely imbalanced classes (>5x) benefit from tree ensembles which
    handle class weights and decision boundaries flexibly. Logistic
    regression often defaults to the majority class.
  tags: [classification, imbalance]

# ... 8 more rules covering: many_classes multiclass, regression
# heavy_tailed targets, forecasting many-series global modeling,
# forecasting high-missing original data, forecasting non-stationary,
# small_or_medium with smooth nonlinear (SVR), etc.
```

The full 13 rules are written into the YAML at SP4 implementation time. The schema is fixed; the content is curated.

---

## Section 4 — Retrieval algorithm

### 4.1 Weighted-overlap matching (`pool.find_similar`)

Hard filter on `problem_type`. Weighted score over the rest. Tie-break by recency.

```python
RETRIEVAL_WEIGHTS: dict[str, int] = {
    # Structural (3) — decides which model family applies
    "n_rows": 3,
    "n_series": 3,
    "history_length": 3,
    "horizon_difficulty": 3,
    "seasonality_detected": 3,

    # Strong influence (2) — within family / edge cases
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
```

`problem_type` is a hard filter, never weighted (a regression experience has zero relevance to a forecasting task).

### 4.2 Algorithm

```python
def find_similar(profile, problem_type, k):
    rows = db.execute(
        "SELECT * FROM experiences WHERE problem_type = ?", (problem_type,)
    ).fetchall()

    scored = []
    for row in rows:
        candidate_profile = json.loads(row["dataset_profile_json"])
        score = 0
        matched_fields = ["problem_type"]
        for field, weight in RETRIEVAL_WEIGHTS.items():
            if field in profile and field in candidate_profile and profile[field] == candidate_profile[field]:
                score += weight
                matched_fields.append(field)
        scored.append((score, row, matched_fields))

    scored.sort(key=lambda x: (-x[0], -isoparse(x[1]["created_at"]).timestamp()))
    return [build_retrieval_view(row, score, matched) for score, row, matched in scored[:k]]
```

Total weight ceilings:
- classification: 13 (3 + 2+2+2 + 1+1+1+1)
- regression: 11 (3 + 2+2 + 1+1+1+1)
- forecasting: 29 (3+3+3+3+3 + 2+2+2+2+2 + 1+1+1+1)

Weights are exposed in `settings.retrieval_weights` so they can be tuned without code changes.

---

## Section 5 — Retrieval tools (LangChain `@tool`)

### 5.1 File: `src/mlops_agents/tools/memory_tools.py`

```python
@tool
def retrieve_similar_experiences(
    dataset_profile_json: str,
    problem_type: str,
    k: int = 5,
) -> str:
    """Retrieve up to k past experience records with the most similar dataset_profile.

    Returns JSON list of trimmed RetrievalView objects, ordered by similarity score
    descending (ties broken by recency).
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
    """Retrieve all curated ML rules whose applies_when is satisfied by the profile.

    Returns JSON list of MLRule objects in YAML order.
    """
    profile = json.loads(dataset_profile_json)
    profile["problem_type"] = problem_type
    rules = match_rules(profile)
    return json.dumps([r.model_dump() for r in rules], default=str)
```

### 5.2 `RetrievalView` schema (trimmed from full `ExperienceRecord`)

```python
class CandidateResultView(BaseModel):
    model_key: str
    status: Literal["successful", "failed"]
    best_score: float | None = None
    complexity_rank: int | None = None
    error_type: str | None = None

class SelectedSolutionView(BaseModel):
    model_key: str
    validation_score: float
    validation_std: float | None
    complexity_rank: int

class RetrievalView(BaseModel):
    task_id: str
    dataset_name: str | None
    dataset_profile: dict
    models_tested: list[CandidateResultView]
    selected_solution: SelectedSolutionView
    models_not_recommended: list[dict] = Field(default_factory=list)
    experience_summary: str | None
    similarity_score: int
    matched_fields: list[str]
```

Excluded from RetrievalView (kept only in SQLite/MLflow): full hyperparameters, MLflow run IDs, traceback paths, training_plan_input, model artifact paths.

If the agent ever needs the full record, a separate (non-tool) helper exists for human inspection: `pool.get(task_id)`.

---

## Section 6 — Benchmark runner

### 6.1 File: `scripts/run_benchmark.py`

Direct executor invocation (no graph, no agent, no data_validator). Logic:

```python
def main(manifest_path: Path = Path("scripts/benchmark_manifest.yaml")) -> None:
    manifest = load_manifest(manifest_path)
    pool = ExperiencePool(settings.experience_db_path, settings.experience_audit_dir)
    n_success = n_fail = 0
    for entry in manifest:
        try:
            df = fetch_dataset(entry)              # sklearn | openml | local
            csv_path = stage_dataset(df, entry)    # write to data/benchmarks/<id>.csv
            task_meta = build_task_metadata(entry)
            profile = build_dataset_profile(csv_path, task_meta)
            plan = default_training_plan(entry["problem_type"], profile)
            result = run_training_plan(
                plan=plan,
                processed_dataset_path=csv_path,
                target_column=entry["target_column"],
                task_metadata=task_meta,
                output_dir=Path("data/benchmarks/_splits"),
                mlflow_experiment="mlops-agents-benchmark",
            )
            record = load_record(result.experience_record_path)
            pool.insert_from_record(record)
            n_success += 1
            logger.info(f"[{entry['dataset_id']}] champion={result.champion_candidate['model_key']}")
        except Exception as e:
            n_fail += 1
            logger.error(f"[{entry['dataset_id']}] FAILED: {e}")
    logger.info(f"Benchmark complete: {n_success} success, {n_fail} failed")
```

Failures are logged but do not abort the batch — the runner is designed for "run them all, see what survives."

### 6.2 Manifest schema (`scripts/benchmark_manifest.yaml`)

```yaml
- dataset_id: iris
  source: sklearn
  source_id: load_iris
  problem_type: classification
  target_column: target

- dataset_id: california_housing
  source: sklearn
  source_id: fetch_california_housing
  problem_type: regression
  target_column: target

- dataset_id: bank_marketing
  source: openml
  source_id: 1461              # OpenML dataset ID
  problem_type: classification
  target_column: y

- dataset_id: air_passengers
  source: local
  source_id: data/benchmarks/air_passengers.csv
  problem_type: forecasting
  target_column: passengers
  datetime_column: month
  series_id_columns: []
  frequency: MS
  forecast_horizon: 12
```

### 6.3 Initial dataset list (~18)

| Problem | Datasets |
|---|---|
| classification (7) | iris, wine, breast_cancer (sklearn); titanic, adult_income, bank_marketing, heart_disease (OpenML/local) |
| regression (5) | california_housing, diabetes (sklearn); bike_sharing, concrete_strength, energy_efficiency (OpenML/local) |
| forecasting (6) | air_passengers, m4_monthly_sample, electricity_demand_sample, sales_sample, weather_sample, stock_sample (all local CSVs in `data/benchmarks/`) |

Forecasting CSVs are bundled in `data/benchmarks/` (committed to the repo, small files). Manifest entries reference them via `source: local`.

### 6.4 Source adapters (`scripts/_dataset_sources.py`)

```python
def fetch_dataset(entry: dict) -> pd.DataFrame:
    src = entry["source"]
    if src == "sklearn":
        from sklearn import datasets
        loader = getattr(datasets, entry["source_id"])
        bunch = loader()
        return pd.DataFrame(bunch.data, columns=bunch.feature_names).assign(target=bunch.target)
    if src == "openml":
        from sklearn.datasets import fetch_openml
        bunch = fetch_openml(data_id=entry["source_id"], as_frame=True)
        return bunch.frame
    if src == "local":
        return pd.read_csv(entry["source_id"])
    raise ValueError(f"Unknown source: {src}")
```

OpenML datasets are cached by sklearn under `~/scikit_learn_data/`. First run downloads; subsequent runs hit cache.

---

## Section 7 — Settings additions

`src/mlops_agents/config/settings.py`:

```python
# Experience pool
experience_db_path: Path = Path("storage/mlops_metadata.db")
experience_audit_dir: Path = Path("experience_pool")
data_benchmarks_dir: Path = Path("data/benchmarks")

# Knowledge base
ml_rules_path: Path = Path("src/mlops_agents/knowledge/ml_rules.yaml")

# Retrieval
retrieval_default_k: int = 5
# retrieval_weights is a module-level constant in retrieval.py;
# overridable via settings if you want to tune without code changes:
retrieval_weights_override: dict[str, int] = Field(default_factory=dict)
```

`storage/` and `experience_pool/` are added to `.gitignore`.

---

## Section 8 — Files (final summary)

### New
```
src/mlops_agents/contracts/
    profile.py                                      # DatasetProfile (NEW; SP3 had a placeholder)
src/mlops_agents/experience/
    __init__.py
    pool.py
    schema.py                                        # ExperienceRecord, RetrievalView
    retrieval.py                                     # find_similar, weights
    migrations/
        __init__.py
        001_init.sql
        _runner.py
src/mlops_agents/knowledge/
    __init__.py
    ml_rules.yaml                                    # ~13 starter rules
    reader.py                                        # match_rules, MLRule loader
src/mlops_agents/tools/
    memory_tools.py                                  # @tool retrieval functions

scripts/
    run_benchmark.py
    benchmark_manifest.yaml
    _dataset_sources.py                              # fetch_dataset adapters

data/benchmarks/
    air_passengers.csv
    m4_monthly_sample.csv
    electricity_demand_sample.csv
    sales_sample.csv
    weather_sample.csv
    stock_sample.csv

tests/test_experience/
    test_pool.py
    test_retrieval.py
    test_migrations.py
tests/test_knowledge/
    test_reader.py
    test_starter_rules.py                            # smoke test that all rules load + validate against registry
tests/test_tools/
    test_memory_tools.py
tests/test_scripts/
    test_benchmark_runner.py                         # integration test on iris only (fast smoke)
```

### Modified
- `src/mlops_agents/config/settings.py` — add storage paths, retrieval settings.
- `src/mlops_agents/tools/__init__.py` — export memory tools.
- `pyproject.toml` — add `pyyaml` if not present; `statsmodels` already pulled by `statsforecast`. No other new deps.
- `.gitignore` — add `storage/`, `experience_pool/`.

### Deleted
- None. SP4 is purely additive over SP3.

---

## Section 9 — Out of scope for SP4

- **SP5**: the `model_agent` planner LLM, HITL on training plan, generalized-lesson extractor from human rejections (those tools call SP4's retrieval but live in SP5).
- **Vector embeddings for retrieval**: bucketed exact-match weighted matching is sufficient at this scale.
- **Real-time pool updates from pipeline runs**: the trainer node calling `pool.insert_from_record(...)` after each user pipeline run is part of SP5's wiring (SP4 only adds the offline benchmark runner).
- **Schema of dataset_profile evolving**: handled later via `schema_version` field in the JSON (SP4 freezes v1).
- **Multi-tenant or distributed pools**: single SQLite file, single user, single machine. Out of scope.

---

## Section 10 — Acceptance criteria

SP4 is complete when:

1. `apply_pending_migrations(...)` creates the 3 tables on a fresh DB; idempotent on subsequent runs.
2. `ExperiencePool.insert_from_record(record)` writes to all 3 tables atomically (one transaction); writes the JSON audit copy on success.
3. `match_rules(profile)` loads `ml_rules.yaml`, validates every `prefer`/`avoid` model_key against the SP3 model registry, returns rules in YAML order whose `applies_when` is fully satisfied.
4. `retrieve_similar_experiences` and `retrieve_ml_knowledge` are registered as LangChain `@tool` functions, return JSON strings, and pass round-trip Pydantic validation.
5. `find_similar(profile, problem_type, k=5)` returns up to k matches sorted by weighted score (recency tie-break); empty list if no experiences match the problem_type.
6. `scripts/run_benchmark.py` runs end-to-end against the manifest, populates the SQLite + JSON audit dir, logs per-dataset success/failure. Single-dataset failures do not abort the batch.
7. After running the benchmark, the pool contains ≥ 12 experience records (≥ 4 per problem type).
8. The starter `ml_rules.yaml` loads cleanly: every rule's `prefer`/`avoid` model_keys validate against the SP3 registry.
9. Test suite passes; no test calls a real LLM.
