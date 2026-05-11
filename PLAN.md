# Project plan — Multi-Agent MLOps System (TFG)

> Status board organized around **sub-projects** (SP1–SP5). Each sub-project has a brainstormed design spec under `docs/superpowers/specs/` and an implementation plan under `docs/superpowers/plans/`. Tasks within a sub-project are tracked as TDD steps in the plan file.

Last updated: 2026-05-11

---

## At a glance

| Sub-project | Description | State | Spec |
|---|---|---|---|
| **SP1** | Schema-driven data validation + HITL auto-fix | ✅ Complete | `2026-04-20-schema-driven-validation-design.md` |
| **SP2** | Forecasting-aware data validator | ✅ Complete | `2026-05-05-forecasting-aware-data-validator-design.md` |
| **SP3** | Model registry + training pipeline | ✅ Complete | `2026-05-06-model-registry-training-pipeline-design.md` |
| **SP4** | Experience pool + benchmark runner (21 datasets) | ✅ Complete | `2026-05-06-experience-pool-benchmark-runner-design.md` |
| **SP4.1** | Forecasting exog handling + leakage-safe validation | ✅ Complete | `2026-05-11-forecasting-exogenous-leakage-safe-validation-design.md` |
| **API/UI** | FastAPI backend + Next.js frontend | 🔄 In progress | `2026-04-23-fastapi-backend-design.md`, `2026-04-24-nextjs-frontend-design.md` |
| **SP5** | LLM model_agent (retrieves experiences, proposes plans) | ⬜ Next | (spec to be written) |
| **Demo** | End-to-end demo + TFG report | ⬜ Pending | — |

Legend: ✅ Complete · 🔄 In progress · ⬜ Pending · ⚠️ Blocked

---

## SP1 — Schema-driven data validation (✅ Complete)

**Goal:** A user uploads a dataset; the `data_agent` validates it against a schema (column names, types, ranges, freshness) and either approves it or proposes auto-fixes via an HITL flow.

**Delivered:**
- ✅ `data_agent` with `validate_schema`, `check_missing_values`, Evidently quality reports
- ✅ Schema JSONs in `data/schemas/`
- ✅ HITL auto-fix loop (user approves/rejects each proposed fix)
- ✅ Streamlit + (later) FastAPI/Next.js endpoints to upload and validate

---

## SP2 — Forecasting-aware data validator (✅ Complete)

**Goal:** When the user declares the problem as forecasting, the validator captures the temporal structure: datetime column, frequency, gaps, series_id (if panel), exogenous columns. This produces a `task_metadata` dict that the training pipeline can rely on.

**Delivered:**
- ✅ Frequency detection (H/D/W/MS/M/QS/YS) + irregular-spacing report
- ✅ Gap analysis (missing dates within the inferred frequency)
- ✅ Multi-series detection (panel vs single-target)
- ✅ Exogenous column detection (non-target/non-date/non-series-id numeric cols)
- ✅ `task_metadata` schema documented for downstream consumers

---

## SP3 — Model registry + training pipeline (✅ Complete)

**Goal:** A deterministic training executor that, given a `TrainingPlan` and a dataset, runs Optuna-tuned model selection for classification / regression / forecasting and produces a `TrainingResult` + an `ExperienceRecord`.

**Delivered:**
- ✅ `src/mlops_agents/models/registry.yaml` — all models with factory, default_params, search_space, complexity_rank, requirements
- ✅ Factory functions per model (`factories.py`): sklearn, LightGBM, XGBoost, CatBoost, statsforecast (AutoETS, AutoARIMA, Naive, SeasonalNaive), skforecast (ForecasterRecursiveMultiSeries wrapping each tabular regressor + SVR)
- ✅ `executor.py` with `_run_candidate_classification` / `_run_candidate_regression` / `_run_candidate_forecasting`
- ✅ Optuna integration with `build_suggest_fn(search_space)`
- ✅ MLflow parent + nested runs; champion selection (`_pick_champion`) with complexity tie-breaker
- ✅ `_retrain_forecasting` / `_retrain_tabular` to refit champion on full train pool
- ✅ Search-space override validation (`override_validation.py`)

---

## SP4 — Experience pool + benchmark runner (✅ Complete)

**Goal:** Every training run writes an `ExperienceRecord` to a SQLite-backed pool. The pool is seeded offline from a manifest of 21 public benchmark datasets so SP5 has a non-empty retrieval source from day one.

**Delivered:**
- ✅ `ExperiencePool` with `INSERT OR REPLACE` upsert + PRAGMA-introspecting idempotent migrations
- ✅ SQL migrations (`001_init.sql`, `002_add_forecasting_columns.sql`)
- ✅ `ExperienceRecord` Pydantic model (schema.py) + JSON audit copies
- ✅ `scripts/run_benchmark.py` with `--trials N` override, label-encoding of categoricals, OpenML-leakage-column dropping, NaN imputation
- ✅ `scripts/_dataset_sources.py` with sklearn / openml / local / yfinance / yfinance_multi fetchers
- ✅ 21-entry manifest covering: 7 classification, 5 regression, 9 forecasting (including 3 large multi-exog yfinance datasets)
- ✅ All 21 datasets seeded successfully

---

## SP4.1 — Forecasting exog handling + leakage-safe validation (✅ Complete)

**Goal:** Fix the temporal leakage bug where realized future exog values were silently fed into `forecaster.predict(exog=val_exog)`. Add typed contracts so the executor enforces leakage protection deterministically.

Spec: [`2026-05-11-forecasting-exogenous-leakage-safe-validation-design.md`](docs/superpowers/specs/2026-05-11-forecasting-exogenous-leakage-safe-validation-design.md)
Plan: [`2026-05-11-forecasting-exogenous-leakage-safe-validation.md`](docs/superpowers/plans/2026-05-11-forecasting-exogenous-leakage-safe-validation.md)

**Tasks (all done):**

| # | Task | State |
|---|---|---|
| 1 | Typed Pydantic models (`ValidationStrategy`, `ExogStrategySettings`, `ForecastingSettings`) | ✅ |
| 2 | Pydantic-ified `DatasetProfile` + `history_length` field | ✅ |
| 3 | `validation_folds.iter_folds` (single_split / rolling / expanding) | ✅ |
| 4 | `exog_extender` (naive_carry / ets / auto_arima + fallback + index align) | ✅ |
| 5 | `validation_policy` (selection + plan guardrails) | ✅ |
| 6 | `ExperienceRecord` + PRAGMA migration (5 new fields) | ✅ |
| 7 | Rewritten `_run_candidate_forecasting` with leakage-safe loop | ✅ |
| 8 | Experience record assembly + MLflow per-fold logging | ✅ |
| 9 | 6 new forecasting rules + `MLRule.recommend` field | ✅ |
| 10 | Benchmark manifest `exogenous_columns` + `expected_drift` annotations | ✅ |
| 11 | End-to-end regression: re-seed 21 datasets, verify no leakage | ✅ |
| + | `TrainingPlan._check_plan_integrity` boundary validator | ✅ |

**Result:** 326 unit tests pass; 21/21 benchmarks re-seeded; `sp500_macro_weekly` verified — all 7 exog declared `unknown_future` and extended via `naive_carry` (no leakage).

---

## API + UI (🔄 In progress)

| Component | State |
|---|---|
| FastAPI backend with SSE for run streaming | ✅ |
| Next.js pages: pipeline, experiments, monitoring | ✅ |
| HITL approval flow via REST endpoint | ✅ |
| Streamlit dashboard (alternative UI) | ✅ |
| Frontend tests (Vitest) | 🔄 Partial |
| Experiment chart panel polish | 🔄 |

---

## SP5 — LLM model_agent (⬜ Next)

**Goal:** Replace `default_training_plan` (rule-based, registry-driven) with an LLM that:
1. Reads the `DatasetProfile`, `task_metadata`, registered models, ML rules, and retrieved similar past `ExperienceRecord`s
2. Produces a `TrainingPlan` with:
   - Ranked `candidates` with `reason` per candidate
   - `models_not_recommended` with reasons (the LLM's judgment, not just hard-rule rejection)
   - `forecasting_settings` (validation strategy + exog strategies) for forecasting tasks
3. Hands the plan to the deterministic executor, which validates and runs it

**Why now:** the experience pool is populated (21 records), the contracts are LLM-safe (`_check_plan_integrity` enforces structure), the rules YAML has the planner-guidance entries.

**Open questions for the design spec:**
- Retrieval: vector similarity on profile dict, or keyword filtering, or both?
- Prompt structure: how to present 5–10 retrieved records concisely?
- Fallback: if the LLM proposes a structurally-invalid plan, retry once with the validation error in the prompt, then fall back to the deterministic planner?
- Token budget: gpt-4.1-mini context window per call

To start: invoke the `superpowers:brainstorming` skill on this.

---

## Demo + TFG report (⬜ Pending)

| Item | State |
|---|---|
| End-to-end demo recording (upload → validate → train → deploy with HITL) | ⬜ |
| TFG written report (architecture, design decisions, results) | ⬜ |
| Architecture diagrams (graph topology, data flow, contracts) | 🔄 In progress (ARCHITECTURE.md has Mermaid-ish ASCII) |
| Defense slides | ⬜ |
| `docker compose up` smoke test | ⬜ |

---

## Cross-cutting hygiene (always-on)

| Item | State |
|---|---|
| `uv run pytest -m "not integration"` passes | ✅ (326/326) |
| `uv run ruff check .` clean | 🔄 Partial (a few pre-existing warnings) |
| `uv run mypy src/` clean | 🔄 Partial |
| Integration test (`@pytest.mark.integration`) with real GITHUB_TOKEN | ⬜ |

---

## What's intentionally out of scope (v1)

- Multi-target panel forecasting with leakage-safe exog (deferred to v2 — single-target with many exog is the primary use case)
- `scenario_based` / `market_implied` / `forecasted` covariate types from the original brief (the binary `known_future` / `unknown_future` covers ≥95% of real cases)
- Auto-detection of `expected_drift` from the data itself (user-provided business hint in v1)
- Season-length-aware `min_train_len` (MVP uses `max(3 × horizon, 30)`)
- Optuna-tuning of validation `n_folds` / `window_size` (these are evaluation protocol, not model hyperparameters)
