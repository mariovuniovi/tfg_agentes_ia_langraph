# Project structure — file reference

Quick reference for what each file does. Use this when navigating the repo. For the *why*, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Root

| File | Purpose |
|---|---|
| `pyproject.toml` | UV configuration: deps, Python 3.12, ruff + mypy + pytest config |
| `uv.lock` | UV lockfile (committed) |
| `.python-version` | Pins Python 3.12 |
| `.env.example` | Documented env-var template |
| `.gitignore` | Excludes `.venv/`, `.env`, `mlruns/`, `storage/`, `experience_pool/`, `CLAUDE.md`, `models/`, etc. |
| `README.md` | Project intro, setup, status |
| `ARCHITECTURE.md` | System shape, agent graph, contracts, data flow |
| `PLAN.md` | Sub-project status board (SP1–SP5) |
| `STRUCTURE.md` | This file |
| `Dockerfile`, `docker-compose.yml` | Multi-stage UV build + MLflow + app services |
| `langgraph.json` | LangGraph CLI config pointing at the compiled graph |

---

## `src/mlops_agents/` — main package

### `state/` — shared graph state
| File | Purpose |
|---|---|
| `agent_state.py` | `AgentState` TypedDict — the shared state every node reads/writes. Read this before editing any agent. |
| `schemas.py` | Pydantic schemas for structured LLM outputs: `RouterOutput`, `ValidationResult`, `TrainingResult`, `EvaluationResult` |

### `config/` — environment + constants
| File | Purpose |
|---|---|
| `settings.py` | `Settings` (pydantic-settings) — reads `.env`. Never hardcode tokens; use `settings.github_token` |
| `constants.py` | Global constants: `MIN_ACCURACY_TO_DEPLOY`, agent names, MLflow aliases |

### `contracts/` — cross-cutting Pydantic contracts
| File | Purpose |
|---|---|
| `training.py` | `TrainingPlan`, `TrainingPlanCandidate`, `RejectedModel`, `ValidationStrategy`, `ExogStrategySettings`, `ForecastingSettings`, `TrialBudget`, `SearchParamOverride`. Includes `_check_plan_integrity` boundary validator. |
| `profile.py` | `DatasetProfile` (Pydantic v2) — the retrieval join key. Includes `history_length` bucket for forecasting. |

### `utils/` — shared utilities
| File | Purpose |
|---|---|
| `llm.py` | LLM factory: `get_llm()` returns the worker model, `get_router_llm()` returns the cheaper supervisor model |
| `logging.py` | loguru setup — `get_logger(__name__)`. Never `print()`. |
| `runners.py` | Entry point for the `mlops-dashboard` console script |

### `tools/` — deterministic `@tool` functions
Pure Python; no LLM calls. Agents call these and interpret results.

| File | Purpose |
|---|---|
| `data_tools.py` | `load_dataset`, `validate_schema`, `check_missing_values` |
| `evidently_tools.py` | `check_data_quality`, `check_data_drift` (Evidently 0.7 API) |
| `mlflow_tools.py` | `log_experiment`, `get_best_run`, `register_model`, `set_model_alias` |
| `memory_tools.py` | `retrieve_ml_knowledge`, `retrieve_similar_experiences` — entry points for SP5 retrieval |

### `prompts/` — YAML system prompts
| File | Purpose |
|---|---|
| `loader.py` | `get_prompt(name)` — loads YAML and returns a `PromptTemplate` |
| `supervisor.yaml` | Supervisor system prompt + routing rules |
| `data_agent.yaml`, `evaluation_agent.yaml`, `deployment_agent.yaml` | Specialist agent prompts |

(Note: `training_agent.yaml` not present yet — training currently runs via the deterministic executor without an LLM in the loop. SP5 will introduce a model_agent prompt.)

### `agents/` — specialist agents
Each agent is `langchain.agents.create_agent(...)` (ReAct loop). The supervisor is a structured-output LLM call (not ReAct).

| File | Purpose |
|---|---|
| `supervisor.py` | `supervisor_node(state)` — structured LLM call returning `RouterOutput`. Logs every routing decision. Force-exits when `remaining_steps <= 2`. |
| `data_agent.py` | `build_data_agent()` — tools: data_tools + evidently_tools |
| `evaluation_agent.py` | `build_evaluation_agent()` — tools: mlflow_tools |
| `deployment_agent.py` | `build_deployment_agent()` — tools: register + set alias |
| `registry.py` | `get_agent(name)` — `@lru_cache` factory; builds agents lazily |

### `graphs/` — LangGraph topology
| File | Purpose |
|---|---|
| `mlops_graph.py` | The main `StateGraph`. Builds nodes (supervisor + 4 specialists), the `deployer_node` with `interrupt()` for HITL, and the compiled `graph`. Also has `main()` for CLI execution. |
| `subgraphs/training_flow.py` | Reserved for an iterative retrain sub-workflow (not wired into the main graph yet) |

### `models/` — model registry
| File | Purpose |
|---|---|
| `registry.yaml` | All available models per problem_type: factory, default_params, search_space, complexity_rank, requirements |
| `loader.py` | `get_model(key)`, `get_models_for(problem_type)` — registry accessors |
| `factories.py` | One factory function per model key. Wraps sklearn / LightGBM / XGBoost / CatBoost / statsforecast / skforecast |
| `search_spaces.py` | `build_suggest_fn(search_space)` — converts YAML search space to an Optuna `suggest_*` callable |

### `training/` — deterministic training spine
| File | Purpose |
|---|---|
| `executor.py` | `run_training_plan(plan, csv, target, task_metadata, output_dir, mlflow_experiment) -> TrainingResult` — the top-level entry. Dispatches to `_run_candidate_classification` / `_run_candidate_regression` / `_run_candidate_forecasting`. Owns the leakage-safe per-fold loop and the MLflow logging. |
| `profiler.py` | `build_dataset_profile(csv, task_metadata) -> DatasetProfile` — computes the bucketed profile |
| `splitter.py` | Train/test split (single-shot; the K-fold backtest is inside the executor) |
| `default_plans.py` | `default_training_plan(problem_type, profile) -> TrainingPlan` — registry-driven fallback when no LLM produces the plan |
| `validation_policy.py` | `select_validation_strategy`, `resolve_rolling_window_size`, `validate_forecasting_plan` — the policy + guardrails for forecasting validation |
| `validation_folds.py` | `iter_folds(train_pool, strategy, dt_col, sid_cols)` — yields (train_idx, val_idx) pairs for single_split / rolling / expanding |
| `exog_extender.py` | `extend_exog`, `align_val_exog_index` — the **leakage firewall** for unknown_future exog columns (naive_carry / ets / auto_arima) |
| `override_validation.py` | Validates user-provided search-space overrides against the registry |
| `trial_budget.py` | `allocate_trials(...)` — distributes Optuna trials across candidates |
| `experience.py` | `build_task_id`, `write_experience_record` — record assembly + filesystem write |

### `experience/` — pool persistence
| File | Purpose |
|---|---|
| `pool.py` | `ExperiencePool(db_path, audit_dir)` — SQLite store with `insert_from_record` and `get(task_id)`. Idempotent migration runner. |
| `schema.py` | `ExperienceRecord`, `SelectedSolution`, `CandidateResult` Pydantic models |
| `retrieval.py` | (placeholder) future SP5 retrieval helpers |
| `migrations/001_init.sql` | Initial schema: `experiences`, `candidate_results`, `model_artifacts` tables |
| `migrations/002_add_forecasting_columns.sql` | SP4.1 additions: `validation_strategy_json`, `exog_availability_json`, `exog_strategies_json`, `per_fold_metrics_json`, `exog_fit_failures_json` |
| `migrations/_runner.py` | PRAGMA-introspecting migration applier (idempotent) |

### `knowledge/` — ML rules
| File | Purpose |
|---|---|
| `ml_rules.yaml` | Domain rules: hard preferences (`prefer`/`avoid`) and forecasting recipe recommendations (`recommend`). Consumed by the deterministic planner today and by SP5 LLM tomorrow. |
| `reader.py` | `MLRule` Pydantic model + `load_rules()` + `match_rules(context)` — matches rules against a merged profile+task_metadata context |

---

## `api/` — FastAPI backend

| Path | Purpose |
|---|---|
| `main.py` | FastAPI app + CORS + router includes |
| `routers/uploads.py` | `POST /uploads` — receive user CSVs |
| `routers/runs.py` | `POST /runs`, `GET /runs/{id}/stream` (SSE), `POST /runs/{id}/approve` |
| `routers/experiments.py` | MLflow experiment browsing |
| `routers/monitoring.py` | Drift report generation |
| `services/pipeline.py` | Invokes the LangGraph compiled graph |
| `services/pipeline_helpers.py` | SSE event formatting + state translation |
| `services/run_store.py` | Active-run registry (in-memory) |
| `services/mlflow_client.py` | Wrapped MLflow client |
| `models/` | API-side Pydantic request/response models (`run.py`, `experiment.py`, `monitoring.py`) |
| `tests/` | API tests (FastAPI TestClient) |

---

## `frontend/` — Next.js UI

| Path | Purpose |
|---|---|
| `app/page.tsx` | Landing |
| `app/pipeline/page.tsx` | Trigger a run, watch SSE log, approve at HITL gate |
| `app/experiments/page.tsx` | MLflow run browser with chart panel |
| `app/monitoring/page.tsx` | Drift detection on uploaded reference + current CSVs |
| `components/pipeline/*` | `TriggerPanel`, `EventLog`, `RunStatusBadge`, `HITLGate`, `ResultsDashboard` |
| `components/experiments/*` | `RunSidebar`, `ChartPanel`, `charts/` |
| `components/monitoring/*` | `AdHocForm`, `DriftTable`, `LatestReport` |
| `hooks/use-run-stream.ts` | SSE subscription hook |
| `hooks/use-approve.ts` | HITL approve/reject mutation |
| `stores/run-store.ts` | Zustand store for active-run state |
| `lib/api.ts`, `lib/format.ts`, `lib/query-client.ts` | API client + helpers |
| `__tests__/` | Vitest unit tests for hooks, components, stores |

---

## `dashboard/` — Streamlit UI (alternative)

| File | Purpose |
|---|---|
| `app.py` | Streamlit entry + navigation |
| `pages/01_pipeline.py` | Run launcher with real-time log streaming |
| `pages/02_experiments.py` | MLflow run table |
| `pages/03_monitoring.py` | Drift report uploader |
| `pages/04_chat.py` | Chat interface (calls agents directly) |
| `components/metrics_display.py`, `components/chat_interface.py` | Reusable widgets |

---

## `scripts/` — utility scripts

| File | Purpose |
|---|---|
| `run_pipeline.py` | CLI: `uv run python scripts/run_pipeline.py [csv]` — runs the full graph |
| `run_benchmark.py` | Seeds the experience pool from `benchmark_manifest.yaml`. Has `_preprocess_benchmark_df` (label-encode categoricals, drop high-cardinality strings, impute NaNs) and `build_task_metadata` (propagates `exogenous_columns`, `expected_drift`). |
| `_dataset_sources.py` | Fetchers: `sklearn`, `openml`, `local`, `yfinance`, `yfinance_multi` |
| `_generate_benchmarks.py` | Builds local benchmark CSVs (air_passengers, m4_monthly_sample, gold_macro_monthly, etc.) |
| `benchmark_manifest.yaml` | 21 dataset entries: 7 classification + 5 regression + 9 forecasting (including 3 with multi-exog yfinance) |
| `seed_mlflow.py` | Creates demo MLflow runs |

---

## `data/`

| Path | Purpose |
|---|---|
| `samples/` | Toy datasets for development (iris.csv, iris_measurements.csv, iris_labels.csv) |
| `schemas/` | Dataset schema JSONs (column types, expected names) |
| `benchmarks/` | Benchmark CSVs (one per manifest entry) |
| `benchmarks/_splits/<dataset>/` | Train/test split + champion `.pkl` per benchmark run |
| `uploads/` | User-uploaded CSVs (timestamped) |
| `processed/` | Processed-canonical CSVs (post-merge, post-encoding) |
| `merged/` | Merged-but-not-yet-processed CSVs |
| `working/` | Intermediate artifacts |

---

## `tests/` — unit + integration

Mirrors `src/` layout. ~326 tests, ~50s to run (excluding integration).

| Path | Purpose |
|---|---|
| `conftest.py` | Shared fixtures: `sample_csv`, `mock_llm`, `iris_schema_file`, `minimal_experience_record` (factory), etc. **Check here before adding new fixtures.** |
| `test_contracts/` | `TrainingPlan`, `ForecastingSettings`, `SearchParamOverride`, `_check_plan_integrity` |
| `test_training/` | `executor`, `profiler`, `splitter`, `validation_folds`, `validation_policy`, `exog_extender`, `executor_forecasting_leakage` |
| `test_experience/` | Pool migration, round-trip |
| `test_models/` | Factory + loader + search_spaces |
| `test_knowledge/` | YAML rules load + match (`forecasting_rules`, `starter_rules`) |
| `test_tools/` | Deterministic tools (no LLM mocks needed for data_tools) |
| `test_agents/` | Agent builders (with `mock_llm`) |
| `test_graphs/` | Graph structure (compile, expected nodes) |
| `test_integration/` | End-to-end; requires real `GITHUB_TOKEN`; `@pytest.mark.integration` |

---

## `docs/`

| Path | Purpose |
|---|---|
| `superpowers/specs/` | Design specs (one per feature, brainstormed before coding) |
| `superpowers/plans/` | Implementation plans (one per feature, bite-sized TDD tasks) |

Most recent (SP4.1 forecasting work):
- `specs/2026-05-11-forecasting-exogenous-leakage-safe-validation-design.md`
- `plans/2026-05-11-forecasting-exogenous-leakage-safe-validation.md`

---

## Generated / gitignored

| Path | Created by | Status |
|---|---|---|
| `.venv/` | `uv sync` | gitignored |
| `mlruns/`, `mlartifacts/` | MLflow tracking | gitignored |
| `storage/mlops_metadata.db` | `ExperiencePool` | gitignored |
| `experience_pool/*.json` | Audit copies | gitignored |
| `models/` (top-level) | SP3 training artifacts | gitignored (note: `src/mlops_agents/models/` is NOT gitignored — only top-level `/models/`) |
| `catboost_info/` | CatBoost training logs | gitignored |
| `data/uploads/`, `data/merged/`, `data/processed/`, `data/working/` | Pipeline-generated | gitignored |
| `CLAUDE.md`, `CLAUDE_backup.md` | Local Claude Code guidance | gitignored |
