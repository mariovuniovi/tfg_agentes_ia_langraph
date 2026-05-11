# Architecture

A reference for how the system is shaped and where responsibilities live. Read this once before touching unfamiliar code.

## Design principles

1. **Deterministic spine, agentic edges.** Data loading, training loops, splitting, metric computation, and validation strategy are pure Python. LLMs are reserved for *interpretation* (failure reports, strategy proposals, natural-language summaries) and *judgment* (model_agent in SP5).
2. **Schema → Plan → Executor.** Data validation captures business truth (`future_availability`, `expected_drift`). The training plan captures modelling choices (validation strategy, exog extension). The executor enforces both — it never trusts the LLM to skip safety checks.
3. **HITL at the deployment gate, not everywhere.** Only `deployer_node` uses `interrupt()`; all code before it must be idempotent.
4. **The experience pool is the memory.** Every training run produces an `ExperienceRecord` keyed by dataset profile. SP5 (the future LLM planner) retrieves similar past records and reasons from them rather than re-deriving rules from scratch.

## Two-layer separation

```
┌──────────────────────────────────────────────────────────────────┐
│                     User-facing surfaces                          │
│   Next.js frontend ── FastAPI backend ── Streamlit dashboard ── CLI│
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                        LangGraph agent graph                      │
│                                                                   │
│      ┌─────────────────────────────────────────────┐              │
│      │  supervisor_node  ◀────── routes via ──────│              │
│      │  (structured        RouterOutput            │              │
│      │   LLM call)                                 │              │
│      └────────┬──────────────────────────────────────┘            │
│               │                                                   │
│      ┌────────┼────────┬─────────┬─────────┐                      │
│      ▼        ▼        ▼         ▼         ▼                      │
│   data_   training_  evaluation_  deployer_node                   │
│   agent   agent      agent        (HITL: interrupt())             │
│      │       │          │            │                            │
│      └───────┴──────────┴────────────┘                            │
│              All agents call deterministic tools                  │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Deterministic core (no LLM)                    │
│                                                                   │
│   training/executor      ← run_training_plan(plan, ...) -> result │
│   training/validation_*  ← policy, folds, exog extender            │
│   models/factories       ← skforecast / statsforecast / sklearn   │
│   experience/pool        ← SQLite + INSERT OR REPLACE              │
│   knowledge/reader       ← ml_rules.yaml matcher                   │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                          Persistence                              │
│                                                                   │
│   storage/mlops_metadata.db   ← ExperiencePool (SQLite)           │
│   mlruns/                     ← MLflow tracking                    │
│   experience_pool/*.json      ← Audit copies of records            │
│   data/benchmarks/_splits/    ← Train/test splits + champion .pkl  │
└──────────────────────────────────────────────────────────────────┘
```

## The agent graph

`StateGraph` with **5 nodes** (`src/mlops_agents/graphs/mlops_graph.py`):

```
START
  │
  ▼
supervisor ── data_validator ──┐
  ▲                            │
  │                            ▼
  └─────────── trainer ──── evaluator ── deployer ── END
                                            │
                                            └── interrupt() before
                                                MLflow Model Registry
                                                promotion
```

- **Supervisor** is a single LLM call with structured output (`RouterOutput { next, reasoning }`). Every routing decision is logged.
- **Specialist agents** (data, training, evaluation, deployment) use `langchain.agents.create_agent` with a ReAct loop. Each agent calls deterministic tools and returns a natural-language summary plus structured fields.
- **deployer_node** is the only `interrupt()` site. It pauses after the model is registered as `challenger` and waits for `Command(resume={"approved": True/False})`.
- **Graceful recursion exit**: supervisor checks `remaining_steps <= 2` and forces `Command(goto=END)`.

## Contracts (the language between layers)

All defined in `src/mlops_agents/contracts/` and `src/mlops_agents/experience/`. Pydantic v2 throughout.

| Contract | Lives in | Used by |
|---|---|---|
| `AgentState` | `state/agent_state.py` | The graph — all nodes read/write this TypedDict |
| `DatasetProfile` | `contracts/profile.py` | Profiler output; rule matcher input; SP5 retrieval key |
| `TrainingPlan` | `contracts/training.py` | What the planner (or LLM) produces; what the executor consumes |
| `TrainingPlanCandidate` | `contracts/training.py` | One model to try with optional search-space override |
| `RejectedModel` | `contracts/training.py` | Goes in `TrainingPlan.models_not_recommended` with a reason |
| `ValidationStrategy` | `contracts/training.py` | `single_split` / `rolling_window` / `expanding_window` + n_folds + horizon |
| `ExogStrategySettings` | `contracts/training.py` | Per-column exog extension strategy + default fallback |
| `ForecastingSettings` | `contracts/training.py` | Bundles validation + exog strategies into the plan |
| `ExperienceRecord` | `experience/schema.py` | Persisted; SQLite + JSON audit |
| `MLRule` | `knowledge/reader.py` | YAML rules with `applies_when`, `prefer`, `recommend`, `reason` |

`TrainingPlan` enforces structural integrity at the boundary (`_check_plan_integrity`): no model_key in both `candidates` and `models_not_recommended`, no empty `reason`, every model_key registered for the plan's `problem_type`.

## Data flow for a forecasting run

```
1. User uploads dataset + schema (or runs benchmark)
       │
       ▼
2. data_agent validates schema, captures future_availability per exog column
       │   writes task_metadata: { exogenous_columns: [...], expected_drift, ... }
       ▼
3. build_dataset_profile(csv, task_metadata) → DatasetProfile (Pydantic)
       │   buckets n_rows, history_length, computes seasonality/trend/stationarity
       ▼
4. training_agent (or default_training_plan) builds TrainingPlan
       │   - select_validation_strategy(profile, task_metadata) → ValidationStrategy
       │   - candidates from registry filtered by _is_eligible
       │   - models_not_recommended with reasons (filled by SP5 LLM in future)
       ▼
5. run_training_plan(plan, csv, task_metadata, ...) — the executor
       │   - validate_forecasting_plan(plan, ...) ← raises on violations
       │   - For each candidate, _run_candidate_forecasting:
       │       * iter_folds(train_pool, strategy) → (train_idx, val_idx) per fold
       │       * For each fold:
       │           - For each exog col:
       │               · known_future → use cand_val[col] (genuinely known)
       │               · unknown_future → extend_exog(cand_train[col], horizon, strategy)
       │               · drop → omit from both train and val exog
       │           - forecaster.fit(series_dict, train_exog)
       │           - forecaster.predict(steps=horizon, exog=val_exog)
       │           - score the fold
       │       * Aggregate via mean → trial_score (back to Optuna)
       ▼
6. _pick_champion(results) → SelectedSolution
       │
       ▼
7. _retrain_forecasting(champion, full train_pool) → champion.pkl
       │
       ▼
8. ExperienceRecord assembled:
       │   - dataset_profile, training_plan_input, models_tested
       │   - selected_solution, validation_strategy, exog_availability,
       │     exog_strategies (actually applied), per_fold_metrics, exog_fit_failures
       │
       ▼
9. ExperiencePool.insert_from_record(record) → SQLite + JSON audit
       │
       ▼
10. evaluation_agent compares champion vs. test set baseline
       │
       ▼
11. deployer_node registers as challenger → interrupt() → wait for human
       │
       ▼
12. On approve: set MLflow alias `champion`. Done.
```

## The leakage firewall

The bug we fixed in SP4.1: previously the executor silently fed realized future exog values into `forecaster.predict(exog=val_exog)`. The new design:

```python
# Inside the per-fold loop, for each exog column:
if availability[col] == "known_future":
    val_exog[col] = cand_val[col]               # genuinely known — safe
else:  # unknown_future
    val_exog[col], failure = extend_exog(
        cand_train[col], horizon, strategy, freq   # ← only training history
    )
```

`extend_exog()` is the **single allowed path** to construct `val_exog` for `unknown_future` columns. The executor cannot bypass it. Strategies: `naive_carry` (cheapest, default), `ets`, `auto_arima`. Failures fall back to `naive_carry` with a logged warning and an entry in `exog_fit_failures`.

## Knowledge layer

`src/mlops_agents/knowledge/ml_rules.yaml` holds **planner guidance** — not executable defaults. Three kinds:

```yaml
# Hard preferences (used by the deterministic default planner today)
- rule_id: forecasting_long_history_with_exogenous_prefers_supervised
  applies_when:
    problem_type: forecasting
    history_length: [medium, long]
    exogenous_features_available: true
  prefer: [lightgbm_forecaster, xgboost_forecaster, gbm_forecaster]
  reason: "..."

# Forecasting recipe recommendations (consumed by SP5 LLM planner)
- rule_id: forecasting_high_drift_rolling_window
  applies_when:
    problem_type: forecasting
    expected_drift: high
  recommend:
    validation_strategy: rolling_window
  reason: "..."

# Exog strategy hints
- rule_id: exog_unknown_default_naive_carry
  applies_when:
    problem_type: forecasting
    exog_future_availability: unknown_future
  recommend:
    exog_strategy: naive_carry
  reason: "..."
```

The rule matcher (`match_rules`) accepts a merged context: profile fields + selected task_metadata fields (`expected_drift`, `exog_*` hints). Rules can be `prefer`-style (candidate selection) or `recommend`-style (strategy proposals). SP5 will read both and produce reasoned plans.

## Sub-project boundaries

| Sub-project | Scope | Outputs |
|---|---|---|
| SP1 | Schema-driven data validation | `data_agent`, validation report, HITL auto-fix flow |
| SP2 | Forecasting-aware data validator | Frequency detection, gap analysis, exog detection in `task_metadata` |
| SP3 | Model registry + training pipeline | `models/registry.yaml`, `factories.py`, `executor.py` (single fold initially) |
| SP4 | Experience pool + benchmark runner | `experience/pool.py`, `scripts/run_benchmark.py`, 21 seeded records |
| SP4.1 | Forecasting exog handling + leakage-safe validation | `validation_policy.py`, `validation_folds.py`, `exog_extender.py`, rewritten `_run_candidate_forecasting` |
| **SP5** | LLM model_agent | (Next) consumes pool + rules + profile; produces `TrainingPlan` with `models_not_recommended` reasons |
| Frontend | Next.js UI on FastAPI | Pipeline triggering, experiments browser, monitoring, HITL approval |

Each sub-project has a design spec under `docs/superpowers/specs/` and an implementation plan under `docs/superpowers/plans/`.

## What goes through MLflow

The executor logs per training run:
- **Parent run params**: `validation_strategy_type`, `validation_n_folds`, `exog_default_strategy`, `expected_drift`, `metric_to_optimize`
- **Parent run metrics**: `fold_0_<metric>`, ..., `fold_N_<metric>`, `fold_mean_<metric>`, `fold_std_<metric>`, champion metrics
- **Nested runs**: one per candidate model, with its best Optuna trial's params + score
- **Artifacts**: `champion_<model_key>.pkl`, JSON record path, MLflow Model Registry entry as `challenger` (champion alias only after HITL approval)

## What goes through the experience pool

SQLite at `storage/mlops_metadata.db`, table `experiences`. Each row corresponds to one full training run with these key columns (plus the 5 SP4.1 additions):

- `task_id`, `problem_type`, `dataset_name`
- `dataset_profile_json` ← retrieval join key
- `training_plan_json` ← what was tried (and what was rejected, with reasons)
- `selected_model_key`, `validation_score`, `validation_std`, `metric_to_optimize`
- `validation_strategy_json`, `exog_availability_json`, `exog_strategies_json` (forecasting only)
- `per_fold_metrics_json`, `exog_fit_failures_json` (forecasting only)
- `mlflow_parent_run_id`, `created_at`

SP5 retrieval will query this table for "similar" past records (same problem_type, similar profile buckets, comparable exog footprint) and feed them as context to the LLM planner.
