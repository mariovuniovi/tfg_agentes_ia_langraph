# Multi-Agent MLOps System — TFG

A multi-agent MLOps system that takes a dataset from validation to production-ready model with a human-in-the-loop deployment gate. Built on a **custom LangGraph supervisor** orchestrating four specialist agents (data, training, evaluation, deployment) plus a deterministic training spine that handles classification, regression, and forecasting (with leakage-safe validation for exogenous time series).

## Status

| Sub-project | Description | State |
|---|---|---|
| **SP1** | Schema-driven data validation + HITL auto-fix | ✅ Complete |
| **SP2** | Forecasting-aware data validator (frequency, gaps, exog) | ✅ Complete |
| **SP3** | Model registry + training pipeline (Optuna, MLflow) | ✅ Complete |
| **SP4** | Experience pool + offline benchmark runner (21 datasets seeded) | ✅ Complete |
| **SP4.1** | Forecasting exogenous handling + leakage-safe validation | ✅ Complete (Tasks 1–11) |
| **SP5** | LLM model_agent (retrieves experiences, proposes plans) | ⬜ Next |
| **Frontend** | Next.js UI on top of the FastAPI backend | 🔄 In progress |

See [PLAN.md](PLAN.md) for the full status board and [ARCHITECTURE.md](ARCHITECTURE.md) for the system shape.

## Tech stack

- **Python 3.12 + UV** (src-layout)
- **LangGraph + LangChain** for agent orchestration
- **GitHub Models** (gpt-4.1-mini) — free LLM tier
- **scikit-learn, LightGBM, XGBoost, CatBoost** for tabular models
- **statsforecast + skforecast** for forecasting (AutoETS, AutoARIMA, recursive multi-series ML)
- **Optuna** for hyperparameter search
- **MLflow** for experiment tracking + model registry
- **Evidently AI** for data quality + drift reports
- **FastAPI + Next.js** for the user-facing API and UI
- **SQLite** for the experience pool (`storage/mlops_metadata.db`)

## Quick start

```bash
uv sync                                    # install all dependencies
cp .env.example .env                       # add GITHUB_TOKEN + GITHUB_MODEL
```

**Run the full stack** (two terminals):

```bash
# Terminal 1 — FastAPI backend (port 8000)
uv run uvicorn api.main:app --reload --port 8000

# Terminal 2 — Next.js frontend (port 3000)
cd frontend && npm install && npm run dev
```

**Alternative — Streamlit dashboard** (single process, less polished):

```bash
uv run streamlit run dashboard/app.py
```

**Alternative — CLI:**

```bash
uv run python scripts/run_pipeline.py data/samples/iris.csv
```

## Seeding the experience pool

The model_agent (SP5) will retrieve from a pool of past training experiences. Seed it with 21 public benchmark datasets:

```bash
uv run python scripts/run_benchmark.py --trials 8
```

This:
- Fetches datasets from sklearn / OpenML / yfinance / local CSVs
- Runs the full training executor on each (classification, regression, forecasting)
- For forecasting: chooses validation strategy (single_split / rolling / expanding) by deterministic policy, applies leakage-safe exogenous extension (naive_carry / ETS / AutoARIMA), records per-fold metrics
- Inserts an `ExperienceRecord` into `storage/mlops_metadata.db` for each dataset

Inspect the pool:

```bash
uv run python -c "
import sqlite3, json
from mlops_agents.config.settings import settings
conn = sqlite3.connect(settings.experience_db_path)
for r in conn.execute('SELECT dataset_name, selected_model_key, validation_score FROM experiences ORDER BY dataset_name'):
    print(r)
"
```

## Tests

```bash
uv run python -m pytest -m "not integration"   # unit (~326 tests, no LLM calls)
uv run python -m pytest                        # all (+ slow LLM-touching integration tests)
uv run ruff check . && uv run ruff format .    # lint + format
```

## Forecasting at a glance

The forecasting executor handles the case you actually care about: **one target series + many exogenous predictors**, where some exogenous values won't be known at prediction time. The user (or the upcoming SP5 LLM planner) tags each exog column with `future_availability`:

```yaml
# In task_metadata for a forecasting task:
exogenous_columns:
  - { name: holiday_flag,  future_availability: known_future }
  - { name: oil_price,     future_availability: unknown_future }
  - { name: usd_index,     future_availability: unknown_future }
expected_drift: high  # → rolling-window backtesting
```

The executor then:
1. Picks a validation strategy based on `history_length` + `expected_drift` (short history → single split; medium+drift=high → rolling 3-fold; medium+low drift → expanding 3-fold).
2. For each fold, extends `unknown_future` columns from training history via the chosen strategy (default `naive_carry`).
3. **Never** uses realized future exog values for `unknown_future` columns — this is the leakage firewall.
4. Records per-fold scores, strategies actually applied, and any extender failures in the `ExperienceRecord`.

See [docs/superpowers/specs/2026-05-11-forecasting-exogenous-leakage-safe-validation-design.md](docs/superpowers/specs/2026-05-11-forecasting-exogenous-leakage-safe-validation-design.md) for the design spec.

## Project structure

```
src/mlops_agents/   ← Python package (agents, graph, contracts, training, knowledge, experience)
api/                ← FastAPI backend (REST + SSE for the frontend)
frontend/           ← Next.js UI (pipeline, experiments, monitoring)
dashboard/          ← Streamlit alternative UI
scripts/            ← run_pipeline, run_benchmark, seed_mlflow
data/               ← benchmark CSVs + user uploads + schemas
docs/superpowers/   ← brainstorming specs and implementation plans
tests/              ← 326 unit tests (mirrors src/ layout)
```

See [STRUCTURE.md](STRUCTURE.md) for the file-by-file breakdown.

## Key documents

- [ARCHITECTURE.md](ARCHITECTURE.md) — system shape, agent graph, contracts, data flow
- [PLAN.md](PLAN.md) — sub-project status and roadmap
- [STRUCTURE.md](STRUCTURE.md) — what each file is for
- [`docs/superpowers/specs/`](docs/superpowers/specs/) — design specs (one per feature)
- [`docs/superpowers/plans/`](docs/superpowers/plans/) — implementation plans (one per feature)

## License

TFG — Universidad de Oviedo, 2026.
