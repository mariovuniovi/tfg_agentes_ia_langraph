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
- **OpenAI API** (gpt-4.1-mini) — LLM provider (GitHub Models free tier also supported, see Getting started)
- **scikit-learn, LightGBM, XGBoost, CatBoost** for tabular models
- **statsforecast + skforecast** for forecasting (AutoETS, AutoARIMA, recursive multi-series ML)
- **Optuna** for hyperparameter search
- **MLflow** for experiment tracking + model registry
- **Evidently AI** for data quality + drift reports
- **FastAPI + Next.js** for the user-facing API and UI
- **SQLite** for the experience pool (`storage/mlops_metadata.db`)

## Getting started

**Prerequisites:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) — no Python, Node, or any other tool needed.

```bash
git clone https://github.com/mariovuniovi/tfg_agentes_ia_langraph.git
cd tfg_agentes_ia_langraph
cp .env.example .env          # then edit .env and fill in OPENAI_API_KEY
docker compose up --build
```

Open **http://localhost:3000** — the pipeline UI is ready.

| Service | URL |
|---|---|
| Next.js UI | http://localhost:3000 |
| FastAPI backend | http://localhost:8000 |
| MLflow | http://localhost:5000 |

**After making code changes:**
```bash
docker compose up --build     # rebuilds only what changed, takes ~1-2 min
```

---

### Using GitHub Models (free alternative)

> **Note on model quality:** This project was developed and tested against OpenAI's API. GitHub Models offers compatible model names but is a different hosted service with a **150 requests/day rate limit** on the free tier. Agent reliability — especially the data validator and planner, which require precise instruction-following — may be lower than with OpenAI's API. Use GitHub Models to explore the system; use OpenAI's API for consistent results.

1. Create a free [GitHub Personal Access Token](https://github.com/settings/tokens) (no special scopes needed)
2. In your `.env`, make these changes:
   ```bash
   OPENAI_API_KEY=your_github_personal_access_token
   OPENAI_BASE_URL=https://models.inference.ai.azure.com
   OPENAI_MODEL_SUPERVISOR=openai/gpt-4.1-mini
   OPENAI_MODEL_DATA_VALIDATOR=openai/gpt-4.1-mini
   OPENAI_MODEL_PLANNER=openai/gpt-4.1-mini
   OPENAI_MODEL_EVALUATOR=openai/gpt-4.1-mini
   OPENAI_MODEL_DEPLOYER=openai/gpt-4.1-mini
   ```
3. Run `docker compose up --build` — no code changes needed. The OpenAI SDK reads `OPENAI_BASE_URL` automatically.

---

### Local development (without Docker)

```bash
uv sync                                    # install all Python dependencies
cp .env.example .env                       # fill in OPENAI_API_KEY
```

Two terminals:
```bash
# Terminal 1 — FastAPI backend (port 8000)
uv run uvicorn api.main:app --reload --port 8000

# Terminal 2 — Next.js frontend (port 3000)
cd frontend && npm install && npm run dev
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
