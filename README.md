# Multi-Agent MLOps System — TFG

Multi-agent MLOps system using a custom LangGraph supervisor pattern.
Four specialist agents (data validation, training, evaluation, deployment)
orchestrated by a supervisor with structured output routing.

Built with Python 3.12, UV, LangGraph, LangChain, GitHub Models (GPT-4.1-mini),
MLflow, Evidently AI, and Streamlit.

## Setup

The app has two processes that must run in separate terminals:

```bash
# Terminal 1 — FastAPI backend (MLOps pipeline, agents, MLflow)
uv run uvicorn api.main:app --reload --port 8000

# Terminal 2 — Next.js frontend (UI at http://localhost:3000)
cd frontend && npm run dev
```

## Quick start

```bash
cp .env.example .env        # add your GITHUB_TOKEN and GITHUB_MODEL
uv sync                     # install dependencies
uv run streamlit run dashboard/app.py   # launch dashboard
uv run python scripts/run_pipeline.py  # run pipeline from CLI
```

See [PLAN.md](PLAN.md) for the work plan and [STRUCTURE.md](STRUCTURE.md) for the full file reference.
