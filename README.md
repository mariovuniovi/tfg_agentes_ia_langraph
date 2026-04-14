# Multi-Agent MLOps System — TFG

Multi-agent MLOps system using a custom LangGraph supervisor pattern.
Four specialist agents (data validation, training, evaluation, deployment)
orchestrated by a supervisor with structured output routing.

Built with Python 3.12, UV, LangGraph, LangChain, GitHub Models (GPT-4.1-mini),
MLflow, Evidently AI, and Streamlit.

## Quick start

```bash
cp .env.example .env        # add your GITHUB_TOKEN and GITHUB_MODEL
uv sync                     # install dependencies
uv run streamlit run dashboard/app.py   # launch dashboard
uv run python scripts/run_pipeline.py  # run pipeline from CLI
```

See [PLAN.md](PLAN.md) for the work plan and [STRUCTURE.md](STRUCTURE.md) for the full file reference.
