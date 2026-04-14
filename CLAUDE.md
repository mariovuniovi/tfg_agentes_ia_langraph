# Multi-Agent MLOps System — Bachelor's Thesis

## Overview
Multi-agent MLOps system using a custom LangGraph supervisor pattern.
Four specialist agents (data validation, training, evaluation, deployment)
orchestrated by a supervisor with structured output routing.
Built with UV, Streamlit dashboard, MLflow, Evidently AI, and GitHub Models (gpt-4.1-mini).

## Commands
```
uv sync                                          # Install all dependencies
uv run pytest                                    # Run all tests
uv run pytest -m "not integration"               # Unit tests only (no LLM calls)
uv run ruff check . && uv run ruff format .      # Lint & format
uv run mypy src/                                 # Type check
uv run streamlit run dashboard/app.py            # Start dashboard (port 8501)
uv run python scripts/run_pipeline.py            # Run full MLOps pipeline
uv run python scripts/seed_mlflow.py             # Seed MLflow with sample runs
docker compose up                                # Full stack: MLflow (5000) + app (8501)
```

## Architecture
```
src/mlops_agents/
├── agents/      supervisor.py + data_agent.py, training_agent.py,
│                evaluation_agent.py, deployment_agent.py, registry.py
├── graphs/      mlops_graph.py (main StateGraph) + subgraphs/
├── state/       agent_state.py (TypedDict) + schemas.py (Pydantic)
├── tools/       data_tools.py, training_tools.py, mlflow_tools.py, evidently_tools.py
├── prompts/     YAML templates per agent + loader.py
├── mcp_servers/ mlflow_server.py, data_server.py
├── config/      settings.py (Pydantic Settings reads .env) + constants.py
└── utils/       llm.py (LLM factory), logging.py, runners.py
dashboard/       Streamlit multi-page app (imports from mlops_agents)
```
## Core Principles

### 1. Think Before Coding
Don't assume. Don't hide confusion. Surface tradeoffs.

LLMs often pick an interpretation silently and run with it. This principle forces explicit reasoning:

- State assumptions explicitly — If uncertain, ask rather than guess
- Present multiple interpretations — Don't pick silently when ambiguity exists
- Push back when warranted — If a simpler approach exists, say so
- Stop when confused — Name what's unclear and ask for clarification
### 2. Simplicity First
Minimum code that solves the problem. Nothing speculative.

Combat the tendency toward overengineering:

- No features beyond what was asked
- No abstractions for single-use code
- No "flexibility" or "configurability" that wasn't requested
- No error handling for impossible scenarios
- If 200 lines could be 50, rewrite it

The test: Would a senior engineer say this is overcomplicated? If yes, simplify.

### 3. Surgical Changes
Touch only what you must. Clean up only your own mess.

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting
- Don't refactor things that aren't broken
- Match existing style, even if you'd do it differently
- If you notice unrelated dead code, mention it — don't delete it

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused
- Don't remove pre-existing dead code unless asked

The test: Every changed line should trace directly to the user's request.
## Conventions
- Type hints everywhere — strict mypy
- State must be TypedDict (LangGraph requirement)
- Pydantic BaseModel for tool I/O schemas
- Use loguru for all logging — never print()
- Environment config via pydantic-settings — never hardcode tokens
- Agent tools return structured dicts, not raw strings
- Nodes return partial state dicts — never mutate state in-place
- Use `uv run` prefix — never activate venv manually
- **Agent creation**: use `from langchain.agents import create_agent` with `system_prompt=` parameter — `create_react_agent` from `langgraph.prebuilt` is deprecated and removed
- **Evidently 0.7.21 API**: use `DataSummaryPreset` (not `DataQualityPreset`), `DataDriftPreset()` (no `method=` arg), and `result.dump_dict()` (not `as_dict()` or `load_dict()`)

## Agent Design Principles
- **Deterministic first**: data loading, training loop, metric computation = pure Python nodes
- **Agents only for**: interpreting failures, reasoning about strategy, natural-language reports
- **HITL at**: deployment gate (interrupt() before MLflow Model Registry promotion)
- **Supervisor routing**: structured RouterOutput (next + reasoning) — every decision is logged
- **Rate limits**: 150 RPD per model on GitHub Models free tier — use different models per agent

## Key Files
- `src/mlops_agents/state/agent_state.py` — shared state schema (read before editing agents)
- `src/mlops_agents/graphs/mlops_graph.py` — graph topology (the source of truth for flow)
- `src/mlops_agents/config/settings.py` — all configuration via env vars
- `tests/conftest.py` — shared fixtures (check before creating new ones)
