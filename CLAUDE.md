# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview
Multi-agent MLOps system using a custom LangGraph supervisor pattern.
Four specialist agents (data validation, training, evaluation, deployment)
orchestrated by a supervisor with structured output routing.
Built with UV, Streamlit dashboard, MLflow, Evidently AI, and OpenAI API (gpt-4.1-mini).

## Commands
```
uv sync                                          # Install all dependencies
uv run pytest                                    # Run all tests
uv run pytest -m "not integration"               # Unit tests only (no LLM calls)
uv run pytest tests/test_tools/test_data_tools.py  # Run a single test file
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
├── config/      settings.py (Pydantic Settings reads .env) + constants.py
└── utils/       llm.py (LLM factory), logging.py, runners.py
dashboard/       Streamlit multi-page app (imports from mlops_agents)
```

### Execution flow
```
.env (GITHUB_TOKEN, GITHUB_MODEL)
    ↓
config/settings.py      ← reads env vars
    ↓
utils/llm.py            ← creates ChatOpenAI pointing to GitHub Models
    ↓
agents/registry.py      ← lazy-builds the 4 agents with @lru_cache
    ↓
graphs/mlops_graph.py   ← StateGraph: START → supervisor → agents → supervisor → END
    ↓
dashboard/app.py        ← Streamlit UI  OR  scripts/run_pipeline.py  ← CLI
```

## Core Principles

### 1. Think Before Coding
Don't assume. Don't hide confusion. Surface tradeoffs.

- State assumptions explicitly — if uncertain, ask rather than guess
- Present multiple interpretations — don't pick silently when ambiguity exists
- Push back when warranted — if a simpler approach exists, say so
- Stop when confused — name what's unclear and ask for clarification

### 2. Simplicity First
Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked
- No abstractions for single-use code
- No "flexibility" or "configurability" that wasn't requested
- No error handling for impossible scenarios
- If 200 lines could be 50, rewrite it

### 3. Surgical Changes
Touch only what you must. Clean up only your own mess.

- Don't "improve" adjacent code, comments, or formatting
- Don't refactor things that aren't broken
- Match existing style, even if you'd do it differently
- Remove imports/variables/functions that YOUR changes made unused
- Don't remove pre-existing dead code unless asked

## Conventions
- Type hints everywhere — strict mypy
- State must be TypedDict (LangGraph requirement)
- Pydantic BaseModel for tool I/O schemas
- Use loguru for all logging — never `print()`
- Environment config via pydantic-settings — never hardcode tokens
- Agent tools return structured dicts, not raw strings
- Nodes return partial state dicts — never mutate state in-place
- Use `uv run` prefix — never activate venv manually
- **Agent creation**: use `from langchain.agents import create_agent` with `system_prompt=` parameter — `create_react_agent` from `langgraph.prebuilt` is deprecated and removed
- **Evidently 0.7.21 API**: use `DataSummaryPreset` (not `DataQualityPreset`), `DataDriftPreset()` (no `method=` arg), and `result.dump_dict()` (not `as_dict()` or `load_dict()`)
- Never commit changes
- **Never add Claude as co-author** in any commit or merge message — do not include `Co-Authored-By: Claude` or any Anthropic co-author trailer

## Agent Design Principles
- **Deterministic first**: data loading, training loop, metric computation = pure Python nodes
- **Agents only for**: interpreting failures, reasoning about strategy, natural-language reports
- **HITL at**: deployment gate — `interrupt()` in `deployer_node` before MLflow Model Registry promotion
- **HITL rule**: all code before `interrupt()` must be idempotent (the node restarts on resume)
- **`interrupt()` placement**: only in graph nodes, never inside react agent tools
- **Supervisor routing**: structured `RouterOutput` (next + reasoning) — every decision is logged
- **Rate limits**: 150 RPD per model on GitHub Models free tier — use different models per agent
- **Agent node names** in the graph: `data_validator`, `trainer`, `evaluator`, `deployer` (not the builder names)
- **Graceful recursion exit**: check `remaining_steps <= 2` in supervisor and force `Command(goto=END)`

## Testing Conventions
- Unit tests must NOT make real LLM calls — mock the LLM with `unittest.mock`
- Integration tests must be marked `@pytest.mark.integration` and `@pytest.mark.slow`
- Data tools are deterministic — test them with real pandas DataFrames (no mocks needed)
- Check `tests/conftest.py` for existing fixtures before creating new ones

## Key Files
- `src/mlops_agents/state/agent_state.py` — shared state schema (read before editing agents)
- `src/mlops_agents/graphs/mlops_graph.py` — graph topology (the source of truth for flow)
- `src/mlops_agents/config/settings.py` — all configuration via env vars
- `src/mlops_agents/agents/registry.py` — lazy agent factory (`get_agent(name)`)
- `tests/conftest.py` — shared fixtures (check before creating new ones)
