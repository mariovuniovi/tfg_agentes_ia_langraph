# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview
Multi-agent MLOps system on a custom LangGraph StateGraph with a deterministic
workflow_controller router. Each pipeline stage is a domain package consumed by
the graph through a thin node wrapper. Two LLM agent stages (data validation,
planning) plus an LLM audit node (report writer); training, evaluation and
deployment are deterministic. Built with UV, FastAPI backend, MLflow, Evidently
AI, and the OpenAI API.

## Commands
```
uv sync                                          # Install all dependencies
uv run pytest                                    # Run all tests
uv run pytest -m "not integration"               # Unit tests only (no LLM calls)
uv run pytest tests/test_tools/test_data_tools.py  # Run a single test file
uv run ruff check . && uv run ruff format .      # Lint & format
uv run mypy src/                                 # Type check
uv run python scripts/run_pipeline.py            # Run full MLOps pipeline (CLI + HITL)
uv run python scripts/seed_mlflow.py             # Seed MLflow with sample runs
docker compose up                                # Full stack: MLflow (5000) + app
```

## Architecture
```
src/mlops_agents/
├── graphs/           mlops_graph.py (topology + thin node wrappers), cli.py (CLI + HITL),
│                     workflow_controller.py (deterministic router), approval_nodes.py
│                     (HITL gates), taxonomy.py (node categories)
├── data_validation/  node.py + agent.py + context.py + schema_contract.py (LLM stage)
├── planning/         node.py + agent.py + tools.py + validation.py + context.py (LLM stage)
├── training/         executor.py + profiler, splitter, validation_policy, … (deterministic)
├── evaluation/       promotion.py (deterministic decision) + report_writer.py (LLM audit)
├── deployment/       deployer.py (deterministic MLflow registration)
├── experience/       pool.py + retrieval.py (experience store for the planner)
├── knowledge/        reader.py (static ML rule base)
├── models/           loader.py (model registry) + factories.py + search_spaces.py
├── forecasting/      seasonality.py (season-length policy, import-cycle-free)
├── contracts/        Pydantic contracts (node→state updates, plans, profiles)
├── state/            agent_state.py (AgentState TypedDict)
├── tools/            data_tools.py, join_discovery_tools.py, mlflow_tools.py
├── prompts/          YAML templates per agent + loader.py
├── config/           settings.py (Pydantic Settings reads .env) + constants.py
├── observability/    pricing.py (token→cost)
└── utils/            llm.py (LLM factory), logging.py
api/                  FastAPI backend (imports graph from mlops_agents)
frontend/             Next.js UI — consumes the run WebSocket event stream (typed PipelineEvent)
```

### Execution flow
```
.env (OPENAI_API_KEY)
    ↓
config/settings.py      ← reads env vars
    ↓
utils/llm.py            ← creates ChatOpenAI per agent (model from prompt YAML)
    ↓
each LLM stage builds and caches its own agent
(data_validation/agent.py, planning/agent.py, evaluation/report_writer.py)
    ↓
graphs/mlops_graph.py   ← StateGraph: START → workflow_controller → stage nodes
                          → workflow_controller → … → END
    ↓
api/ (FastAPI)  OR  scripts/run_pipeline.py → graphs/cli.py  ← CLI
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
- Committing is allowed; never merge or push (no `git merge`, `git push`) unless explicitly asked
- **Never add Claude as co-author** in any commit or merge message — do not include `Co-Authored-By: Claude` or any Anthropic co-author trailer

## Agent Design Principles
- **Deterministic first**: data loading, training loop, metric computation = pure Python nodes
- **Agents only for**: interpreting failures, reasoning about strategy, natural-language reports
- **HITL at**: deployment gate — `interrupt()` in `deployer_node` before MLflow Model Registry promotion
- **HITL rule**: all code before `interrupt()` must be idempotent (the node restarts on resume)
- **`interrupt()` placement**: only in graph nodes, never inside react agent tools
- **Routing**: deterministic `workflow_controller` (no LLM) reads state and returns `Command(goto=...)` — every decision is logged
- **Node names** in the graph: `data_validator`, `planner`, `executor`, `evaluation`, `report_writer`, `deployer` + gates `dataset_approval`, `deployment_approval`
- **Agent ownership**: each LLM stage builds and caches its own agent (`data_validation/agent.py: get_data_agent`, `planning/agent.py: build_planner_agent` per-run, `evaluation/report_writer.py: get_report_writer_agent`) — there is no central registry

## Testing Conventions
- Unit tests must NOT make real LLM calls — mock the LLM with `unittest.mock`
- Integration tests must be marked `@pytest.mark.integration` and `@pytest.mark.slow`
- Data tools are deterministic — test them with real pandas DataFrames (no mocks needed)
- Check `tests/conftest.py` for existing fixtures before creating new ones

## Key Files
- `src/mlops_agents/state/agent_state.py` — shared state schema (read before editing agents)
- `src/mlops_agents/graphs/mlops_graph.py` — graph topology (the source of truth for flow)
- `src/mlops_agents/config/settings.py` — all configuration via env vars
- `src/mlops_agents/graphs/workflow_controller.py` — deterministic router (routing rules live here)
- `tests/conftest.py` — shared fixtures (check before creating new ones)
