# Multi-agent LangGraph project structure with UV

**The optimal setup combines UV's `--package` src layout, PEP 735 dependency groups, a domain-organized agent architecture, and multi-stage Docker builds.** This configuration gives you reproducible builds via `uv.lock`, clean separation between agents/tools/graphs/UI, and fast Docker rebuilds through layer caching. The following is a complete, copy-paste-ready reference for a multi-agent MLOps system using LangGraph, Streamlit, MCP, MLflow, and GitHub Models.

---

## The complete directory tree

This structure synthesizes LangGraph's official application layout, the agent-service-toolkit pattern (the most mature LangGraph reference project at 4.1k GitHub stars), and UV's `--package` conventions for src-layout projects.

```
mlops-multi-agent/
├── pyproject.toml                    # UV project config (all deps + tool config)
├── uv.lock                          # UV lockfile (commit to git)
├── .python-version                  # Python version pin (e.g., 3.12)
├── .env.example                     # Template for environment variables
├── .gitignore
├── README.md
├── Dockerfile
├── docker-compose.yml
├── langgraph.json                   # LangGraph deployment config
├── CLAUDE.md                        # Claude Code project instructions
│
├── .claude/                         # Claude Code configuration
│   ├── settings.json                # Permissions (committed)
│   ├── .mcp.json                    # MCP server configs for Claude
│   ├── rules/
│   │   ├── langgraph-agents.md      # Path-scoped rules for agent code
│   │   └── testing.md               # Path-scoped rules for tests
│   ├── commands/
│   │   ├── test-agent.md            # /project:test-agent <name>
│   │   └── run-pipeline.md          # /project:run-pipeline
│   └── skills/
│       └── create-agent/
│           └── SKILL.md             # Auto-invoked when creating agents
│
├── src/
│   └── mlops_agents/                # Main package (matches project name)
│       ├── __init__.py
│       ├── agents/                  # 🤖 Agent definitions
│       │   ├── __init__.py
│       │   ├── supervisor.py        # Supervisor/orchestrator agent
│       │   ├── data_agent.py        # Data ingestion & validation
│       │   ├── training_agent.py    # Model training & tuning
│       │   ├── evaluation_agent.py  # Model evaluation & comparison
│       │   ├── monitoring_agent.py  # Drift detection & alerting
│       │   └── registry.py          # Agent registry & factory
│       │
│       ├── graphs/                  # 📊 LangGraph graph definitions
│       │   ├── __init__.py
│       │   ├── mlops_graph.py       # Main compiled supervisor graph
│       │   └── subgraphs/
│       │       ├── __init__.py
│       │       └── training_flow.py # Sub-workflow for training loop
│       │
│       ├── state/                   # 📦 State & schema definitions
│       │   ├── __init__.py
│       │   ├── agent_state.py       # Main AgentState TypedDict
│       │   └── schemas.py           # Pydantic models for tool I/O
│       │
│       ├── tools/                   # 🔧 LangChain @tool functions
│       │   ├── __init__.py
│       │   ├── mlflow_tools.py      # Experiment tracking tools
│       │   ├── data_tools.py        # Dataset loading/validation
│       │   ├── training_tools.py    # scikit-learn training tools
│       │   └── evidently_tools.py   # Drift monitoring tools
│       │
│       ├── prompts/                 # 💬 Prompt templates (YAML)
│       │   ├── __init__.py
│       │   ├── supervisor.yaml
│       │   ├── data_agent.yaml
│       │   ├── training_agent.yaml
│       │   ├── evaluation_agent.yaml
│       │   ├── monitoring_agent.yaml
│       │   └── loader.py            # Prompt loading utility
│       │
│       ├── mcp_servers/             # 🔌 MCP server implementations
│       │   ├── __init__.py
│       │   ├── mlflow_server.py     # MLflow tools via MCP
│       │   └── data_server.py       # Data access via MCP
│       │
│       ├── config/                  # ⚙️ Configuration
│       │   ├── __init__.py
│       │   ├── settings.py          # Pydantic Settings (env-based)
│       │   └── constants.py         # App constants
│       │
│       └── utils/                   # 🛠️ Shared utilities
│           ├── __init__.py
│           ├── llm.py               # LLM initialization helpers
│           └── logging.py           # Logging setup
│
├── dashboard/                       # 📊 Streamlit dashboard
│   ├── app.py                       # Main entry point
│   ├── pages/
│   │   ├── 01_pipeline.py           # Pipeline overview
│   │   ├── 02_experiments.py        # MLflow experiment browser
│   │   ├── 03_monitoring.py         # Drift/performance monitoring
│   │   └── 04_chat.py              # Chat interface to agents
│   └── components/
│       ├── chat_interface.py
│       └── metrics_display.py
│
├── data/                            # 📁 Sample datasets & schemas
│   ├── samples/
│   │   └── iris.csv
│   └── schemas/
│       └── dataset_schema.json
│
├── tests/                           # 🧪 Test suite (mirrors src/)
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_agents/
│   │   ├── test_supervisor.py
│   │   └── test_data_agent.py
│   ├── test_tools/
│   │   └── test_mlflow_tools.py
│   ├── test_graphs/
│   │   └── test_mlops_graph.py
│   └── test_integration/
│       └── test_end_to_end.py
│
├── notebooks/                       # 📓 Exploration notebooks
│   └── exploration.ipynb
│
├── docs/                            # 📚 Project documentation
│   ├── architecture.md
│   └── setup.md
│
└── scripts/                         # 🔧 Utility scripts
    ├── seed_mlflow.py
    └── run_pipeline.py
```

**Key architectural decisions** behind this layout: agents are separated from graphs (topology vs. compute), MCP servers live inside the package since they share dependencies, Streamlit is a top-level directory because it's a standalone UI that imports from the package, and prompts use YAML for clean version control diffs and non-code editability.

---

## The complete pyproject.toml

Initialize with `uv init --package mlops-multi-agent` to get the src layout with build system, then replace the generated file with this:

```toml
[project]
name = "mlops-multi-agent"
version = "0.1.0"
description = "Multi-agent MLOps system using LangGraph — Bachelor's thesis"
readme = "README.md"
requires-python = ">=3.12"
license = { text = "MIT" }
authors = [{ name = "Your Name", email = "you@example.com" }]

dependencies = [
    # LangGraph / LangChain ecosystem
    "langgraph>=0.4",
    "langgraph-supervisor>=0.0.30",
    "langchain>=0.3.25",
    "langchain-core>=0.3.25",
    "langchain-openai>=0.3.17",

    # MCP (Model Context Protocol)
    "mcp[cli]>=1.2",
    "langchain-mcp-adapters>=0.2.2",

    # Dashboard & MLOps
    "streamlit>=1.40",
    "mlflow>=2.18",
    "evidently>=0.5",

    # ML & Data
    "scikit-learn>=1.6",
    "pandas>=2.2",
    "numpy>=2.0",

    # Configuration & Validation
    "pydantic>=2.10",
    "pydantic-settings>=2.7",
    "python-dotenv>=1.0",

    # HTTP & Logging
    "httpx>=0.28",
    "loguru>=0.7",
]

[project.scripts]
mlops-dashboard = "mlops_agents.utils.runners:run_dashboard"
mlops-pipeline = "mlops_agents.graphs.mlops_graph:main"

[dependency-groups]
dev = [
    "pytest>=8.3",
    "pytest-asyncio>=0.24",
    "ruff>=0.9",
    "mypy>=1.13",
    "pre-commit>=4.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

# =============================================================================
# Tool Configuration
# =============================================================================

[tool.ruff]
target-version = "py312"
line-length = 120

[tool.ruff.lint]
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "F",      # pyflakes
    "I",      # isort
    "N",      # pep8-naming
    "UP",     # pyupgrade
    "B",      # flake8-bugbear
    "C4",     # flake8-comprehensions
    "SIM",    # flake8-simplify
    "ASYNC",  # flake8-async (important for LangGraph)
    "S",      # flake8-bandit (security)
    "T20",    # flake8-print
]
ignore = [
    "E501",   # line-too-long (handled by formatter)
    "S101",   # assert used (needed for tests)
    "B008",   # function-call-in-default-argument (Pydantic pattern)
]

[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = ["S101", "PLR0913"]

[tool.ruff.lint.isort]
known-first-party = ["mlops_agents"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = ["-v", "--strict-markers", "--tb=short"]
markers = [
    "slow: marks tests as slow",
    "integration: marks integration tests",
]

[tool.mypy]
python_version = "3.12"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[[tool.mypy.overrides]]
module = [
    "streamlit.*", "mlflow.*", "evidently.*",
    "langchain.*", "langchain_core.*", "langchain_openai.*",
    "langgraph.*", "langchain_mcp_adapters.*", "mcp.*", "loguru.*",
]
ignore_missing_imports = true
```

**UV uses PEP 735 `[dependency-groups]`** instead of the older `[tool.uv.dev-dependencies]` approach. The `dev` group is special — it syncs by default with `uv sync`. Add custom groups (e.g., `docs`, `test`) for finer-grained control. UV's lockfile (`uv.lock`) pins exact versions across all platforms, so the `>=` constraints in `pyproject.toml` define floors while `uv.lock` ensures reproducibility.

---

## UV workflow and essential commands

```bash
# Project setup
uv init --package mlops-multi-agent    # Create src-layout project
uv python pin 3.12                     # Pin Python version
uv sync                                # Install all deps + dev group
uv sync --no-dev                       # Production deps only
uv sync --group docs                   # Include a custom group

# Dependency management
uv add langgraph streamlit mlflow      # Add production dependencies
uv add --dev pytest ruff mypy          # Add to dev group
uv add --group test pytest-cov         # Add to custom group

# Lock file management (commit uv.lock to git!)
uv lock                                # Update lockfile
uv lock --check                        # CI: verify lockfile is current
uv lock --upgrade                      # Upgrade all dependencies
uv lock --upgrade-package langgraph    # Upgrade single package

# Running code (preferred over venv activation)
uv run streamlit run dashboard/app.py  # Run Streamlit
uv run pytest                          # Run tests
uv run ruff check .                    # Lint
uv run mypy src/                       # Type check
uv run python scripts/seed_mlflow.py   # Run any script

# One-off tool execution (no install)
uvx ruff check .                       # Run ruff without installing
```

**The `uv run` command is the recommended way to execute everything** — it automatically verifies the lockfile, syncs the environment, and runs the command. You never need to manually activate the virtual environment. UV creates `.venv/` in the project root automatically.

---

## The Dockerfile for UV projects

This multi-stage build follows Astral's official Docker guide. Dependencies install in a cached layer separate from source code, so code changes don't trigger a full reinstall.

```dockerfile
# =============================================================================
# Stage 1: Build — install deps with UV
# =============================================================================
FROM python:3.12-slim-bookworm AS build

COPY --from=ghcr.io/astral-sh/uv:0.10 /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Layer 1: Install dependencies only (cached unless lock changes)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Layer 2: Copy source and install the project itself
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# =============================================================================
# Stage 2: Runtime — minimal image without UV or build tools
# =============================================================================
FROM python:3.12-slim-bookworm AS runtime

RUN groupadd -g 1001 app && useradd -u 1001 -g app -m app
WORKDIR /app

COPY --from=build --chown=app:app /app /app
ENV PATH="/app/.venv/bin:$PATH"

USER app
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
```

**Three critical UV Docker patterns**: first, `COPY --from=ghcr.io/astral-sh/uv:0.10` copies the UV binary without curl or pip; second, the two-step sync (`--no-install-project` then full sync) separates dependency layers from source layers; third, `--locked` fails the build if `uv.lock` is outdated, guaranteeing reproducibility.

---

## The docker-compose.yml with MLflow and Streamlit

```yaml
services:
  # =========================================================================
  # MLflow Tracking Server (SQLite backend — simple for thesis)
  # =========================================================================
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.18.0
    container_name: mlops-mlflow
    restart: unless-stopped
    command: >
      mlflow server
        --host 0.0.0.0
        --port 5000
        --backend-store-uri sqlite:///mlflow/mlflow.db
        --artifacts-destination /mlflow/artifacts
        --serve-artifacts
    volumes:
      - mlflow_data:/mlflow
    ports:
      - "5000:5000"
    networks:
      - mlops-net

  # =========================================================================
  # Streamlit Dashboard + LangGraph Agents
  # =========================================================================
  app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: mlops-app
    restart: unless-stopped
    depends_on:
      - mlflow
    env_file:
      - .env
    environment:
      MLFLOW_TRACKING_URI: http://mlflow:5000
      PYTHONUNBUFFERED: "1"
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
    networks:
      - mlops-net
    develop:
      watch:
        - action: sync
          path: ./src
          target: /app/src
        - action: sync
          path: ./dashboard
          target: /app/dashboard
        - action: rebuild
          path: ./pyproject.toml

volumes:
  mlflow_data:

networks:
  mlops-net:
    driver: bridge
```

The `develop.watch` section enables **Docker Compose Watch** — source code changes sync instantly into the container without rebuilding, while `pyproject.toml` changes trigger a full rebuild. For production or a more robust setup, swap the SQLite backend for PostgreSQL by adding a `postgres` service and changing the backend store URI.

---

## The .env.example file

```bash
# LLM Configuration (GitHub Models)
GITHUB_TOKEN=your_github_pat_here
OPENAI_API_BASE=https://models.inference.ai.azure.com
MODEL_NAME=gpt-4.1-mini

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# LangSmith (optional tracing)
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=mlops-multi-agent
```

---

## Configuration management with Pydantic Settings

```python
# src/mlops_agents/config/settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LLM
    github_token: str
    openai_api_base: str = "https://models.inference.ai.azure.com"
    model_name: str = "gpt-4.1-mini"

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"

    # Evidently
    evidently_workspace: str = "./evidently_workspace"

    # LangSmith
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

settings = Settings()
```

---

## Core LangGraph patterns for the supervisor graph

```python
# src/mlops_agents/state/agent_state.py
import operator
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    current_dataset: str | None
    experiment_id: str | None
    model_metrics: dict | None
```

```python
# src/mlops_agents/graphs/mlops_graph.py
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from mlops_agents.config.settings import settings
from mlops_agents.tools.data_tools import load_dataset, validate_data
from mlops_agents.tools.training_tools import train_model
from mlops_agents.tools.mlflow_tools import log_experiment
from mlops_agents.prompts.loader import get_prompt

model = ChatOpenAI(
    model=settings.model_name,
    api_key=settings.github_token,
    base_url=settings.openai_api_base,
)

data_agent = create_react_agent(
    model=model,
    tools=[load_dataset, validate_data],
    name="data_agent",
    prompt=get_prompt("data_agent").template,
)

training_agent = create_react_agent(
    model=model,
    tools=[train_model, log_experiment],
    name="training_agent",
    prompt=get_prompt("training_agent").template,
)

graph = create_supervisor(
    [data_agent, training_agent],
    model=model,
    prompt=get_prompt("supervisor").template,
).compile()
```

```python
# src/mlops_agents/prompts/loader.py
from pathlib import Path
from langchain_core.prompts import load_prompt

PROMPTS_DIR = Path(__file__).parent

def get_prompt(name: str):
    return load_prompt(PROMPTS_DIR / f"{name}.yaml")
```

```yaml
# src/mlops_agents/prompts/supervisor.yaml
_type: "prompt"
template: |
  You are the MLOps Supervisor coordinating a team of specialized agents.

  Your team:
  - data_agent: Data ingestion, validation, and preprocessing
  - training_agent: Model training, hyperparameter tuning, MLflow logging
  - evaluation_agent: Model evaluation and comparison
  - monitoring_agent: Data drift detection and alerting

  Delegate tasks to the appropriate agent. Synthesize results when done.
input_variables: []
```

**YAML prompts are the recommended approach** — they provide clean version control diffs, separation of concerns between prompts and code, and native LangChain support via `load_prompt()`.

---

## The langgraph.json deployment config

```json
{
  "dependencies": ["./"],
  "graphs": {
    "mlops_supervisor": "./src/mlops_agents/graphs/mlops_graph.py:graph"
  },
  "env": "./.env"
}
```

---

## CLAUDE.md for the project

This file loads into every Claude Code session automatically. Keep it under **150 lines** — instruction-following degrades as length increases.

```markdown
# Multi-Agent MLOps System — Bachelor's Thesis

## Overview
Multi-agent MLOps system using LangGraph supervisor pattern. Agents handle data
ingestion, model training, evaluation, and monitoring. Built with UV, Streamlit,
MLflow, MCP servers, and GitHub Models (gpt-4.1-mini).

## Commands
uv sync                                         # Install all dependencies
uv run pytest                                    # Run tests
uv run pytest -m "not integration"               # Unit tests only
uv run ruff check . && uv run ruff format .      # Lint & format
uv run mypy src/                                 # Type check
uv run streamlit run dashboard/app.py            # Start dashboard
docker compose up                                # Full stack (MLflow + app)

## Architecture
src/mlops_agents/
├── agents/          Agent definitions (one per file, supervisor.py is entry)
├── graphs/          LangGraph graph topology (mlops_graph.py = main graph)
├── state/           TypedDict state + Pydantic schemas
├── tools/           @tool functions grouped by domain
├── prompts/         YAML prompt templates + loader.py
├── mcp_servers/     FastMCP server implementations
├── config/          Pydantic Settings (reads .env)
└── utils/           Shared helpers (LLM init, logging)
dashboard/           Streamlit multi-page app (imports from mlops_agents)

## Conventions
- Type hints everywhere — strict mypy
- State must be TypedDict (LangGraph requirement)
- Pydantic BaseModel for configs and data schemas
- Use loguru for logging, never print()
- Environment config via pydantic-settings, never hardcode
- Google-style docstrings on all public functions
- Agent tools return structured dicts, not raw strings
- MCP servers use FastMCP from mcp.server.fastmcp
- Always use `uv run` prefix — never activate venv manually

## Agent Patterns
- Each agent: function taking State → returning dict of updated keys
- Graph edges use conditional routing based on state["next"]
- Tools defined in src/mlops_agents/tools/, registered per agent
- Agent communication only through shared LangGraph state
- Use create_react_agent for tool-using agents
- Use create_supervisor from langgraph-supervisor for orchestration

## Important
- Never commit .env files — use .env.example as template
- MLflow URI set via MLFLOW_TRACKING_URI env var
- MCP servers run via stdio transport as separate processes
- Check conftest.py fixtures before creating new test fixtures
- Streamlit state uses st.session_state exclusively
```

---

## The .claude/ directory configuration

### .claude/settings.json
```json
{
  "permissions": {
    "allow": [
      "Bash(uv run *)",
      "Bash(uv sync)",
      "Bash(git status)",
      "Bash(git diff *)",
      "Bash(git log *)",
      "Read", "Write", "Edit", "Glob", "Grep"
    ],
    "deny": [
      "Bash(rm -rf *)",
      "Bash(pip install *)",
      "Read(.env)",
      "Read(.env.*)"
    ]
  }
}
```

### .claude/.mcp.json
```json
{
  "mcpServers": {
    "mlflow": {
      "command": "uv",
      "args": ["run", "python", "-m", "mlops_agents.mcp_servers.mlflow_server"],
      "env": { "MLFLOW_TRACKING_URI": "${MLFLOW_TRACKING_URI}" }
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": { "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}" }
    }
  }
}
```

### .claude/rules/langgraph-agents.md
```markdown
---
paths:
  - "src/mlops_agents/agents/**/*.py"
  - "src/mlops_agents/graphs/**/*.py"
---
# LangGraph Agent Rules
- Agent function signature: `def agent_name(state: AgentState) -> dict`
- Return only updated keys, never the full state
- Use Command(goto="next_node") for dynamic routing
- Handle errors by setting state["error"] field
- Import tools from mlops_agents.tools, register with create_react_agent
```

### .claude/commands/test-agent.md
```markdown
---
description: Run tests for a specific agent module
argument-hint: [agent-name e.g. data_agent]
---
!`uv run pytest tests/ -k "$ARGUMENTS" -v --tb=short`

If tests fail, analyze failures and suggest fixes.
If no tests exist, create them in the appropriate test file.
```

### .claude/skills/create-agent/SKILL.md
```markdown
---
name: create-agent
description: Create a new LangGraph agent. Use when adding agents to the pipeline.
allowed-tools: Read, Write, Bash, Grep, Glob
---
1. Read src/mlops_agents/state/agent_state.py for current state schema
2. Read an existing agent as template (e.g., src/mlops_agents/agents/data_agent.py)
3. Create new agent file in src/mlops_agents/agents/
4. Add tools in src/mlops_agents/tools/ if needed
5. Create YAML prompt in src/mlops_agents/prompts/
6. Register in src/mlops_agents/graphs/mlops_graph.py
7. Create test in tests/test_agents/
```

---

## The .gitignore

```gitignore
# UV / Python
.venv/
__pycache__/
*.pyc
*.egg-info/
dist/
build/

# Environment
.env
.env.*
!.env.example

# IDE
.vscode/
.idea/

# Tools
.mypy_cache/
.pytest_cache/
.ruff_cache/

# MLflow local
mlruns/
mlartifacts/

# Claude Code
CLAUDE.local.md
.claude/settings.local.json

# Data (large files)
data/raw/
*.parquet
```

**Do not ignore `uv.lock`** — Astral's official guidance is to commit it to version control. It's a human-readable TOML file that captures exact dependency versions across all platforms, ensuring every developer and CI run uses identical packages.

---

## Conclusion

This setup gives a bachelor's thesis project a production-grade foundation without unnecessary complexity. The **src-layout with `uv init --package`** ensures clean imports and test isolation. **PEP 735 dependency groups** replace the older `[tool.uv.dev-dependencies]` approach with a standard that other tools will adopt. The **two-step Docker build** (dependencies first, source second) means code-only changes rebuild in seconds rather than minutes. YAML prompts, Pydantic Settings, and the domain-organized agent structure all follow patterns from the most mature LangGraph projects in production today. The CLAUDE.md keeps Claude Code effective by staying under 150 lines with only universally relevant instructions — task-specific guidance lives in `.claude/rules/` with path-scoped frontmatter so it activates only when needed.