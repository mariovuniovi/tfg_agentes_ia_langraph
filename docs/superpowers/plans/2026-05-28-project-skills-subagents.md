# Project Skills & Subagents Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 5 skills, 2 subagents, infrastructure prerequisites, and 2 missing agent integration tests to the project to reduce repetitive manual work and enable professional thesis documentation generation.

**Architecture:** Prerequisites first (Loguru file sink, pool snapshot script, agent integration tests), then subagent spec files, then skills created via `/write-a-skill`. Skills dispatch subagents via the Agent tool; they don't call each other directly.

**Tech Stack:** Python 3.12, Loguru, SQLite (experience pool), LangChain `@tool`, pytest, LaTeX/BibLaTeX, Claude Code `/write-a-skill` skill.

---

> **Spec correction:** The spec lists 4 integration test files including `test_training_agent_integration.py` and `test_deployment_agent_integration.py`. Reading `agents/registry.py` confirms only `data_validator` and `evaluator` are react agents — the deployer was replaced with a deterministic node and training runs through the deterministic executor. Only 2 integration test files are needed.

---

## File Map

| Action | Path | Purpose |
|--------|------|---------|
| Modify | `src/mlops_agents/utils/logging.py` | Add file sink |
| Modify | `.gitignore` | Ignore `logs/` |
| Create | `logs/.gitkeep` | Keep folder in repo |
| Create | `scripts/pool_snapshot.py` | Pool state snapshot for run-benchmark |
| Create | `tests/test_agents/test_data_agent_integration.py` | Integration test for data_validator |
| Create | `tests/test_agents/test_evaluation_agent_integration.py` | Integration test for evaluator |
| Create | `.claude/agents/log-reader.md` | Haiku subagent spec |
| Create | `.claude/agents/test-runner.md` | Haiku subagent spec |
| Create | `.claude/skills/add-tool/SKILL.md` | Via `/write-a-skill` |
| Create | `.claude/skills/debug-pipeline/SKILL.md` | Via `/write-a-skill` |
| Create | `.claude/skills/tune-prompt/SKILL.md` | Via `/write-a-skill` |
| Create | `.claude/skills/run-benchmark/SKILL.md` | Via `/write-a-skill` |
| Create | `.claude/skills/thesis-docs/SKILL.md` | Via `/write-a-skill` |
| Create | `docs/thesis/THESIS_INDEX.md` | Coherence file (empty initial state) |

---

## Task 1: Add Loguru File Sink

**Files:**
- Modify: `src/mlops_agents/utils/logging.py`
- Modify: `.gitignore`
- Create: `logs/.gitkeep`

- [ ] **Step 1: Add file sink to logging.py**

Replace the contents of `src/mlops_agents/utils/logging.py` with:

```python
"""Loguru-based logging setup.

Handler registration happens once at module level. get_logger() only
binds a name into the context — it never touches global handler state.
"""

import sys
from pathlib import Path
from typing import Any

from loguru import logger
from mlops_agents.config.settings import settings

_LOG_FILE = Path("logs/pipeline.log")

logger.remove()
logger.add(
    sys.stderr,
    level=settings.log_level,
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{extra[name]}</cyan> | "
        "{message}"
    ),
)
logger.add(
    _LOG_FILE,
    level="DEBUG",
    rotation="10 MB",
    retention=3,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[name]} | {message}",
    encoding="utf-8",
)


def get_logger(name: str) -> Any:
    """Return a loguru logger bound with the module name.

    Safe to call many times — does not alter the global handler list.
    """
    return logger.bind(name=name)
```

- [ ] **Step 2: Add logs/ to .gitignore**

Append to `.gitignore`:
```
# Log files (rotation-managed, not committed)
logs/*.log
logs/*.log.*
```

- [ ] **Step 3: Create logs/.gitkeep so the folder exists in the repo**

```bash
mkdir -p logs && touch logs/.gitkeep
```

- [ ] **Step 4: Verify the file sink is created on import**

```bash
uv run python -c "from mlops_agents.utils.logging import get_logger; get_logger('test').info('smoke test'); import os; assert os.path.exists('logs/pipeline.log'), 'log file not created'"
echo "OK"
```

Expected: `OK` with no assertion error.

- [ ] **Step 5: Run existing logging tests to confirm no regression**

```bash
uv run pytest tests/test_utils/test_logging.py -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/mlops_agents/utils/logging.py .gitignore logs/.gitkeep
git commit -m "feat: add loguru file sink (logs/pipeline.log, 10 MB rotation, keep 3)"
```

---

## Task 2: Create scripts/pool_snapshot.py

**Files:**
- Create: `scripts/pool_snapshot.py`

- [ ] **Step 1: Write the snapshot script**

Create `scripts/pool_snapshot.py`:

```python
"""Print a JSON snapshot of the current experience pool state.

Used by the run-benchmark skill to compare pool before/after a benchmark run.

Usage:
    uv run python scripts/pool_snapshot.py
"""
from __future__ import annotations
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mlops_agents.config.settings import settings


def snapshot() -> dict:
    db_path = settings.experience_db_path
    if not db_path.exists():
        return {"total": 0, "by_problem_type": {}, "entries": []}

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    total = conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]

    by_type: dict[str, int] = {}
    for row in conn.execute(
        "SELECT problem_type, COUNT(*) as cnt FROM experiences GROUP BY problem_type"
    ):
        by_type[row["problem_type"]] = row["cnt"]

    entries = []
    for row in conn.execute(
        "SELECT task_id, dataset_name, problem_type, selected_model_key, "
        "validation_score, validation_std, created_at FROM experiences ORDER BY created_at DESC"
    ):
        entries.append({
            "task_id": row["task_id"],
            "dataset_name": row["dataset_name"],
            "problem_type": row["problem_type"],
            "champion_model": row["selected_model_key"],
            "validation_score": row["validation_score"],
            "validation_std": row["validation_std"],
            "created_at": row["created_at"],
        })

    conn.close()
    return {"total": total, "by_problem_type": by_type, "entries": entries}


if __name__ == "__main__":
    print(json.dumps(snapshot(), indent=2))
```

- [ ] **Step 2: Run the script to verify it works (pool may be empty — that's fine)**

```bash
uv run python scripts/pool_snapshot.py
```

Expected: valid JSON like `{"total": 0, "by_problem_type": {}, "entries": []}` or actual entries if pool has data. No exceptions.

- [ ] **Step 3: Commit**

```bash
git add scripts/pool_snapshot.py
git commit -m "feat: add pool_snapshot.py for before/after benchmark comparison"
```

---

## Task 3: Data Agent Integration Test

**Files:**
- Create: `tests/test_agents/test_data_agent_integration.py`

- [ ] **Step 1: Write the integration test**

Create `tests/test_agents/test_data_agent_integration.py`:

```python
"""Integration test for the data validation agent — requires GITHUB_TOKEN."""
import shutil
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage


@pytest.mark.integration
@pytest.mark.slow
def test_data_agent_loads_dataset_and_responds(tmp_path: Path) -> None:
    """Real LLM call — verifies the data_validator agent can use load_dataset tool."""
    from mlops_agents.agents.data_agent import build_data_agent

    src = Path("data/samples/iris.csv")
    if not src.exists():
        pytest.skip("data/samples/iris.csv not found")

    dst = tmp_path / "iris.csv"
    shutil.copy(src, dst)

    agent = build_data_agent()
    result = agent.invoke({
        "messages": [HumanMessage(content=(
            f"Load the dataset at {dst} and report its shape, column names, and data types. "
            "Do not perform any further validation steps."
        ))]
    })

    messages = result.get("messages", [])
    assert len(messages) > 1, "Agent must produce at least one response beyond the input"
    last = messages[-1]
    assert hasattr(last, "content") and last.content.strip(), (
        "Last message must have non-empty content"
    )
    content_lower = last.content.lower()
    assert any(kw in content_lower for kw in ("row", "column", "feature", "iris", "shape")), (
        "Response should mention dataset properties"
    )
```

- [ ] **Step 2: Run the test to confirm it works (requires GITHUB_TOKEN in .env)**

```bash
uv run pytest tests/test_agents/test_data_agent_integration.py -v -m integration
```

Expected: PASSED. If `GITHUB_TOKEN` is missing, it will fail with an auth error — ensure `.env` is configured.

- [ ] **Step 3: Confirm unit tests still pass (no regressions)**

```bash
uv run pytest tests/test_agents/test_data_agent.py -v
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_agents/test_data_agent_integration.py
git commit -m "test: add integration test for data_validator agent"
```

---

## Task 4: Evaluation Agent Integration Test

**Files:**
- Create: `tests/test_agents/test_evaluation_agent_integration.py`

- [ ] **Step 1: Write the integration test**

Create `tests/test_agents/test_evaluation_agent_integration.py`:

```python
"""Integration test for the evaluation agent — requires GITHUB_TOKEN."""
import pytest
from langchain_core.messages import HumanMessage


@pytest.mark.integration
@pytest.mark.slow
def test_evaluation_agent_responds_to_classification_metrics() -> None:
    """Real LLM call — verifies the evaluator agent responds to a classification scenario."""
    from mlops_agents.agents.evaluation_agent import build_evaluation_agent

    agent = build_evaluation_agent()
    result = agent.invoke({
        "messages": [HumanMessage(content=(
            "Problem type: classification\n"
            "Training run ID: test-run-nonexistent\n"
            "Training metrics: {\"macro_f1\": 0.87, \"accuracy\": 0.91}\n"
            "NOTE: MLflow stores F1 under 'macro_f1'. Call get_best_run with "
            "metric='macro_f1' (ascending=False). "
            "If get_best_run returns an error because the run does not exist in this "
            "test environment, base your recommendation on the provided metrics alone "
            "and state that evaluation_passed=True if macro_f1 >= 0.75."
        ))]
    })

    messages = result.get("messages", [])
    assert len(messages) > 1, "Agent must produce at least one response beyond the input"
    last = messages[-1]
    assert hasattr(last, "content") and last.content.strip(), (
        "Last message must have non-empty content"
    )
    content_lower = last.content.lower()
    assert any(kw in content_lower for kw in ("evaluat", "promot", "approv", "f1", "metric", "pass")), (
        "Response should reference evaluation outcome or metrics"
    )
```

- [ ] **Step 2: Run the test**

```bash
uv run pytest tests/test_agents/test_evaluation_agent_integration.py -v -m integration
```

Expected: PASSED.

- [ ] **Step 3: Commit**

```bash
git add tests/test_agents/test_evaluation_agent_integration.py
git commit -m "test: add integration test for evaluator agent"
```

---

## Task 5: Create log-reader Subagent Spec

**Files:**
- Create: `.claude/agents/log-reader.md`

- [ ] **Step 1: Create the agents directory and subagent spec**

Create `.claude/agents/log-reader.md`:

```markdown
---
description: Locate and surface errors from Loguru log files. Returns a structured summary of errors found: message, file, line, frequency. Use when debugging pipeline failures.
model: claude-haiku-4-5-20251001
allowed-tools: Grep, Glob, Read
---

You are a log analysis specialist. Your only job is to find error patterns in log files.

## How to search

Log files live in `logs/pipeline.log` (and rotated variants `logs/pipeline.log.1`, etc.).

Log line format:
```
YYYY-MM-DD HH:mm:ss | LEVEL    | module.name | message
```

## Steps

1. Glob for all log files: `logs/pipeline.log*`
2. Grep for ERROR and WARNING lines matching the input keyword or run ID
3. For each match, extract: timestamp, level, module, message
4. Count occurrences per unique message pattern
5. Return a structured bullet list

## Output format

```
## Log Analysis Results

**Search term:** <keyword>
**Files searched:** <list>
**Total matches:** <n>

### Errors found:
- [TIMESTAMP] MODULE | MESSAGE (×N occurrences)
- ...

### Warnings found:
- [TIMESTAMP] MODULE | MESSAGE (×N occurrences)

### Summary:
<1-2 sentence diagnosis>
```

## Rules

- Never read full files — only matched lines with `-C 2` context
- Never write to any file
- Never run Bash commands
- If no log files exist, report: "No log files found in logs/. Ensure the file sink is configured in utils/logging.py."
```

- [ ] **Step 2: Verify the file is valid Markdown with correct frontmatter**

```bash
uv run python -c "
import re, sys
content = open('.claude/agents/log-reader.md').read()
assert content.startswith('---'), 'Missing frontmatter'
assert 'claude-haiku' in content, 'Missing model'
assert 'allowed-tools' in content, 'Missing allowed-tools'
print('OK')
"
```

Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add .claude/agents/log-reader.md
git commit -m "feat: add log-reader Haiku subagent spec"
```

---

## Task 6: Create test-runner Subagent Spec

**Files:**
- Create: `.claude/agents/test-runner.md`

- [ ] **Step 1: Create the subagent spec**

Create `.claude/agents/test-runner.md`:

```markdown
---
description: Execute pytest and return a structured failure report. Does not fix failures — only reports. Use when running benchmarks or targeted test suites.
model: claude-haiku-4-5-20251001
allowed-tools: Bash, Read
---

You are a test execution specialist. Your only job is to run pytest and report results clearly.

## Rules

- Only run `uv run pytest ...` commands via Bash — no other shell commands
- Never attempt to fix failing tests
- Never write to any file
- Read test output files only if explicitly needed for parsing

## Steps

1. Run the pytest command provided by the caller
2. Parse stdout for: total collected, passed, failed, error, skipped, duration
3. For each FAILED test, extract: test name + short failure reason (first assertion error line)
4. Return the structured report below

## Output format

```
## Test Results

**Command:** `<pytest command run>`
**Duration:** Xs

| Result | Count |
|--------|-------|
| Passed | N |
| Failed | N |
| Errors | N |
| Skipped | N |

### Failed tests:
- `test_module::test_name` — <first assertion error, one line>
- ...

### Summary:
<1 sentence: pass/fail verdict and most important failure if any>
```

## If no failures

Report all tests passed with duration. No further analysis needed.
```

- [ ] **Step 2: Verify**

```bash
uv run python -c "
content = open('.claude/agents/test-runner.md').read()
assert 'claude-haiku' in content
assert 'allowed-tools: Bash, Read' in content
print('OK')
"
```

Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add .claude/agents/test-runner.md
git commit -m "feat: add test-runner Haiku subagent spec"
```

---

## Task 7: add-tool Skill

**Files:**
- Create: `.claude/skills/add-tool/SKILL.md` (via `/write-a-skill`)

- [ ] **Step 1: Invoke the write-a-skill skill**

Run `/write-a-skill` and use the following brief when prompted:

```
Skill name: add-tool
Trigger: User wants to add a new tool to an existing agent in the MLOps pipeline.

Steps:
1. Read the target agent file (src/mlops_agents/agents/<name>_agent.py) and src/mlops_agents/state/agent_state.py
2. Check existing tools in src/mlops_agents/tools/ to avoid duplication — list them for the user
3. Ask the user which domain file the tool should live in, showing options:
   data_tools.py, training_tools.py, mlflow_tools.py, evidently_tools.py, memory_tools.py
4. Implement the tool using @tool decorator + Python type hints + docstring (existing pattern). Only use Pydantic args_schema for complex nested input.
5. Register the tool in the agent's create_agent(..., tools=[...]) call
6. Write a unit test: deterministic tools use real DataFrames (no LLM mock); tools calling external services use unittest.mock
7. Run: uv run pytest tests/ -k "<tool_name>" -v

Guards:
- Return type must be str containing JSON — use json.dumps(), never return plain text
- No LLM calls inside tools
- Tool name must not already exist in tools/
- Tools are imported from mlops_agents.tools, not defined inline in agent files
```

- [ ] **Step 2: Verify SKILL.md was created**

```bash
test -f .claude/skills/add-tool/SKILL.md && echo "OK" || echo "MISSING"
```

Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/add-tool/
git commit -m "feat: add add-tool skill"
```

---

## Task 8: debug-pipeline Skill

**Files:**
- Create: `.claude/skills/debug-pipeline/SKILL.md` (via `/write-a-skill`)

- [ ] **Step 1: Invoke /write-a-skill with this brief**

```
Skill name: debug-pipeline
Trigger: A pipeline run failed or produced unexpected output.

Steps:
0. Check .env for LANGCHAIN_TRACING_V2=true. If absent or false, prompt the user to enable it and re-run for full node-by-node traces. Proceed without it only if user explicitly skips.
1. Use the Agent tool to dispatch the log-reader subagent defined in .claude/agents/log-reader.md, passing the error keyword or run ID
2. Identify the failing node from the LangSmith trace (preferred) or log summary (fallback). Node names: data_validator, planner, trainer (deterministic executor), evaluator, deployer.
3. Read that node's source file in src/mlops_agents/graphs/mlops_graph.py or src/mlops_agents/agents/
4. Read the node's relevant state fields from src/mlops_agents/state/agent_state.py
5. Propose a minimal reproduction (inline script or pytest case)
6. Propose a fix and explain the root cause
7. After fixing, run: uv run pytest -m "not integration" to confirm no regressions

Notes:
- If the failure is in an LLM agent (data_validator or evaluator nodes), check the prompt in src/mlops_agents/prompts/ before the code
- Deterministic nodes (planner/executor, deployer): failures are code bugs, not prompt issues
- Future: deeper observability via Langfuse or LangSmith full integration
```

- [ ] **Step 2: Verify**

```bash
test -f .claude/skills/debug-pipeline/SKILL.md && echo "OK" || echo "MISSING"
```

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/debug-pipeline/
git commit -m "feat: add debug-pipeline skill"
```

---

## Task 9: tune-prompt Skill

**Files:**
- Create: `.claude/skills/tune-prompt/SKILL.md` (via `/write-a-skill`)

- [ ] **Step 1: Invoke /write-a-skill with this brief**

```
Skill name: tune-prompt
Trigger: An agent's natural-language output or routing decision is wrong or suboptimal.

Steps:
1. Read src/mlops_agents/prompts/<agent>.yaml
2. Read the agent's recent output from logs/pipeline.log or user-provided sample
3. Identify the specific failure mode: hallucination, wrong routing, missing field, wrong tone
4. Propose 2-3 prompt variants with brief reasoning for each
5. Check if tests/test_agents/test_<agent>_integration.py exists. If not, offer to create a minimal one first (single input → assert expected output fields present). Most agents currently lack integration tests.
6. Warn the user upfront: each round makes 1 real LLM call against GitHub Models quota (max 3 rounds = up to 3 calls). Confirm before starting each round.
7. Run the integration test for the target agent with the new prompt variant
8. Compare outputs and select the best variant
9. Write the winning variant back to the YAML file

Guards:
- Maximum 3 iteration rounds; if not converging, escalate to an architecture question
- Never change model parameters (temperature, max_tokens) without explicit user approval
- Keep prompt changes isolated to one agent at a time
- Always confirm with user before each round (they may stop early to save quota)

Agent prompt files: src/mlops_agents/prompts/data_agent.yaml, evaluation_agent.yaml, supervisor.yaml, planner.yaml, deployment_agent.yaml
```

- [ ] **Step 2: Verify**

```bash
test -f .claude/skills/tune-prompt/SKILL.md && echo "OK" || echo "MISSING"
```

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/tune-prompt/
git commit -m "feat: add tune-prompt skill"
```

---

## Task 10: run-benchmark Skill

**Files:**
- Create: `.claude/skills/run-benchmark/SKILL.md` (via `/write-a-skill`)

- [ ] **Step 1: Invoke /write-a-skill with this brief**

```
Skill name: run-benchmark
Trigger: Before or after significant changes, or when comparing agent strategies.

Steps:
1. Run: uv run python scripts/pool_snapshot.py — capture the JSON output as the pre-run snapshot
2. Use the Agent tool to dispatch the test-runner subagent defined in .claude/agents/test-runner.md, instructing it to execute: uv run python scripts/run_benchmark.py
3. Run: uv run python scripts/pool_snapshot.py again — capture post-run JSON
4. Diff the two snapshots: for each dataset_id, compare champion_model and validation_score
5. Summarize: wins (higher score or new champion), regressions (score drop), unchanged, new entries
6. Flag any regression above 5% validation score drop as a blocking issue requiring user decision

Notes:
- Benchmark results live in SQLite (settings.experience_db_path) and MLflow experiment mlops-agents-benchmark — there is no diagnostics file
- If the pool was empty before the run, the post-run snapshot is the new baseline — no regression to report
- Does not auto-fix regressions; surfaces them for user decision
```

- [ ] **Step 2: Verify**

```bash
test -f .claude/skills/run-benchmark/SKILL.md && echo "OK" || echo "MISSING"
```

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/run-benchmark/
git commit -m "feat: add run-benchmark skill"
```

---

## Task 11: thesis-docs Skill

**Files:**
- Create: `.claude/skills/thesis-docs/SKILL.md` (via `/write-a-skill`)
- Create: `docs/thesis/THESIS_INDEX.md`

- [ ] **Step 1: Invoke /write-a-skill with this brief**

```
Skill name: thesis-docs
Trigger: User wants to generate a thesis chapter or section in LaTeX.

Steps:
1. Ask the user which chapter/section they want to generate
2. Read docs/thesis/THESIS_INDEX.md (coherence: terminology, prior sections, cross-refs). If it doesn't exist, create it from the template in this skill (see below) and ask the user to confirm the chapter outline.
3. Run a short targeted brainstorm — ask 2-4 questions specific to that section to gather personal voice, motivation, conclusions, or opinions that cannot be derived from code or specs. Wait for answers before proceeding. For purely technical sections, ask the user if they have notes to add or if the spec is sufficient.
4. Read relevant specs from docs/superpowers/specs/ first (dense design summaries). Announce which files will be read before reading them so the user can redirect. Only read source code for specific implementation details not covered by specs.
5. Announce the target file path (docs/thesis/capXX_<topic>.tex) and whether it already exists (overwrite) before writing.
6. Overwrite docs/thesis/capXX_<topic>.tex in place — git history is the safety net.
7. Update THESIS_INDEX.md: update chapter row status to done, add new terminology and cross-references.

Output conventions:
- Language: academic Spanish
- Format: standalone LaTeX — \section{} is top-level, \subsection{} below (no \documentclass)
- Bibliography: BibLaTeX + Biber backend; cite with \cite{} referencing Biblio.bib
- Style: formal academic prose, Universidad de Oviedo TFG conventions
- Figures: \begin{figure}[H] with \caption{} and \label{fig:XX}
- Code listings: \begin{lstlisting}[language=Python]
- Self-contained: compiles when \input{}-ed into a parent document

For personal-voice sections (Motivación, Conclusiones, Limitaciones): brainstorm is mandatory.
For purely technical sections: skip brainstorm if user says notes are not needed.

THESIS_INDEX.md template:
---
# Thesis Index

## Chapter Outline
| # | Title (ES) | Subsections | File | Status |
|---|-----------|-------------|------|--------|
| 1 | Introducción | Motivación, Objetivos, Alcance del trabajo, Estructura del Trabajo | — | pending |
| 2 | Conceptos Básicos | MLOps, Sistemas multi-agente basados en LLMs, API de OpenAI como capa de acceso al modelo, LangGraph, Patrón ReAct y tools, Human-in-the-Loop, Principio de restraint agéntico: cuándo no usar agentes, Herramientas auxiliares | — | pending |
| 3 | Estudio de Alternativas | Alternativas a LangGraph, Alternativas al proveedor de modelos, Alternativas al almacén de experiencia | — | pending |
| 4 | Arquitectura | Visión general del sistema, Patrón supervisor-trabajador, Diseño del estado compartido, Flujo de ejecución end-to-end, Human-in-the-Loop en el nodo de despliegue | — | pending |
| 5 | Desarrollo e Implementación | Agente de validación de datos, Agente de entrenamiento, Agente de evaluación, Agente de despliegue, Pool de experiencia, API REST y frontend, Contenerización | — | pending |
| 6 | Resultados | Métricas de rendimiento del pipeline, Evaluación del pool de experiencia (benchmarks), Análisis del comportamiento agéntico | — | pending |
| 7 | Conclusiones y Trabajo Futuro | Conclusiones, Limitaciones, Escalabilidad del proyecto, Trabajo futuro | — | pending |

## Terminology (ES)
| English term | Spanish term used |
|---|---|

## Cross-references
| Label | Introduced in | Description |
|---|---|---|
---
```

- [ ] **Step 2: Create the initial THESIS_INDEX.md**

Create `docs/thesis/THESIS_INDEX.md`:

```markdown
# Thesis Index

## Chapter Outline
| # | Title (ES) | Subsections | File | Status |
|---|-----------|-------------|------|--------|
| 1 | Introducción | Motivación, Objetivos, Alcance del trabajo, Estructura del Trabajo | — | pending |
| 2 | Conceptos Básicos | MLOps, Sistemas multi-agente basados en LLMs, API de OpenAI como capa de acceso al modelo, LangGraph, Patrón ReAct y tools, Human-in-the-Loop, Principio de restraint agéntico: cuándo no usar agentes, Herramientas auxiliares | — | pending |
| 3 | Estudio de Alternativas | Alternativas a LangGraph, Alternativas al proveedor de modelos, Alternativas al almacén de experiencia | — | pending |
| 4 | Arquitectura | Visión general del sistema, Patrón supervisor-trabajador, Diseño del estado compartido, Flujo de ejecución end-to-end, Human-in-the-Loop en el nodo de despliegue | — | pending |
| 5 | Desarrollo e Implementación | Agente de validación de datos, Agente de entrenamiento, Agente de evaluación, Agente de despliegue, Pool de experiencia, API REST y frontend, Contenerización | — | pending |
| 6 | Resultados | Métricas de rendimiento del pipeline, Evaluación del pool de experiencia (benchmarks), Análisis del comportamiento agéntico | — | pending |
| 7 | Conclusiones y Trabajo Futuro | Conclusiones, Limitaciones, Escalabilidad del proyecto, Trabajo futuro | — | pending |

## Terminology (ES)
| English term | Spanish term used |
|---|---|

## Cross-references
| Label | Introduced in | Description |
|---|---|---|
```

- [ ] **Step 3: Verify SKILL.md and THESIS_INDEX.md exist**

```bash
test -f .claude/skills/thesis-docs/SKILL.md && echo "skill OK" || echo "skill MISSING"
test -f docs/thesis/THESIS_INDEX.md && echo "index OK" || echo "index MISSING"
```

Expected: both `OK`.

- [ ] **Step 4: Commit**

```bash
git add .claude/skills/thesis-docs/ docs/thesis/THESIS_INDEX.md
git commit -m "feat: add thesis-docs skill and initial THESIS_INDEX.md"
```

---

## Self-Review

**Spec coverage check:**
- ✅ Loguru file sink → Task 1
- ✅ pool_snapshot.py → Task 2
- ✅ data_agent integration test → Task 3
- ✅ evaluation_agent integration test → Task 4
- ✅ Corrected: training + deployment agent tests removed (not react agents)
- ✅ log-reader subagent → Task 5
- ✅ test-runner subagent → Task 6
- ✅ add-tool skill → Task 7
- ✅ debug-pipeline skill → Task 8
- ✅ tune-prompt skill → Task 9
- ✅ run-benchmark skill → Task 10
- ✅ thesis-docs skill + THESIS_INDEX.md → Task 11
- ✅ manage-experience removed (per user decision during grilling)

**Placeholder scan:** No TBDs. All code blocks are complete and runnable.

**Type consistency:** No cross-task type dependencies — each task is standalone.
