# Project Skills & Subagents Design

**Date:** 2026-05-27
**Branch:** feature/container
**Scope:** All custom `.claude/` skills and subagents for the MLOps thesis project

---

## Goal

Reduce repetitive manual multi-step work during development by encoding recurring workflows as Claude Code skills, and offload cheap grep/test execution to Haiku-powered subagents. Skills are invoked via `/skill-name`; subagents are dispatched programmatically by skills or directly.

---

## File Layout

```
.claude/
├── agents/
│   ├── log-reader.md
│   └── test-runner.md
└── skills/
    ├── add-tool/SKILL.md
    ├── debug-pipeline/SKILL.md
    ├── tune-prompt/SKILL.md
    ├── run-benchmark/SKILL.md
    └── thesis-docs/SKILL.md

docs/thesis/
├── THESIS_INDEX.md          ← coherence file, auto-maintained by thesis-docs skill
└── capXX_<topic>.tex        ← generated chapter files
```

Skills are created using `/write-a-skill` (not hand-authored). Subagent specs in `.claude/agents/` are plain Markdown written directly.

---

## Skills

### `add-tool`

**Trigger:** User wants to add a new tool to an existing agent.

**Steps:**
1. Read the target agent file and `src/mlops_agents/state/agent_state.py`
2. Check existing tools in `src/mlops_agents/tools/` to avoid duplication
3. Ask the user which domain file the tool should live in, showing the existing options (`data_tools.py`, `training_tools.py`, `mlflow_tools.py`, `evidently_tools.py`, `memory_tools.py`) — then implement there
4. Implement using `@tool` decorator + Python type hints + docstring (existing pattern). Only introduce Pydantic `args_schema` if the tool needs complex nested input.
5. Register the tool in the agent's `create_react_agent` call
6. Write a unit test (deterministic tools use real DataFrames, no LLM mock)
7. Run `uv run pytest tests/ -k "<tool_name>" -v` to verify

**Guards:**
- Return type must be `str` containing JSON — use `json.dumps()` for structured data, never return arbitrary plain text (consistent with all existing tools)
- No LLM calls inside tools
- Tool name must not already exist in `tools/`
- **Future improvement:** LangChain 0.2+ supports `dict` returns from `@tool` (auto-serialized) — migrate all tools at once if a refactor is planned

---

### `debug-pipeline`

**Trigger:** A pipeline run failed or produced unexpected output.

**Steps:**
0. Check `.env` for `LANGCHAIN_TRACING_V2=true`. If absent or false, prompt the user to enable it and re-run the pipeline first — LangSmith provides full node-by-node traces that make step 2 reliable. Proceed without it only if the user explicitly skips this step.
1. Use the Agent tool to dispatch the `log-reader` subagent defined in `.claude/agents/log-reader.md`, passing the error keyword or run ID
2. Identify the failing node from the LangSmith trace (preferred) or log summary (fallback)
3. Read that node's source file
4. Read the node's relevant state fields from `agent_state.py`
5. Propose a minimal reproduction (inline script or pytest case)
6. Propose a fix and explain the root cause
7. After fixing, run `uv run pytest -m "not integration"` to confirm no regressions

**Notes:**
- Follows the `diagnose` skill pattern but MLflow/LangGraph-aware
- If the failure is in an LLM agent (not a deterministic node), check the prompt first before the code
- **Future:** deeper observability via Langfuse or LangSmith full integration (currently optional tracing only)

---

### `tune-prompt`

**Trigger:** An agent's natural-language output or routing decision is wrong or suboptimal.

**Steps:**
1. Read `src/mlops_agents/prompts/<agent>.yaml`
2. Read the agent's recent output (from `logs/pipeline.log` or user-provided sample)
3. Identify the specific failure mode (hallucination, wrong routing, missing field, wrong tone)
4. Propose 2–3 prompt variants with brief reasoning for each
5. Check if `tests/test_agents/test_<agent>_integration.py` exists. If not, offer to create a minimal one first (single input → assert expected output fields present) before proceeding — most agents currently lack integration tests.
6. Warn the user upfront: each round makes 1 real LLM call against GitHub Models quota (max 3 rounds = up to 3 calls); confirm before starting
7. Run the integration test for the target agent with the new prompt variant
8. Compare outputs and select the best variant
8. Write the winning variant back to the YAML file

**Guards:**
- Maximum 3 iteration rounds; if not converging, escalate to an architecture question
- Never change model parameters (temperature, max_tokens) without explicit user approval
- Keep prompt changes isolated to one agent at a time
- Always confirm with user before each round (they may stop early to save quota)

---

### `run-benchmark`

**Trigger:** Before or after significant changes, or when comparing agent strategies.

**Steps:**
1. Run `uv run python scripts/pool_snapshot.py` to snapshot the experience pool before running (prints JSON: count per problem type, champion model + validation score per dataset). A helper script `scripts/pool_snapshot.py` must exist for this step.
2. Use the Agent tool to dispatch the `test-runner` subagent defined in `.claude/agents/test-runner.md`, instructing it to execute `uv run python scripts/run_benchmark.py`
3. Run `uv run python scripts/pool_snapshot.py` again after the run; diff JSON output against the pre-run snapshot
4. Summarize: wins, regressions, unchanged metrics — per dataset (champion model, validation score delta)
5. Flag any regression above 5% validation score drop as a blocking issue

**Notes:**
- Does not auto-fix regressions; surfaces them for user decision
- If no prior snapshot exists (pool was empty), baseline is established from the current run
- Benchmark results live in SQLite (`settings.experience_db_path`) and MLflow experiment `mlops-agents-benchmark` — there is no diagnostics file

---

### `thesis-docs`

**Trigger:** User wants to generate a thesis chapter or section in LaTeX.

**Steps:**
1. Ask the user which chapter/section they want to generate
2. Read `docs/thesis/THESIS_INDEX.md` (coherence: terminology, prior sections, cross-refs)
3. Run a short targeted brainstorm — ask 2–4 questions specific to that section to gather personal voice, motivation, conclusions, or opinions that cannot be derived from code or specs. Examples: "¿Cuál fue la principal dificultad técnica?" / "¿Qué te motivó a usar LangGraph en lugar de otra herramienta?" / "¿Qué limitaciones identificas en el enfoque actual?" Wait for answers before proceeding.
4. Read relevant specs from `docs/superpowers/specs/` first (already dense design summaries); announce which files will be read before reading them so the user can redirect. Only read source code for specific implementation details not covered by specs.
5. Announce the target file path (`docs/thesis/capXX_<topic>.tex`) and whether it already exists (overwrite) before writing
6. Overwrite `docs/thesis/capXX_<topic>.tex` in place — git history is the safety net for previous versions
7. Update `THESIS_INDEX.md`: update the chapter row status to `done`, add new terminology and cross-references

**Note:** For purely technical sections (e.g., Arquitectura subsections), skip the brainstorm if the user indicates notes are not needed. For personal-voice sections (Motivación, Conclusiones, Limitaciones), the brainstorm is mandatory.

**Output conventions:**
- Language: academic Spanish
- Format: standalone LaTeX (no `\documentclass` — `\section{}` is top-level, `\subsection{}` below)
- Bibliography: BibLaTeX + Biber backend; cite with `\cite{}` referencing `Biblio.bib`
- Style: formal academic prose, Universidad de Oviedo TFG conventions
- Figures: use `\begin{figure}[H]` with `\caption{}` and `\label{fig:XX}`
- Code listings: use `\begin{lstlisting}[language=Python]`
- Self-contained: the file must compile when `\input{}`-ed into a parent document

**THESIS_INDEX.md structure:**
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

On first `thesis-docs` run: create `THESIS_INDEX.md` with the outline above; ask user to confirm or adjust before generating any chapter.

---

## Subagents

### `log-reader`

**File:** `.claude/agents/log-reader.md`
**Model:** Haiku (low reasoning, pattern matching only)
**Allowed tools:** `Grep`, `Glob`, `Read` (no writes, no Bash)

**Purpose:** Locate and surface errors from Loguru log files. Returns a structured summary: error messages, file + line, frequency. Never loads full files — only matched lines with context windows.

**Prerequisite:** `src/mlops_agents/utils/logging.py` must include a file sink writing to `logs/pipeline.log` with `rotation="10 MB"` and `retention=3` (max ~30 MB on disk). The `logs/` folder is gitignored.

**Input:** Error keyword, run ID, or timestamp range (passed by caller).
**Output:** Bullet list of findings — source, message, count.

---

### `test-runner`

**File:** `.claude/agents/test-runner.md`
**Model:** Haiku
**Allowed tools:** `Bash` (pytest only), `Read` (test output files)

**Purpose:** Execute pytest and return a structured failure report. Does not attempt to fix failures.

**Input:** Pytest command or filter expression (e.g., `-k "data_agent" -m "not integration"`).
**Output:** Pass/fail counts, list of failed test names + short failure reason, total duration.

---

## Implementation Notes

- All skills are created via `/write-a-skill` invocations — not hand-authored
- Subagent specs are plain Markdown written directly to `.claude/agents/`
- `docs/thesis/THESIS_INDEX.md` is created empty on first `thesis-docs` run
- Skills do not call each other directly; `debug-pipeline` dispatches `log-reader` via the Agent tool, `run-benchmark` dispatches `test-runner` — this is the only inter-skill coupling
- **Prerequisites to build as part of this plan (before skills):**
  - Add file sink to `src/mlops_agents/utils/logging.py` (`logs/pipeline.log`, `rotation="10 MB"`, `retention=3`) + gitignore `logs/`
  - Create `scripts/pool_snapshot.py` helper
  - Create missing agent integration tests: `tests/test_agents/test_data_agent_integration.py`, `test_evaluation_agent_integration.py`, `test_deployment_agent_integration.py`, `test_training_agent_integration.py` (each marked `@pytest.mark.integration`)
