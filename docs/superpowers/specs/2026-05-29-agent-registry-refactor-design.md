# Agent Architecture Refactor — Design Spec

**Date:** 2026-05-29
**Status:** Approved

## Problem

The current pipeline is structurally confused:

1. `mlops_graph.py` is a fat god-file mixing business logic, graph topology, HITL handling, and CLI runner.
2. Nodes use inconsistent patterns: some via registry (`data_validator`, `evaluator`), some embedded directly in the graph (`planner`, `supervisor`, `executor`, `deployer`).
3. `deployment_agent.py` exists but is never called — dead code.
4. Settings reference LLM agents (`trainer`, `deployer`) that don't actually have an LLM.
5. The **evaluator is an LLM react agent that makes deterministic decisions**: comparing two numbers against fixed thresholds. The LLM has freedom to pick the wrong metric, wrong direction, or compare incomparable numbers — silent correctness risk.
6. The **supervisor is an LLM router** for a flow that is fundamentally a state machine: every routing decision is derivable from state fields.

The system was drifting into a "multi-LLM system" rather than a true multi-agent architecture. This refactor restores the original principle from `CLAUDE.md`:

> Deterministic first; agents only for interpreting failures, reasoning about strategy, natural-language reports.

## Goals

1. Only use LLMs where genuine reasoning is required.
2. Every LLM-based node has its own module with a single `build_*()` function.
3. Every LLM-based node is accessible via `get_agent(name)`.
4. Every graph node in `mlops_graph.py` is a thin wrapper.
5. Dead code is removed.
6. The architecture is honest: 2 true agents + 1 evidence-alignment LLM node + deterministic spine.

## Final Architecture

```
START
  ↓
Deterministic Workflow Controller        [no LLM]
  ↓
Data Mapping & Validation Agent          [react agent — true agent]
  ↓
Human Approval Gate 1: validate dataset  [HITL]
  ↓
Model Planner Agent                      [structured output LLM — true agent]
  ↓
Deterministic Training Executor          [no LLM]
  ↓
Deterministic Evaluation & Promotion     [no LLM]
  ↓
Evaluation Report LLM Node               [structured output LLM — audit node]
  ↓
Human Approval Gate 2: approve deploy    [HITL]
  ↓
Deterministic Deployment Module          [no LLM]
  ↓
Experience Pool Update                   [no LLM]
END
```

## Node Classification

| Node | Type | Why |
|---|---|---|
| `workflow_controller` | Pure Python state machine | Routing derivable from state — no reasoning needed |
| `data_validator` | React agent (`create_agent()`) | Genuine tool loop: maps schema, proposes joins, validates quality, decides fix strategy based on intermediate findings |
| `planner` | Structured-output LLM (`with_structured_output(PlannerOutput)`) | One-shot reasoning over pre-assembled context (dataset profile + experiences + rules) to produce a constrained `TrainingPlan` |
| `executor` | Pure Python | Runs Optuna + scikit-learn + MLflow logging — no reasoning |
| `evaluation` | Pure Python | Apply deterministic thresholds against baseline/current champion — pure number comparison |
| `report_writer` | Structured-output LLM (`with_structured_output(EvaluationReport)`) | Evidence-alignment audit: triangulates planner reasoning, training results, and empirical metrics |
| `deployer` | Pure Python + HITL | MLflow registration + human approval gate — no reasoning |

**Rule:** an LLM is justified only when the node must interpret evidence, weigh competing signals, or write natural-language explanations grounded in multiple sources. Comparing numbers, applying thresholds, calling APIs in fixed sequence — these are Python.

## True Agents (2)

### 1. Data Mapping & Validation Agent (`data_validator`)

Already implemented as `create_agent()` with 10 tools. Maps raw files to target schema, proposes joins, validates data quality, detects leakage/schema issues, generates the target dataset proposal. The tool loop is justified because the agent decides which tools to call based on what it finds (number of input files, presence of missing values, mismatched columns, etc.).

**Refactor:** unchanged except for cleanup of node wrapper in `mlops_graph.py`.

### 2. Model Planner Agent (`planner`)

Currently a custom LLM node. **Stays as structured output**, not converted to `create_agent()`.

**Rationale:** the context-gathering steps (`build_dataset_profile`, `pool.find_similar`, `match_rules`, `get_models_for`) are always called, always in the same order, with no branching. Wrapping them as tools would add complexity with zero functional benefit — there is no real react loop happening, just deterministic Python masquerading as tools.

**Pattern:**
```python
def build_planner_agent():
    return get_llm("planner", max_tokens=16000).with_structured_output(
        PlannerOutput, method="function_calling"
    )
```

Context is assembled in the node, passed to the LLM, validated `PlannerOutput` comes back.

## LLM Node — Not an Agent

### Evaluation Report LLM Node (`report_writer`)

**This is an evidence-aligned audit node, not a free narrative generator.** It does NOT decide promotion — promotion is decided deterministically before this node runs. The LLM's job is to triangulate planner reasoning, deterministic results, and empirical metrics.

**Output schema:**
```python
class EvaluationReport(BaseModel):
    summary: str
    champion_model: str
    why_champion_won: str
    planner_alignment: str
    deviations_from_planner_expectations: list[str]
    evidence_consistency_warnings: list[str]
    risks_and_warnings: list[str]
    promotion_decision_explanation: str
    human_review_notes: list[str]
```

**Required reasoning by the LLM:**
1. Why was the model originally selected as a candidate (planner evidence)?
2. Why did it actually become champion after training (empirical metrics)?
3. Was the outcome consistent with retrieved experiences?
4. Are there contradictions, weak evidence, or signs retrieval was misleading?

**Guardrails:**

The LLM must NOT say things like:
> "The model won because similar experiences proved it was best."

It must say things like:
> "The model was selected as a candidate because similar experiences suggested it was promising. It became champion because it achieved the best deterministic validation/test metric among the trained candidates."

This prevents conflating planning evidence with empirical results.

**Scenarios the LLM must handle correctly:**

| Scenario | Expected report |
|---|---|
| Planner expected LightGBM; LightGBM won | Consistent with retrieved experiences and empirical validation |
| Planner expected LightGBM; Random Forest won | Empirical result diverged from prior experience; possible dataset-specific behavior |
| Retrieved experiences had low similarity scores | Historical evidence was weak; result should rely mainly on current validation |
| Planner rejected XGBoost but executor trained XGBoost | Inconsistency between TrainingPlan and execution artifacts — flag |

**Input to the LLM:**
- Deterministic evaluation result (`evaluation_passed`, `candidate_metrics`, `champion_metrics`, `thresholds_applied`)
- Full `planner_output_record` (selected models, rejected models, evidence used, retrieved experiences, matched rules)
- `TrainingPlan` actually executed
- Champion model key and its trial history

**State field written:**
- `evaluation_report_audit: dict` — full `EvaluationReport.model_dump()` from the LLM.
- `evaluation_report_audit_status: "ok" | "retry_ok" | "stub"` — provenance flag.

The existing `evaluation_report` field keeps its current shape (`candidate_metrics`, `candidate_run_id`, `baseline_metrics`) and is written by the deterministic `evaluation_node`. This preserves backward compatibility with the SSE event shape and frontend rendering. The new audit field is rendered additively by the frontend when present (follow-up issue).

**Failure handling — retry once, then soft fail:**

The `report_writer_node` follows the same pattern as `planner` today:
1. First attempt: call the LLM, validate `EvaluationReport` schema.
2. On `ValidationError`, timeout, or LLM exception → retry once with the validation error injected as feedback.
3. On second failure → write a **stub report** with `summary: "Audit report unavailable due to LLM error"`, log the error, set `evaluation_report_audit_status: "stub"`, and continue to Gate 2.

This preserves the human's ability to approve/reject deployment based on the deterministic `evaluation_report` even when the audit narrative is unavailable. The deterministic promotion decision is the source of truth; the audit is enrichment.

## Deterministic Modules

### Workflow Controller (replaces `supervisor`)

Pure Python state machine. Routes based on state fields only — no LLM, no reasoning.

```python
def workflow_controller(state: AgentState) -> Command:
    counts = state.get("agent_attempt_counts") or {}
    max_attempts = settings.max_attempts_per_agent

    if state.get("error_message"):
        return Command(goto=END)
    if not state.get("validation_passed"):
        if counts.get("data_validator", 0) >= max_attempts:
            return Command(goto=END, update={"error_message": "data_validator: max attempts reached"})
        return Command(goto="data_validator")
    if state.get("dataset_approved") is None:
        return Command(goto="dataset_approval")
    if state.get("dataset_approved") is False:
        # Gate 1 rejection — retry data_validator with rejection feedback up to max_attempts
        if counts.get("data_validator", 0) >= max_attempts:
            return Command(goto=END, update={"error_message": "Dataset rejected after max attempts"})
        return Command(goto="data_validator", update={"dataset_approved": None, "validation_passed": False})
    if not state.get("training_plan"):
        return Command(goto="planner")
    if not state.get("training_run_id"):
        return Command(goto="executor")
    if state.get("evaluation_passed") is None:
        return Command(goto="evaluation")
    if state.get("evaluation_report_audit") is None:
        return Command(goto="report_writer")
    if state.get("evaluation_passed") is False:
        return Command(goto=END)            # Deterministic rejection — audit written, no Gate 2
    if state.get("deployment_approved") is None:
        return Command(goto="deployment_approval")
    if state.get("deployment_approved") is False:
        return Command(goto=END)            # Gate 2 rejection — terminal, no retry
    if state.get("deployment_decision") == "pending":
        return Command(goto="deployer")
    return Command(goto=END)
```

Each node manages its own internal retries (e.g. `data_validator` already does 3 attempts). The controller only sees final pass/fail and aborts on error_message.

### Training Executor

Already lives in `training/executor.py`. Unchanged except for a thin graph wrapper.

### Evaluation & Promotion Module (new)

New module `evaluation/promotion.py`. Pure Python — fetches champion via MLflow, applies thresholds, sets `evaluation_passed`.

```python
def evaluate_promotion(state: AgentState) -> dict:
    problem_type = state["problem_type"]
    metric, ascending = _metric_for_problem_type(problem_type)
    candidate = state["training_metrics"]
    champion = _fetch_current_champion(metric, ascending)
    passed = _apply_thresholds(problem_type, candidate, champion)
    return {
        "evaluation_passed": passed,
        "candidate_metrics": candidate,
        "champion_metrics": champion,
        "thresholds_applied": _thresholds_for(problem_type),
    }
```

### Deployment Module

Already exists inline in `mlops_graph.py`. Extracted to `deployment/deployer.py` as `run_deployer(state) -> dict`. Pure MLflow API calls + HITL `interrupt()` handled at node level.

### Experience Pool Update

Already lives in `training/experience.py`. Called as final deterministic step.

## Module Structure Changes

```
src/mlops_agents/
├── agents/
│   ├── data_agent.py            # unchanged (build_data_agent())
│   ├── planner.py               # CHANGE: add build_planner_agent(); strip node logic
│   ├── evaluation_agent.py      # DELETE — replaced by deterministic evaluation + report_writer
│   ├── deployment_agent.py      # DELETE — dead code
│   ├── supervisor.py            # DELETE — replaced by workflow_controller
│   └── registry.py              # CHANGE: only LLM-using nodes
├── evaluation/                  # NEW
│   ├── promotion.py             # evaluate_promotion(state) -> dict (deterministic)
│   └── report_writer.py         # build_report_writer() + EvaluationReport schema (LLM node)
├── deployment/                  # NEW
│   └── deployer.py              # run_deployer(state) -> dict (no LLM, just MLflow + HITL)
├── graphs/
│   └── mlops_graph.py           # CHANGE: thin wrappers + workflow_controller + new topology
└── state/
    └── agent_state.py           # CHANGE: add dataset_approved, evaluation_report, etc.
```

## Registry

Only LLM-using nodes are registered:

```python
@lru_cache(maxsize=None)
def get_agent(name: str) -> Any:
    if name == "data_validator":
        from mlops_agents.agents.data_agent import build_data_agent
        return build_data_agent()
    if name == "planner":
        from mlops_agents.agents.planner import build_planner_agent
        return build_planner_agent()
    if name == "report_writer":
        from mlops_agents.evaluation.report_writer import build_report_writer
        return build_report_writer()
    raise ValueError(
        f"Unknown agent: '{name}'. Valid: data_validator, planner, report_writer"
    )
```

## Graph Node Template

LLM nodes:
```python
def planner_node(state: AgentState) -> Command[Literal["workflow_controller"]]:
    agent = get_agent("planner")
    ctx = _build_planner_context(state)
    output = agent.invoke(ctx)
    return Command(update=_planner_state_updates(output), goto="workflow_controller")
```

Deterministic nodes:
```python
def evaluation_node(state: AgentState) -> Command[Literal["workflow_controller"]]:
    result = evaluate_promotion(state)
    return Command(update=result, goto="workflow_controller")
```

All nodes route back to `workflow_controller`.

## Settings & LLM Cleanup

**Remove from `settings.py`:**
- `openai_model_supervisor` — no LLM supervisor anymore
- `openai_model_trainer` — executor is pure Python
- `openai_model_evaluator` — evaluation is pure Python
- `openai_model_deployer` — deployer is deterministic

**Add to `settings.py`:**
- `openai_model_report_writer` — for the audit LLM node

**Final `model_map` in `llm.py`:**
```python
model_map = {
    "data_validator": settings.openai_model_data_validator,
    "planner":        settings.openai_model_planner,
    "report_writer":  settings.openai_model_report_writer,
}
```

**Update `api/services/pipeline.py` run_info event:**
```python
"models": {
    "data_validator": settings.openai_model_data_validator,
    "planner":        settings.openai_model_planner,
    "report_writer":  settings.openai_model_report_writer,
}
```

The UI will show 3 LLM-using nodes (down from 5). The deterministic nodes (`workflow_controller`, `executor`, `evaluation`, `deployer`) appear in the pipeline visualization but without a model label.

## HITL Gates

Each HITL gate is its own dedicated graph node — a pure HITL node that calls `interrupt()` and writes an approval field to state. This keeps agents focused on their domain work and makes gates explicit in the graph topology.

| Gate | Node | Sets state field | Purpose |
|---|---|---|---|
| **Gate 1** | `dataset_approval_node` | `dataset_approved: bool` | Human approves the target dataset before training begins |
| **Gate 2** | `deployment_approval_node` | `deployment_approved: bool` | Human approves deployment based on the evidence-aligned audit report |

**Pattern:**
```python
def dataset_approval_node(state: AgentState) -> Command:
    approval = interrupt({
        "type": "dataset_approval",
        "preview": state["dataset_summary"],
        "validation_report": state["validation_report"],
    })
    approved = bool(approval.get("approved", False))
    return Command(
        update={
            "dataset_approved": approved,
            "dataset_rejection_comment": "" if approved else approval.get("comment", ""),
        },
        goto="workflow_controller",
    )
```

The `data_validator` no longer embeds its own HITL — its internal retry loop for validation failures still exists, but the human approval gate is extracted. On rejection, the controller routes back to `data_validator`, which reads `dataset_rejection_comment` from state and injects it into the agent's input context. The controller clears `dataset_rejection_comment` once the retry completes successfully.

**Rejection feedback contract:**
- `dataset_approval_node` writes `dataset_rejection_comment` on rejection.
- `data_validator_node` reads it on retry and prepends it to the agent's input message as: "Your previous attempt was rejected by a human reviewer. Their feedback: {comment}. Please try a different approach."
- After a successful retry, the field is cleared by the controller via state update.

**Retry artifact behavior:**
- `data_validator_node` writes its processed CSV to the same `processed_dataset_path` on every attempt — the previous attempt's CSV is overwritten on retry.
- This matches the spec's "rejection feedback in state" principle: history of rejected attempts lives in `dataset_rejection_comment` + messages, not as parallel artifacts on disk.
- The thesis cares about the final approved dataset, not intermediate rejected attempts.

## Dead Code Removed

| Item | Location | Reason |
|---|---|---|
| `build_deployment_agent()` | `agents/deployment_agent.py` | Never called; deployer is deterministic |
| `build_evaluation_agent()` | `agents/evaluation_agent.py` | Replaced by deterministic `evaluate_promotion()` + `report_writer` |
| `supervisor_node` (LLM) | `agents/supervisor.py` | Replaced by `workflow_controller` state machine |
| `RouterOutput` | `state/schemas.py` | No more LLM routing |
| `openai_model_supervisor` | `settings.py` | No LLM supervisor |
| `openai_model_trainer` | `settings.py` | No trainer LLM |
| `openai_model_evaluator` | `settings.py` | Evaluation is deterministic |
| `openai_model_deployer` | `settings.py` | Deployer is deterministic |
| `"trainer"` / `"deployer"` / `"evaluator"` / `"supervisor"` | `llm.py` model_map | No matching LLM agents |
| `get_router_llm()` | `utils/llm.py` | No more routing LLM |
| `evaluation_agent.yaml` prompt | `prompts/` | Replaced by report_writer prompt |

## Thesis Framing

The architecture uses **two true agents** (data validation, model planning) and **one evidence-alignment LLM node** (evaluation report). Human approval is required after target dataset generation and before deployment. Routing, training, promotion, and deployment remain deterministic to preserve reproducibility and control.

The sophisticated design choice is **knowing when NOT to use an agent**. Forcing LLMs into deterministic decisions (number comparison, fixed-order context assembly, API calls) trades reproducibility and correctness for the appearance of agentic behavior. This architecture restricts LLM use to:

1. **Multi-step exploration with branching** (data validation)
2. **Reasoning over evidence to produce structured plans** (model planning)
3. **Triangulating multiple evidence sources for human-readable audits** (evaluation report)

Everything else is Python.

## Migration: Hard Cut

This is a **breaking change to the runtime architecture only**. No migration script needed.

**Invalidated by this refactor:**
- LangGraph checkpoint state (uses `InMemorySaver` — resets per process; no on-disk checkpoints exist yet)
- `RouterOutput` schema and any state with `next: "<old-supervisor-route>"`

**Untouched by this refactor:**
- MLflow experiment data, run history, registered models
- Experience pool: `storage/mlops_metadata.db`, `experience_pool/*.json`, `ExperienceRecord` schema
- ML rules knowledge base
- All training executor logic and outputs

The executor continues writing the same `ExperienceRecord` shape; the planner continues calling `pool.find_similar()` over the same data. Historical experience pool entries remain queryable by the new planner agent without changes.

## Scope: Backend-Only

This refactor is **backend-only**. The frontend keeps consuming the existing SSE event shapes (`hitl_request`, `routing`, `agent_reasoning`, `planner_context`, `run_info`, `run_complete`). The new approval nodes emit `hitl_request` events with the same payload structure as before:

| Old emitter | New emitter | `hitl_request.agent` value (preserved) |
|---|---|---|
| `data_validator_node` (embedded HITL) | `dataset_approval_node` | `"data_validation"` |
| `deployer_node` (embedded HITL) | `deployment_approval_node` | `"deployer"` |

The `run_info` event's `models` map is reduced to the 3 LLM-using nodes (`data_validator`, `planner`, `report_writer`). The UI will simply show fewer model labels — existing rendering code handles missing keys gracefully.

A follow-up issue will be filed for frontend updates:
- Pipeline visualization showing the new node graph (controller + gates + LLM nodes + deterministic modules)
- `EvaluationReport` panel rendering (new structured fields)
- Updated model-label row in the run header

## Testing

- Existing unit tests for `data_agent` remain unchanged.
- New unit tests for `build_planner_agent()` and `build_report_writer()` — mock the LLM, assert constructor succeeds.
- New unit tests for `workflow_controller` — pure state machine, deterministic, exhaustive case coverage.
- New unit tests for `evaluate_promotion` — real metric dicts, no mocks needed.
- New unit tests for `run_deployer` — mock MLflow client.
- Delete tests for removed supervisor LLM routing and evaluation_agent react logic.
