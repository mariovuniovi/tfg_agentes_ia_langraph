# Agent Context Isolation Design

**Date:** 2026-05-03
**Status:** Approved
**Branch:** claude/develop

## Problem

All four worker agents currently receive `list(state["messages"])` as their context when invoked. This means every subsequent agent inherits the full conversation history of all prior agents — including their internal summaries, tool results, and reasoning — even though none of it is relevant to their task.

As the pipeline progresses, each agent receives more irrelevant context:
- Trainer sees data_validator's column mapping reasoning
- Evaluator sees trainer's hyperparameter tuning notes
- Deployer sees everything from all three prior stages

This wastes tokens, pollutes agent reasoning with irrelevant history, and couples agents that should be isolated. It also conflates two distinct concerns: **workflow state** (the facts the pipeline has established) and **conversation history** (the audit trail of how it got there).

## Design Principles

1. **Artifacts by reference** — agents receive file paths, run IDs, and model URIs. Never file contents, never CSV rows.
2. **State as memory** — each node reads from typed `AgentState` fields. Not from prior messages.
3. **Summaries as audit** — each node writes one compact `HumanMessage` to `state["messages"]`. The supervisor reads this for routing. No tool call payloads ever reach `messages`.
4. **Full schema, summarized data** — the schema JSON is passed in full to the data_validator (it is the validation contract; summarising it causes missed constraints). Raw dataset content stays on disk and is accessed via tools.

## What Does Not Change

- `state["messages"]` and `InMemorySaver` are untouched — they remain the HITL resumability mechanism and the supervisor's audit trail.
- The supervisor continues to read `state["messages"]` (which already contains only the compact one-line summaries each node writes). Its logic does not change.
- The data_validator's context content is unchanged — it still receives the full schema JSON and raw file paths. What changes is that it no longer also receives `list(state["messages"])` prepended.
- The internal ReAct loop of each agent (tool calls, tool results) is unaffected — these have never been written to `state["messages"]`.

## AgentState Changes

One new field added to `agent_state.py`:

```python
dataset_summary: dict  # {row_count, column_names, dtypes, null_counts}
```

Built deterministically in `data_validator_node` after the agent runs, by reading the canonical CSV with pandas. Written to state. Read by the trainer's context builder.

No other state fields are added or removed.

## Per-Agent Payload Contracts

### data_validator
```
Raw files: [list of raw CSV file paths]
Schema path: /absolute/path/to/schema.json
Target schema: {full schema JSON — columns, types, nullable, min/max, allowed_values, mapping_hints, is_key}
```
Source: `state["dataset_paths"]`, schema file on disk.

### trainer
```
Canonical dataset: data/processed/iris_classification.csv
Dataset summary: {row_count, column_names, dtypes, null_counts}
```
Source: `state["dataset_path"]`, `state["dataset_summary"]`.

### evaluator
```
Training run ID: <mlflow run id>
Trained model path: <local model artifact path>
Training metrics: {model_type, train_accuracy, val_accuracy}
```
Source: `state["training_run_id"]`, `state["trained_model_path"]`, `state["training_metrics"]`.

### deployer
```
Best model URI: runs:/<run_id>/model  (MLflow artifact reference)
Training run ID: <mlflow run id>
Evaluation report: {candidate_metrics, candidate_run_id, baseline_metrics}
```
Source: `state["best_model_uri"]`, `state["training_run_id"]`, `state["evaluation_report"]`.

### supervisor
```
state["messages"]   — compact one-line HumanMessage summaries (narrative audit trail)
state_snapshot      — structured routing-relevant state fields injected as final message:
  {
    "validation_passed":   bool,
    "evaluation_passed":   bool,
    "deployment_decision": "pending" | "approved" | "rejected",
    "error_message":       str,
    "training_run_id":     str
  }
```

The snapshot is injected as a `HumanMessage` appended after `state["messages"]` when building the supervisor's LLM input. This ensures routing decisions are made on typed facts, not inferred from text summaries.

**Why both:** the narrative summaries give the supervisor reasoning context ("why did this fail?"); the structured snapshot gives it deterministic routing signals (`error_message` set → FINISH, `validation_passed=False` → do not retry). Without the snapshot, the supervisor infers typed facts from text — fragile. Without the summaries, it has no reasoning context for edge cases.

## Context Builder Functions

Four private functions added to `mlops_graph.py`, one per worker agent. Each returns a single `HumanMessage` built from state fields.

```python
def _build_data_validator_context(state: AgentState) -> HumanMessage
def _build_trainer_context(state: AgentState) -> HumanMessage
def _build_evaluator_context(state: AgentState) -> HumanMessage
def _build_deployer_context(state: AgentState) -> HumanMessage
```

Each node changes its `agent.invoke()` call from:
```python
agent.invoke({"messages": list(state["messages"]) + [context_message]})
```
to:
```python
agent.invoke({"messages": [_build_*_context(state)]})
```

## dataset_summary Construction

Built in `data_validator_node` after the agent has produced the canonical CSV, using pure pandas — no LLM call, no tool invocation:

```python
df = pd.read_csv(processed_path)
dataset_summary = {
    "row_count": len(df),
    "column_names": list(df.columns),
    "dtypes": df.dtypes.astype(str).to_dict(),
    "null_counts": df.isnull().sum().to_dict(),
}
```

If `processed_path` is empty (validation failed), `dataset_summary` is set to `{}`.

## Files Changed

| File | Change |
|------|--------|
| `src/mlops_agents/state/agent_state.py` | Add `dataset_summary: dict` field |
| `src/mlops_agents/graphs/mlops_graph.py` | Add 4 `_build_*_context` functions; update all 4 worker nodes to use them; build `dataset_summary` in `data_validator_node`; inject structured state snapshot in `supervisor_node` |
| `src/mlops_agents/agents/supervisor.py` | Inject structured state snapshot into supervisor LLM input |
| `src/mlops_agents/prompts/supervisor.yaml` | Clarify rule 5 references `error_message` field directly (already present but reinforce) |
| `tests/test_graphs/test_node_state_extraction.py` | Update node tests to verify context isolation and `dataset_summary` in state |
| `tests/test_agents/test_supervisor.py` | Add test verifying snapshot fields appear in supervisor input |

## What This Enables

- **Token efficiency**: each agent receives the minimum context needed, not an ever-growing history
- **Reasoning clarity**: agents cannot hallucinate based on prior agents' tool outputs
- **Testability**: each context builder is a pure function of state — trivial to unit test
- **Auditability**: `state["messages"]` remains a clean, compact record of pipeline progress
- **Research clarity**: makes the separation of workflow state vs conversation history explicit and demonstrable
