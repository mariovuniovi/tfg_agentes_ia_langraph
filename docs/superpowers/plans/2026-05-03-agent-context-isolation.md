# Agent Context Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `list(state["messages"])` in all four worker nodes with purpose-built context messages derived from typed AgentState fields, and inject a structured state snapshot into the supervisor's LLM input.

**Architecture:** Each worker node calls a private `_build_*_context(state)` pure function that constructs a single `HumanMessage` from typed state fields only. The supervisor node appends a structured `HumanMessage` snapshot after `state["messages"]` so routing decisions are based on typed facts, not text inference. A new `dataset_summary` field is built deterministically with pandas in `data_validator_node` and passed to the trainer by reference.

**Tech Stack:** LangGraph `AgentState` (TypedDict), `HumanMessage` / `SystemMessage` from langchain-core, pandas for `dataset_summary`, existing `_extract_tool_json` helper.

---

## File Map

| File | Change |
|------|--------|
| `src/mlops_agents/state/agent_state.py` | Add `dataset_summary: dict` field |
| `src/mlops_agents/graphs/mlops_graph.py` | Add 4 `_build_*_context` functions; update all 4 worker nodes; build `dataset_summary` in `data_validator_node` |
| `src/mlops_agents/agents/supervisor.py` | Inject structured state snapshot into LLM input |
| `src/mlops_agents/prompts/supervisor.yaml` | Reinforce rule 5 with explicit field reference |
| `tests/test_graphs/test_node_state_extraction.py` | Add `dataset_summary` to `_make_state`; add isolation + `dataset_summary` tests |
| `tests/test_agents/test_supervisor.py` | Add `dataset_summary` to `make_state`; add snapshot injection test |

---

### Task 1: Add `dataset_summary` to AgentState and update test helpers

**Files:**
- Modify: `src/mlops_agents/state/agent_state.py`
- Modify: `tests/test_graphs/test_node_state_extraction.py` (lines 80-98)
- Modify: `tests/test_agents/test_supervisor.py` (lines 11-30)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_graphs/test_node_state_extraction.py` after the existing `_make_state` function:

```python
def test_agent_state_has_dataset_summary_field():
    from mlops_agents.state.agent_state import AgentState
    import typing
    hints = typing.get_type_hints(AgentState)
    assert "dataset_summary" in hints
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_graphs/test_node_state_extraction.py::test_agent_state_has_dataset_summary_field -v
```

Expected: FAIL with `AssertionError`

- [ ] **Step 3: Add `dataset_summary` to AgentState**

In `src/mlops_agents/state/agent_state.py`, add after `error_message: str`:

```python
    # Context isolation — built deterministically by data_validator_node
    dataset_summary: dict  # {row_count, column_names, dtypes, null_counts}
```

Full file after change:

```python
"""Shared LangGraph state definition for the MLOps pipeline."""

import operator
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """Shared state passed between all nodes in the MLOps graph.

    Fields updated by reducers (operator.add) accumulate across nodes.
    All other plain fields overwrite on update — this is intentional.
    """

    # Message history — operator.add appends instead of overwriting
    messages: Annotated[list[BaseMessage], operator.add]

    # Supervisor routing — which node to visit next
    next: str

    # Pipeline inputs
    dataset_paths: list[str]   # raw CSV files provided by user
    dataset_path: str          # canonical CSV written by data_validator_node

    # Stage outputs (set by each agent node)
    validation_passed: bool
    validation_report: dict  # Evidently AI report as dict
    trained_model_path: str
    training_run_id: str      # MLflow run ID
    training_metrics: dict
    evaluation_passed: bool
    evaluation_report: dict
    best_model_uri: str

    # Deployment
    deployment_decision: str  # "approved" | "rejected" | "pending"
    deployment_status: str

    # Error tracking
    error_message: str
    agent_attempt_counts: dict[str, int]  # {"data_validator": 1, "trainer": 2, …}

    # Context isolation — built deterministically by data_validator_node
    dataset_summary: dict  # {row_count, column_names, dtypes, null_counts}
```

- [ ] **Step 4: Update `_make_state()` in `test_node_state_extraction.py`**

Find the `_make_state` function (lines 80-98) and add `"dataset_summary": {}` to the returned dict:

```python
def _make_state() -> dict:
    return {
        "messages": [HumanMessage(content="Run pipeline on iris.csv")],
        "next": "",
        "dataset_paths": ["./data/samples/iris.csv"],
        "dataset_path": "./data/samples/iris.csv",
        "validation_passed": False,
        "validation_report": {},
        "trained_model_path": "",
        "training_run_id": "",
        "training_metrics": {},
        "evaluation_passed": False,
        "evaluation_report": {},
        "best_model_uri": "",
        "deployment_decision": "pending",
        "deployment_status": "",
        "error_message": "",
        "agent_attempt_counts": {},
        "dataset_summary": {},
    }
```

- [ ] **Step 5: Update `make_state()` in `test_supervisor.py`**

Find the `make_state` function (lines 11-30) and add `"dataset_summary": {}`:

```python
def make_state(messages=None, **kwargs):
    base = {
        "messages": messages or [HumanMessage(content="Run the pipeline.")],
        "next": "",
        "dataset_path": "test.csv",
        "validation_passed": False,
        "validation_report": {},
        "trained_model_path": "",
        "training_run_id": "",
        "training_metrics": {},
        "evaluation_passed": False,
        "evaluation_report": {},
        "best_model_uri": "",
        "deployment_decision": "pending",
        "deployment_status": "",
        "error_message": "",
        "agent_attempt_counts": {},
        "dataset_summary": {},
    }
    base.update(kwargs)
    return base
```

- [ ] **Step 6: Run tests to verify they pass**

```
uv run pytest tests/test_graphs/test_node_state_extraction.py tests/test_agents/test_supervisor.py -v
```

Expected: all existing tests pass + new `test_agent_state_has_dataset_summary_field` passes

- [ ] **Step 7: Commit**

```bash
git add src/mlops_agents/state/agent_state.py tests/test_graphs/test_node_state_extraction.py tests/test_agents/test_supervisor.py
git commit -m "feat: add dataset_summary field to AgentState"
```

---

### Task 2: Add four `_build_*_context` pure functions

**Files:**
- Modify: `src/mlops_agents/graphs/mlops_graph.py` (after line 49, before `data_validator_node`)
- Modify: `tests/test_graphs/test_node_state_extraction.py` (add tests at end)

- [ ] **Step 1: Write the failing tests**

Add to the end of `tests/test_graphs/test_node_state_extraction.py`:

```python
# ---------------------------------------------------------------------------
# Context builder pure functions
# ---------------------------------------------------------------------------


def test_build_data_validator_context_includes_raw_files():
    from mlops_agents.graphs.mlops_graph import _build_data_validator_context

    state = _make_state()
    msg = _build_data_validator_context(state)
    assert "./data/samples/iris.csv" in msg.content
    assert "Raw files:" in msg.content


def test_build_data_validator_context_includes_schema_path():
    from mlops_agents.graphs.mlops_graph import _build_data_validator_context

    state = _make_state()
    msg = _build_data_validator_context(state)
    assert "Schema path:" in msg.content
    assert "Target schema:" in msg.content


def test_build_trainer_context_includes_dataset_path_and_summary():
    from mlops_agents.graphs.mlops_graph import _build_trainer_context

    state = _make_state()
    state["dataset_path"] = "data/processed/iris.csv"
    state["dataset_summary"] = {"row_count": 150, "column_names": ["a", "b"]}
    msg = _build_trainer_context(state)
    assert "data/processed/iris.csv" in msg.content
    assert "row_count" in msg.content
    assert "150" in msg.content


def test_build_evaluator_context_includes_run_id_and_metrics():
    from mlops_agents.graphs.mlops_graph import _build_evaluator_context

    state = _make_state()
    state["training_run_id"] = "abc123"
    state["trained_model_path"] = "models/rf.pkl"
    state["training_metrics"] = {"val_accuracy": 0.95}
    msg = _build_evaluator_context(state)
    assert "abc123" in msg.content
    assert "models/rf.pkl" in msg.content
    assert "0.95" in msg.content


def test_build_deployer_context_includes_model_uri_and_report():
    from mlops_agents.graphs.mlops_graph import _build_deployer_context

    state = _make_state()
    state["best_model_uri"] = "runs:/abc123/model"
    state["training_run_id"] = "abc123"
    state["evaluation_report"] = {"candidate_metrics": {"accuracy": 0.97}}
    msg = _build_deployer_context(state)
    assert "runs:/abc123/model" in msg.content
    assert "abc123" in msg.content
    assert "0.97" in msg.content
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_graphs/test_node_state_extraction.py::test_build_data_validator_context_includes_raw_files -v
```

Expected: FAIL with `ImportError` (function not yet defined)

- [ ] **Step 3: Add the four context builder functions to `mlops_graph.py`**

After line 49 (after `_extract_tool_json`), before `data_validator_node`, add:

```python
def _build_data_validator_context(state: AgentState) -> HumanMessage:
    from pathlib import Path as _Path
    from mlops_agents.config.settings import settings

    schema_file = _Path("data/schemas") / f"{settings.dataset_schema}.json"
    schema_json = schema_file.read_text() if schema_file.exists() else "{}"
    return HumanMessage(content=(
        f"Raw files: {json.dumps(state.get('dataset_paths', []))}\n"
        f"Schema path: {str(schema_file.resolve())}\n"
        f"Target schema:\n{schema_json}"
    ))


def _build_trainer_context(state: AgentState) -> HumanMessage:
    return HumanMessage(content=(
        f"Canonical dataset: {state.get('dataset_path', '')}\n"
        f"Dataset summary: {json.dumps(state.get('dataset_summary') or {})}"
    ))


def _build_evaluator_context(state: AgentState) -> HumanMessage:
    return HumanMessage(content=(
        f"Training run ID: {state.get('training_run_id', '')}\n"
        f"Trained model path: {state.get('trained_model_path', '')}\n"
        f"Training metrics: {json.dumps(state.get('training_metrics') or {})}"
    ))


def _build_deployer_context(state: AgentState) -> HumanMessage:
    return HumanMessage(content=(
        f"Best model URI: {state.get('best_model_uri', '')}\n"
        f"Training run ID: {state.get('training_run_id', '')}\n"
        f"Evaluation report: {json.dumps(state.get('evaluation_report') or {})}"
    ))
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/test_graphs/test_node_state_extraction.py -k "build_" -v
```

Expected: all 5 context builder tests pass

- [ ] **Step 5: Commit**

```bash
git add src/mlops_agents/graphs/mlops_graph.py tests/test_graphs/test_node_state_extraction.py
git commit -m "feat: add _build_*_context pure functions for agent context isolation"
```

---

### Task 3: Update `data_validator_node` — extract context builder, build `dataset_summary`

**Files:**
- Modify: `src/mlops_agents/graphs/mlops_graph.py` (`data_validator_node`, lines 52-160)
- Modify: `tests/test_graphs/test_node_state_extraction.py` (add `dataset_summary` output test)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_graphs/test_node_state_extraction.py` after the context builder tests:

```python
def test_data_validator_node_builds_dataset_summary_on_success():
    """data_validator_node must set dataset_summary in state when validation passes."""
    import tempfile, os
    import pandas as pd
    from mlops_agents.graphs.mlops_graph import data_validator_node

    # Write a small CSV so pandas can actually read it
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("a,b\n1,2\n3,4\n")
        tmp_path = f.name

    validation_json = json.dumps({"passed": True, "output_path": tmp_path})
    mock_result = {
        "messages": [
            ToolMessage(content=validation_json, tool_call_id="1", name="validate_against_schema"),
            AIMessage(content="Validation passed."),
        ]
    }

    try:
        with patch("mlops_agents.graphs.mlops_graph.get_agent") as mock_get_agent, \
             patch("mlops_agents.graphs.mlops_graph.interrupt", return_value={"approved": True, "comment": ""}):
            mock_agent = MagicMock()
            mock_agent.invoke.return_value = mock_result
            mock_get_agent.return_value = mock_agent

            state = _make_state()
            state["dataset_paths"] = [tmp_path]
            command = data_validator_node(state)
    finally:
        os.unlink(tmp_path)

    assert "dataset_summary" in command.update
    assert command.update["dataset_summary"]["row_count"] == 2
    assert "a" in command.update["dataset_summary"]["column_names"]
    assert "b" in command.update["dataset_summary"]["column_names"]


def test_data_validator_node_sets_empty_dataset_summary_on_failure():
    """dataset_summary must be {} when validation fails."""
    from mlops_agents.graphs.mlops_graph import data_validator_node

    mock_result = {"messages": [AIMessage(content="Could not validate.")]}
    with patch("mlops_agents.graphs.mlops_graph.get_agent") as mock_get_agent:
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = mock_result
        mock_get_agent.return_value = mock_agent

        command = data_validator_node(_make_state())

    assert command.update.get("dataset_summary") == {}


def test_data_validator_node_invokes_agent_with_isolated_context():
    """data_validator_node must NOT pass state['messages'] to agent.invoke."""
    from mlops_agents.graphs.mlops_graph import data_validator_node

    validation_json = json.dumps({"passed": True, "output_path": ""})
    mock_result = {
        "messages": [
            ToolMessage(content=validation_json, tool_call_id="1", name="validate_against_schema"),
            AIMessage(content="Validation passed."),
        ]
    }
    with patch("mlops_agents.graphs.mlops_graph.get_agent") as mock_get_agent, \
         patch("mlops_agents.graphs.mlops_graph.interrupt", return_value={"approved": True, "comment": ""}):
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = mock_result
        mock_get_agent.return_value = mock_agent

        state = _make_state()
        state["messages"] = [
            HumanMessage(content="Prior supervisor message 1"),
            HumanMessage(content="Prior supervisor message 2"),
        ]
        data_validator_node(state)

    call_messages = mock_agent.invoke.call_args[0][0]["messages"]
    assert len(call_messages) == 1, (
        f"Expected exactly 1 context message, got {len(call_messages)}. "
        "Prior state['messages'] must not be forwarded to worker agents."
    )
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_graphs/test_node_state_extraction.py::test_data_validator_node_builds_dataset_summary_on_success -v
```

Expected: FAIL (no `dataset_summary` in `command.update`)

- [ ] **Step 3: Update `data_validator_node`**

Replace the existing `data_validator_node` body in `mlops_graph.py`. The new version:
1. Uses `_build_data_validator_context(state)` instead of the inline context build
2. Passes `[_build_data_validator_context(state)]` (not `list(state["messages"]) + [context_message]`) to `agent.invoke`
3. Builds `dataset_summary` with pandas after the agent runs
4. Includes `dataset_summary` in every `Command.update`

```python
def data_validator_node(state: AgentState) -> Command[Literal["supervisor"]]:
    import pandas as pd

    agent = get_agent("data_validator")
    result = agent.invoke({"messages": [_build_data_validator_context(state)]})
    final_message = result["messages"][-1].content

    quality_report: dict = _extract_tool_json(result["messages"], "check_data_quality")
    mapping_result: dict = _extract_tool_json(result["messages"], "apply_column_mapping")
    validation_result: dict = _extract_tool_json(result["messages"], "validate_against_schema")
    imputation_result: dict = _extract_tool_json(result["messages"], "impute_missing_values")

    processed_path = mapping_result.get("output_path", "")
    validation_passed = bool(validation_result.get("passed", False))

    dataset_summary: dict = {}
    if processed_path:
        try:
            df = pd.read_csv(processed_path)
            dataset_summary = {
                "row_count": len(df),
                "column_names": list(df.columns),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "null_counts": df.isnull().sum().to_dict(),
            }
        except Exception:
            pass

    base_update = {
        "messages": [HumanMessage(content=final_message, name="data_validator")],
        "validation_report": quality_report,
        "validation_passed": validation_passed,
        "dataset_path": processed_path,
        "dataset_summary": dataset_summary,
    }

    if not validation_passed:
        error_msg = f"Data validation failed after auto-fix attempt: {final_message}"
        logger.warning("[data_validator] validation failed — aborting without HITL")
        return Command(
            update={**base_update, "error_message": error_msg},
            goto="supervisor",
        )

    # Validation passed — build preview and surface HITL for human sign-off.
    preview: dict = {"shape": [0, 0], "columns": [], "sample_rows": []}
    if processed_path:
        try:
            df = pd.read_csv(processed_path)
            preview = {
                "shape": list(df.shape),
                "columns": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns],
                "sample_rows": json.loads(df.head(20).to_json(orient="records")),
            }
        except Exception:
            pass

    counts = dict(state.get("agent_attempt_counts") or {})
    attempt = counts.get("data_validator", 1)

    missing_vals: dict = {}
    if isinstance(quality_report, dict):
        missing_vals = quality_report.get("missing_values", {})

    approval = interrupt({
        "type": "data_validation",
        "question": "Review the processed dataset before training begins.",
        "attempt": attempt,
        "dataset_preview": preview,
        "validation_summary": {
            "passed": True,
            "missing_values": missing_vals,
            "schema_validated": True,
        },
        "imputation_applied": imputation_result,
    })

    if approval.get("approved", False):
        logger.info("[data_validator] approved — routing back to supervisor")
        return Command(update=base_update, goto="supervisor")

    comment = approval.get("comment", "")
    rejection_text = (
        f"Dataset rejected by human reviewer. Comment: {comment}"
        if comment
        else "Dataset rejected by human reviewer."
    )
    logger.info(f"[data_validator] rejected — comment: {comment!r}")
    return Command(
        update={
            **base_update,
            "messages": [
                HumanMessage(content=final_message, name="data_validator"),
                HumanMessage(content=rejection_text, name="data_validator"),
            ],
            "validation_passed": False,
            "error_message": rejection_text,
        },
        goto="supervisor",
    )
```

Note: The `from pathlib import Path as _Path` and `from mlops_agents.config.settings import settings` local imports are removed from the node body — they are now inside `_build_data_validator_context`. Only `import pandas as pd` remains as a local import.

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/test_graphs/test_node_state_extraction.py -v
```

Expected: all tests pass including the 3 new ones

- [ ] **Step 5: Commit**

```bash
git add src/mlops_agents/graphs/mlops_graph.py tests/test_graphs/test_node_state_extraction.py
git commit -m "feat: context-isolate data_validator_node, build dataset_summary in state"
```

---

### Task 4: Update trainer, evaluator, and deployer nodes

**Files:**
- Modify: `src/mlops_agents/graphs/mlops_graph.py` (`trainer_node`, `evaluator_node`, `deployer_node`)
- Modify: `tests/test_graphs/test_node_state_extraction.py` (add isolation tests)

- [ ] **Step 1: Write the failing tests**

Add to the end of `tests/test_graphs/test_node_state_extraction.py`:

```python
def test_trainer_node_invokes_agent_with_isolated_context():
    """trainer_node must pass exactly one context message — not state['messages']."""
    from mlops_agents.graphs.mlops_graph import trainer_node

    train_json = json.dumps({
        "model_type": "random_forest", "model_path": "rf.pkl",
        "train_accuracy": 0.98, "val_accuracy": 0.95,
    })
    mock_result = {
        "messages": [
            ToolMessage(content=train_json, tool_call_id="1", name="train_model"),
            AIMessage(content="Training complete."),
        ]
    }
    with patch("mlops_agents.graphs.mlops_graph.get_agent") as mock_get_agent:
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = mock_result
        mock_get_agent.return_value = mock_agent

        state = _make_state()
        state["messages"] = [
            HumanMessage(content="supervisor msg 1"),
            HumanMessage(content="supervisor msg 2"),
        ]
        trainer_node(state)

    call_messages = mock_agent.invoke.call_args[0][0]["messages"]
    assert len(call_messages) == 1


def test_evaluator_node_invokes_agent_with_isolated_context():
    """evaluator_node must pass exactly one context message — not state['messages']."""
    from mlops_agents.graphs.mlops_graph import evaluator_node

    runs_json = json.dumps([
        {"run_id": "run1", "metrics": {"accuracy": 0.97}, "params": {}, "model_uri": "runs:/run1/model"},
    ])
    mock_result = {
        "messages": [
            ToolMessage(content=runs_json, tool_call_id="1", name="get_best_run"),
            AIMessage(content="Evaluation complete."),
        ]
    }
    with patch("mlops_agents.graphs.mlops_graph.get_agent") as mock_get_agent:
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = mock_result
        mock_get_agent.return_value = mock_agent

        state = _make_state()
        state["messages"] = [HumanMessage(content="prior msg 1"), HumanMessage(content="prior msg 2")]
        evaluator_node(state)

    call_messages = mock_agent.invoke.call_args[0][0]["messages"]
    assert len(call_messages) == 1


def test_deployer_node_invokes_agent_with_isolated_context():
    """deployer_node must pass exactly one context message — not state['messages']."""
    from mlops_agents.graphs.mlops_graph import deployer_node

    mock_result = {
        "messages": [AIMessage(content="Model registered as challenger.")]
    }
    with patch("mlops_agents.graphs.mlops_graph.get_agent") as mock_get_agent, \
         patch("mlops_agents.graphs.mlops_graph.interrupt", return_value={"approved": True}):
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = mock_result
        mock_get_agent.return_value = mock_agent

        state = _make_state()
        state["messages"] = [HumanMessage(content="prior 1"), HumanMessage(content="prior 2")]
        deployer_node(state)

    call_messages = mock_agent.invoke.call_args[0][0]["messages"]
    assert len(call_messages) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_graphs/test_node_state_extraction.py::test_trainer_node_invokes_agent_with_isolated_context -v
```

Expected: FAIL (trainer currently passes `list(state["messages"])` → 3 messages, not 1)

- [ ] **Step 3: Update `trainer_node`**

Replace the `agent.invoke` call in `trainer_node` (line 165):

```python
def trainer_node(state: AgentState) -> Command[Literal["supervisor"]]:
    agent = get_agent("trainer")
    result = agent.invoke({"messages": [_build_trainer_context(state)]})
    final_message = result["messages"][-1].content
    # ... rest unchanged
```

- [ ] **Step 4: Update `evaluator_node`**

Replace the `agent.invoke` call in `evaluator_node` (line 191):

```python
def evaluator_node(state: AgentState) -> Command[Literal["supervisor"]]:
    agent = get_agent("evaluator")
    result = agent.invoke({"messages": [_build_evaluator_context(state)]})
    final_message = result["messages"][-1].content
    # ... rest unchanged
```

- [ ] **Step 5: Update `deployer_node`**

Replace the `agent.invoke` call in `deployer_node` (line 226):

```python
    agent = get_agent("deployer")
    result = agent.invoke({"messages": [_build_deployer_context(state)]})
    registration_summary = result["messages"][-1].content
    # ... rest unchanged
```

- [ ] **Step 6: Run all node tests**

```
uv run pytest tests/test_graphs/ -v
```

Expected: all tests pass

- [ ] **Step 7: Commit**

```bash
git add src/mlops_agents/graphs/mlops_graph.py tests/test_graphs/test_node_state_extraction.py
git commit -m "feat: context-isolate trainer, evaluator, deployer nodes"
```

---

### Task 5: Inject structured state snapshot into supervisor LLM input

**Files:**
- Modify: `src/mlops_agents/agents/supervisor.py`
- Modify: `tests/test_agents/test_supervisor.py` (add snapshot test)

- [ ] **Step 1: Write the failing test**

Add to the end of `tests/test_agents/test_supervisor.py`:

```python
@patch("mlops_agents.agents.supervisor._router_llm")
def test_supervisor_injects_state_snapshot_into_llm_input(mock_llm):
    """supervisor_node must append a structured state snapshot after state['messages']."""
    import json

    captured_messages = []

    mock_structured = MagicMock()

    def capture_invoke(messages):
        captured_messages.extend(messages)
        return RouterOutput(next="FINISH", reasoning="done")

    mock_structured.invoke.side_effect = capture_invoke
    mock_llm.with_structured_output.return_value = mock_structured

    from mlops_agents.agents.supervisor import supervisor_node
    from langchain_core.messages import SystemMessage

    state = make_state(
        validation_passed=True,
        evaluation_passed=True,
        deployment_decision="approved",
        error_message="",
        training_run_id="abc123",
    )
    supervisor_node(state)

    # Last message must be the structured snapshot (HumanMessage with JSON)
    last_msg = captured_messages[-1]
    assert isinstance(last_msg, HumanMessage)
    snapshot = json.loads(last_msg.content.replace("Pipeline state:\n", ""))
    assert snapshot["validation_passed"] is True
    assert snapshot["evaluation_passed"] is True
    assert snapshot["deployment_decision"] == "approved"
    assert snapshot["error_message"] == ""
    assert snapshot["training_run_id"] == "abc123"
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_agents/test_supervisor.py::test_supervisor_injects_state_snapshot_into_llm_input -v
```

Expected: FAIL — snapshot not currently in LLM input

- [ ] **Step 3: Update `supervisor_node` in `supervisor.py`**

Add `HumanMessage` to imports and update the `messages` construction:

```python
from langchain_core.messages import HumanMessage, SystemMessage
```

Then replace line 42 in `supervisor.py`:

```python
    messages = [SystemMessage(content=_supervisor_prompt)] + list(state["messages"])
```

with:

```python
    state_snapshot = HumanMessage(content=(
        f"Pipeline state:\n{json.dumps({"
        f'"validation_passed": {json.dumps(state.get("validation_passed"))}, '
        f'"evaluation_passed": {json.dumps(state.get("evaluation_passed"))}, '
        f'"deployment_decision": {json.dumps(state.get("deployment_decision", "pending"))}, '
        f'"error_message": {json.dumps(state.get("error_message", ""))}, '
        f'"training_run_id": {json.dumps(state.get("training_run_id", ""))}'
        f"}})"
    ))
    messages = [SystemMessage(content=_supervisor_prompt)] + list(state["messages"]) + [state_snapshot]
```

The cleaner way (avoid nested f-string quotes):

```python
import json as _json  # add at top of file alongside other imports

    snapshot_data = {
        "validation_passed": state.get("validation_passed"),
        "evaluation_passed": state.get("evaluation_passed"),
        "deployment_decision": state.get("deployment_decision", "pending"),
        "error_message": state.get("error_message", ""),
        "training_run_id": state.get("training_run_id", ""),
    }
    state_snapshot = HumanMessage(content=f"Pipeline state:\n{_json.dumps(snapshot_data)}")
    messages = [SystemMessage(content=_supervisor_prompt)] + list(state["messages"]) + [state_snapshot]
```

Full updated `supervisor.py`:

```python
"""Supervisor node — LLM-based router that orchestrates the 4 specialist agents.

Uses structured output (RouterOutput) so every routing decision is auditable.
The supervisor uses a cheaper model (gpt-4.1-nano) to conserve rate limit budget.
"""

import json
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.managed.is_last_step import RemainingSteps
from langgraph.types import Command

from mlops_agents.config.constants import AGENT_SUPERVISOR
from mlops_agents.config.settings import settings
from mlops_agents.prompts import get_prompt
from mlops_agents.state.agent_state import AgentState
from mlops_agents.state.schemas import RouterOutput
from mlops_agents.utils.llm import get_router_llm
from mlops_agents.utils.logging import get_logger

logger = get_logger(__name__)

_router_llm = get_router_llm()
_supervisor_prompt = get_prompt("supervisor").template


def supervisor_node(
    state: AgentState,
) -> Command[Literal["data_validator", "trainer", "evaluator", "deployer", "__end__"]]:
    """Supervisor node: reads state and decides which agent to call next.

    Uses structured output to enforce a valid routing decision.
    The reasoning field is logged for thesis analysis.
    """
    # Graceful exit if approaching recursion limit
    remaining: RemainingSteps | None = state.get("remaining_steps")  # type: ignore[assignment]
    if remaining is not None and remaining <= 2:
        logger.warning("Approaching recursion limit — forcing FINISH")
        return Command(goto=END, update={"next": "FINISH"})

    snapshot_data = {
        "validation_passed": state.get("validation_passed"),
        "evaluation_passed": state.get("evaluation_passed"),
        "deployment_decision": state.get("deployment_decision", "pending"),
        "error_message": state.get("error_message", ""),
        "training_run_id": state.get("training_run_id", ""),
    }
    state_snapshot = HumanMessage(content=f"Pipeline state:\n{json.dumps(snapshot_data)}")
    messages = [SystemMessage(content=_supervisor_prompt)] + list(state["messages"]) + [state_snapshot]
    response: RouterOutput = _router_llm.with_structured_output(RouterOutput).invoke(messages)

    logger.info(f"[{AGENT_SUPERVISOR}] → {response.next} | reason: {response.reasoning}")

    goto = END if response.next == "FINISH" else response.next

    if goto != END:
        counts = dict(state.get("agent_attempt_counts") or {})
        if counts.get(goto, 0) >= settings.max_attempts_per_agent:
            logger.warning(f"[supervisor] max attempts reached for {goto} — forcing END")
            return Command(goto=END, update={"next": "FINISH"})
        counts[goto] = counts.get(goto, 0) + 1
        return Command(goto=goto, update={"next": response.next, "agent_attempt_counts": counts})

    return Command(goto=END, update={"next": response.next})
```

- [ ] **Step 4: Run all supervisor tests**

```
uv run pytest tests/test_agents/test_supervisor.py -v
```

Expected: all 5 tests pass

- [ ] **Step 5: Commit**

```bash
git add src/mlops_agents/agents/supervisor.py tests/test_agents/test_supervisor.py
git commit -m "feat: inject structured state snapshot into supervisor LLM input"
```

---

### Task 6: Reinforce supervisor.yaml rule 5 and run full test suite

**Files:**
- Modify: `src/mlops_agents/prompts/supervisor.yaml`

- [ ] **Step 1: Update rule 5 in `supervisor.yaml`**

Replace rule 5 in `src/mlops_agents/prompts/supervisor.yaml`:

```yaml
  5. If error_message is set in state, always select FINISH — do not retry any agent.
     If validation_passed=False after data_validator has already run, select FINISH —
     imputation is handled automatically inside the agent, not by retrying the node.
```

with:

```yaml
  5. Check the "Pipeline state:" message (always the last message) for routing signals:
     - If error_message is non-empty → always select FINISH, no exceptions.
     - If validation_passed=False → select FINISH, never retry data_validator.
       Imputation is automatic inside the agent; retrying cannot change the outcome.
     - Use deployment_decision, evaluation_passed, and training_run_id to confirm
       pipeline stage — do not infer these from narrative summaries alone.
```

- [ ] **Step 2: Run full unit test suite**

```
uv run pytest -m "not integration" -v
```

Expected: all tests pass

- [ ] **Step 3: Run lint and type check**

```
uv run ruff check . && uv run ruff format . && uv run mypy src/
```

Expected: no errors

- [ ] **Step 4: Commit**

```bash
git add src/mlops_agents/prompts/supervisor.yaml
git commit -m "docs: reinforce supervisor rule 5 to reference Pipeline state snapshot fields"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] `dataset_summary: dict` added to AgentState (Task 1)
- [x] 4 `_build_*_context` functions with tests (Task 2)
- [x] `dataset_summary` built in `data_validator_node` (Task 3)
- [x] All 4 worker nodes use context builders (Tasks 3-4)
- [x] Supervisor receives `state["messages"]` + structured snapshot (Task 5)
- [x] `supervisor.yaml` rule 5 updated (Task 6)
- [x] `test_node_state_extraction.py` covers isolation + `dataset_summary` (Tasks 3-4)
- [x] `test_supervisor.py` covers snapshot injection (Task 5)

**Type consistency:**
- `_build_data_validator_context(state: AgentState) -> HumanMessage` — matches usage in `data_validator_node`
- `_build_trainer_context` / `_build_evaluator_context` / `_build_deployer_context` — same signature
- `dataset_summary: dict` — matches `state.get("dataset_summary") or {}` in `_build_trainer_context`
- `state_snapshot` is `HumanMessage` — appended to `list(state["messages"])` which is `list[BaseMessage]` ✓

**No placeholders:** All code blocks contain complete implementations.
