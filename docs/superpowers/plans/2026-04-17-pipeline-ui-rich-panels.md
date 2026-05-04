# Pipeline UI Rich Panels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade `dashboard/pages/01_pipeline.py` to a two-column split layout where a live results panel (Data / Training / Evaluation tabs) fills in progressively as each agent completes.

**Architecture:** Three independent changes wired together: (1) a pure `extract_panel_data` helper reads displayable data from LangGraph state; (2) the three worker nodes in `mlops_graph.py` are updated to populate `validation_report`, `training_metrics`, and `evaluation_report` in shared state by parsing tool output messages; (3) the pipeline page is rewritten with a two-column layout that streams log updates left and tab content updates right during a run, then shows both in their final state after completion.

**Tech Stack:** Python 3.12, Streamlit ≥1.40, LangGraph Command API, pandas, langchain-core ToolMessage, pytest, unittest.mock.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `dashboard/pipeline_helpers.py` | Add `extract_panel_data` pure function |
| Modify | `tests/test_dashboard/test_pipeline_helpers.py` | Tests for `extract_panel_data` |
| Modify | `src/mlops_agents/graphs/mlops_graph.py` | Add `_extract_tool_json`; update `data_validator_node`, `trainer_node`, `evaluator_node` to populate state fields |
| Create | `tests/test_graphs/test_node_state_extraction.py` | Tests for `_extract_tool_json` and updated nodes |
| Modify | `dashboard/pages/01_pipeline.py` | Full rewrite: two-column layout, tabs, live panel updates |

---

## Task 1: Add `extract_panel_data` to `pipeline_helpers.py`

**Files:**
- Modify: `dashboard/pipeline_helpers.py`
- Modify: `tests/test_dashboard/test_pipeline_helpers.py`

- [ ] **Step 1: Add failing tests for `extract_panel_data`**

Append to `tests/test_dashboard/test_pipeline_helpers.py`:

```python
import pandas as pd
from dashboard.pipeline_helpers import extract_panel_data


def test_extract_panel_data_empty_state_returns_all_empty():
    result = extract_panel_data({})
    assert result["validation_report"] == {}
    assert result["training_metrics"] == {}
    assert result["evaluation_report"] == {}
    assert result["dataset_preview"] == []


def test_extract_panel_data_returns_validation_report():
    report = {"passed": True, "row_count": 150, "column_count": 5}
    result = extract_panel_data({"validation_report": report, "dataset_path": ""})
    assert result["validation_report"] == report


def test_extract_panel_data_returns_training_metrics():
    metrics = {"model_type": "random_forest", "val_accuracy": 0.95}
    result = extract_panel_data({"training_metrics": metrics})
    assert result["training_metrics"] == metrics


def test_extract_panel_data_returns_evaluation_report():
    report = {"candidate_metrics": {"accuracy": 0.97}, "baseline_metrics": {}}
    result = extract_panel_data({"evaluation_report": report})
    assert result["evaluation_report"] == report


def test_extract_panel_data_no_preview_when_validation_report_empty():
    result = extract_panel_data({"validation_report": {}, "dataset_path": "./data/samples/iris.csv"})
    assert result["dataset_preview"] == []


def test_extract_panel_data_no_preview_when_path_missing():
    result = extract_panel_data({"validation_report": {"passed": True}, "dataset_path": "/nonexistent/file.csv"})
    assert result["dataset_preview"] == []


def test_extract_panel_data_no_preview_when_path_empty():
    result = extract_panel_data({"validation_report": {"passed": True}, "dataset_path": ""})
    assert result["dataset_preview"] == []


def test_extract_panel_data_loads_preview_when_valid(tmp_path):
    csv_path = tmp_path / "sample.csv"
    df = pd.DataFrame({"a": range(20), "b": range(20), "target": range(20)})
    df.to_csv(csv_path, index=False)
    result = extract_panel_data({
        "validation_report": {"passed": True},
        "dataset_path": str(csv_path),
    })
    assert len(result["dataset_preview"]) == 10
    assert result["dataset_preview"][0]["a"] == 0
```

- [ ] **Step 2: Run tests — verify they FAIL**

```bash
uv run pytest tests/test_dashboard/test_pipeline_helpers.py -k "extract_panel_data" -v
```

Expected: `ImportError` or `AttributeError` — `extract_panel_data` does not exist yet.

- [ ] **Step 3: Implement `extract_panel_data` in `pipeline_helpers.py`**

Add at the bottom of `dashboard/pipeline_helpers.py` (keep existing functions untouched):

```python
from pathlib import Path

import pandas as pd


def extract_panel_data(state: dict) -> dict:
    """Extract displayable panel data from a raw LangGraph state dict.

    Returns a dict with keys:
      validation_report  — dict from check_data_quality tool output
      training_metrics   — dict with model_type, train_accuracy, val_accuracy
      evaluation_report  — dict with candidate_metrics, baseline_metrics
      dataset_preview    — list of row dicts (first 10 rows), loaded once after validation
    All values are empty ({} / []) when the corresponding stage has not yet completed.
    """
    validation_report: dict = state.get("validation_report") or {}
    training_metrics: dict = state.get("training_metrics") or {}
    evaluation_report: dict = state.get("evaluation_report") or {}

    dataset_preview: list = []
    if validation_report:
        dataset_path = state.get("dataset_path", "")
        if dataset_path and Path(dataset_path).exists():
            try:
                dataset_preview = pd.read_csv(dataset_path).head(10).to_dict("records")
            except Exception:
                dataset_preview = []

    return {
        "validation_report": validation_report,
        "training_metrics": training_metrics,
        "evaluation_report": evaluation_report,
        "dataset_preview": dataset_preview,
    }
```

- [ ] **Step 4: Run tests — verify they PASS**

```bash
uv run pytest tests/test_dashboard/test_pipeline_helpers.py -v
```

Expected: all tests pass (original 11 + new 8 = 19 passed).

- [ ] **Step 5: Run full unit suite to check for regressions**

```bash
uv run pytest -m "not integration" -v
```

Expected: all previously passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add dashboard/pipeline_helpers.py tests/test_dashboard/test_pipeline_helpers.py
git commit -m "feat: add extract_panel_data helper for pipeline UI panels"
```

---

## Task 2: Update agent nodes to populate structured state fields

The current `_wrap_agent` generic wrapper only updates `messages`. This task adds a `_extract_tool_json` helper and rewrites the three worker nodes (`data_validator_node`, `trainer_node`, `evaluator_node`) so they also set `validation_report`, `training_metrics`, and `evaluation_report` in shared state by parsing the react agent's internal `ToolMessage` outputs.

**Files:**
- Modify: `src/mlops_agents/graphs/mlops_graph.py`
- Create: `tests/test_graphs/test_node_state_extraction.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_graphs/test_node_state_extraction.py`:

```python
"""Tests for _extract_tool_json and updated worker node state extraction."""

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


# ---------------------------------------------------------------------------
# _extract_tool_json
# ---------------------------------------------------------------------------

def test_extract_tool_json_finds_matching_message():
    from mlops_agents.graphs.mlops_graph import _extract_tool_json

    msgs = [
        ToolMessage(
            content='{"passed": true, "row_count": 150}',
            tool_call_id="call_1",
            name="check_data_quality",
        )
    ]
    result = _extract_tool_json(msgs, "check_data_quality")
    assert result == {"passed": True, "row_count": 150}


def test_extract_tool_json_returns_empty_dict_when_no_match():
    from mlops_agents.graphs.mlops_graph import _extract_tool_json

    result = _extract_tool_json([], "check_data_quality")
    assert result == {}


def test_extract_tool_json_returns_last_matching_message():
    from mlops_agents.graphs.mlops_graph import _extract_tool_json

    msgs = [
        ToolMessage(content='{"row_count": 100}', tool_call_id="1", name="check_data_quality"),
        ToolMessage(content='{"row_count": 200}', tool_call_id="2", name="check_data_quality"),
    ]
    result = _extract_tool_json(msgs, "check_data_quality")
    assert result["row_count"] == 200


def test_extract_tool_json_handles_invalid_json_gracefully():
    from mlops_agents.graphs.mlops_graph import _extract_tool_json

    msgs = [ToolMessage(content="not valid json", tool_call_id="1", name="my_tool")]
    result = _extract_tool_json(msgs, "my_tool")
    assert result == {}


def test_extract_tool_json_handles_list_result():
    from mlops_agents.graphs.mlops_graph import _extract_tool_json

    payload = json.dumps([{"run_id": "abc", "metrics": {"accuracy": 0.95}}])
    msgs = [ToolMessage(content=payload, tool_call_id="1", name="get_best_run")]
    result = _extract_tool_json(msgs, "get_best_run")
    assert isinstance(result, list)
    assert result[0]["run_id"] == "abc"


def test_extract_tool_json_skips_non_tool_messages():
    from mlops_agents.graphs.mlops_graph import _extract_tool_json

    msgs = [
        HumanMessage(content="run pipeline"),
        AIMessage(content="calling tool"),
        ToolMessage(content='{"found": true}', tool_call_id="1", name="my_tool"),
    ]
    result = _extract_tool_json(msgs, "my_tool")
    assert result == {"found": True}


# ---------------------------------------------------------------------------
# data_validator_node
# ---------------------------------------------------------------------------

def _make_state() -> dict:
    return {
        "messages": [HumanMessage(content="Run pipeline on iris.csv")],
        "next": "",
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
        "retry_count": 0,
    }


def test_data_validator_node_populates_validation_report():
    from mlops_agents.graphs.mlops_graph import data_validator_node

    quality_json = json.dumps({
        "passed": True,
        "row_count": 150,
        "column_count": 5,
        "missing_values_total": 0,
        "max_missing_pct": 0.0,
        "duplicate_rows": 0,
    })
    mock_result = {
        "messages": [
            ToolMessage(content=quality_json, tool_call_id="1", name="check_data_quality"),
            AIMessage(content="Data validation passed."),
        ]
    }
    with patch("mlops_agents.graphs.mlops_graph.get_agent") as mock_get_agent:
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = mock_result
        mock_get_agent.return_value = mock_agent

        command = data_validator_node(_make_state())

    assert command.update["validation_report"]["passed"] is True
    assert command.update["validation_report"]["row_count"] == 150
    assert command.update["validation_passed"] is True
    assert command.goto == "supervisor"


def test_data_validator_node_passed_false_when_no_tool_output():
    from mlops_agents.graphs.mlops_graph import data_validator_node

    mock_result = {"messages": [AIMessage(content="Could not validate.")]}
    with patch("mlops_agents.graphs.mlops_graph.get_agent") as mock_get_agent:
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = mock_result
        mock_get_agent.return_value = mock_agent

        command = data_validator_node(_make_state())

    assert command.update["validation_report"] == {}
    assert command.update["validation_passed"] is False


# ---------------------------------------------------------------------------
# trainer_node
# ---------------------------------------------------------------------------

def test_trainer_node_populates_training_metrics():
    from mlops_agents.graphs.mlops_graph import trainer_node

    train_json = json.dumps({
        "model_type": "random_forest",
        "model_path": "./models/random_forest_model.pkl",
        "hyperparameters": {"n_estimators": 100},
        "train_accuracy": 0.98,
        "val_accuracy": 0.95,
        "classification_report": {},
    })
    mlflow_json = json.dumps({"run_id": "abc123", "model_uri": "runs:/abc123/model"})
    mock_result = {
        "messages": [
            ToolMessage(content=train_json, tool_call_id="1", name="train_model"),
            ToolMessage(content=mlflow_json, tool_call_id="2", name="log_experiment"),
            AIMessage(content="Training complete."),
        ]
    }
    with patch("mlops_agents.graphs.mlops_graph.get_agent") as mock_get_agent:
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = mock_result
        mock_get_agent.return_value = mock_agent

        command = trainer_node(_make_state())

    assert command.update["training_metrics"]["model_type"] == "random_forest"
    assert command.update["training_metrics"]["val_accuracy"] == 0.95
    assert command.update["training_run_id"] == "abc123"
    assert command.update["trained_model_path"] == "./models/random_forest_model.pkl"
    assert command.goto == "supervisor"


# ---------------------------------------------------------------------------
# evaluator_node
# ---------------------------------------------------------------------------

def test_evaluator_node_populates_evaluation_report():
    from mlops_agents.graphs.mlops_graph import evaluator_node

    runs_json = json.dumps([
        {"run_id": "run1", "metrics": {"accuracy": 0.97, "f1_score": 0.96}, "params": {}, "model_uri": "runs:/run1/model"},
        {"run_id": "run0", "metrics": {"accuracy": 0.93, "f1_score": 0.92}, "params": {}, "model_uri": "runs:/run0/model"},
    ])
    mock_result = {
        "messages": [
            ToolMessage(content=runs_json, tool_call_id="1", name="get_best_run"),
            AIMessage(content="Candidate beats baseline. Recommend promote."),
        ]
    }
    with patch("mlops_agents.graphs.mlops_graph.get_agent") as mock_get_agent:
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = mock_result
        mock_get_agent.return_value = mock_agent

        command = evaluator_node(_make_state())

    assert command.update["evaluation_report"]["candidate_metrics"]["accuracy"] == 0.97
    assert command.update["evaluation_report"]["baseline_metrics"]["accuracy"] == 0.93
    assert command.update["evaluation_report"]["candidate_run_id"] == "run1"
    assert command.goto == "supervisor"
```

- [ ] **Step 2: Run tests — verify they FAIL**

```bash
uv run pytest tests/test_graphs/test_node_state_extraction.py -v
```

Expected: `ImportError` for `_extract_tool_json` — it does not exist yet.

- [ ] **Step 3: Add `_extract_tool_json` and update the three worker nodes in `mlops_graph.py`**

Add the following imports at the top of `src/mlops_agents/graphs/mlops_graph.py` (after existing imports):

```python
import json
from typing import Any

from langchain_core.messages import ToolMessage
```

Then add the helper function right before the worker node wrappers (before `_wrap_agent`):

```python
def _extract_tool_json(messages: list, tool_name: str) -> Any:
    """Return the parsed JSON content of the last ToolMessage matching tool_name.

    Returns {} if no matching message is found or JSON parsing fails.
    Returns a list when the tool responded with a JSON array (e.g. get_best_run).
    """
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and getattr(msg, "name", None) == tool_name:
            try:
                return json.loads(msg.content)
            except (json.JSONDecodeError, TypeError):
                return {}
    return {}
```

Then replace the three specific worker node functions (keep `_wrap_agent` and `deployer_node` unchanged):

```python
def data_validator_node(state: AgentState) -> Command[Literal["supervisor"]]:
    agent = get_agent("data_validator")
    result = agent.invoke({"messages": list(state["messages"])})
    final_message = result["messages"][-1].content

    quality_report: dict = _extract_tool_json(result["messages"], "check_data_quality")

    logger.info("[data_validator] completed — routing back to supervisor")
    return Command(
        update={
            "messages": [HumanMessage(content=final_message, name="data_validator")],
            "validation_report": quality_report,
            "validation_passed": bool(quality_report.get("passed", False)),
        },
        goto="supervisor",
    )


def trainer_node(state: AgentState) -> Command[Literal["supervisor"]]:
    agent = get_agent("trainer")
    result = agent.invoke({"messages": list(state["messages"])})
    final_message = result["messages"][-1].content

    train_result: dict = _extract_tool_json(result["messages"], "train_model")
    mlflow_result: dict = _extract_tool_json(result["messages"], "log_experiment")

    training_metrics = {
        "model_type": train_result.get("model_type", ""),
        "train_accuracy": train_result.get("train_accuracy", 0.0),
        "val_accuracy": train_result.get("val_accuracy", 0.0),
    }

    logger.info("[trainer] completed — routing back to supervisor")
    return Command(
        update={
            "messages": [HumanMessage(content=final_message, name="trainer")],
            "training_metrics": training_metrics,
            "training_run_id": mlflow_result.get("run_id", ""),
            "trained_model_path": train_result.get("model_path", ""),
        },
        goto="supervisor",
    )


def evaluator_node(state: AgentState) -> Command[Literal["supervisor"]]:
    agent = get_agent("evaluator")
    result = agent.invoke({"messages": list(state["messages"])})
    final_message = result["messages"][-1].content

    best_runs_raw = _extract_tool_json(result["messages"], "get_best_run")
    runs_list: list = best_runs_raw if isinstance(best_runs_raw, list) else []
    candidate = runs_list[0] if runs_list else {}
    baseline = runs_list[1] if len(runs_list) > 1 else {}

    evaluation_report = {
        "candidate_metrics": candidate.get("metrics", {}),
        "candidate_run_id": candidate.get("run_id", ""),
        "baseline_metrics": baseline.get("metrics", {}),
    }

    logger.info("[evaluator] completed — routing back to supervisor")
    return Command(
        update={
            "messages": [HumanMessage(content=final_message, name="evaluator")],
            "evaluation_report": evaluation_report,
            "evaluation_passed": bool(candidate),
        },
        goto="supervisor",
    )
```

- [ ] **Step 4: Run tests — verify they PASS**

```bash
uv run pytest tests/test_graphs/test_node_state_extraction.py -v
```

Expected: all 13 tests pass.

- [ ] **Step 5: Run full unit suite to check for regressions**

```bash
uv run pytest -m "not integration" -v
```

Expected: all previously passing tests still pass (no regressions from node rewrites).

- [ ] **Step 6: Commit**

```bash
git add src/mlops_agents/graphs/mlops_graph.py tests/test_graphs/test_node_state_extraction.py
git commit -m "feat: populate validation_report, training_metrics, evaluation_report in worker nodes"
```

---

## Task 3: Rewrite pipeline page with two-column layout and tabs

**Files:**
- Modify: `dashboard/pages/01_pipeline.py`

- [ ] **Step 1: Replace the entire file**

Overwrite `dashboard/pages/01_pipeline.py` with:

```python
"""Pipeline page — run and monitor the MLOps agent pipeline.

Three-phase state machine stored in st.session_state:
  idle             → dataset selector + Run button
  awaiting_approval → full-width frozen log + HITL approval panel
  complete         → two-column log (left) + results tabs (right) + outcome banner
"""

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from langgraph.types import Command

sys.path.insert(0, str(Path(__file__).parents[2]))

from dashboard.pipeline_helpers import build_initial_state, event_to_log_line, extract_panel_data
from mlops_agents.config.constants import GRAPH_RECURSION_LIMIT
from mlops_agents.graphs.mlops_graph import graph

st.set_page_config(page_title="Pipeline", layout="wide")
st.title("🤖 MLOps Pipeline")

# ── Session state initialisation ──────────────────────────────────────────────
_DEFAULTS: dict = {
    "phase": "idle",            # idle | awaiting_approval | complete
    "log_lines": [],
    "pipeline_config": None,
    "interrupt_value": {},
    "deployment_decision": "pending",
    "final_message": "",
    "reject_mode": False,
    "validation_report": {},
    "training_metrics": {},
    "evaluation_report": {},
    "dataset_preview": [],
    "training_run_id": "",
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Helpers ───────────────────────────────────────────────────────────────────

def _log(line: str) -> None:
    st.session_state["log_lines"].append(line)


def _render_log(placeholder) -> None:
    placeholder.markdown("\n\n".join(st.session_state["log_lines"]))


def _update_panel_data(config: dict) -> None:
    """Read current graph state and update panel session fields."""
    state_vals = graph.get_state(config).values
    panel = extract_panel_data(state_vals)
    for key, val in panel.items():
        if val:
            st.session_state[key] = val
    run_id = state_vals.get("training_run_id", "")
    if run_id:
        st.session_state["training_run_id"] = run_id


def _render_tabs(right_placeholder) -> None:
    """Render the three results tabs into right_placeholder."""
    with right_placeholder.container():
        tab1, tab2, tab3 = st.tabs(["📊 Data", "🏋️ Training", "📈 Evaluation"])

        with tab1:
            report = st.session_state["validation_report"]
            if not report:
                st.info("Waiting for data validation to complete...")
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Rows", report.get("row_count", "—"))
                c2.metric("Columns", report.get("column_count", "—"))
                c3.metric("Missing %", f"{report.get('max_missing_pct', 0):.1f}%")
                c4.metric("Status", "✅ Passed" if report.get("passed") else "❌ Failed")
                preview = st.session_state["dataset_preview"]
                if preview:
                    st.dataframe(pd.DataFrame(preview), use_container_width=True)
                with st.expander("Full Validation Report"):
                    st.json(report)

        with tab2:
            metrics = st.session_state["training_metrics"]
            run_id = st.session_state["training_run_id"]
            if not metrics:
                st.info("Waiting for model training to complete...")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Model", metrics.get("model_type", "—"))
                c2.metric("Train Acc", f"{metrics.get('train_accuracy', 0):.2%}")
                c3.metric("Val Acc", f"{metrics.get('val_accuracy', 0):.2%}")
                if run_id:
                    st.caption(f"MLflow Run ID: `{run_id}`")
                num_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                if num_metrics:
                    st.dataframe(
                        pd.DataFrame([num_metrics]).T.rename(columns={0: "Value"}),
                        use_container_width=True,
                    )

        with tab3:
            eval_report = st.session_state["evaluation_report"]
            if not eval_report:
                st.info("Waiting for model evaluation to complete...")
            else:
                candidate = eval_report.get("candidate_metrics", {})
                baseline = eval_report.get("baseline_metrics", {})
                if candidate or baseline:
                    all_keys = sorted(set(list(candidate.keys()) + list(baseline.keys())))
                    rows = [
                        {
                            "Metric": k,
                            "Candidate": candidate.get(k, "—"),
                            "Baseline": baseline.get(k, "—"),
                        }
                        for k in all_keys
                    ]
                    st.dataframe(pd.DataFrame(rows).set_index("Metric"), use_container_width=True)
                cand_run = eval_report.get("candidate_run_id", "")
                if cand_run:
                    st.caption(f"Candidate Run ID: `{cand_run}`")
                if not candidate and not baseline:
                    st.json(eval_report)


def _resume_pipeline(resume: dict) -> None:
    """Resume the paused graph with the operator decision and stream remaining events."""
    config = st.session_state["pipeline_config"]
    log_placeholder = st.empty()
    _render_log(log_placeholder)

    for event in graph.stream(Command(resume=resume), config=config):
        line = event_to_log_line(event)
        if line:
            _log(line)
            _render_log(log_placeholder)

    _update_panel_data(config)

    final = graph.get_state(config).values
    st.session_state["deployment_decision"] = final.get("deployment_decision", "pending")
    msgs = final.get("messages", [])
    if msgs:
        last = msgs[-1]
        st.session_state["final_message"] = last.content if hasattr(last, "content") else str(last)

    st.session_state["reject_mode"] = False
    st.session_state["phase"] = "complete"
    st.rerun()


# ── Phase: idle ───────────────────────────────────────────────────────────────

if st.session_state["phase"] == "idle":
    from mlops_agents.config.settings import settings

    data_dir = Path(settings.data_dir)
    csvs = sorted(data_dir.glob("*.csv")) if data_dir.exists() else []
    options = [str(f) for f in csvs] or ["./data/samples/iris.csv"]

    col1, col2 = st.columns([3, 1])
    with col1:
        dataset_path = st.selectbox(
            "Select dataset",
            options=options,
            help="CSV file with a 'target' column",
        )
    with col2:
        run_button = st.button("▶ Run Pipeline", type="primary", use_container_width=True)

    if run_button:
        thread_id = f"streamlit-{int(time.time())}"
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": GRAPH_RECURSION_LIMIT,
        }
        st.session_state["pipeline_config"] = config

        left_col, right_col = st.columns([4, 6])
        with left_col:
            st.subheader("Pipeline Log")
            log_placeholder = st.empty()
        with right_col:
            st.subheader("Live Results")
            right_placeholder = st.empty()

        interrupt_detected = False
        for event in graph.stream(build_initial_state(dataset_path), config=config):
            if "__interrupt__" in event:
                st.session_state["interrupt_value"] = event["__interrupt__"][0].value
                st.session_state["phase"] = "awaiting_approval"
                _log("⏸ **Pipeline paused — awaiting human approval**")
                interrupt_detected = True
                break
            else:
                line = event_to_log_line(event)
                if line:
                    _log(line)
                    _render_log(log_placeholder)
                _update_panel_data(config)
                _render_tabs(right_placeholder)

        if interrupt_detected:
            _render_log(log_placeholder)
            st.rerun()
        elif st.session_state["phase"] == "idle":
            _update_panel_data(config)
            final = graph.get_state(config).values
            st.session_state["deployment_decision"] = final.get("deployment_decision", "pending")
            msgs = final.get("messages", [])
            if msgs:
                last = msgs[-1]
                st.session_state["final_message"] = (
                    last.content if hasattr(last, "content") else str(last)
                )
            st.session_state["phase"] = "complete"
            st.rerun()


# ── Phase: awaiting_approval ──────────────────────────────────────────────────

elif st.session_state["phase"] == "awaiting_approval":
    st.subheader("Pipeline Log")
    st.markdown("\n\n".join(st.session_state["log_lines"]))

    st.divider()

    iv = st.session_state["interrupt_value"]
    st.subheader("⚠️ Human Approval Required")
    st.markdown(f"**{iv.get('question', 'Approve this action?')}**")

    summary = iv.get("registration_summary", "")
    if summary:
        with st.expander("Registration details"):
            st.text(summary)

    if not st.session_state["reject_mode"]:
        col_a, col_r = st.columns(2)
        with col_a:
            if st.button("✅ Approve", type="primary", use_container_width=True):
                _resume_pipeline({"approved": True})
        with col_r:
            if st.button("❌ Reject", use_container_width=True):
                st.session_state["reject_mode"] = True
                st.rerun()
    else:
        reason = st.text_input("Rejection reason (optional)")
        if st.button("Confirm Rejection", type="primary"):
            _resume_pipeline({"approved": False, "reason": reason or "Rejected by operator"})


# ── Phase: complete ───────────────────────────────────────────────────────────

elif st.session_state["phase"] == "complete":
    left_col, right_col = st.columns([4, 6])
    with left_col:
        st.subheader("Pipeline Log")
        st.markdown("\n\n".join(st.session_state["log_lines"]))
    with right_col:
        st.subheader("Results")
        right_placeholder = st.empty()
        _render_tabs(right_placeholder)

    st.divider()

    decision = st.session_state["deployment_decision"]
    msg = st.session_state["final_message"]

    if decision == "approved":
        st.success("Pipeline complete. Model promoted to champion.")
    elif decision == "rejected":
        st.warning(f"Deployment rejected. {msg}" if msg else "Deployment rejected.")
    else:
        st.error(f"Pipeline stopped early. {msg}" if msg else "Pipeline stopped early.")

    if st.button("🔄 Run Again"):
        for k, v in _DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()
```

- [ ] **Step 2: Verify the file parses cleanly**

```bash
uv run python -c "import ast, pathlib; ast.parse(pathlib.Path('dashboard/pages/01_pipeline.py').read_text()); print('OK')"
```

Expected output: `OK`

- [ ] **Step 3: Run the full unit suite**

```bash
uv run pytest -m "not integration" -v
```

Expected: all tests pass (0 failures).

- [ ] **Step 4: Start the dashboard and verify idle phase**

```bash
uv run streamlit run dashboard/app.py
```

Open `http://localhost:8501`, navigate to **Pipeline**.

Expected: dataset selector shows available CSVs, "▶ Run Pipeline" button visible, no errors in terminal.

- [ ] **Step 5: Run the pipeline and verify live two-column layout**

Click "▶ Run Pipeline". Observe the browser.

Expected:
- Left column ("Pipeline Log") fills with log lines as agents complete:
  ```
  🔀 [supervisor] → data_validator
  ✅ [data_validator] completed
  🔀 [supervisor] → trainer
  ✅ [trainer] completed
  ...
  ```
- Right column ("Live Results") — tabs appear; after `data_validator` completes, **📊 Data** tab shows row count, column count, missing % metrics and a 10-row dataframe preview; after `trainer` completes, **🏋️ Training** tab shows model type, train/val accuracy, and MLflow run ID; after `evaluator` completes, **📈 Evaluation** tab shows candidate vs. baseline metrics table.

- [ ] **Step 6: Verify the HITL approval panel is full-width**

When the pipeline reaches the deployer, the page should re-render to `awaiting_approval` phase: full-width log + `⚠️ Human Approval Required` panel. No tabs visible in this phase.

- [ ] **Step 7: Approve and verify complete phase**

Click "✅ Approve".

Expected:
- Remaining log lines appear
- Page re-renders to `complete` phase: two columns return
- Left: full log; Right: all three tabs populated with final data
- Below divider: green `st.success("Pipeline complete. Model promoted to champion.")`
- "🔄 Run Again" button at the bottom

- [ ] **Step 8: Verify Run Again resets cleanly**

Click "🔄 Run Again". Expected: page returns to idle — dataset selector visible, all tabs cleared.

- [ ] **Step 9: Commit**

```bash
git add dashboard/pages/01_pipeline.py
git commit -m "feat: two-column pipeline page with live data, training, and evaluation tabs"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Covered by |
|-----------------|-----------|
| Left column (40%) streaming log | Task 3 — `left_col` in all run phases |
| Right column (60%) three tabs | Task 3 — `_render_tabs` called in streaming loop and complete phase |
| Tabs fill in progressively | Task 3 — `_update_panel_data` + `_render_tabs` after every event |
| HITL panel full-width | Task 3 — `awaiting_approval` phase has no columns |
| `validation_report` from `data_validator` | Task 2 — `data_validator_node` extracts `check_data_quality` ToolMessage |
| `training_metrics` + `training_run_id` from `trainer` | Task 2 — `trainer_node` extracts `train_model` + `log_experiment` ToolMessages |
| `evaluation_report` from `evaluator` | Task 2 — `evaluator_node` extracts `get_best_run` ToolMessage |
| `extract_panel_data` pure helper | Task 1 — `pipeline_helpers.py` |
| `dataset_preview` 10-row sample | Task 1 — loaded in `extract_panel_data` when validation_report is non-empty |
| Empty state shows "Waiting..." placeholder | Task 3 — each tab branch checks for empty dict/list |
| Safe fallback for missing keys | Task 3 — all tab renders use `.get()` with defaults |
| Run Again resets all panel fields | Task 3 — `_DEFAULTS` includes all 4 new fields, loop resets them |

**Placeholder scan:** None found.

**Type consistency:**
- `extract_panel_data(state: dict) -> dict` — defined Task 1, used in `_update_panel_data` in Task 3 ✅
- `_extract_tool_json(messages: list, tool_name: str) -> Any` — defined Task 2, used in the three nodes ✅
- `training_run_id` stored separately in session state and populated via `_update_panel_data` ✅
- `right_placeholder` created as `st.empty()` within right_col context, passed to `_render_tabs` ✅
- `_render_tabs` uses `right_placeholder.container()` — valid Streamlit pattern for live placeholder updates ✅
