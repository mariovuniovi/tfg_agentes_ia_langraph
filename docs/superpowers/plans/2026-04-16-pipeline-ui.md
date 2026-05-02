# Pipeline UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `dashboard/pages/01_pipeline.py` into a polished, fully-functional pipeline demo page with live agent log, proper HITL interrupt detection, and an approval panel.

**Architecture:** Extract pure helper functions into `dashboard/pipeline_helpers.py` (tested with pytest), then rewrite the page as a three-phase state machine (`idle → awaiting_approval → complete`) using `st.session_state` to persist state across Streamlit re-runs. The compiled graph (with `InMemorySaver`) lives at module level in `mlops_graph.py` and persists within the Streamlit session.

**Tech Stack:** Python 3.12, Streamlit ≥1.40, LangGraph `Command` + `interrupt()`, `st.session_state`, `st.empty()` for live log updates.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `dashboard/__init__.py` | Makes `dashboard` a package (required for test imports) |
| Create | `dashboard/pipeline_helpers.py` | Pure functions: `event_to_log_line`, `build_initial_state` |
| Create | `tests/test_dashboard/__init__.py` | Test package |
| Create | `tests/test_dashboard/test_pipeline_helpers.py` | Unit tests for helpers |
| Modify | `pyproject.toml` | Add `pythonpath = ["."]` so pytest can import `dashboard` |
| Modify | `dashboard/pages/01_pipeline.py` | Full rewrite: state machine + HITL panel |

---

## Task 1: Make `dashboard` importable and add helper tests

**Files:**
- Create: `dashboard/__init__.py`
- Create: `dashboard/pipeline_helpers.py`
- Create: `tests/test_dashboard/__init__.py`
- Create: `tests/test_dashboard/test_pipeline_helpers.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add `pythonpath` to pytest config**

In `pyproject.toml`, add one line to `[tool.pytest.ini_options]`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
pythonpath = ["."]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = ["-v", "--strict-markers", "--tb=short"]
markers = [
    "slow: marks tests as slow",
    "integration: marks integration tests requiring real LLM calls",
]
```

- [ ] **Step 2: Create `dashboard/__init__.py`**

Create an empty file at `dashboard/__init__.py`:

```python
```

(Empty file — just marks the directory as a Python package.)

- [ ] **Step 3: Create `tests/test_dashboard/__init__.py`**

Create an empty file at `tests/test_dashboard/__init__.py`:

```python
```

- [ ] **Step 4: Write the failing tests**

Create `tests/test_dashboard/test_pipeline_helpers.py`:

```python
"""Unit tests for Pipeline page helper functions."""

from dashboard.pipeline_helpers import build_initial_state, event_to_log_line


def test_event_to_log_line_supervisor_routes_to_agent():
    event = {"supervisor": {"next": "data_validator", "messages": []}}
    assert event_to_log_line(event) == "🔀 `[supervisor]` → **data_validator**"


def test_event_to_log_line_supervisor_finish():
    event = {"supervisor": {"next": "FINISH", "messages": []}}
    assert event_to_log_line(event) == "🏁 `[supervisor]` → **FINISH**"


def test_event_to_log_line_worker_node():
    event = {"data_validator": {"messages": []}}
    assert event_to_log_line(event) == "✅ `[data_validator]` completed"


def test_event_to_log_line_interrupt_returns_none():
    assert event_to_log_line({"__interrupt__": []}) is None


def test_event_to_log_line_supervisor_no_next_returns_none():
    assert event_to_log_line({"supervisor": {"messages": []}}) is None


def test_build_initial_state_sets_dataset_path():
    state = build_initial_state("./data/samples/iris.csv")
    assert state["dataset_path"] == "./data/samples/iris.csv"


def test_build_initial_state_has_human_message():
    from langchain_core.messages import HumanMessage
    state = build_initial_state("./data/samples/iris.csv")
    assert len(state["messages"]) == 1
    assert isinstance(state["messages"][0], HumanMessage)
    assert "iris.csv" in state["messages"][0].content


def test_build_initial_state_deployment_pending():
    state = build_initial_state("./data/samples/iris.csv")
    assert state["deployment_decision"] == "pending"


def test_build_initial_state_validation_false():
    state = build_initial_state("./data/samples/iris.csv")
    assert state["validation_passed"] is False
    assert state["evaluation_passed"] is False
```

- [ ] **Step 5: Run tests — verify they FAIL**

```bash
uv run pytest tests/test_dashboard/test_pipeline_helpers.py -v
```

Expected: `ImportError` — `dashboard.pipeline_helpers` does not exist yet.

- [ ] **Step 6: Create `dashboard/pipeline_helpers.py`**

```python
"""Pure helper functions for the Pipeline Streamlit page.

No Streamlit imports — these are extracted for testability.
"""

from langchain_core.messages import HumanMessage


def event_to_log_line(event: dict) -> str | None:
    """Convert a LangGraph stream event dict to a UI log line.

    Returns None for events that should be silently skipped.

    Event shapes:
      {"supervisor": {"next": "data_validator", ...}}  → routing line
      {"supervisor": {"next": "FINISH", ...}}          → finish line
      {"data_validator": {...}}                         → worker line
      {"__interrupt__": [...]}                         → None (handled by caller)
    """
    if "__interrupt__" in event:
        return None

    if "supervisor" in event:
        next_agent = event["supervisor"].get("next", "")
        if next_agent == "FINISH":
            return "🏁 `[supervisor]` → **FINISH**"
        if next_agent:
            return f"🔀 `[supervisor]` → **{next_agent}**"
        return None

    node_name = next(iter(event))
    return f"✅ `[{node_name}]` completed"


def build_initial_state(dataset_path: str) -> dict:
    """Build the initial LangGraph state dict for a pipeline run."""
    return {
        "messages": [HumanMessage(content=f"Run the full MLOps pipeline on dataset: {dataset_path}")],
        "next": "",
        "dataset_path": dataset_path,
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
```

- [ ] **Step 7: Run tests — verify they PASS**

```bash
uv run pytest tests/test_dashboard/test_pipeline_helpers.py -v
```

Expected: `10 passed`.

- [ ] **Step 8: Run full unit suite to check for regressions**

```bash
uv run pytest -m "not integration" -v
```

Expected: all previously-passing tests still pass.

- [ ] **Step 9: Commit**

```bash
git add dashboard/__init__.py dashboard/pipeline_helpers.py tests/test_dashboard/__init__.py tests/test_dashboard/test_pipeline_helpers.py pyproject.toml
git commit -m "feat: extract pipeline helpers and add unit tests"
```

---

## Task 2: Rewrite `01_pipeline.py` with state machine and HITL panel

**Files:**
- Modify: `dashboard/pages/01_pipeline.py`

- [ ] **Step 1: Replace the entire file**

Overwrite `dashboard/pages/01_pipeline.py` with:

```python
"""Pipeline page — run and monitor the MLOps agent pipeline.

Three-phase state machine stored in st.session_state:
  idle             → shows dataset selector + Run button
  awaiting_approval → shows frozen log + HITL approval panel
  complete         → shows full log + outcome banner + Run Again button
"""

import time
from pathlib import Path

import streamlit as st
from langgraph.types import Command

from dashboard.pipeline_helpers import build_initial_state, event_to_log_line

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
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Helpers ───────────────────────────────────────────────────────────────────

def _log(line: str) -> None:
    st.session_state["log_lines"].append(line)


def _render_log(placeholder) -> None:
    placeholder.markdown("\n\n".join(st.session_state["log_lines"]))


def _resume_pipeline(resume: dict) -> None:
    """Resume the paused graph with the operator decision and stream remaining events."""
    from mlops_agents.graphs.mlops_graph import graph

    config = st.session_state["pipeline_config"]
    st.subheader("Pipeline Log")
    log_placeholder = st.empty()
    _render_log(log_placeholder)

    for event in graph.stream(Command(resume=resume), config=config):
        line = event_to_log_line(event)
        if line:
            _log(line)
            _render_log(log_placeholder)

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
    data_dir = Path("./data/samples")
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
        from mlops_agents.graphs.mlops_graph import graph
        from mlops_agents.config.constants import GRAPH_RECURSION_LIMIT

        thread_id = f"streamlit-{int(time.time())}"
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": GRAPH_RECURSION_LIMIT,
        }
        st.session_state["pipeline_config"] = config

        st.subheader("Pipeline Log")
        log_placeholder = st.empty()

        for event in graph.stream(build_initial_state(dataset_path), config=config):
            if "__interrupt__" in event:
                st.session_state["interrupt_value"] = event["__interrupt__"][0].value
                st.session_state["phase"] = "awaiting_approval"
                _log("⏸ **Pipeline paused — awaiting human approval**")
                _render_log(log_placeholder)
                st.rerun()
            else:
                line = event_to_log_line(event)
                if line:
                    _log(line)
                    _render_log(log_placeholder)

        # Stream ended without interrupt — pipeline finished inline
        if st.session_state["phase"] == "idle":
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
    st.subheader("Pipeline Log")
    st.markdown("\n\n".join(st.session_state["log_lines"]))

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
        st.session_state.clear()
        st.rerun()
```

- [ ] **Step 2: Verify the page imports cleanly**

```bash
uv run python -c "import ast, pathlib; ast.parse(pathlib.Path('dashboard/pages/01_pipeline.py').read_text()); print('OK')"
```

Expected output: `OK`

- [ ] **Step 3: Run the full unit suite**

```bash
uv run pytest -m "not integration" -v
```

Expected: all tests pass (0 failures).

- [ ] **Step 4: Start the dashboard and manually verify the idle phase**

```bash
uv run streamlit run dashboard/app.py
```

Open `http://localhost:8501`, navigate to **Pipeline** in the sidebar.

Expected: dataset selector shows `iris.csv`, "▶ Run Pipeline" button visible, no errors in terminal.

- [ ] **Step 5: Run the pipeline and verify the log streams**

Click "▶ Run Pipeline". Watch the terminal and browser.

Expected log lines appear one by one:
```
🔀 [supervisor] → data_validator
✅ [data_validator] completed
🔀 [supervisor] → trainer
✅ [trainer] completed
🔀 [supervisor] → evaluator
✅ [evaluator] completed
🔀 [supervisor] → deployer
⏸ Pipeline paused — awaiting human approval
```

- [ ] **Step 6: Verify the HITL approval panel**

After the pipeline pauses, the page should show:
- "⚠️ Human Approval Required" subheader
- The approval question text
- "Registration details" expander
- "✅ Approve" and "❌ Reject" buttons

- [ ] **Step 7: Approve and verify completion**

Click "✅ Approve".

Expected:
- Remaining log lines appear (`✅ [deployer] completed`, `🏁 [supervisor] → FINISH`)
- Green success banner: "Pipeline complete. Model promoted to champion."
- "🔄 Run Again" button appears

- [ ] **Step 8: Verify Run Again resets the page**

Click "🔄 Run Again".

Expected: page returns to idle state — dataset selector and Run button visible, log cleared.

- [ ] **Step 9: Commit**

```bash
git add dashboard/pages/01_pipeline.py
git commit -m "feat: rewrite pipeline page with state machine and HITL approval panel"
```

---

## Self-Review

**Spec coverage:**
- Three-phase state machine (`idle`, `awaiting_approval`, `complete`) → Task 2 ✅
- Live log with supervisor routing + worker completions → Task 2 (`event_to_log_line`) ✅
- HITL panel: question, registration details expander, Approve/Reject buttons → Task 2 ✅
- Reject flow with optional reason → Task 2 ✅
- Completion banner (approved/rejected/early-stop) → Task 2 ✅
- Run Again button → Task 2 ✅
- `event_to_log_line` and `build_initial_state` helpers tested → Task 1 ✅

**Placeholder scan:** None found. All code blocks are complete.

**Type consistency:**
- `event_to_log_line(event: dict) -> str | None` — defined in Task 1, used in Task 2 ✅
- `build_initial_state(dataset_path: str) -> dict` — defined in Task 1, used in Task 2 ✅
- `st.session_state["interrupt_value"]` set in idle phase, read in `awaiting_approval` phase ✅
- `Command(resume=resume)` — `resume` is `dict`, matches LangGraph signature ✅
