# Schema Upload UI & Pydantic Contract Validation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users upload a JSON schema from the Streamlit UI, validate it immediately with Pydantic, and pass it through `AgentState` into `data_validator_node` — replacing the hardcoded file-based schema load.

**Architecture:** `SchemaContract` (Pydantic) validates the uploaded JSON in the dashboard; the JSON string travels as `AgentState.schema_json`; `data_validator_node` reads from state instead of disk and `_validate_schema_contract` still runs as a second check. Run button is disabled until a valid schema is present.

**Tech Stack:** Pydantic v2, Streamlit `st.file_uploader`, LangGraph `AgentState` TypedDict.

---

### Task 1: Add `SchemaContract` and `ColumnSchema` Pydantic models

**Files:**
- Modify: `src/mlops_agents/state/schemas.py`
- Create: `tests/test_state/__init__.py`
- Create: `tests/test_state/test_schemas.py`

- [ ] **Step 1: Create the test directory and file**

```bash
# PowerShell
New-Item -ItemType Directory -Path tests\test_state -Force
New-Item -ItemType File -Path tests\test_state\__init__.py -Force
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_state/test_schemas.py`:

```python
"""Unit tests for SchemaContract Pydantic model."""

import pytest
from pydantic import ValidationError

from mlops_agents.state.schemas import SchemaContract


def _classification_schema(**overrides):
    base = {
        "problem_type": "classification",
        "target_column": "label",
        "columns": [{"name": "feature_a", "dtype": "float"}, {"name": "label", "dtype": "str"}],
    }
    base.update(overrides)
    return base


def _forecasting_schema(**overrides):
    base = {
        "problem_type": "forecasting",
        "target_column": "sales",
        "datetime_column": "date",
        "series_id_columns": ["store_id"],
        "forecast_horizon": 30,
        "frequency": "D",
        "columns": [
            {"name": "date", "dtype": "datetime"},
            {"name": "store_id", "dtype": "str"},
            {"name": "sales", "dtype": "float"},
        ],
    }
    base.update(overrides)
    return base


def test_valid_classification_schema_passes():
    contract = SchemaContract.model_validate(_classification_schema())
    assert contract.problem_type == "classification"
    assert contract.target_column == "label"


def test_valid_regression_schema_passes():
    schema = {
        "problem_type": "regression",
        "target_column": "price",
        "columns": [{"name": "size", "dtype": "float"}, {"name": "price", "dtype": "float"}],
    }
    contract = SchemaContract.model_validate(schema)
    assert contract.problem_type == "regression"


def test_valid_forecasting_schema_passes():
    contract = SchemaContract.model_validate(_forecasting_schema())
    assert contract.problem_type == "forecasting"
    assert contract.forecast_horizon == 30


def test_extra_column_fields_allowed():
    schema = _classification_schema()
    schema["columns"][0]["nullable"] = False
    schema["columns"][0]["description"] = "A feature"
    schema["columns"][0]["mapping_hint"] = "some hint"
    # Must not raise
    SchemaContract.model_validate(schema)


def test_extra_top_level_fields_allowed():
    schema = _classification_schema()
    schema["name"] = "my_dataset"
    schema["description"] = "for testing"
    SchemaContract.model_validate(schema)


def test_missing_problem_type_raises():
    schema = _classification_schema()
    del schema["problem_type"]
    with pytest.raises(ValidationError):
        SchemaContract.model_validate(schema)


def test_invalid_problem_type_raises():
    with pytest.raises(ValidationError):
        SchemaContract.model_validate(_classification_schema(problem_type="clustering"))


def test_target_column_not_in_columns_raises():
    with pytest.raises(ValidationError, match="target_column"):
        SchemaContract.model_validate(_classification_schema(target_column="nonexistent"))


def test_missing_target_column_raises():
    schema = _classification_schema()
    del schema["target_column"]
    with pytest.raises(ValidationError):
        SchemaContract.model_validate(schema)


def test_forecasting_missing_datetime_column_raises():
    schema = _forecasting_schema()
    del schema["datetime_column"]
    with pytest.raises(ValidationError, match="datetime_column"):
        SchemaContract.model_validate(schema)


def test_forecasting_datetime_column_not_in_columns_raises():
    with pytest.raises(ValidationError, match="datetime_column"):
        SchemaContract.model_validate(_forecasting_schema(datetime_column="nonexistent"))


def test_forecasting_zero_horizon_raises():
    with pytest.raises(ValidationError, match="forecast_horizon"):
        SchemaContract.model_validate(_forecasting_schema(forecast_horizon=0))


def test_forecasting_negative_horizon_raises():
    with pytest.raises(ValidationError, match="forecast_horizon"):
        SchemaContract.model_validate(_forecasting_schema(forecast_horizon=-1))


def test_forecasting_missing_frequency_raises():
    schema = _forecasting_schema()
    del schema["frequency"]
    with pytest.raises(ValidationError, match="frequency"):
        SchemaContract.model_validate(schema)


def test_forecasting_series_id_not_in_columns_raises():
    with pytest.raises(ValidationError, match="series_id_columns"):
        SchemaContract.model_validate(_forecasting_schema(series_id_columns=["missing_col"]))


def test_forecasting_empty_series_id_columns_passes():
    # Single-series forecasting: series_id_columns may be []
    schema = _forecasting_schema(series_id_columns=[])
    SchemaContract.model_validate(schema)
```

- [ ] **Step 3: Run tests to verify they fail**

```
uv run pytest tests/test_state/test_schemas.py -v
```

Expected: FAIL with `ImportError` (`SchemaContract` not found)

- [ ] **Step 4: Add models to `src/mlops_agents/state/schemas.py`**

Add after the existing imports at the top of the file:

```python
from pydantic import ConfigDict, model_validator
```

Add these two classes after the existing `EvaluationResult` class:

```python
class ColumnSchema(BaseModel):
    """Single column entry in a schema JSON."""
    model_config = ConfigDict(extra="allow")  # nullable, description, mapping_hint, etc. passed through

    name: str
    dtype: str


class SchemaContract(BaseModel):
    """Top-level ML dataset schema contract — validated on upload before the pipeline runs."""
    model_config = ConfigDict(extra="allow")  # name, description top-level fields allowed

    problem_type: Literal["classification", "regression", "forecasting"]
    target_column: str
    columns: list[ColumnSchema]
    datetime_column: str | None = None
    series_id_columns: list[str] = []
    forecast_horizon: int | None = None
    frequency: str | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> "SchemaContract":
        column_names = {c.name for c in self.columns}
        if self.target_column not in column_names:
            raise ValueError(
                f"'target_column' '{self.target_column}' not found in columns."
            )
        if self.problem_type == "forecasting":
            if not self.datetime_column:
                raise ValueError("'datetime_column' required for forecasting.")
            if self.datetime_column not in column_names:
                raise ValueError(
                    f"'datetime_column' '{self.datetime_column}' not found in columns."
                )
            if not self.forecast_horizon or self.forecast_horizon <= 0:
                raise ValueError("'forecast_horizon' must be a positive integer.")
            if not self.frequency:
                raise ValueError("'frequency' required for forecasting.")
            for col in self.series_id_columns:
                if col not in column_names:
                    raise ValueError(
                        f"'series_id_columns' entry '{col}' not found in columns."
                    )
        return self
```

- [ ] **Step 5: Run tests to verify they pass**

```
uv run pytest tests/test_state/test_schemas.py -v
```

Expected: all 16 tests PASS

- [ ] **Step 6: Run full suite for regressions**

```
uv run pytest -m "not integration" -v
```

Expected: all tests pass

- [ ] **Step 7: Commit**

```bash
git add src/mlops_agents/state/schemas.py tests/test_state/__init__.py tests/test_state/test_schemas.py
git commit -m "feat: add SchemaContract and ColumnSchema Pydantic models for schema upload validation"
```

---

### Task 2: Add `schema_json` to `AgentState` and update test fixtures

**Files:**
- Modify: `src/mlops_agents/state/agent_state.py`
- Modify: `tests/test_graphs/test_node_state_extraction.py`
- Modify: `tests/test_agents/test_supervisor.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_graphs/test_node_state_extraction.py` after `test_agent_state_has_task_metadata_field`:

```python
def test_agent_state_has_schema_json_field():
    import typing
    from mlops_agents.state.agent_state import AgentState
    hints = typing.get_type_hints(AgentState)
    assert "schema_json" in hints
    assert hints["schema_json"] is str
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_graphs/test_node_state_extraction.py::test_agent_state_has_schema_json_field -v
```

Expected: FAIL with `AssertionError`

- [ ] **Step 3: Add field to `AgentState`**

In `src/mlops_agents/state/agent_state.py`, add after `task_metadata`:

```python
    # Raw schema JSON string — uploaded by user via UI and passed through state.
    # data_validator_node reads this instead of loading from disk.
    schema_json: str
```

- [ ] **Step 4: Update `_make_state()` in `test_node_state_extraction.py`**

Add `"schema_json": "{}"` to the returned dict in `_make_state()`:

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
        "problem_type": "",
        "task_metadata": {},
        "schema_json": "{}",
    }
```

- [ ] **Step 5: Update `make_state()` in `test_supervisor.py`**

Add `"schema_json": ""` to the `base` dict in `make_state()`:

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
        "problem_type": "",
        "task_metadata": {},
        "schema_json": "",
    }
    base.update(kwargs)
    return base
```

- [ ] **Step 6: Run the new test and full suite**

```
uv run pytest tests/test_graphs/test_node_state_extraction.py::test_agent_state_has_schema_json_field -v
uv run pytest -m "not integration" -v
```

Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add src/mlops_agents/state/agent_state.py tests/test_graphs/test_node_state_extraction.py tests/test_agents/test_supervisor.py
git commit -m "feat: add schema_json field to AgentState"
```

---

### Task 3: Update `data_validator_node` to read schema from state

**Files:**
- Modify: `src/mlops_agents/graphs/mlops_graph.py`
- Modify: `tests/test_graphs/test_node_state_extraction.py`

The node currently reads the schema like this (lines 131–141):

```python
def data_validator_node(state: AgentState) -> Command[Literal["supervisor"]]:
    from pathlib import Path as _Path

    import pandas as pd

    from mlops_agents.config.settings import settings

    schema_file = _Path("data/schemas") / f"{settings.dataset_schema}.json"
    schema_json = schema_file.read_text() if schema_file.exists() else "{}"
    schema_path = str(schema_file.resolve())
    schema_data = json.loads(schema_json) if schema_json != "{}" else {}
```

- [ ] **Step 1: Write failing tests**

Add to `tests/test_graphs/test_node_state_extraction.py` after the existing `data_validator_node` tests:

```python
def test_data_validator_node_reads_schema_json_from_state():
    """data_validator_node must use state['schema_json'] instead of loading from disk."""
    import os
    import tempfile

    from mlops_agents.graphs.mlops_graph import data_validator_node

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("sepal_length,target\n5.1,setosa\n6.3,versicolor\n")
        tmp_path = f.name

    schema = json.dumps({
        "problem_type": "classification",
        "target_column": "target",
        "columns": [{"name": "sepal_length", "dtype": "float"}, {"name": "target", "dtype": "str"}],
    })
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
            state["schema_json"] = schema
            command = data_validator_node(state)
    finally:
        os.unlink(tmp_path)

    assert command.update.get("problem_type") == "classification"
    assert command.update.get("task_metadata") == {"target_column": "target"}


def test_data_validator_node_aborts_when_schema_json_empty():
    """data_validator_node must abort immediately when schema_json is empty."""
    from mlops_agents.graphs.mlops_graph import data_validator_node

    with patch("mlops_agents.graphs.mlops_graph.get_agent") as mock_get_agent, \
         patch("mlops_agents.graphs.mlops_graph.interrupt") as mock_interrupt:
        mock_agent = MagicMock()
        mock_get_agent.return_value = mock_agent

        state = _make_state()
        state["schema_json"] = ""  # no schema uploaded
        command = data_validator_node(state)

    mock_agent.invoke.assert_not_called()
    mock_interrupt.assert_not_called()
    assert command.update.get("validation_passed") is False
    assert "schema" in command.update.get("error_message", "").lower()
    assert command.goto == "supervisor"
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_graphs/test_node_state_extraction.py::test_data_validator_node_reads_schema_json_from_state tests/test_graphs/test_node_state_extraction.py::test_data_validator_node_aborts_when_schema_json_empty -v
```

Expected: FAIL

- [ ] **Step 3: Update `data_validator_node` in `mlops_graph.py`**

Replace the schema-reading block and the `from pathlib import Path as _Path` / `from mlops_agents.config.settings import settings` local imports. The updated node opening:

```python
def data_validator_node(state: AgentState) -> Command[Literal["supervisor"]]:
    import pandas as pd

    schema_json: str = state.get("schema_json") or ""
    schema_path = "(uploaded via UI)" if schema_json else "(none)"
    schema_data = json.loads(schema_json) if schema_json else {}

    if not schema_json:
        error_msg = "No schema uploaded. Upload a schema JSON before running the pipeline."
        logger.error(f"[data_validator] {error_msg}")
        return Command(
            update={
                "messages": [HumanMessage(content=error_msg, name="data_validator")],
                "validation_passed": False,
                "error_message": error_msg,
                "problem_type": "",
                "task_metadata": {},
                "dataset_summary": {},
                "validation_report": {},
                "dataset_path": "",
                "schema_json": "",
            },
            goto="supervisor",
        )

    try:
        _validate_schema_contract(schema_data)
    except ValueError as exc:
        ...  # existing block unchanged
```

The rest of the function body (from `try: _validate_schema_contract` onward) is **unchanged**. Only the schema-reading lines at the top are replaced.

Also add `"schema_json": schema_json` to `base_update` so it is preserved in state across all return paths:

```python
    base_update = {
        "messages": [HumanMessage(content=final_message, name="data_validator")],
        "validation_report": quality_report,
        "validation_passed": validation_passed,
        "dataset_path": processed_path,
        "dataset_summary": dataset_summary,
        "problem_type": problem_type,
        "task_metadata": task_metadata,
        "schema_json": schema_json,
    }
```

Also add `"schema_json": ""` to the `ValueError` early-return Command (the contract-violation path):

```python
        return Command(
            update={
                "messages": [HumanMessage(content=error_msg, name="data_validator")],
                "validation_passed": False,
                "error_message": error_msg,
                "problem_type": "",
                "task_metadata": {},
                "dataset_summary": {},
                "validation_report": {},
                "dataset_path": "",
                "schema_json": schema_json,
            },
            goto="supervisor",
        )
```

Finally, add `"schema_json": ""` to `main()`'s `initial_state`:

```python
    initial_state: dict = {
        "messages": [...],
        "next": "",
        "dataset_paths": dataset_paths,
        "dataset_path": "",
        "dataset_summary": {},
        "problem_type": "",
        "task_metadata": {},
        "schema_json": "",        # ← add this line
        "validation_passed": False,
        ...
    }
```

- [ ] **Step 4: Run the new tests**

```
uv run pytest tests/test_graphs/test_node_state_extraction.py::test_data_validator_node_reads_schema_json_from_state tests/test_graphs/test_node_state_extraction.py::test_data_validator_node_aborts_when_schema_json_empty -v
```

Expected: PASS

- [ ] **Step 5: Run full suite**

```
uv run pytest -m "not integration" -v
```

Expected: all pass. Note: the previously existing tests that patched `pathlib.Path.read_text` and `pathlib.Path.exists` now test dead code paths — they will still pass because `_make_state()` now provides `"schema_json": "{}"` which is an empty/default JSON object. Those tests (`test_data_validator_node_sets_problem_type_and_task_metadata_in_state`, `test_data_validator_node_aborts_on_contract_violation`) should be updated to set `state["schema_json"]` directly instead of patching `pathlib.Path`. Do this update now:

In `test_data_validator_node_sets_problem_type_and_task_metadata_in_state`: remove the `patch("pathlib.Path.read_text", ...)` and `patch("pathlib.Path.exists", ...)` lines; instead set `state["schema_json"] = schema` before calling `data_validator_node(state)`.

In `test_data_validator_node_aborts_on_contract_violation`: remove the `patch("pathlib.Path.read_text", ...)` and `patch("pathlib.Path.exists", ...)` lines; instead set `state["schema_json"] = bad_schema` before calling `data_validator_node(state)`.

- [ ] **Step 6: Run full suite again**

```
uv run pytest -m "not integration" -v
```

Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add src/mlops_agents/graphs/mlops_graph.py tests/test_graphs/test_node_state_extraction.py
git commit -m "feat: data_validator_node reads schema_json from state instead of disk"
```

---

### Task 4: Update `build_initial_state` in `pipeline_helpers.py`

**Files:**
- Modify: `dashboard/pipeline_helpers.py`
- Modify: `tests/test_dashboard/test_pipeline_helpers.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_dashboard/test_pipeline_helpers.py`:

```python
def test_build_initial_state_includes_schema_json():
    schema = '{"problem_type": "classification"}'
    state = build_initial_state(["./data/samples/iris.csv"], schema_json=schema)
    assert state["schema_json"] == schema


def test_build_initial_state_schema_json_defaults_to_empty():
    state = build_initial_state(["./data/samples/iris.csv"])
    assert state["schema_json"] == ""


def test_build_initial_state_includes_problem_type_and_task_metadata():
    state = build_initial_state(["./data/samples/iris.csv"])
    assert "problem_type" in state
    assert state["problem_type"] == ""
    assert "task_metadata" in state
    assert state["task_metadata"] == {}


def test_build_initial_state_includes_dataset_summary():
    state = build_initial_state(["./data/samples/iris.csv"])
    assert "dataset_summary" in state
    assert state["dataset_summary"] == {}
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_dashboard/test_pipeline_helpers.py::test_build_initial_state_includes_schema_json tests/test_dashboard/test_pipeline_helpers.py::test_build_initial_state_schema_json_defaults_to_empty tests/test_dashboard/test_pipeline_helpers.py::test_build_initial_state_includes_problem_type_and_task_metadata tests/test_dashboard/test_pipeline_helpers.py::test_build_initial_state_includes_dataset_summary -v
```

Expected: FAIL

- [ ] **Step 3: Update `build_initial_state` in `dashboard/pipeline_helpers.py`**

```python
def build_initial_state(dataset_paths: list[str], schema_json: str = "") -> dict:
    """Build the initial LangGraph state dict for a pipeline run."""
    paths_display = ", ".join(dataset_paths)
    return {
        "messages": [HumanMessage(content=f"Run the full MLOps pipeline on these raw files: {paths_display}")],
        "next": "",
        "dataset_paths": dataset_paths,
        "dataset_path": "",
        "schema_json": schema_json,
        "dataset_summary": {},
        "problem_type": "",
        "task_metadata": {},
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
    }
```

- [ ] **Step 4: Run all tests**

```
uv run pytest tests/test_dashboard/test_pipeline_helpers.py -v
uv run pytest -m "not integration" -v
```

Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add dashboard/pipeline_helpers.py tests/test_dashboard/test_pipeline_helpers.py
git commit -m "feat: add schema_json parameter to build_initial_state"
```

---

### Task 5: Add schema uploader to the dashboard

**Files:**
- Modify: `dashboard/pages/01_pipeline.py`

This task modifies the Streamlit UI only — no unit tests are possible for Streamlit widgets. Manual verification steps are provided instead.

- [ ] **Step 1: Add imports to `dashboard/pages/01_pipeline.py`**

Add to the existing import block at the top:

```python
import json

from pydantic import ValidationError

from mlops_agents.state.schemas import SchemaContract
```

- [ ] **Step 2: Add schema uploader in the idle phase**

The idle phase starts at line 182 with `if st.session_state["phase"] == "idle":`. Inside that block, **before** the `col1, col2 = st.columns([3, 1])` line, add:

```python
    # ── Schema upload ─────────────────────────────────────────────────────────
    st.subheader("Schema")
    uploaded_schema = st.file_uploader(
        "Upload schema JSON",
        type=["json"],
        help=(
            "JSON file declaring problem_type, target_column, and column definitions. "
            "Required before the pipeline can run."
        ),
    )
    if uploaded_schema is not None:
        try:
            raw = uploaded_schema.read()
            schema_data = json.loads(raw)
            SchemaContract.model_validate(schema_data)
            st.session_state["schema_json"] = raw.decode("utf-8")
            st.success(f"Schema valid — problem type: **{schema_data['problem_type']}**")
        except json.JSONDecodeError as exc:
            st.error(f"Not valid JSON: {exc}")
            st.session_state.pop("schema_json", None)
        except ValidationError as exc:
            first_error = exc.errors()[0]["msg"]
            st.error(f"Schema contract violation: {first_error}")
            st.session_state.pop("schema_json", None)

    schema_json = st.session_state.get("schema_json", "")
```

- [ ] **Step 3: Update the `st.subheader` and dataset section**

Change the `col1, col2 = st.columns([3, 1])` block to add `st.subheader("Dataset")` before it:

```python
    # ── Dataset selection ─────────────────────────────────────────────────────
    st.subheader("Dataset")
    col1, col2 = st.columns([3, 1])
```

- [ ] **Step 4: Gate the Run button on schema presence**

Update the Run button line from:

```python
        run_button = st.button("▶ Run Pipeline", type="primary", use_container_width=True)
```

to:

```python
        run_button = st.button(
            "▶ Run Pipeline",
            type="primary",
            use_container_width=True,
            disabled=not schema_json,
            help=None if schema_json else "Upload a valid schema JSON to enable the pipeline.",
        )
```

- [ ] **Step 5: Pass `schema_json` to `build_initial_state`**

Update the `build_initial_state` call (currently line 225):

```python
        for chunk in graph.stream(
            build_initial_state(dataset_paths, schema_json=schema_json),
            config=config,
            ...
        ):
```

- [ ] **Step 6: Add `"schema_json"` to `_DEFAULTS`**

In the `_DEFAULTS` dict near the top of the file, add:

```python
_DEFAULTS: dict = {
    "phase": "idle",
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
    "run_events": [],
    "schema_json": "",   # ← add this
}
```

- [ ] **Step 7: Manual verification**

Start the dashboard:

```bash
uv run streamlit run dashboard/app.py
```

Verify:
1. Navigate to the Pipeline page — a "Schema" section with file uploader appears above the dataset selector
2. Without uploading a schema, the "▶ Run Pipeline" button is greyed out with helper text
3. Upload `data/schemas/iris_classification.json` — success message shows `"Schema valid — problem type: **classification**"`
4. Button becomes active
5. Upload a broken JSON file (e.g., a `.txt` renamed to `.json`) — `"Not valid JSON"` error appears, button disabled again
6. Upload a JSON missing `problem_type` — `"Schema contract violation"` error, button disabled
7. With a valid schema and dataset selected, click Run — pipeline executes normally

- [ ] **Step 8: Run the unit test suite one final time**

```
uv run pytest -m "not integration" -v
```

Expected: all pass

- [ ] **Step 9: Commit**

```bash
git add dashboard/pages/01_pipeline.py
git commit -m "feat: add schema upload widget with Pydantic validation to pipeline UI"
```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| No hardcoded schema, no fallback | Task 3 (empty `schema_json` → immediate abort with error) |
| Schema upload via `st.file_uploader` | Task 5 |
| Pydantic validation on upload | Tasks 1, 5 |
| Run button disabled without valid schema | Task 5 |
| Clear error messages for invalid schema | Task 5 |
| `schema_json: str` in `AgentState` | Task 2 |
| `data_validator_node` reads from state | Task 3 |
| `_validate_schema_contract` still runs as second check | Task 3 (unchanged — function not touched) |
| `build_initial_state` passes `schema_json` | Task 4 |
| `main()` initial_state gets `schema_json: ""` | Task 3 |
| Extra column fields allowed by Pydantic | Task 1 (`ConfigDict(extra="allow")`) |
| Extra top-level fields allowed | Task 1 (`ConfigDict(extra="allow")`) |
| Tests updated for `_make_state()` and `make_state()` | Task 2 |
| Existing node tests cleaned up (remove pathlib patches) | Task 3 |

**No placeholders found.**

**Type consistency:** `schema_json: str` throughout — `AgentState` field, `build_initial_state` parameter, `state.get("schema_json")` return, `st.session_state["schema_json"]` storage. Consistent.
