# Foundation: ML Dataset Contract & State Model — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `problem_type` and `task_metadata` to `AgentState`, enforce a machine-readable schema contract, and propagate task metadata to all downstream context builders.

**Architecture:** `_validate_schema_contract` runs deterministically before the agent in `data_validator_node`; on success the metadata (`problem_type`, `task_metadata`) is extracted after the agent and written to state once. All four context builders gain one or two new lines reading those typed fields. Supervisor snapshot gains `problem_type`. Graph topology is unchanged.

**Tech Stack:** Python 3.12, LangGraph `TypedDict` state, `unittest.mock` for all LLM tests.

---

### Task 1: Add `problem_type` and `task_metadata` to `AgentState`

**Files:**
- Modify: `src/mlops_agents/state/agent_state.py`
- Test: `tests/test_graphs/test_node_state_extraction.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_graphs/test_node_state_extraction.py` after the existing `test_agent_state_has_dataset_summary_field` test:

```python
def test_agent_state_has_problem_type_field():
    import typing
    from mlops_agents.state.agent_state import AgentState
    hints = typing.get_type_hints(AgentState)
    assert "problem_type" in hints
    assert hints["problem_type"] is str


def test_agent_state_has_task_metadata_field():
    import typing
    from mlops_agents.state.agent_state import AgentState
    hints = typing.get_type_hints(AgentState)
    assert "task_metadata" in hints
    assert hints["task_metadata"] is dict
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_graphs/test_node_state_extraction.py::test_agent_state_has_problem_type_field tests/test_graphs/test_node_state_extraction.py::test_agent_state_has_task_metadata_field -v
```

Expected: FAIL with `AssertionError`

- [ ] **Step 3: Add the fields to `AgentState`**

In `src/mlops_agents/state/agent_state.py`, after the `dataset_summary` line:

```python
    # Context isolation — built deterministically by data_validator_node
    dataset_summary: dict  # {row_count, column_names, dtypes, null_counts}

    # Task type — written once by data_validator_node before agent invocation
    problem_type: str   # "classification" | "regression" | "forecasting"

    # Task-level metadata — written once by data_validator_node
    task_metadata: dict
    # classification/regression: {"target_column": str}
    # forecasting: {
    #   "target_column": str,
    #   "datetime_column": str,
    #   "series_id_columns": list[str],
    #   "forecast_horizon": int,
    #   "frequency": str,
    # }
```

- [ ] **Step 4: Add `problem_type` and `task_metadata` to `_make_state()` in both test files**

In `tests/test_graphs/test_node_state_extraction.py`, update `_make_state()`:

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
    }
```

In `tests/test_agents/test_supervisor.py`, update `make_state()`:

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
    }
    base.update(kwargs)
    return base
```

- [ ] **Step 5: Run tests to verify they pass**

```
uv run pytest tests/test_graphs/test_node_state_extraction.py::test_agent_state_has_problem_type_field tests/test_graphs/test_node_state_extraction.py::test_agent_state_has_task_metadata_field -v
```

Expected: PASS

- [ ] **Step 6: Run the full test suite to verify no regressions**

```
uv run pytest -m "not integration" -v
```

Expected: all existing tests pass

- [ ] **Step 7: Commit**

```bash
git add src/mlops_agents/state/agent_state.py tests/test_graphs/test_node_state_extraction.py tests/test_agents/test_supervisor.py
git commit -m "feat: add problem_type and task_metadata fields to AgentState"
```

---

### Task 2: Update iris schema with top-level contract fields

**Files:**
- Modify: `data/schemas/iris_classification.json`

- [ ] **Step 1: Add `problem_type` and `target_column` to the schema**

Replace the contents of `data/schemas/iris_classification.json` with:

```json
{
  "problem_type": "classification",
  "target_column": "target",
  "name": "iris_classification",
  "description": "Iris flower classification dataset",
  "columns": [
    {
      "name": "sepal_length",
      "dtype": "float",
      "description": "Sepal length in cm",
      "required": true,
      "nullable": false,
      "min": 0.0,
      "max": 30.0,
      "mapping_hint": "Usually named 'SepalLengthCm', 'sepal length (cm)' or similar in raw files"
    },
    {
      "name": "sepal_width",
      "dtype": "float",
      "description": "Sepal width in cm",
      "required": true,
      "nullable": false,
      "min": 0.0,
      "max": 30.0,
      "mapping_hint": "Usually named 'SepalWidthCm', 'sepal width (cm)' or similar"
    },
    {
      "name": "petal_length",
      "dtype": "float",
      "description": "Petal length in cm",
      "required": true,
      "nullable": false,
      "min": 0.0,
      "max": 30.0,
      "mapping_hint": "Usually named 'PetalLengthCm', 'petal length (cm)' or similar"
    },
    {
      "name": "petal_width",
      "dtype": "float",
      "description": "Petal width in cm",
      "required": true,
      "nullable": false,
      "min": 0.0,
      "max": 30.0,
      "mapping_hint": "Usually named 'PetalWidthCm', 'petal width (cm)' or similar"
    },
    {
      "name": "sample_id",
      "dtype": "int",
      "description": "Unique sample identifier used to join measurement files",
      "required": true,
      "nullable": false,
      "is_key": true,
      "mapping_hint": "Usually named 'id', 'sample_id', 'Id' or similar"
    },
    {
      "name": "target",
      "dtype": "str",
      "description": "Class label for the flower species",
      "required": true,
      "nullable": false,
      "allowed_values": ["setosa", "versicolor", "virginica"],
      "mapping_hint": "Often named 'species', 'class', 'label', or 'variety' in raw datasets"
    }
  ]
}
```

- [ ] **Step 2: Commit**

```bash
git add data/schemas/iris_classification.json
git commit -m "feat: add problem_type and target_column to iris schema contract"
```

---

### Task 3: Implement `_validate_schema_contract`

**Files:**
- Modify: `src/mlops_agents/graphs/mlops_graph.py`
- Test: `tests/test_graphs/test_node_state_extraction.py`

- [ ] **Step 1: Write failing tests for contract validation**

Add to `tests/test_graphs/test_node_state_extraction.py` after the AgentState field tests:

```python
# ---------------------------------------------------------------------------
# _validate_schema_contract
# ---------------------------------------------------------------------------


def test_validate_schema_contract_passes_for_valid_classification():
    from mlops_agents.graphs.mlops_graph import _validate_schema_contract

    schema = {
        "problem_type": "classification",
        "target_column": "label",
        "columns": [
            {"name": "feature_a"},
            {"name": "label"},
        ],
    }
    _validate_schema_contract(schema)  # must not raise


def test_validate_schema_contract_passes_for_valid_regression():
    from mlops_agents.graphs.mlops_graph import _validate_schema_contract

    schema = {
        "problem_type": "regression",
        "target_column": "price",
        "columns": [{"name": "size"}, {"name": "price"}],
    }
    _validate_schema_contract(schema)  # must not raise


def test_validate_schema_contract_passes_for_valid_forecasting():
    from mlops_agents.graphs.mlops_graph import _validate_schema_contract

    schema = {
        "problem_type": "forecasting",
        "target_column": "sales",
        "datetime_column": "date",
        "series_id_columns": ["store_id"],
        "forecast_horizon": 30,
        "frequency": "D",
        "columns": [
            {"name": "date"},
            {"name": "store_id"},
            {"name": "sales"},
        ],
    }
    _validate_schema_contract(schema)  # must not raise


def test_validate_schema_contract_raises_on_missing_problem_type():
    import pytest
    from mlops_agents.graphs.mlops_graph import _validate_schema_contract

    schema = {"target_column": "label", "columns": [{"name": "label"}]}
    with pytest.raises(ValueError, match="problem_type"):
        _validate_schema_contract(schema)


def test_validate_schema_contract_raises_on_unknown_problem_type():
    import pytest
    from mlops_agents.graphs.mlops_graph import _validate_schema_contract

    schema = {
        "problem_type": "clustering",
        "target_column": "label",
        "columns": [{"name": "label"}],
    }
    with pytest.raises(ValueError, match="problem_type"):
        _validate_schema_contract(schema)


def test_validate_schema_contract_raises_when_target_column_missing_from_schema():
    import pytest
    from mlops_agents.graphs.mlops_graph import _validate_schema_contract

    schema = {
        "problem_type": "classification",
        "target_column": "nonexistent",
        "columns": [{"name": "feature_a"}],
    }
    with pytest.raises(ValueError, match="target_column"):
        _validate_schema_contract(schema)


def test_validate_schema_contract_raises_when_target_column_not_declared():
    import pytest
    from mlops_agents.graphs.mlops_graph import _validate_schema_contract

    schema = {
        "problem_type": "regression",
        "columns": [{"name": "price"}],
    }
    with pytest.raises(ValueError, match="target_column"):
        _validate_schema_contract(schema)


def test_validate_schema_contract_raises_on_missing_forecasting_fields():
    import pytest
    from mlops_agents.graphs.mlops_graph import _validate_schema_contract

    schema = {
        "problem_type": "forecasting",
        "target_column": "sales",
        "columns": [{"name": "sales"}],
        # missing datetime_column, forecast_horizon, frequency
    }
    with pytest.raises(ValueError, match="datetime_column|forecast_horizon|frequency"):
        _validate_schema_contract(schema)


def test_validate_schema_contract_raises_when_forecast_horizon_not_positive():
    import pytest
    from mlops_agents.graphs.mlops_graph import _validate_schema_contract

    schema = {
        "problem_type": "forecasting",
        "target_column": "sales",
        "datetime_column": "date",
        "forecast_horizon": 0,
        "frequency": "D",
        "columns": [{"name": "date"}, {"name": "sales"}],
    }
    with pytest.raises(ValueError, match="forecast_horizon"):
        _validate_schema_contract(schema)


def test_validate_schema_contract_raises_when_datetime_column_not_in_columns():
    import pytest
    from mlops_agents.graphs.mlops_graph import _validate_schema_contract

    schema = {
        "problem_type": "forecasting",
        "target_column": "sales",
        "datetime_column": "nonexistent_date",
        "forecast_horizon": 7,
        "frequency": "D",
        "columns": [{"name": "sales"}],
    }
    with pytest.raises(ValueError, match="datetime_column"):
        _validate_schema_contract(schema)


def test_validate_schema_contract_raises_when_series_id_column_not_in_columns():
    import pytest
    from mlops_agents.graphs.mlops_graph import _validate_schema_contract

    schema = {
        "problem_type": "forecasting",
        "target_column": "sales",
        "datetime_column": "date",
        "series_id_columns": ["missing_store"],
        "forecast_horizon": 7,
        "frequency": "D",
        "columns": [{"name": "date"}, {"name": "sales"}],
    }
    with pytest.raises(ValueError, match="series_id_columns"):
        _validate_schema_contract(schema)
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_graphs/test_node_state_extraction.py -k "validate_schema_contract" -v
```

Expected: FAIL with `ImportError` (`_validate_schema_contract` not found)

- [ ] **Step 3: Implement `_validate_schema_contract` in `mlops_graph.py`**

Insert this function after `_build_deployer_context` and before `data_validator_node` in `src/mlops_agents/graphs/mlops_graph.py`:

```python
def _validate_schema_contract(schema_data: dict) -> None:
    """Validate ML dataset contract fields. Raises ValueError on any violation."""
    column_names = {c["name"] for c in schema_data.get("columns", [])}

    problem_type = schema_data.get("problem_type")
    if problem_type not in ("classification", "regression", "forecasting"):
        raise ValueError(
            f"Schema missing or invalid 'problem_type'. Got: {problem_type!r}. "
            "Must be 'classification', 'regression', or 'forecasting'."
        )

    target_column = schema_data.get("target_column")
    if not target_column or target_column not in column_names:
        raise ValueError(
            f"'target_column' must be declared and exist in columns. Got: {target_column!r}."
        )

    if problem_type == "forecasting":
        required = ["datetime_column", "forecast_horizon", "frequency"]
        missing = [f for f in required if not schema_data.get(f)]
        if missing:
            raise ValueError(f"Forecasting schema missing required fields: {missing}")

        if not isinstance(schema_data["forecast_horizon"], int) or schema_data["forecast_horizon"] <= 0:
            raise ValueError(
                f"'forecast_horizon' must be a positive integer. Got: {schema_data['forecast_horizon']!r}."
            )

        if schema_data["datetime_column"] not in column_names:
            raise ValueError(
                f"'datetime_column' '{schema_data['datetime_column']}' not found in columns."
            )

        for col in schema_data.get("series_id_columns", []):
            if col not in column_names:
                raise ValueError(f"'series_id_columns' entry '{col}' not found in columns.")
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/test_graphs/test_node_state_extraction.py -k "validate_schema_contract" -v
```

Expected: PASS (all 11 contract validation tests)

- [ ] **Step 5: Run the full test suite**

```
uv run pytest -m "not integration" -v
```

Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add src/mlops_agents/graphs/mlops_graph.py tests/test_graphs/test_node_state_extraction.py
git commit -m "feat: implement _validate_schema_contract for ML dataset contract"
```

---

### Task 4: Integrate contract validation and metadata extraction into `data_validator_node`

**Files:**
- Modify: `src/mlops_agents/graphs/mlops_graph.py`
- Test: `tests/test_graphs/test_node_state_extraction.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_graphs/test_node_state_extraction.py`:

```python
# ---------------------------------------------------------------------------
# data_validator_node — contract validation and metadata extraction
# ---------------------------------------------------------------------------


def test_data_validator_node_sets_problem_type_and_task_metadata_in_state():
    """data_validator_node must write problem_type and task_metadata to state after agent succeeds."""
    import tempfile, os

    from mlops_agents.graphs.mlops_graph import data_validator_node

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("sepal_length,target\n5.1,setosa\n6.3,versicolor\n")
        tmp_path = f.name

    schema = json.dumps({
        "problem_type": "classification",
        "target_column": "target",
        "columns": [{"name": "sepal_length"}, {"name": "target"}],
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
             patch("mlops_agents.graphs.mlops_graph.interrupt", return_value={"approved": True, "comment": ""}), \
             patch("builtins.open", side_effect=lambda p, *a, **kw: open(p, *a, **kw) if p != "data/schemas/.json" else __import__("io").StringIO(schema)), \
             patch("pathlib.Path.read_text", return_value=schema), \
             patch("pathlib.Path.exists", return_value=True):
            mock_agent = MagicMock()
            mock_agent.invoke.return_value = mock_result
            mock_get_agent.return_value = mock_agent

            command = data_validator_node(_make_state())
    finally:
        os.unlink(tmp_path)

    assert command.update.get("problem_type") == "classification"
    assert command.update.get("task_metadata") == {"target_column": "target"}


def test_data_validator_node_aborts_on_contract_violation():
    """data_validator_node must return error Command immediately when schema contract is invalid."""
    from mlops_agents.graphs.mlops_graph import data_validator_node

    bad_schema = json.dumps({"columns": [{"name": "feature_a"}]})  # no problem_type

    interrupt_called = []

    def fail_if_called(payload: dict) -> dict:
        interrupt_called.append(payload)
        return {}

    with patch("mlops_agents.graphs.mlops_graph.get_agent") as mock_get_agent, \
         patch("mlops_agents.graphs.mlops_graph.interrupt", side_effect=fail_if_called), \
         patch("pathlib.Path.read_text", return_value=bad_schema), \
         patch("pathlib.Path.exists", return_value=True):
        mock_agent = MagicMock()
        mock_get_agent.return_value = mock_agent

        command = data_validator_node(_make_state())

    mock_agent.invoke.assert_not_called()
    assert len(interrupt_called) == 0
    assert "problem_type" in command.update.get("error_message", "")
    assert command.update.get("validation_passed") is False
    assert command.update.get("problem_type") == ""
    assert command.update.get("task_metadata") == {}
    assert command.goto == "supervisor"
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_graphs/test_node_state_extraction.py::test_data_validator_node_sets_problem_type_and_task_metadata_in_state tests/test_graphs/test_node_state_extraction.py::test_data_validator_node_aborts_on_contract_violation -v
```

Expected: FAIL

- [ ] **Step 3: Update `data_validator_node` with validation and extraction**

In `src/mlops_agents/graphs/mlops_graph.py`, update `data_validator_node`. The full updated function body:

```python
def data_validator_node(state: AgentState) -> Command[Literal["supervisor"]]:
    import json as _json
    from pathlib import Path as _Path

    import pandas as pd

    from mlops_agents.config.settings import settings

    schema_file = _Path("data/schemas") / f"{settings.dataset_schema}.json"
    schema_json = schema_file.read_text() if schema_file.exists() else "{}"
    schema_path = str(schema_file.resolve())
    schema_data = _json.loads(schema_json) if schema_json != "{}" else {}

    try:
        _validate_schema_contract(schema_data)
    except ValueError as exc:
        error_msg = f"Schema contract violation: {exc}"
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
            },
            goto="supervisor",
        )

    agent = get_agent("data_validator")
    result = agent.invoke({"messages": [_build_data_validator_context(state, schema_json=schema_json, schema_path=schema_path)]})
    final_message = result["messages"][-1].content

    quality_report: dict = _extract_tool_json(result["messages"], "check_data_quality")
    mapping_result: dict = _extract_tool_json(result["messages"], "apply_column_mapping")
    validation_result: dict = _extract_tool_json(result["messages"], "validate_against_schema")
    imputation_result: dict = _extract_tool_json(result["messages"], "impute_missing_values")

    processed_path = (
        mapping_result.get("output_path", "")
        or validation_result.get("output_path", "")
    )
    validation_passed = bool(validation_result.get("passed", False))

    dataset_summary: dict = {}
    preview: dict = {"shape": [0, 0], "columns": [], "sample_rows": []}
    if processed_path:
        try:
            df = pd.read_csv(processed_path)
            dataset_summary = {
                "row_count": len(df),
                "column_names": list(df.columns),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "null_counts": df.isnull().sum().to_dict(),
            }
            preview = {
                "shape": list(df.shape),
                "columns": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns],
                "sample_rows": json.loads(df.head(20).to_json(orient="records")),
            }
        except Exception:
            pass

    problem_type: str = schema_data.get("problem_type", "")
    task_metadata: dict = {"target_column": schema_data.get("target_column", "")}
    if problem_type == "forecasting":
        task_metadata.update({
            "datetime_column": schema_data.get("datetime_column", ""),
            "series_id_columns": schema_data.get("series_id_columns", []),
            "forecast_horizon": schema_data.get("forecast_horizon"),
            "frequency": schema_data.get("frequency", ""),
        })

    base_update = {
        "messages": [HumanMessage(content=final_message, name="data_validator")],
        "validation_report": quality_report,
        "validation_passed": validation_passed,
        "dataset_path": processed_path,
        "dataset_summary": dataset_summary,
        "problem_type": problem_type,
        "task_metadata": task_metadata,
    }

    if not validation_passed:
        error_msg = f"Data validation failed after auto-fix attempt: {final_message}"
        logger.warning("[data_validator] validation failed — aborting without HITL")
        return Command(
            update={**base_update, "error_message": error_msg},
            goto="supervisor",
        )

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
    error_msg = f"Human rejected validated dataset. Comment: {comment}"
    logger.warning("[data_validator] human rejected — aborting pipeline")
    return Command(
        update={**base_update, "error_message": error_msg},
        goto="supervisor",
    )
```

Note: the `import json as _json` avoids shadowing the module-level `json` import already at the top of the file. Actually, `json` is already imported at the top of `mlops_graph.py`, so use `json.loads` directly and remove the local alias. The local `_Path` and `pd` imports remain inside the function since they're only needed in this node.

> **Important:** Check the existing imports at the top of `mlops_graph.py` — `json` is already imported at the module level. In the function body, write `json.loads(schema_json)` directly, not `_json.loads`. Remove the `import json as _json` line — use the existing `import json`.

- [ ] **Step 4: Run the new tests**

```
uv run pytest tests/test_graphs/test_node_state_extraction.py::test_data_validator_node_sets_problem_type_and_task_metadata_in_state tests/test_graphs/test_node_state_extraction.py::test_data_validator_node_aborts_on_contract_violation -v
```

Expected: PASS

- [ ] **Step 5: Run the full test suite**

```
uv run pytest -m "not integration" -v
```

Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add src/mlops_agents/graphs/mlops_graph.py tests/test_graphs/test_node_state_extraction.py
git commit -m "feat: validate schema contract and extract problem_type/task_metadata in data_validator_node"
```

---

### Task 5: Update the four context builders with `problem_type` and `task_metadata`

**Files:**
- Modify: `src/mlops_agents/graphs/mlops_graph.py`
- Test: `tests/test_graphs/test_node_state_extraction.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_graphs/test_node_state_extraction.py`:

```python
def test_build_trainer_context_includes_problem_type_and_task_metadata():
    from mlops_agents.graphs.mlops_graph import _build_trainer_context

    state = _make_state()
    state["problem_type"] = "classification"
    state["task_metadata"] = {"target_column": "target"}
    msg = _build_trainer_context(state)
    assert "Problem type: classification" in msg.content
    assert "target_column" in msg.content


def test_build_evaluator_context_includes_problem_type_and_task_metadata():
    from mlops_agents.graphs.mlops_graph import _build_evaluator_context

    state = _make_state()
    state["problem_type"] = "regression"
    state["task_metadata"] = {"target_column": "price"}
    msg = _build_evaluator_context(state)
    assert "Problem type: regression" in msg.content
    assert "target_column" in msg.content


def test_build_deployer_context_includes_problem_type():
    from mlops_agents.graphs.mlops_graph import _build_deployer_context

    state = _make_state()
    state["problem_type"] = "forecasting"
    msg = _build_deployer_context(state)
    assert "Problem type: forecasting" in msg.content
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_graphs/test_node_state_extraction.py::test_build_trainer_context_includes_problem_type_and_task_metadata tests/test_graphs/test_node_state_extraction.py::test_build_evaluator_context_includes_problem_type_and_task_metadata tests/test_graphs/test_node_state_extraction.py::test_build_deployer_context_includes_problem_type -v
```

Expected: FAIL with `AssertionError`

- [ ] **Step 3: Update the three context builders in `mlops_graph.py`**

Replace `_build_trainer_context`:

```python
def _build_trainer_context(state: AgentState) -> HumanMessage:
    return HumanMessage(content=(
        f"Problem type: {state.get('problem_type', '')}\n"
        f"Task metadata: {json.dumps(state.get('task_metadata') or {})}\n"
        f"Canonical dataset: {state.get('dataset_path', '')}\n"
        f"Dataset summary: {json.dumps(state.get('dataset_summary') or {})}"
    ))
```

Replace `_build_evaluator_context`:

```python
def _build_evaluator_context(state: AgentState) -> HumanMessage:
    return HumanMessage(content=(
        f"Problem type: {state.get('problem_type', '')}\n"
        f"Task metadata: {json.dumps(state.get('task_metadata') or {})}\n"
        f"Training run ID: {state.get('training_run_id', '')}\n"
        f"Trained model path: {state.get('trained_model_path', '')}\n"
        f"Training metrics: {json.dumps(state.get('training_metrics') or {})}"
    ))
```

Replace `_build_deployer_context`:

```python
def _build_deployer_context(state: AgentState) -> HumanMessage:
    return HumanMessage(content=(
        f"Problem type: {state.get('problem_type', '')}\n"
        f"Best model URI: {state.get('best_model_uri', '')}\n"
        f"Training run ID: {state.get('training_run_id', '')}\n"
        f"Evaluation report: {json.dumps(state.get('evaluation_report') or {})}"
    ))
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/test_graphs/test_node_state_extraction.py -k "build_trainer or build_evaluator or build_deployer" -v
```

Expected: all pass including the previously existing tests

- [ ] **Step 5: Run the full test suite**

```
uv run pytest -m "not integration" -v
```

Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add src/mlops_agents/graphs/mlops_graph.py tests/test_graphs/test_node_state_extraction.py
git commit -m "feat: propagate problem_type and task_metadata through trainer, evaluator, deployer context builders"
```

---

### Task 6: Add `problem_type` to supervisor state snapshot

**Files:**
- Modify: `src/mlops_agents/agents/supervisor.py`
- Test: `tests/test_agents/test_supervisor.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_agents/test_supervisor.py`:

```python
@patch("mlops_agents.agents.supervisor._router_llm")
def test_supervisor_snapshot_includes_problem_type(mock_llm):
    """Supervisor snapshot must include problem_type from state."""
    import json

    captured_messages = []

    mock_structured = MagicMock()

    def capture_invoke(messages):
        captured_messages.extend(messages)
        return RouterOutput(next="FINISH", reasoning="done")

    mock_structured.invoke.side_effect = capture_invoke
    mock_llm.with_structured_output.return_value = mock_structured

    from mlops_agents.agents.supervisor import supervisor_node

    state = make_state(
        problem_type="classification",
        agent_attempt_counts={"data_validator": 1},
    )
    supervisor_node(state)

    last_msg = captured_messages[-1]
    snapshot = json.loads(last_msg.content.replace("Pipeline state:\n", ""))
    assert "problem_type" in snapshot
    assert snapshot["problem_type"] == "classification"
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_agents/test_supervisor.py::test_supervisor_snapshot_includes_problem_type -v
```

Expected: FAIL with `AssertionError` (`problem_type` not in snapshot)

- [ ] **Step 3: Update `supervisor_node` to include `problem_type` in snapshot**

In `src/mlops_agents/agents/supervisor.py`, update `snapshot_data`:

```python
    snapshot_data = {
        "problem_type": state.get("problem_type", ""),
        "validation_passed": state.get("validation_passed") if dv_has_run else None,
        "evaluation_passed": state.get("evaluation_passed"),
        "deployment_decision": state.get("deployment_decision", "pending"),
        "error_message": state.get("error_message", ""),
        "training_run_id": state.get("training_run_id", ""),
    }
```

- [ ] **Step 4: Run all supervisor tests**

```
uv run pytest tests/test_agents/test_supervisor.py -v
```

Expected: all pass

- [ ] **Step 5: Run the full test suite**

```
uv run pytest -m "not integration" -v
```

Expected: all tests pass

- [ ] **Step 6: Lint and type-check**

```
uv run ruff check . && uv run ruff format . && uv run mypy src/
```

Expected: no errors

- [ ] **Step 7: Commit**

```bash
git add src/mlops_agents/agents/supervisor.py tests/test_agents/test_supervisor.py
git commit -m "feat: add problem_type to supervisor state snapshot"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| `problem_type: str` and `task_metadata: dict` in `AgentState` | Task 1 |
| `iris_classification.json` gets `problem_type` and `target_column` | Task 2 |
| `_validate_schema_contract` implemented | Task 3 |
| Contract validation runs before agent in `data_validator_node` | Task 4 |
| Metadata extraction runs after agent — `problem_type` + `task_metadata` written to all Command.update paths | Task 4 |
| `_build_trainer_context` gains `problem_type` + `task_metadata` | Task 5 |
| `_build_evaluator_context` gains `problem_type` + `task_metadata` | Task 5 |
| `_build_deployer_context` gains `problem_type` | Task 5 |
| Supervisor snapshot gains `problem_type` | Task 6 |
| Tests updated for both test files | Tasks 1, 3, 4, 5, 6 |

**No placeholders found.**

**Type consistency:** `problem_type: str` declared in Task 1, used as `str` in all later tasks. `task_metadata: dict` declared in Task 1, built as `dict` in Task 4, passed as `dict` in Task 5.

**Note on Task 4 patching:** The tests for `data_validator_node` with contract validation use `patch("pathlib.Path.read_text", ...)` and `patch("pathlib.Path.exists", ...)` to inject a controlled schema without touching the real filesystem. If tests fail because the patch doesn't apply to the right code path, check whether `mlops_graph.py` uses `schema_file.read_text()` — confirm exact method name and adjust patch target accordingly.
