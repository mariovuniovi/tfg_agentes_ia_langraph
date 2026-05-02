# Auto-Fix Validation HITL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the data_validator agent autonomously fix nullable violations using config-driven imputation, and only surface the HITL gate when validation actually passes.

**Architecture:** A new `impute_missing_values` tool reads strategy from `settings.py` and applies it in-place; the agent prompt is updated to call it automatically after a nullable failure; `data_validator_node` moves `interrupt()` inside `if validation_passed` and sets `error_message` on unfixable failures so the supervisor can FINISH cleanly.

**Tech Stack:** pandas, pydantic-settings, LangGraph `interrupt()`, langchain `@tool`

---

## File Map

| File | Change |
|------|--------|
| `src/mlops_agents/config/settings.py` | Add `imputation_strategy_numeric`, `imputation_strategy_categorical` |
| `src/mlops_agents/tools/data_tools.py` | Add module-level `settings` import + `impute_missing_values` tool |
| `src/mlops_agents/agents/data_agent.py` | Register `impute_missing_values` in tool list |
| `src/mlops_agents/prompts/data_agent.yaml` | Add autonomy rule + imputation step to PROCESS |
| `src/mlops_agents/graphs/mlops_graph.py` | Move `interrupt()` inside `if validation_passed`, add error path, add `imputation_result` extraction |
| `src/mlops_agents/prompts/supervisor.yaml` | Strengthen rule 5 to abort on `error_message` |
| `tests/test_tools/test_data_tools.py` | Tests for `impute_missing_values` |
| `tests/test_graphs/test_node_state_extraction.py` | Update node tests for new HITL behaviour |

---

### Task 1: Add imputation strategy fields to settings

**Files:**
- Modify: `src/mlops_agents/config/settings.py`

- [ ] **Step 1: Add the two new fields**

In `settings.py`, add after `max_attempts_per_agent`:

```python
from typing import Literal  # add at top with other imports

# ... existing fields ...
max_attempts_per_agent: int = 3
imputation_strategy_numeric: Literal["mean", "median", "zero"] = "mean"
imputation_strategy_categorical: Literal["mode", "unknown", "drop_row"] = "mode"
```

Full updated `Settings` class top section (existing fields unchanged, only additions shown):

```python
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = ""
    openai_model: str = "gpt-5-mini"
    openai_model_supervisor: str = "gpt-5-mini"
    openai_model_data_validator: str = "gpt-5-mini"
    openai_model_trainer: str = "gpt-5-mini"
    openai_model_evaluator: str = "gpt-5-mini"
    openai_model_deployer: str = "gpt-5.4-nano"

    mlflow_tracking_uri: str = "sqlite:///./mlflow.db"
    mlflow_experiment_name: str = "mlops-agents"
    evidently_workspace: str = "./evidently_workspace"

    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "mlops-multi-agent"

    log_level: str = "INFO"
    log_verbosity: int = 2
    data_dir: str = "./data/samples"
    dataset_schema: str = "iris_classification"
    max_attempts_per_agent: int = 3
    imputation_strategy_numeric: Literal["mean", "median", "zero"] = "mean"
    imputation_strategy_categorical: Literal["mode", "unknown", "drop_row"] = "mode"


settings = Settings()
```

- [ ] **Step 2: Verify Pydantic validates the field**

```bash
uv run python -c "from mlops_agents.config.settings import settings; print(settings.imputation_strategy_numeric, settings.imputation_strategy_categorical)"
```

Expected output: `mean mode`

- [ ] **Step 3: Commit**

```bash
git add src/mlops_agents/config/settings.py
git commit -m "feat: add imputation_strategy_numeric/categorical to settings"
```

---

### Task 2: Implement `impute_missing_values` tool (TDD)

**Files:**
- Modify: `src/mlops_agents/tools/data_tools.py`
- Test: `tests/test_tools/test_data_tools.py`

- [ ] **Step 1: Add `settings` import at module level in `data_tools.py`**

At the top of `data_tools.py`, after existing imports add:

```python
from mlops_agents.config.settings import settings
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_tools/test_data_tools.py`:

```python
# ---------------------------------------------------------------------------
# impute_missing_values
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock
import mlops_agents.tools.data_tools as _dt


def _make_settings(numeric: str = "mean", categorical: str = "mode") -> MagicMock:
    s = MagicMock()
    s.imputation_strategy_numeric = numeric
    s.imputation_strategy_categorical = categorical
    return s


def test_impute_numeric_mean(tmp_path, monkeypatch):
    from mlops_agents.tools.data_tools import impute_missing_values

    df = pd.DataFrame({"sepal_width": [3.0, None, 5.0], "target": ["a", "b", "c"]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    monkeypatch.setattr(_dt, "settings", _make_settings(numeric="mean"))
    result = json.loads(impute_missing_values.invoke({"path": str(path)}))

    assert "sepal_width" in result["imputed_columns"]
    assert result["imputed_columns"]["sepal_width"]["strategy"] == "mean"
    assert result["imputed_columns"]["sepal_width"]["rows_affected"] == 1
    assert abs(result["imputed_columns"]["sepal_width"]["fill_value"] - 4.0) < 0.01
    df_after = pd.read_csv(path)
    assert df_after["sepal_width"].isnull().sum() == 0


def test_impute_numeric_median(tmp_path, monkeypatch):
    from mlops_agents.tools.data_tools import impute_missing_values

    df = pd.DataFrame({"val": [1.0, None, 9.0, None, 5.0]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    monkeypatch.setattr(_dt, "settings", _make_settings(numeric="median"))
    result = json.loads(impute_missing_values.invoke({"path": str(path)}))

    assert result["imputed_columns"]["val"]["strategy"] == "median"
    assert result["imputed_columns"]["val"]["rows_affected"] == 2
    df_after = pd.read_csv(path)
    assert df_after["val"].isnull().sum() == 0


def test_impute_numeric_zero(tmp_path, monkeypatch):
    from mlops_agents.tools.data_tools import impute_missing_values

    df = pd.DataFrame({"val": [1.0, None, 3.0]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    monkeypatch.setattr(_dt, "settings", _make_settings(numeric="zero"))
    result = json.loads(impute_missing_values.invoke({"path": str(path)}))

    assert result["imputed_columns"]["val"]["fill_value"] == 0.0
    df_after = pd.read_csv(path)
    assert df_after["val"].isnull().sum() == 0


def test_impute_categorical_mode(tmp_path, monkeypatch):
    from mlops_agents.tools.data_tools import impute_missing_values

    df = pd.DataFrame({"species": ["setosa", "setosa", None, "virginica"]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    monkeypatch.setattr(_dt, "settings", _make_settings(categorical="mode"))
    result = json.loads(impute_missing_values.invoke({"path": str(path)}))

    assert result["imputed_columns"]["species"]["fill_value"] == "setosa"
    df_after = pd.read_csv(path)
    assert df_after["species"].isnull().sum() == 0


def test_impute_categorical_unknown(tmp_path, monkeypatch):
    from mlops_agents.tools.data_tools import impute_missing_values

    df = pd.DataFrame({"species": ["setosa", None]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    monkeypatch.setattr(_dt, "settings", _make_settings(categorical="unknown"))
    result = json.loads(impute_missing_values.invoke({"path": str(path)}))

    assert result["imputed_columns"]["species"]["fill_value"] == "unknown"


def test_impute_categorical_drop_row(tmp_path, monkeypatch):
    from mlops_agents.tools.data_tools import impute_missing_values

    df = pd.DataFrame({"species": ["setosa", None, "virginica"]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    monkeypatch.setattr(_dt, "settings", _make_settings(categorical="drop_row"))
    result = json.loads(impute_missing_values.invoke({"path": str(path)}))

    assert result["imputed_columns"]["species"]["strategy"] == "drop_row"
    df_after = pd.read_csv(path)
    assert len(df_after) == 2


def test_impute_no_missing_is_noop(tmp_path, monkeypatch):
    from mlops_agents.tools.data_tools import impute_missing_values

    df = pd.DataFrame({"val": [1.0, 2.0, 3.0]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    monkeypatch.setattr(_dt, "settings", _make_settings())
    result = json.loads(impute_missing_values.invoke({"path": str(path)}))

    assert result["imputed_columns"] == {}


def test_impute_returns_output_path(tmp_path, monkeypatch):
    from mlops_agents.tools.data_tools import impute_missing_values

    df = pd.DataFrame({"val": [1.0, None]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    monkeypatch.setattr(_dt, "settings", _make_settings())
    result = json.loads(impute_missing_values.invoke({"path": str(path)}))

    assert result["output_path"] == str(path)


def test_impute_file_not_found(tmp_path, monkeypatch):
    from mlops_agents.tools.data_tools import impute_missing_values

    monkeypatch.setattr(_dt, "settings", _make_settings())
    result = json.loads(impute_missing_values.invoke({"path": "/nonexistent/file.csv"}))

    assert "error" in result
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/test_tools/test_data_tools.py -k "impute" -v
```

Expected: `ImportError` or `AttributeError` — `impute_missing_values` does not exist yet.

- [ ] **Step 4: Implement `impute_missing_values` in `data_tools.py`**

Append to `data_tools.py` (after the last existing `@tool`):

```python
@tool
def impute_missing_values(path: str) -> str:
    """Impute missing values in a canonical CSV using strategies from settings.

    Numeric columns (float64, int64): uses settings.imputation_strategy_numeric
    Categorical columns (object): uses settings.imputation_strategy_categorical

    Writes the result back to the same path (in-place).

    Args:
        path: Path to the canonical CSV file to impute.

    Returns:
        JSON with {output_path, imputed_columns} where each imputed column
        maps to {strategy, fill_value, rows_affected}.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        return json.dumps({"error": f"File not found: {path}"})

    df = pd.read_csv(csv_path)
    imputed: dict[str, dict] = {}

    numeric_strategy = settings.imputation_strategy_numeric
    categorical_strategy = settings.imputation_strategy_categorical

    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        if null_count == 0:
            continue

        if df[col].dtype in ("float64", "int64"):
            if numeric_strategy == "mean":
                fill_value = float(df[col].mean())
            elif numeric_strategy == "median":
                fill_value = float(df[col].median())
            else:  # "zero"
                fill_value = 0.0
            df[col] = df[col].fillna(fill_value)
            imputed[col] = {"strategy": numeric_strategy, "fill_value": fill_value, "rows_affected": null_count}

        elif df[col].dtype == object:
            if categorical_strategy == "mode":
                fill_value = str(df[col].mode().iloc[0]) if not df[col].mode().empty else "unknown"
                df[col] = df[col].fillna(fill_value)
                imputed[col] = {"strategy": "mode", "fill_value": fill_value, "rows_affected": null_count}
            elif categorical_strategy == "unknown":
                df[col] = df[col].fillna("unknown")
                imputed[col] = {"strategy": "unknown", "fill_value": "unknown", "rows_affected": null_count}
            else:  # "drop_row"
                df = df.dropna(subset=[col])
                imputed[col] = {"strategy": "drop_row", "fill_value": None, "rows_affected": null_count}

    df.to_csv(csv_path, index=False)
    logger.info(f"Imputed {len(imputed)} column(s) in {csv_path.name}")
    return json.dumps({"output_path": str(csv_path), "imputed_columns": imputed}, default=str)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_tools/test_data_tools.py -k "impute" -v
```

Expected: all impute tests PASS.

- [ ] **Step 6: Run full unit test suite to check for regressions**

```bash
uv run pytest -m "not integration" -q
```

Expected: all existing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add src/mlops_agents/tools/data_tools.py tests/test_tools/test_data_tools.py
git commit -m "feat: add impute_missing_values tool with config-driven strategy"
```

---

### Task 3: Register `impute_missing_values` in the data agent

**Files:**
- Modify: `src/mlops_agents/agents/data_agent.py`

- [ ] **Step 1: Add the import and register the tool**

Replace the current `data_agent.py` with:

```python
"""Data Validation Agent — validates datasets before they enter the pipeline."""

from langchain.agents import create_agent

from mlops_agents.prompts import get_prompt
from mlops_agents.tools.data_tools import (
    apply_column_mapping,
    check_missing_values,
    impute_missing_values,
    load_dataset,
    merge_datasets,
    validate_against_schema,
)
from mlops_agents.tools.evidently_tools import check_data_drift, check_data_quality
from mlops_agents.utils.llm import get_llm


def build_data_agent():
    """Build and return the data validation react agent."""
    return create_agent(
        model=get_llm("data_validator"),
        tools=[
            load_dataset,
            merge_datasets,
            apply_column_mapping,
            validate_against_schema,
            check_missing_values,
            check_data_quality,
            check_data_drift,
            impute_missing_values,
        ],
        name="data_validator",
        system_prompt=get_prompt("data_agent").template,
    )
```

- [ ] **Step 2: Verify import resolves**

```bash
uv run python -c "from mlops_agents.agents.data_agent import build_data_agent; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/mlops_agents/agents/data_agent.py
git commit -m "feat: register impute_missing_values in data_validator agent"
```

---

### Task 4: Update the data agent prompt

**Files:**
- Modify: `src/mlops_agents/prompts/data_agent.yaml`

- [ ] **Step 1: Replace the prompt file content**

Replace the entire content of `src/mlops_agents/prompts/data_agent.yaml`:

```yaml
_type: "prompt"
input_variables: []
template: |
  You are a Data Validation Specialist responsible for gating the MLOps pipeline.

  **AUTONOMY RULE: Never ask the user questions. Never ask for confirmation before
  calling a tool or modifying a file. Complete the full task and stop. The pipeline
  is fully automated — there is no one to answer your questions.**

  Your context message contains:
  - "Raw files": a JSON list of raw CSV file paths provided by the user
  - "Schema path": the full path to the target schema JSON file
  - "Target schema": the full schema JSON defining canonical columns, types, constraints, and mapping hints

  Your job is to merge the raw files into the canonical dataset, fix nullable violations
  automatically, validate all constraints, and report the result clearly.

  TOOLS:
  - load_dataset: Load and summarise a single CSV — use this on each raw file first.
  - merge_datasets: Join multiple CSVs on a common key column.
  - apply_column_mapping: Rename/drop columns to match the canonical schema.
  - validate_against_schema: Check all schema constraints (nullability, min/max, allowed values, required columns).
  - impute_missing_values: Fill missing values using the configured imputation strategy. Call this on the canonical file when validate_against_schema reports nullable violations.
  - check_missing_values: Compute missing value statistics per column.
  - check_data_quality: Run an Evidently AI quality report.
  - check_data_drift: Compare two CSVs for statistical drift (optional, only if a reference dataset is available).

  PROCESS:
  1. Call load_dataset on EACH raw file to inspect its column names and types.
  2. Read the schema. Use the "mapping_hint" and "is_key": true fields to decide:
     - Which file contains which canonical columns.
     - Which column in each file corresponds to the join key.
  3. Call merge_datasets with your join specification. If any file lacks a matching key column, stop and report the error.
  4. Call apply_column_mapping on the merged file:
     - Build a mapping {"raw_or_merged_col": "canonical_col"} for every column you can match.
     - Use "data/processed/<schema_name>.csv" as the output path.
  5. Call validate_against_schema on the canonical output file, passing the schema path from your context.
  5b. If validate_against_schema reports nullable violations, call impute_missing_values on the canonical
      file, then call validate_against_schema again once. Do not repeat this more than once. If validation
      still fails after imputation, report the remaining violations and stop.
  6. Optionally call check_data_quality for an Evidently summary.
  7. Report clearly:
     - PASSED or FAILED
     - Which files were merged on which key columns
     - Which raw columns were mapped to which canonical names
     - Any constraint violations with detail
     - If imputation was applied, which columns were imputed and with what value

  Be specific. The supervisor uses your output to decide whether to proceed to training.
  If validation fails after auto-fix, clearly explain what the data engineer needs to fix
  in the source files — do not suggest they reply to you.
```

- [ ] **Step 2: Verify prompt loads without error**

```bash
uv run python -c "from mlops_agents.prompts import get_prompt; p = get_prompt('data_agent'); print(p.template[:80])"
```

Expected: prints first 80 chars of the prompt (should start with "You are a Data Validation Specialist").

- [ ] **Step 3: Commit**

```bash
git add src/mlops_agents/prompts/data_agent.yaml
git commit -m "feat: add autonomy rule and imputation step to data_agent prompt"
```

---

### Task 5: Update `data_validator_node` — HITL only fires on pass

**Files:**
- Modify: `src/mlops_agents/graphs/mlops_graph.py:52-139`

- [ ] **Step 1: Replace the `data_validator_node` function**

Replace lines 52–139 (`def data_validator_node` through its closing `return`) with:

```python
def data_validator_node(state: AgentState) -> Command[Literal["supervisor"]]:
    import pandas as pd
    from pathlib import Path as _Path
    from mlops_agents.config.settings import settings

    schema_file = _Path("data/schemas") / f"{settings.dataset_schema}.json"
    schema_json = schema_file.read_text() if schema_file.exists() else "{}"
    schema_path = str(schema_file.resolve())

    dataset_paths = state.get("dataset_paths", [])
    context_message = HumanMessage(
        content=(
            f"Raw files: {json.dumps(dataset_paths)}\n"
            f"Schema path: {schema_path}\n"
            f"Target schema:\n{schema_json}"
        )
    )

    agent = get_agent("data_validator")
    result = agent.invoke({"messages": list(state["messages"]) + [context_message]})
    final_message = result["messages"][-1].content

    quality_report: dict = _extract_tool_json(result["messages"], "check_data_quality")
    mapping_result: dict = _extract_tool_json(result["messages"], "apply_column_mapping")
    validation_result: dict = _extract_tool_json(result["messages"], "validate_against_schema")
    imputation_result: dict = _extract_tool_json(result["messages"], "impute_missing_values")

    processed_path = mapping_result.get("output_path", "")
    validation_passed = bool(validation_result.get("passed", False))

    base_update = {
        "messages": [HumanMessage(content=final_message, name="data_validator")],
        "validation_report": quality_report,
        "validation_passed": validation_passed,
        "dataset_path": processed_path,
    }

    if not validation_passed:
        # Validation failed after agent's auto-fix attempt — abort without HITL.
        # The supervisor will see error_message set and select FINISH.
        error_msg = f"Data validation failed after auto-fix attempt: {final_message}"
        logger.warning("[data_validator] validation failed — aborting without HITL")
        return Command(
            update={**base_update, "error_message": error_msg},
            goto="supervisor",
        )

    # Validation passed — build preview and surface HITL for human sign-off.
    # Use df.to_json + json.loads so NaN/Inf become null — plain to_dict() leaves
    # Python float('nan') which serialises to the bare token NaN, invalid JSON.
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

    # Human rejected a validated+imputed dataset. Abort — retrying cannot help
    # because tools and strategy are deterministic.
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
            "messages": [HumanMessage(content=rejection_text, name="data_validator")],
            "validation_passed": False,
            "error_message": rejection_text,
        },
        goto="supervisor",
    )
```

- [ ] **Step 2: Run existing node tests to check they still pass**

```bash
uv run pytest tests/test_graphs/test_node_state_extraction.py -v
```

Expected: some tests may need updating (Task 7 handles that) — at minimum the tests that mock `interrupt` with `validation_passed=False` paths will now not reach `interrupt()`.

- [ ] **Step 3: Run full unit suite**

```bash
uv run pytest -m "not integration" -q
```

- [ ] **Step 4: Commit**

```bash
git add src/mlops_agents/graphs/mlops_graph.py
git commit -m "feat: HITL only fires when validation passes, abort with error_message on failure"
```

---

### Task 6: Strengthen supervisor prompt rule 5

**Files:**
- Modify: `src/mlops_agents/prompts/supervisor.yaml`

- [ ] **Step 1: Replace rule 5**

In `supervisor.yaml`, replace:

```yaml
  5. If any stage reports an error or failure, select FINISH and report the failure clearly.
```

with:

```yaml
  5. If error_message is set in state, always select FINISH — do not retry any agent.
     If validation_passed=False after data_validator has already run, select FINISH —
     imputation is handled automatically inside the agent, not by retrying the node.
```

- [ ] **Step 2: Verify prompt loads**

```bash
uv run python -c "from mlops_agents.prompts import get_prompt; print(get_prompt('supervisor').template)"
```

Expected: prints the full supervisor prompt with the updated rule 5.

- [ ] **Step 3: Commit**

```bash
git add src/mlops_agents/prompts/supervisor.yaml
git commit -m "feat: supervisor rule 5 — always FINISH on error_message, no data_validator retry"
```

---

### Task 7: Update node tests for new HITL behaviour

**Files:**
- Modify: `tests/test_graphs/test_node_state_extraction.py`

- [ ] **Step 1: Update the test for `validation_passed=False` path**

The test `test_data_validator_node_passed_false_when_no_tool_output` previously patched `interrupt` because the node always called it. After Task 5, `interrupt` is NOT called when `validation_passed=False`. Update the test to:
1. Remove the `interrupt` patch (no longer needed)
2. Assert `error_message` is set in the command update

Replace the existing test:

```python
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
    assert "error_message" in command.update
    assert len(command.update["error_message"]) > 0
    assert command.goto == "supervisor"
```

- [ ] **Step 2: Add test for imputation_result extraction in HITL payload**

Append:

```python
def test_data_validator_node_includes_imputation_in_hitl_payload():
    from mlops_agents.graphs.mlops_graph import data_validator_node

    imputation_json = json.dumps({
        "output_path": "./data/processed/iris.csv",
        "imputed_columns": {
            "sepal_width": {"strategy": "mean", "fill_value": 3.5, "rows_affected": 1}
        },
    })
    validation_json = json.dumps({"passed": True})
    mock_result = {
        "messages": [
            ToolMessage(content=validation_json, tool_call_id="1", name="validate_against_schema"),
            ToolMessage(content=imputation_json, tool_call_id="2", name="impute_missing_values"),
            AIMessage(content="Validation passed after imputation."),
        ]
    }

    captured_payload: dict = {}

    def fake_interrupt(payload: dict) -> dict:
        captured_payload.update(payload)
        return {"approved": True, "comment": ""}

    with patch("mlops_agents.graphs.mlops_graph.get_agent") as mock_get_agent, \
         patch("mlops_agents.graphs.mlops_graph.interrupt", side_effect=fake_interrupt):
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = mock_result
        mock_get_agent.return_value = mock_agent

        command = data_validator_node(_make_state())

    assert "imputation_applied" in captured_payload
    assert "sepal_width" in captured_payload["imputation_applied"]["imputed_columns"]
    assert command.update["validation_passed"] is True
    assert command.goto == "supervisor"


def test_data_validator_node_no_hitl_when_validation_fails():
    from mlops_agents.graphs.mlops_graph import data_validator_node

    validation_json = json.dumps({"passed": False, "violations": [{"column": "target", "rule": "allowed_values", "detail": "Unexpected values: ['bad']"}]})
    mock_result = {
        "messages": [
            ToolMessage(content=validation_json, tool_call_id="1", name="validate_against_schema"),
            AIMessage(content="Validation failed. Target column has invalid values."),
        ]
    }

    interrupt_called = []

    def fail_if_called(payload: dict) -> dict:
        interrupt_called.append(payload)
        return {}

    with patch("mlops_agents.graphs.mlops_graph.get_agent") as mock_get_agent, \
         patch("mlops_agents.graphs.mlops_graph.interrupt", side_effect=fail_if_called):
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = mock_result
        mock_get_agent.return_value = mock_agent

        command = data_validator_node(_make_state())

    assert len(interrupt_called) == 0
    assert command.update["validation_passed"] is False
    assert "error_message" in command.update
    assert command.goto == "supervisor"
```

- [ ] **Step 3: Run all node tests**

```bash
uv run pytest tests/test_graphs/test_node_state_extraction.py -v
```

Expected: all tests PASS.

- [ ] **Step 4: Run full unit suite**

```bash
uv run pytest -m "not integration" -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_graphs/test_node_state_extraction.py
git commit -m "test: update node tests for HITL-only-on-pass and imputation_applied payload"
```

---

## Self-Review

**Spec coverage:**
- ✅ Config fields `imputation_strategy_numeric` / `imputation_strategy_categorical` — Task 1
- ✅ `impute_missing_values` tool reads strategy from settings, writes in-place, returns report — Task 2
- ✅ Tool registered in agent — Task 3
- ✅ Autonomy rule + imputation step in prompt — Task 4
- ✅ `interrupt()` inside `if validation_passed` — Task 5
- ✅ `error_message` set on unfixable failure — Task 5
- ✅ `imputation_applied` in interrupt payload — Task 5
- ✅ Human rejection sets `error_message` + `validation_passed=False` — Task 5
- ✅ Supervisor rule 5 strengthened — Task 6
- ✅ Node tests updated — Task 7

**Type consistency:**
- `impute_missing_values.invoke({"path": str(path)})` — matches `@tool` arg name `path: str` ✅
- `_extract_tool_json(result["messages"], "impute_missing_values")` returns `dict` — matches `imputation_result: dict` ✅
- `imputation_result` passed as `"imputation_applied"` in interrupt payload ✅

**No placeholders:** all steps contain concrete code. ✅
