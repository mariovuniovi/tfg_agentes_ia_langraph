# Schema Upload UI & Pydantic Contract Validation

**Date:** 2026-05-04
**Status:** Approved
**Branch:** to be created from `claude/develop`

## Problem

The pipeline currently hard-codes the schema via `settings.dataset_schema = "iris_classification"`, which points to a file on disk. There is no UI for schema selection or upload. The Run button works whether or not a schema is meaningfully configured, and there is no user-facing validation of schema content before the pipeline starts.

Users need to upload their own schema JSON from local files — analogous to uploading raw CSV data — and receive immediate feedback if the schema is structurally invalid before the pipeline is ever invoked.

## Design Principles

- **No hardcoded schema, no fallback** — if no schema has been uploaded and validated, the Run button is disabled. There is no default.
- **Fail fast in the UI** — Pydantic validates the uploaded JSON immediately on upload, before any pipeline run. The user sees a clear error message without waiting for the pipeline to start.
- **Two validation layers** — Pydantic in the dashboard (UX feedback), `_validate_schema_contract` in `data_validator_node` (pipeline safety net). Both must pass.
- **Schema travels through state** — the uploaded JSON string is passed as `AgentState.schema_json`, keeping `AgentState` as the single source of truth. No disk writes, no settings mutation.
- **Minimal blast radius** — only 5 files change. Graph topology, agent logic, and tool code are untouched.

## Data Flow

```
User uploads schema.json (st.file_uploader)
  → Dashboard parses JSON (json.loads)
  → SchemaContract (Pydantic) validates immediately
      ✓ valid  → st.success("Schema valid — problem type: classification")
                 schema_json stored in st.session_state["schema_json"]
                 Run button enabled
      ✗ invalid → st.error("Invalid schema: <reason>")
                  schema_json removed from session_state
                  Run button remains disabled
  → User clicks Run
  → build_initial_state(dataset_paths, schema_json) called
  → schema_json in initial AgentState
  → data_validator_node reads state["schema_json"] (no disk read)
  → _validate_schema_contract() runs as second check
  → LLM data_validator agent proceeds normally
```

## Pydantic Model (`src/mlops_agents/state/schemas.py`)

Two new models added alongside the existing `RouterOutput`, `ValidationResult`, etc.:

```python
class ColumnSchema(BaseModel):
    model_config = ConfigDict(extra="allow")  # nullable, description, mapping_hint, etc. ignored by contract
    name: str
    dtype: str

class SchemaContract(BaseModel):
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

`ColumnSchema` requires only `name` and `dtype`. All other column-level fields (`nullable`, `description`, `mapping_hint`, `min`, `max`, `allowed_values`, `is_key`) are allowed by Pydantic via `model_config = ConfigDict(extra="allow")` but not validated — they are consumed by the data_validator agent, not the contract check.

`SchemaContract` similarly allows extra top-level fields (`name`, `description`) without error.

## AgentState Addition (`src/mlops_agents/state/agent_state.py`)

One new field, added after `task_metadata`:

```python
# Raw schema JSON — uploaded by user and passed through state
# data_validator_node reads this instead of loading from disk
schema_json: str
```

Initialized to `""` in all `initial_state` dicts and test fixtures.

## `data_validator_node` Update (`src/mlops_agents/graphs/mlops_graph.py`)

Replace the disk-read block with a state-read:

```python
# Before:
schema_file = _Path("data/schemas") / f"{settings.dataset_schema}.json"
schema_json = schema_file.read_text() if schema_file.exists() else "{}"
schema_path = str(schema_file.resolve())
schema_data = json.loads(schema_json) if schema_json != "{}" else {}

# After:
schema_json = state.get("schema_json") or "{}"
schema_path = "(uploaded via UI)" if schema_json != "{}" else "(none)"
schema_data = json.loads(schema_json) if schema_json != "{}" else {}
```

The `from pathlib import Path as _Path` and `from mlops_agents.config.settings import settings` local imports inside the node are removed since they are no longer needed.

`_validate_schema_contract(schema_data)` runs unchanged immediately after, providing the pipeline-level safety net.

## `build_initial_state` Update (`dashboard/pipeline_helpers.py`)

```python
def build_initial_state(dataset_paths: list[str], schema_json: str = "") -> dict:
    return {
        ...
        "schema_json": schema_json,
        "dataset_summary": {},
        "problem_type": "",
        "task_metadata": {},
    }
```

## Dashboard UI (`dashboard/pages/01_pipeline.py`)

Schema uploader placed above the dataset multiselect in the configuration section:

```python
st.subheader("Schema")
uploaded_schema = st.file_uploader(
    "Upload schema JSON",
    type=["json"],
    help="JSON file declaring problem_type, target_column, and column definitions.",
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
        # Show only the first error message — Pydantic error lists are verbose
        first_error = exc.errors()[0]["msg"]
        st.error(f"Schema contract violation: {first_error}")
        st.session_state.pop("schema_json", None)

schema_json = st.session_state.get("schema_json", "")
```

Run button gated on schema presence:

```python
run_disabled = not dataset_paths or not schema_json
st.button("Run Pipeline", disabled=run_disabled, ...)
```

When Run is clicked:

```python
initial_state = build_initial_state(dataset_paths, schema_json=schema_json)
```

Imports added to `01_pipeline.py`: `from pydantic import ValidationError` and `from mlops_agents.state.schemas import SchemaContract`.

## `main()` Update (`src/mlops_agents/graphs/mlops_graph.py`)

The `initial_state` dict in `main()` gains `"schema_json": ""`.

## Files Changed

| File | Change |
|---|---|
| `src/mlops_agents/state/schemas.py` | Add `ColumnSchema` and `SchemaContract` Pydantic models |
| `src/mlops_agents/state/agent_state.py` | Add `schema_json: str` |
| `src/mlops_agents/graphs/mlops_graph.py` | `data_validator_node` reads `state["schema_json"]`; remove local `Path`/`settings` imports in node; `main()` initial_state gets `schema_json: ""` |
| `dashboard/pipeline_helpers.py` | `build_initial_state()` gains `schema_json` parameter; dict includes `schema_json`, `dataset_summary`, `problem_type`, `task_metadata` |
| `dashboard/pages/01_pipeline.py` | Add schema uploader widget; gate Run button on schema presence; pass `schema_json` to `build_initial_state()` |

## Testing

- `tests/test_graphs/test_node_state_extraction.py` — update `_make_state()` to include `"schema_json": "{}"` for existing tests; add tests for `data_validator_node` reading from state (patch `state["schema_json"]` instead of `pathlib.Path.read_text`)
- `tests/test_agents/test_supervisor.py` — update `make_state()` to include `"schema_json": ""`
- `tests/test_state/test_schemas.py` (new) — unit tests for `SchemaContract`: valid classification, valid regression, valid forecasting, missing `problem_type`, invalid `target_column`, missing forecasting fields, non-positive `forecast_horizon`, extra fields allowed

## What This Does NOT Change

- Graph topology — no new nodes, no routing changes
- Agent tools — `check_data_quality`, `validate_against_schema`, etc. unchanged
- `_validate_schema_contract` — unchanged, still runs inside the node
- Supervisor prompt — unchanged
- Monitoring, chat, experiments, logs pages — unchanged
