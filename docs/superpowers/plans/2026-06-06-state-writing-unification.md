# State-Writing Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every domain/gate node write to the shared `AgentState` through one uniform, typed pattern — `return Command(update=<Node>StateUpdate(...).to_update(...), goto="workflow_controller")` — delete the dead `state/schemas.py` result classes and the dead `next` field, and enforce that the contracts stay in sync with the TypedDict via one binding test.

**Architecture:** A new `contracts/outputs.py` holds a `StateUpdate` base (`extra="forbid"`, a `to_update()` serializer) plus one `<Node>StateUpdate` contract per node, every field defaulted, field names matching `AgentState` keys exactly. **Domain modules stay state-agnostic** (Philosophy 2): `evaluate_promotion` / `run_report_writer` / `run_deployer` keep returning plain dicts and `run_training_plan` keeps returning `TrainingResult`; the *graph node* constructs the contract (`EvaluationStateUpdate(**evaluate_promotion(state)).to_update()`, etc.). `SchemaContract` moves to `contracts/schema.py` so all Pydantic contracts live under `contracts/` and `state/` holds only `AgentState`. The pure-router `workflow_controller` is the one documented exception — it writes routing-control updates inline.

**Tech Stack:** Python 3.12, Pydantic v2, LangGraph (`Command`, `interrupt`), pytest, UV.

---

## Resolved design decisions (from design review)

1. **Goal:** consistency + structural type-safety + single-source-of-truth + thesis framing.
2. **Contracts ↔ TypedDict:** the TypedDict must stay (LangGraph builds channels + the `messages` reducer from it). Bind the two with a consistency test (Task 11) — the *enforceable* form of single-source-of-truth.
3. **Strictness:** structural-strict only — fields are required/optional with names enforced, value types stay loose (`dict`, not `dict[str,float]`; `str`, not `Literal[...]`). `extra="forbid"` on the base catches typo'd/stray keys. **No** runtime error-boundary is added (out of scope; the deterministic path stays as-is).
4. **data_validator:** one contract covers all three return paths (two early-error + main); every field defaulted.
5. **Construction site (Philosophy 2):** domain helpers stay state-agnostic; the node builds the contract. No domain module imports `contracts/outputs`.
6. **workflow_controller:** exempt — it is a router. Its only exclusive key is `agent_attempt_counts`.
7. **`next`:** deleted (dead legacy field, zero references).
8. **Location:** single `contracts/outputs.py`. **Naming:** base `StateUpdate`; per-node `<Node>StateUpdate`.

---

## Current state (what we're fixing)

Five inconsistent ways nodes write `AgentState` today:

| Node | Current pattern | File |
|---|---|---|
| `data_validator` | `_extract_tool_json()` scrapes agent messages → loose dicts, across **3** return paths | `graphs/mlops_graph.py:203-344` |
| `planner` | builds Pydantic outputs, `.model_dump()` per field into an inline dict | `planning/node.py:118-131` |
| `executor` | gets `TrainingResult`, manually copies `.attr` → state keys | `graphs/mlops_graph.py:374-387` |
| `evaluation` / `report_writer` / `deployer` | helper returns a plain `dict`, spread into `Command` | `evaluation/promotion.py`, `evaluation/report_writer.py`, `deployment/deployer.py` |
| approvals | inline dict literals | `graphs/approval_nodes.py` |
| `workflow_controller` | inline dict literals (**stays — exempt**) | `graphs/workflow_controller.py` |

Dead code removed: `ValidationResult`, `TrainingResult`, `EvaluationResult` in `state/schemas.py` (only referenced by `state/__init__.py`'s re-export + a never-asserted import); the `next` field in `AgentState`. `SchemaContract` + `ColumnSchema` are alive and move to `contracts/schema.py`.

---

## File Structure

**Created:**
- `src/mlops_agents/contracts/schema.py` — `SchemaContract`, `ColumnSchema` (moved verbatim).
- `src/mlops_agents/contracts/outputs.py` — `StateUpdate` base + 9 node-update contracts.
- `tests/test_contracts/__init__.py` — package marker (if missing).
- `tests/test_contracts/test_schema.py` — moved from `tests/test_state/test_schemas.py`.
- `tests/test_contracts/test_outputs.py` — contract unit tests.
- `tests/test_contracts/test_state_binding.py` — the contracts↔TypedDict consistency test.

**Modified (graph layer only — domain modules untouched):**
- `src/mlops_agents/state/__init__.py` — export only `AgentState`.
- `src/mlops_agents/state/agent_state.py` — delete the `next` field.
- `api/routers/uploads.py` — import `SchemaContract` from the new location.
- `src/mlops_agents/graphs/mlops_graph.py` — data_validator / executor / evaluation / report_writer / deployer nodes build contracts; refresh stale module docstring.
- `src/mlops_agents/planning/node.py` — planner builds `PlannerStateUpdate`.
- `src/mlops_agents/graphs/approval_nodes.py` — gates build approval contracts.
- `src/mlops_agents/contracts/__init__.py` — refresh stale docstring.

**Deleted:**
- `src/mlops_agents/state/schemas.py`.
- `tests/test_state/test_schemas.py` (moved).

**Explicitly NOT changed (Philosophy 2):** `evaluation/promotion.py`, `evaluation/report_writer.py`, `deployment/deployer.py`, `training/executor.py` and all their tests keep returning/asserting dicts/`TrainingResult`.

---

### Task 1: Move SchemaContract, delete dead `*Result` classes + dead `next` field

**Files:**
- Create: `src/mlops_agents/contracts/schema.py`
- Modify: `src/mlops_agents/state/__init__.py`, `src/mlops_agents/state/agent_state.py`, `api/routers/uploads.py:10`
- Delete: `src/mlops_agents/state/schemas.py`
- Move: `tests/test_state/test_schemas.py` → `tests/test_contracts/test_schema.py`

- [ ] **Step 1: Create `contracts/schema.py`** with the live classes moved verbatim:

```python
"""Canonical target-schema contract, validated on dataset upload.

Threaded through the pipeline as ``AgentState.schema_json`` (serialised form).
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


class ColumnSchema(BaseModel):
    """A single canonical column. ``extra="allow"`` keeps optional metadata
    (nullable, unique, mapping hints) supplied in the uploaded schema."""

    model_config = ConfigDict(extra="allow")

    name: str
    dtype: str


class SchemaContract(BaseModel):
    """Full target schema: problem type, target, columns, and forecasting keys."""

    model_config = ConfigDict(extra="allow")

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
            if self.forecast_horizon is None or self.forecast_horizon <= 0:
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

- [ ] **Step 2: Replace `state/__init__.py`** with only the live export:

```python
from mlops_agents.state.agent_state import AgentState

__all__ = ["AgentState"]
```

- [ ] **Step 3: Delete the dead `next` field** from `src/mlops_agents/state/agent_state.py`. Remove this line from the `# === Framework / routing ===` block:

```python
    next: str  # UNUSED legacy routing key from the old supervisor pattern (routing now via Command(goto=...))
```
Leave the `messages` line and the section header.

- [ ] **Step 4: Update the one production import** in `api/routers/uploads.py:10`:

Change `from mlops_agents.state.schemas import SchemaContract` to:
```python
from mlops_agents.contracts.schema import SchemaContract
```

- [ ] **Step 5: Delete `src/mlops_agents/state/schemas.py`.**

```bash
git rm src/mlops_agents/state/schemas.py
```

- [ ] **Step 6: Move the test file and fix its import.**

```bash
mkdir -p tests/test_contracts
git mv tests/test_state/test_schemas.py tests/test_contracts/test_schema.py
test -f tests/test_contracts/__init__.py || touch tests/test_contracts/__init__.py
```
In `tests/test_contracts/test_schema.py` change line 6 to:
```python
from mlops_agents.contracts.schema import SchemaContract
```
(Leave `test_agent_state_has_refactor_fields` unchanged — it imports `AgentState` directly and never references `next`.)

- [ ] **Step 7: Verify no remaining references to the deleted symbols.**

Run: `grep -rn "state.schemas\|ValidationResult\|EvaluationResult\|\"next\"\|state\[.next.\]\|\.get(.next." src/ api/ tests/ --include=*.py`
Expected: no matches.

- [ ] **Step 8: Run the moved test + the upload import path.**

Run: `uv run pytest tests/test_contracts/test_schema.py -q`
Expected: PASS (17 tests).

Run: `uv run python -c "import api.routers.uploads; from mlops_agents.contracts.schema import SchemaContract; print('OK')"`
Expected: prints `OK`.

- [ ] **Step 9: Commit.**

```bash
git add -A
git commit -m "refactor(state): move SchemaContract to contracts/, drop dead schemas and next field"
```

---

### Task 2: `StateUpdate` base

**Files:**
- Create: `src/mlops_agents/contracts/outputs.py`
- Test: `tests/test_contracts/test_outputs.py`

- [ ] **Step 1: Write the failing test** in `tests/test_contracts/test_outputs.py`:

```python
"""Unit tests for node→state update contracts."""

import pytest
from pydantic import ValidationError

from mlops_agents.contracts.outputs import StateUpdate


class _Sample(StateUpdate):
    foo: str = "x"


def test_to_update_returns_plain_dict():
    assert _Sample(foo="hello").to_update() == {"foo": "hello"}


def test_to_update_merges_messages_when_provided():
    assert _Sample(foo="hello").to_update(messages=["m1"]) == {"foo": "hello", "messages": ["m1"]}


def test_to_update_omits_messages_key_when_none():
    assert "messages" not in _Sample(foo="hello").to_update()


def test_extra_keys_are_forbidden():
    with pytest.raises(ValidationError):
        _Sample(foo="hello", bogus=1)
```

- [ ] **Step 2: Run it to confirm it fails.**

Run: `uv run pytest tests/test_contracts/test_outputs.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mlops_agents.contracts.outputs'`.

- [ ] **Step 3: Implement the base** in `src/mlops_agents/contracts/outputs.py`:

```python
"""Node→state update contracts.

Each contract's field names match ``AgentState`` keys exactly (the binding test
in tests/test_contracts/test_state_binding.py enforces this). A graph node builds
one contract for the state-slice it owns and writes it with ``.to_update()``:

    return Command(update=EvaluationStateUpdate(**evaluate_promotion(state)).to_update(),
                   goto="workflow_controller")

Design rules:
- ``extra="forbid"``: a stray/typo'd key (e.g. from a helper dict) fails loudly.
- every field is defaulted, so a node can build a partial/failure variant.
- ``to_update()`` uses ``by_alias=True`` so leading-underscore state keys
  (e.g. ``_planner_output_record``) are emitted via ``serialization_alias``.
- nodes that also append chat history pass ``messages=`` (merged via the reducer).
- domain modules (evaluation/, deployment/, training/) MUST NOT import this module;
  contract construction happens in the graph node layer only (Philosophy 2).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class StateUpdate(BaseModel):
    """Base for all node→state update contracts."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    def to_update(self, messages: list[Any] | None = None) -> dict[str, Any]:
        update = self.model_dump(by_alias=True)
        if messages:
            update["messages"] = messages
        return update
```

- [ ] **Step 4: Run the test to confirm it passes.**

Run: `uv run pytest tests/test_contracts/test_outputs.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit.**

```bash
git add src/mlops_agents/contracts/outputs.py tests/test_contracts/test_outputs.py
git commit -m "feat(contracts): add StateUpdate base (extra=forbid) for state-write unification"
```

---

### Task 3: All nine node-update contracts

**Files:**
- Modify: `src/mlops_agents/contracts/outputs.py`
- Test: `tests/test_contracts/test_outputs.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_contracts/test_outputs.py`):

```python
from mlops_agents.contracts.outputs import (
    AuditStateUpdate,
    DataValidationStateUpdate,
    DatasetApprovalStateUpdate,
    DeploymentApprovalStateUpdate,
    DeploymentStateUpdate,
    EvaluationStateUpdate,
    PlannerErrorStateUpdate,
    PlannerStateUpdate,
    TrainingStateUpdate,
)
from mlops_agents.contracts.training import TrainingResult


def test_evaluation_contract_accepts_helper_dict_shape():
    helper_dict = {
        "evaluation_passed": True,
        "candidate_metrics": {"rmse": 1.0},
        "champion_metrics": {"rmse": 2.0},
        "thresholds_applied": {"min_delta": 0.0},
        "evaluation_report": {"candidate_metrics": {"rmse": 1.0}},
    }
    update = EvaluationStateUpdate(**helper_dict).to_update()
    assert update["evaluation_passed"] is True
    assert update["evaluation_report"]["candidate_metrics"] == {"rmse": 1.0}


def test_audit_contract_accepts_helper_dict_shape():
    helper_dict = {"evaluation_report_audit": {"x": 1}, "evaluation_report_audit_status": "ok"}
    assert AuditStateUpdate(**helper_dict).to_update() == helper_dict


def test_deployment_contract_accepts_helper_dict_shape():
    helper_dict = {
        "deployment_status": "deployed",
        "deployment_decision": "deployed",
        "best_model_uri": "models:/m/1",
    }
    assert DeploymentStateUpdate(**helper_dict).to_update() == helper_dict


def test_training_contract_maps_result_fields():
    result = TrainingResult(
        champion_candidate={"model_key": "ridge"},
        champion_model_path="/tmp/model.pkl",
        train_pool_path="/tmp/train.csv",
        test_path="/tmp/test.csv",
        split_metadata_path="/tmp/split.json",
        mlflow_parent_run_id="run123",
        experience_record_path="/tmp/exp.json",
        champion_metrics={"rmse": 1.5},
    )
    update = TrainingStateUpdate.from_training_result(
        result, training_plan={"problem_type": "regression"}
    ).to_update()
    assert update["trained_model_path"] == "/tmp/model.pkl"
    assert update["training_run_id"] == "run123"
    assert update["training_metrics"] == {"rmse": 1.5}
    assert update["training_plan"] == {"problem_type": "regression"}


def test_planner_contract_emits_underscore_alias():
    update = PlannerStateUpdate(
        planner_status="ok",
        training_plan={"problem_type": "regression"},
        planner_output_record={"k": "v"},
    ).to_update()
    assert update["_planner_output_record"] == {"k": "v"}
    assert "planner_output_record" not in update


def test_planner_error_contract_keys():
    assert PlannerErrorStateUpdate(error_message="boom").to_update() == {
        "planner_status": "failed",
        "planner_retry_used": True,
        "error_message": "boom",
    }


def test_data_validation_contract_failure_variant_uses_defaults():
    update = DataValidationStateUpdate(
        validation_passed=False, error_message="bad", schema_json=""
    ).to_update()
    assert update["validation_passed"] is False
    assert update["error_message"] == "bad"
    assert update["data_join_plan"] is None
    assert update["data_join_evaluations"] == []
    assert update["dataset_rejection_comment"] == ""


def test_approval_contracts_keys():
    assert DatasetApprovalStateUpdate(
        dataset_approved=False, dataset_rejection_comment="fix"
    ).to_update() == {"dataset_approved": False, "dataset_rejection_comment": "fix"}
    assert DeploymentApprovalStateUpdate(deployment_approved=True).to_update() == {
        "deployment_approved": True
    }
```

- [ ] **Step 2: Run it to confirm it fails.**

Run: `uv run pytest tests/test_contracts/test_outputs.py -q`
Expected: FAIL with `ImportError: cannot import name 'EvaluationStateUpdate'`.

- [ ] **Step 3: Append all nine contracts** to `src/mlops_agents/contracts/outputs.py`.

First, update the import block at the top of the file (adds `Field`, and `TrainingResult` under a `TYPE_CHECKING` guard so there is no runtime import — it is only a type annotation, and `from __future__ import annotations` is already in effect, so no import cycle can form):

```python
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # annotation-only; avoids any contracts/ import cycle
    from mlops_agents.contracts.training import TrainingResult
```

Then append the nine contract classes:

```python
class DataValidationStateUpdate(StateUpdate):
    """data_validator node — covers all three return paths (two early-error + main).

    All fields default to their reset value, so the early-error paths build the
    same contract with only error_message/schema_json set.
    """

    validation_passed: bool = False
    validation_report: dict = Field(default_factory=dict)
    processed_dataset_path: str = ""
    dataset_summary: dict = Field(default_factory=dict)
    problem_type: str = ""
    task_metadata: dict = Field(default_factory=dict)
    schema_json: str = ""
    data_join_plan: dict | None = None
    data_join_base_nrows: int | None = None
    data_join_evaluations: list[dict] = Field(default_factory=list)
    error_message: str = ""
    dataset_rejection_comment: str = ""


class DatasetApprovalStateUpdate(StateUpdate):
    """dataset_approval HITL gate (gate 1)."""

    dataset_approved: bool | None = None
    dataset_rejection_comment: str = ""


class PlannerStateUpdate(StateUpdate):
    """planner node — success path."""

    planner_analysis: str | None = None
    planner_evidence_used: list[dict] = Field(default_factory=list)
    planner_warnings: list[str] = Field(default_factory=list)
    planner_status: str | None = None
    planner_retry_used: bool | None = None
    training_plan: dict | None = None
    planner_tool_trace: dict = Field(default_factory=dict)
    planner_validation_context: dict = Field(default_factory=dict)
    # State key has a leading underscore — emit via serialization alias.
    planner_output_record: dict | None = Field(
        default=None, serialization_alias="_planner_output_record"
    )


class PlannerErrorStateUpdate(StateUpdate):
    """planner error wrapper — when planning fails after retry."""

    planner_status: str = "failed"
    planner_retry_used: bool = True
    error_message: str = ""


class TrainingStateUpdate(StateUpdate):
    """executor node — maps TrainingResult → state keys."""

    training_plan: dict | None = None
    train_pool_path: str | None = None
    test_path: str | None = None
    split_metadata_path: str | None = None
    trained_model_path: str = ""
    training_run_id: str = ""
    training_metrics: dict = Field(default_factory=dict)
    champion_candidate: dict | None = None
    experience_record_path: str | None = None

    @classmethod
    def from_training_result(
        cls, result: TrainingResult, *, training_plan: dict
    ) -> "TrainingStateUpdate":
        return cls(
            training_plan=training_plan,
            train_pool_path=result.train_pool_path,
            test_path=result.test_path,
            split_metadata_path=result.split_metadata_path,
            trained_model_path=result.champion_model_path,
            training_run_id=result.mlflow_parent_run_id,
            training_metrics=result.champion_metrics,
            champion_candidate=result.champion_candidate,
            experience_record_path=result.experience_record_path,
        )


class EvaluationStateUpdate(StateUpdate):
    """evaluation node — deterministic promotion decision."""

    evaluation_passed: bool | None = None
    candidate_metrics: dict = Field(default_factory=dict)
    champion_metrics: dict = Field(default_factory=dict)
    thresholds_applied: dict = Field(default_factory=dict)
    evaluation_report: dict = Field(default_factory=dict)


class AuditStateUpdate(StateUpdate):
    """report_writer node — LLM audit report."""

    evaluation_report_audit: dict | None = None
    evaluation_report_audit_status: str = ""


class DeploymentApprovalStateUpdate(StateUpdate):
    """deployment_approval HITL gate (gate 2)."""

    deployment_approved: bool | None = None


class DeploymentStateUpdate(StateUpdate):
    """deployer node — MLflow Model Registry promotion."""

    deployment_status: str = ""
    deployment_decision: str = ""
    best_model_uri: str = ""
```

- [ ] **Step 4: Run the contract tests to confirm they pass.**

Run: `uv run pytest tests/test_contracts/test_outputs.py -q`
Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/mlops_agents/contracts/outputs.py tests/test_contracts/test_outputs.py
git commit -m "feat(contracts): add nine per-node StateUpdate contracts"
```

---

### Task 4: Wire the evaluation node (Philosophy 2)

**Files:**
- Modify: `src/mlops_agents/graphs/mlops_graph.py` (imports + `evaluation_node`, lines 409-413)

`evaluation/promotion.py` and its tests are **unchanged** — the helper still returns a dict.

- [ ] **Step 1: Add the contract import** near the other contract imports at the top of `mlops_graph.py`:

```python
from mlops_agents.contracts.outputs import EvaluationStateUpdate
```

- [ ] **Step 2: Replace `evaluation_node`** (lines 409-413):

```python
def evaluation_node(state: AgentState) -> Command[Literal["workflow_controller"]]:
    """Deterministic promotion decision — no LLM."""
    result = evaluate_promotion(state)
    logger.info(f"[evaluation] passed={result['evaluation_passed']}")
    return Command(
        update=EvaluationStateUpdate(**result).to_update(), goto="workflow_controller"
    )
```

- [ ] **Step 3: Run the evaluation + node-extraction suites.**

Run: `uv run pytest tests/test_evaluation/test_promotion.py tests/test_graphs/test_node_state_extraction.py -q`
Expected: PASS (helper dict shape matches the contract exactly; the two patched mocks carry only the five contract fields, so `extra="forbid"` is satisfied).

- [ ] **Step 4: Commit.**

```bash
git add src/mlops_agents/graphs/mlops_graph.py
git commit -m "refactor(evaluation): build EvaluationStateUpdate in the node"
```

---

### Task 5: Wire the report_writer node (Philosophy 2)

**Files:**
- Modify: `src/mlops_agents/graphs/mlops_graph.py` (imports + `report_writer_node`, lines 416-419)

`evaluation/report_writer.py` and its tests are **unchanged**.

- [ ] **Step 1: Add the import:**

```python
from mlops_agents.contracts.outputs import AuditStateUpdate
```

- [ ] **Step 2: Replace `report_writer_node`** (lines 416-419):

```python
def report_writer_node(state: AgentState) -> Command[Literal["workflow_controller"]]:
    """Audit LLM node — produces structured EvaluationReport."""
    result = run_report_writer(state)
    return Command(
        update=AuditStateUpdate(**result).to_update(), goto="workflow_controller"
    )
```

- [ ] **Step 3: Run the report_writer + graph suites.**

Run: `uv run pytest tests/test_evaluation/test_report_writer.py tests/test_graphs/ -q`
Expected: PASS.

- [ ] **Step 4: Commit.**

```bash
git add src/mlops_agents/graphs/mlops_graph.py
git commit -m "refactor(report_writer): build AuditStateUpdate in the node"
```

---

### Task 6: Wire the deployer node (Philosophy 2)

**Files:**
- Modify: `src/mlops_agents/graphs/mlops_graph.py` (imports + `deployer_node`, lines 422-425)

`deployment/deployer.py` and its tests are **unchanged**.

- [ ] **Step 1: Add the import:**

```python
from mlops_agents.contracts.outputs import DeploymentStateUpdate
```

- [ ] **Step 2: Replace `deployer_node`** (lines 422-425):

```python
def deployer_node(state: AgentState) -> Command[Literal["workflow_controller"]]:
    """Deterministic deployment — Gate 2 has already approved upstream."""
    result = run_deployer_module(state)
    return Command(
        update=DeploymentStateUpdate(**result).to_update(), goto="workflow_controller"
    )
```

- [ ] **Step 3: Run the deployment + graph suites.**

Run: `uv run pytest tests/test_deployment/ tests/test_graphs/ -q`
Expected: PASS.

- [ ] **Step 4: Commit.**

```bash
git add src/mlops_agents/graphs/mlops_graph.py
git commit -m "refactor(deployer): build DeploymentStateUpdate in the node"
```

---

### Task 7: Wire the executor node

**Files:**
- Modify: `src/mlops_agents/graphs/mlops_graph.py` (imports + `executor_node`, lines 373-387)

`run_training_plan` (returns `TrainingResult`) and the executor tests are **unchanged**.

- [ ] **Step 1: Add the import:**

```python
from mlops_agents.contracts.outputs import TrainingStateUpdate
```

- [ ] **Step 2: Replace the logging + return** at the end of `executor_node` (lines 373-387):

```python
    logger.info("[executor] completed — routing back to workflow_controller")
    output = TrainingStateUpdate.from_training_result(result, training_plan=plan.model_dump())
    return Command(goto="workflow_controller", update=output.to_update())
```

- [ ] **Step 3: Run the executor + node-extraction suites.**

Run: `uv run pytest tests/test_training/ tests/test_graphs/test_node_state_extraction.py -q`
Expected: PASS.

- [ ] **Step 4: Commit.**

```bash
git add src/mlops_agents/graphs/mlops_graph.py
git commit -m "refactor(executor): map TrainingResult via TrainingStateUpdate contract"
```

---

### Task 8: Wire the planner + planner-error nodes

**Files:**
- Modify: `src/mlops_agents/planning/node.py` (planner_node, lines 118-131)
- Modify: `src/mlops_agents/graphs/mlops_graph.py` (`_planner_node_with_error_handling`, lines 390-406)

- [ ] **Step 1: Update `planner_node`** in `src/mlops_agents/planning/node.py`. Add import near the top:

```python
from mlops_agents.contracts.outputs import PlannerStateUpdate
```
Replace the `return Command(...)` block (lines 118-131):
```python
    output_state = PlannerStateUpdate(
        planner_analysis=output.planning_analysis,
        planner_evidence_used=[e.model_dump() for e in output.evidence_used],
        planner_warnings=output.risks_or_warnings,
        planner_status=planner_status,
        planner_retry_used=retry_used,
        training_plan=output.plan.model_dump(),
        planner_tool_trace=trace.model_dump(),
        planner_validation_context=_audit_subset(validation_ctx),
        planner_output_record=record,
    )
    return Command(goto="workflow_controller", update=output_state.to_update())
```

- [ ] **Step 2: Update `_planner_node_with_error_handling`** in `mlops_graph.py`. Add import:

```python
from mlops_agents.contracts.outputs import PlannerErrorStateUpdate
```
Replace the `except` block's return (lines 396-406):
```python
    except PlannerError as exc:
        logger.error(f"[planner] failed after retry: {exc}")
        output = PlannerErrorStateUpdate(error_message=f"Model planner failed: {exc}")
        return Command(
            goto="workflow_controller",
            update=output.to_update(
                messages=[HumanMessage(content=f"Planner failed: {exc}", name="planner")]
            ),
        )
```

- [ ] **Step 3: Run the planner + graph suites.**

Run: `uv run pytest tests/test_planning/ tests/test_graphs/ -q`
Expected: PASS. The `_planner_output_record` alias is preserved, so any test asserting on `update["_planner_output_record"]` still works.

- [ ] **Step 4: Commit.**

```bash
git add src/mlops_agents/planning/node.py src/mlops_agents/graphs/mlops_graph.py
git commit -m "refactor(planner): build PlannerStateUpdate / PlannerErrorStateUpdate"
```

---

### Task 9: Wire the data_validator node (all three return paths)

**Files:**
- Modify: `src/mlops_agents/graphs/mlops_graph.py` (`data_validator_node`, lines 203-344)

- [ ] **Step 1: Add the import:**

```python
from mlops_agents.contracts.outputs import DataValidationStateUpdate
```

- [ ] **Step 2: Replace the two early-error returns.** First (no schema uploaded, lines 206-219):

```python
        return Command(
            update=DataValidationStateUpdate(
                validation_passed=False, error_message=error_msg, schema_json=""
            ).to_update(messages=[HumanMessage(content=error_msg, name="data_validator")]),
            goto="workflow_controller",
        )
```
Second (schema contract violation, lines 226-239):
```python
        return Command(
            update=DataValidationStateUpdate(
                validation_passed=False, error_message=error_msg, schema_json=schema_json
            ).to_update(messages=[HumanMessage(content=error_msg, name="data_validator")]),
            goto="workflow_controller",
        )
```

- [ ] **Step 3: Replace the main path.** From line 257 (`final_message = ...`) through the end of the function (both `return Command(...)` branches, ending line 344), with the extraction kept and a single contract build + return:

```python
    final_message = result["messages"][-1].content

    quality_report: dict = _extract_tool_json(result["messages"], "check_data_quality")
    mapping_result: dict = _extract_tool_json(result["messages"], "apply_column_mapping")
    validation_result: dict = _extract_tool_json(result["messages"], "validate_against_schema")
    imputation_result: dict = _extract_tool_json(result["messages"], "impute_missing_values")
    join_exec_result: dict = _extract_tool_json(result["messages"], "execute_join_plan")
    eval_result: dict = _extract_tool_json(result["messages"], "evaluate_join_candidates")

    data_join_plan = join_exec_result.get("join_plan")
    data_join_evaluations = eval_result.get("evaluations", [])

    data_join_base_nrows: int | None = None
    if data_join_plan:
        base_name = data_join_plan.get("base_dataset", {}).get("dataset_name")
        if base_name:
            raw_paths = {Path(p).stem: p for p in (state.get("dataset_paths") or [])}
            base_path = raw_paths.get(base_name)
            if base_path and Path(base_path).exists():
                try:
                    data_join_base_nrows = len(pd.read_csv(base_path))
                except Exception:
                    pass

    processed_path = (
        imputation_result.get("output_path", "")
        or mapping_result.get("output_path", "")
        or validation_result.get("output_path", "")
    )
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

    problem_type: str = schema_data.get("problem_type", "")
    task_metadata: dict[str, Any] = {"target_column": schema_data.get("target_column", "")}
    if problem_type == "forecasting":
        task_metadata.update({
            "datetime_column": schema_data.get("datetime_column", ""),
            "series_id_columns": schema_data.get("series_id_columns", []),
            "forecast_horizon": schema_data.get("forecast_horizon"),
            "frequency": schema_data.get("frequency", ""),
            "exogenous_columns": schema_data.get("exogenous_columns"),
        })

    error_message = (
        "" if validation_passed
        else f"Data validation failed after auto-fix attempt: {final_message}"
    )
    if not validation_passed:
        logger.warning("[data_validator] validation failed — aborting without HITL")

    output = DataValidationStateUpdate(
        validation_report=quality_report,
        validation_passed=validation_passed,
        processed_dataset_path=processed_path,
        dataset_summary=dataset_summary,
        problem_type=problem_type,
        task_metadata=task_metadata,
        schema_json=schema_json,
        data_join_plan=data_join_plan,
        data_join_base_nrows=data_join_base_nrows,
        data_join_evaluations=data_join_evaluations,
        error_message=error_message,
    )
    return Command(
        goto="workflow_controller",
        update=output.to_update(
            messages=[HumanMessage(content=final_message, name="data_validator")]
        ),
    )
```

(Behavior note: the success path previously did not write `dataset_rejection_comment`; the contract now writes its default `""`, which equals the reset value the controller assigns on a rejection retry — so the resume/retry flow is unchanged.)

- [ ] **Step 4: Run the graph suites.**

Run: `uv run pytest tests/test_graphs/ -q`
Expected: PASS. If a test asserted the failure path omitted `dataset_rejection_comment`, update it to accept `""`.

- [ ] **Step 5: Commit.**

```bash
git add src/mlops_agents/graphs/mlops_graph.py
git commit -m "refactor(data_validator): single DataValidationStateUpdate across all 3 paths"
```

---

### Task 10: Wire the approval gates

**Files:**
- Modify: `src/mlops_agents/graphs/approval_nodes.py` (imports + both returns, lines 71-77 and 104-107)

- [ ] **Step 1: Add the import** at the top of `approval_nodes.py`:

```python
from mlops_agents.contracts.outputs import DatasetApprovalStateUpdate, DeploymentApprovalStateUpdate
```

- [ ] **Step 2: Replace the `dataset_approval_node` return** (lines 71-77):

```python
    return Command(
        goto="workflow_controller",
        update=DatasetApprovalStateUpdate(
            dataset_approved=approved,
            dataset_rejection_comment="" if approved else comment,
        ).to_update(),
    )
```

- [ ] **Step 3: Replace the `deployment_approval_node` return** (lines 104-107):

```python
    return Command(
        goto="workflow_controller",
        update=DeploymentApprovalStateUpdate(deployment_approved=approved).to_update(),
    )
```

- [ ] **Step 4: Run the graph suites.**

Run: `uv run pytest tests/test_graphs/ -q`
Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/mlops_agents/graphs/approval_nodes.py
git commit -m "refactor(hitl): build approval StateUpdate contracts in the gate nodes"
```

---

### Task 11: Binding test — contracts cover every writable state key

**Files:**
- Create: `tests/test_contracts/test_state_binding.py`

This is the enforceable single-source-of-truth guarantee: the union of all contract fields (by state-key, honouring serialization aliases) must equal the set of `AgentState` keys minus the keys no domain/gate node owns.

- [ ] **Step 1: Write the test** in `tests/test_contracts/test_state_binding.py`:

```python
"""The state-update contracts must stay in sync with AgentState.

Excluded keys are the ones no domain/gate node owns:
  messages              — the reducer channel (operator.add)
  dataset_paths         — a pipeline input, never written by a node
  agent_attempt_counts  — owned by the workflow_controller router (exempt)
"""

from mlops_agents.contracts import outputs as o
from mlops_agents.state.agent_state import AgentState

_ALL_STATE_UPDATES = [
    o.DataValidationStateUpdate,
    o.DatasetApprovalStateUpdate,
    o.PlannerStateUpdate,
    o.PlannerErrorStateUpdate,
    o.TrainingStateUpdate,
    o.EvaluationStateUpdate,
    o.AuditStateUpdate,
    o.DeploymentApprovalStateUpdate,
    o.DeploymentStateUpdate,
]

_EXCLUDED = {"messages", "dataset_paths", "agent_attempt_counts"}


def _keys_written_by_contracts() -> set[str]:
    keys: set[str] = set()
    for model in _ALL_STATE_UPDATES:
        for name, field in model.model_fields.items():
            keys.add(field.serialization_alias or name)
    return keys


def test_contracts_cover_every_writable_state_key():
    state_keys = set(AgentState.__annotations__) - _EXCLUDED
    written = _keys_written_by_contracts()
    assert written == state_keys, {
        "missing_from_contracts": sorted(state_keys - written),
        "extra_in_contracts": sorted(written - state_keys),
    }
```

- [ ] **Step 2: Run it.**

Run: `uv run pytest tests/test_contracts/test_state_binding.py -q`
Expected: PASS. (If it fails, the assertion message lists exactly which keys are missing/extra — fix the contract or the exclusion set.)

- [ ] **Step 3: Commit.**

```bash
git add tests/test_contracts/test_state_binding.py
git commit -m "test(contracts): bind StateUpdate contracts to AgentState keys"
```

---

### Task 12: Full-suite verification + stale-docstring cleanup

**Files:**
- Modify: `src/mlops_agents/graphs/mlops_graph.py` (module docstring, lines 1-12)
- Modify: `src/mlops_agents/contracts/__init__.py` (docstring, line 1)

- [ ] **Step 1: Replace the stale `mlops_graph.py` module docstring** (it still describes the removed LLM supervisor):

```python
"""Main LangGraph StateGraph — the MLOps pipeline topology.

Architecture:
  START → workflow_controller → [data_validator | dataset_approval | planner |
  executor | evaluation | report_writer | deployment_approval | deployer]
  → workflow_controller → … → END

workflow_controller is a deterministic router (no LLM): it reads state and
returns Command(goto=...), writing only routing-control updates inline. Every
other node returns its state slice via a typed contract from
mlops_agents.contracts.outputs (`.to_update()`), then routes back to
workflow_controller. HITL interrupts live in the two approval gate nodes.

Run with:
    uv run python scripts/run_pipeline.py
"""
```

- [ ] **Step 2: Replace the stale `contracts/__init__.py` docstring:**

```python
"""Cross-cutting Pydantic contracts shared across the data, planning, training,
evaluation, and deployment stages, plus the per-node state-update contracts."""
```

- [ ] **Step 3: Lint & type-check.**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy src/`
Expected: clean. Fix any now-unused import the refactor left (only if `ruff` flags it).

- [ ] **Step 4: Run the entire unit suite (no LLM calls).**

Run: `uv run pytest -m "not integration" -q`
Expected: all green.

- [ ] **Step 5: Sanity-check the graph imports & compiles.**

Run: `uv run python -c "from mlops_agents.graphs.mlops_graph import graph; print('graph OK')"`
Expected: prints `graph OK`.

- [ ] **Step 6: Commit.**

```bash
git add -A
git commit -m "docs(graph): refresh module docstrings after state-write unification"
```

---

## Self-Review

**1. Spec coverage (resolved decisions):**
- Goal mix → structural contracts + binding test (Task 11) + thesis-ready docstrings (Task 12). ✓
- Bind contracts↔TypedDict → Task 11. ✓
- Structural-strict + `extra="forbid"`, no value-types, no error boundary → Task 2 base + Task 3 contracts (loose `dict`/`str` types). ✓
- data_validator one contract for all 3 paths, all defaulted → Task 3 + Task 9. ✓
- Philosophy 2 (node builds; domain modules untouched) → Tasks 4-6 edit only `mlops_graph.py`; promotion/report_writer/deployer + their tests unchanged. ✓
- workflow_controller exempt → not modified; `agent_attempt_counts` in the binding exclusion set + documented in docstring. ✓
- Delete `next` → Task 1 Step 3. ✓
- Single `contracts/outputs.py`, naming `<Node>StateUpdate` → Tasks 2-3. ✓

**2. Placeholder scan:** every code/test step has literal content; no "TBD"/"handle edge cases"/"similar to". ✓

**3. Type consistency:**
- `StateUpdate.to_update(messages=None)` (Task 2) used with `messages=[...]` in Tasks 8 and 9, no-arg elsewhere. ✓
- `TrainingStateUpdate.from_training_result(result, *, training_plan)` (Task 3) called with `training_plan=plan.model_dump()` (Task 7). ✓
- `planner_output_record` field + `serialization_alias="_planner_output_record"` (Task 3) asserted via the alias (Task 3 test) and used by the binding test's `serialization_alias` lookup (Task 11). ✓
- Helper dict shapes match contract fields exactly (verified): `evaluate_promotion`→5 keys, `run_report_writer`→2, `run_deployer`→3; with `extra="forbid"`, `<Contract>(**helper_dict)` is valid. ✓
- Binding-test union closes exactly: 9 contracts cover all 41 non-excluded `AgentState` keys (after deleting `next`). ✓

## Notes for the implementer
- Do **not** edit `evaluation/promotion.py`, `evaluation/report_writer.py`, `deployment/deployer.py`, `training/executor.py`, or their tests — Philosophy 2 keeps them returning dicts/`TrainingResult`.
- Keep the `TrainingResult` import in `outputs.py` under the `TYPE_CHECKING` guard — do **not** promote it to a top-level runtime import. It is used only as an annotation; the guard pre-empts any future `contracts/` import cycle. (No cycle exists today, but `outputs.py` is the natural place one would later appear.)
- Task 11's binding closure was verified against the live `AgentState` before writing this plan: 41 keys to cover, all owned, zero missing/extra. If the test ever fails, **inspect first** — a genuinely new state field needs a contract owner; only add to the exclusion set if the key is a reducer channel, a pipeline input, or router-owned.
- The dual-naming aliases in `contracts/training.py` (`TrainingPlanCandidate`/`RejectedModel`) are out of scope.
- Keep the legacy `evaluation_report` nested shape (`candidate_metrics`, `candidate_run_id`, `baseline_metrics`) — frontend/SSE depends on it; it lives inside the dict `evaluate_promotion` returns, untouched.
