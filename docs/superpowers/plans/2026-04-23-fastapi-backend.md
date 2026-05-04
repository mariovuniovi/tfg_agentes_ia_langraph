# FastAPI Backend API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a FastAPI HTTP + WebSocket backend in `api/` that exposes the existing LangGraph pipeline, MLflow experiments, and Evidently drift detection so a Next.js frontend can consume them.

**Architecture:** FastAPI app with three routers (runs, experiments, monitoring). Pipeline runs execute as asyncio background tasks; a per-run `asyncio.Queue` feeds WebSocket clients in real time. HITL interrupts are resumed via a separate POST endpoint that sets an `asyncio.Event`. The existing Streamlit dashboard in `dashboard/` is not touched.

**Tech Stack:** FastAPI, uvicorn, python-multipart, LangGraph (`graph.astream()`), MLflow (`MlflowClient`), Evidently (`DataDriftPreset`), pytest + httpx (`AsyncClient`), unittest.mock.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `pyproject.toml` | Modify | Add fastapi, uvicorn[standard], python-multipart, httpx |
| `api/__init__.py` | Create | Package marker |
| `api/main.py` | Create | FastAPI app, CORS, router registration, `/health` |
| `api/models/__init__.py` | Create | Package marker |
| `api/models/run.py` | Create | `RunCreate`, `RunStatus`, `PipelineEventModel`, `HITLDecision` |
| `api/models/experiment.py` | Create | `MetricSeries`, `RunOut`, `ExperimentOut` |
| `api/models/monitoring.py` | Create | `ColumnDriftResult`, `DriftReport` |
| `api/services/__init__.py` | Create | Package marker |
| `api/services/run_store.py` | Create | `RunEntry` dataclass, `_store` dict, `latest_drift_report` |
| `api/services/pipeline.py` | Create | `pipeline_task()` async background function |
| `api/services/mlflow_client.py` | Create | `MlflowService` — typed wrapper over `MlflowClient` |
| `api/routers/__init__.py` | Create | Package marker |
| `api/routers/runs.py` | Create | `POST /runs`, `WS /ws/{run_id}`, `POST /runs/{run_id}/approve`, `GET /runs/{run_id}`, `GET /runs/{run_id}/events` |
| `api/routers/experiments.py` | Create | `GET /experiments`, `GET /experiments/{exp_id}/runs` |
| `api/routers/monitoring.py` | Create | `GET /monitoring/latest`, `POST /monitoring/drift` |
| `api/tests/__init__.py` | Create | Package marker |
| `api/tests/conftest.py` | Create | `async_client` fixture, `mock_graph` fixture, sample DataFrames |
| `api/tests/test_models.py` | Create | Pydantic model validation tests |
| `api/tests/test_run_store.py` | Create | RunStore unit tests |
| `api/tests/test_pipeline.py` | Create | pipeline_task unit tests (mocked graph) |
| `api/tests/test_runs.py` | Create | Runs router tests (mocked pipeline_task) |
| `api/tests/test_experiments.py` | Create | Experiments router tests (mocked MlflowService) |
| `api/tests/test_monitoring.py` | Create | Monitoring router tests (mocked Evidently) |

---

## Task 0: Create branch and add dependencies

**Files:**
- Modify: `pyproject.toml`
- Create: `api/__init__.py`, `api/models/__init__.py`, `api/services/__init__.py`, `api/routers/__init__.py`, `api/tests/__init__.py`

- [ ] **Step 1: Create the feature branch**

```bash
git checkout -b feature/fastapi-backend
```

Expected: `Switched to a new branch 'feature/fastapi-backend'`

- [ ] **Step 2: Add dependencies to pyproject.toml**

In `pyproject.toml`, add to the `dependencies` list (after the existing entries):

```toml
    # FastAPI backend
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "python-multipart>=0.0.12",
    "httpx>=0.27",
```

- [ ] **Step 3: Install dependencies**

```bash
uv sync
```

Expected: resolves and installs fastapi, uvicorn, python-multipart, httpx with no conflicts.

- [ ] **Step 4: Create package markers**

```bash
mkdir -p api/models api/services api/routers api/tests
touch api/__init__.py api/models/__init__.py api/services/__init__.py api/routers/__init__.py api/tests/__init__.py
```

- [ ] **Step 5: Commit scaffold**

```bash
git add pyproject.toml uv.lock api/
git commit -m "chore: scaffold api/ package and add fastapi dependencies"
```

---

## Task 1: Pydantic models

**Files:**
- Create: `api/models/run.py`
- Create: `api/models/experiment.py`
- Create: `api/models/monitoring.py`
- Create: `api/tests/test_models.py`

- [ ] **Step 1: Write the failing tests**

Create `api/tests/test_models.py`:

```python
"""Tests for Pydantic API models."""
from datetime import datetime
from api.models.run import RunCreate, RunStatus, PipelineEventModel, HITLDecision
from api.models.experiment import MetricSeries, RunOut, ExperimentOut
from api.models.monitoring import ColumnDriftResult, DriftReport


def test_run_create_requires_dataset_paths():
    rc = RunCreate(dataset_paths=["data/samples/iris_measurements.csv"])
    assert rc.dataset_paths == ["data/samples/iris_measurements.csv"]


def test_hitl_decision_valid_values():
    d = HITLDecision(decision="approve")
    assert d.decision == "approve"
    d2 = HITLDecision(decision="reject", reason="too low accuracy")
    assert d2.reason == "too low accuracy"


def test_hitl_decision_rejects_invalid():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        HITLDecision(decision="maybe")


def test_run_status_model():
    rs = RunStatus(run_id="abc", status="running")
    assert rs.interrupt_value is None
    rs2 = RunStatus(run_id="abc", status="awaiting_approval", interrupt_value={"model": "v1"})
    assert rs2.interrupt_value == {"model": "v1"}


def test_pipeline_event_model():
    ev = PipelineEventModel(
        type="tool_call",
        agent="trainer",
        timestamp_ms=1000.0,
        data={"tool_name": "train_model", "arguments": {}},
    )
    assert ev.type == "tool_call"


def test_metric_series_line_styles():
    ms = MetricSeries(name="accuracy", steps=[1, 2], values=[0.8, 0.9], line_style="solid")
    assert ms.line_style == "solid"


def test_drift_report_model():
    col = ColumnDriftResult(column="feature_1", drift_detected=True, score=0.3, method="psi")
    dr = DriftReport(
        dataset_drift=True,
        drift_share=0.5,
        columns=[col],
        generated_at=datetime.utcnow(),
    )
    assert dr.drift_share == 0.5
    assert dr.columns[0].column == "feature_1"
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest api/tests/test_models.py -v
```

Expected: `ImportError` — modules don't exist yet.

- [ ] **Step 3: Create api/models/run.py**

```python
from typing import Any, Literal
from pydantic import BaseModel


class RunCreate(BaseModel):
    dataset_paths: list[str]


class RunStatus(BaseModel):
    run_id: str
    status: Literal["running", "awaiting_approval", "complete", "failed"]
    interrupt_value: dict[str, Any] | None = None


class PipelineEventModel(BaseModel):
    type: Literal[
        "routing", "tool_call", "tool_result", "agent_reasoning",
        "hitl_request", "run_complete",
    ]
    agent: str
    timestamp_ms: float
    data: dict[str, Any]


class HITLDecision(BaseModel):
    decision: Literal["approve", "reject"]
    reason: str = ""
```

- [ ] **Step 4: Create api/models/experiment.py**

```python
from datetime import datetime
from typing import Literal
from pydantic import BaseModel


class MetricSeries(BaseModel):
    name: str
    steps: list[int]
    values: list[float]
    line_style: Literal["solid", "dashed", "dotted"]


class RunOut(BaseModel):
    run_id: str
    run_name: str
    status: str
    start_time: datetime
    params: dict[str, str]
    metrics: dict[str, float]
    metric_series: list[MetricSeries]


class ExperimentOut(BaseModel):
    experiment_id: str
    name: str
```

- [ ] **Step 5: Create api/models/monitoring.py**

```python
from datetime import datetime
from pydantic import BaseModel


class ColumnDriftResult(BaseModel):
    column: str
    drift_detected: bool
    score: float
    method: str


class DriftReport(BaseModel):
    dataset_drift: bool
    drift_share: float
    columns: list[ColumnDriftResult]
    generated_at: datetime
```

- [ ] **Step 6: Run tests — confirm pass**

```bash
uv run pytest api/tests/test_models.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add api/models/ api/tests/test_models.py
git commit -m "feat: add Pydantic models for runs, experiments, monitoring"
```

---

## Task 2: RunStore

**Files:**
- Create: `api/services/run_store.py`
- Create: `api/tests/test_run_store.py`

- [ ] **Step 1: Write the failing tests**

Create `api/tests/test_run_store.py`:

```python
"""Tests for in-memory RunStore."""
import asyncio
import pytest
from api.services.run_store import RunEntry, create_entry, get_entry, set_latest_drift_report, get_latest_drift_report


def _make_entry(run_id: str = "test-run") -> RunEntry:
    return create_entry(run_id, graph_config={"configurable": {"thread_id": run_id}})


def test_create_entry_sets_defaults():
    entry = _make_entry("r1")
    assert entry.run_id == "r1"
    assert entry.status == "running"
    assert isinstance(entry.queue, asyncio.Queue)
    assert isinstance(entry.hitl_event, asyncio.Event)
    assert entry.events == []
    assert entry.last_drift_report is None


def test_get_entry_returns_none_for_unknown():
    assert get_entry("nonexistent") is None


def test_create_and_retrieve_entry():
    entry = _make_entry("r2")
    assert get_entry("r2") is entry


def test_latest_drift_report_starts_none():
    assert get_latest_drift_report() is None


def test_set_and_get_latest_drift_report():
    report = {"dataset_drift": True, "drift_share": 0.5}
    set_latest_drift_report(report)
    assert get_latest_drift_report() == report
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest api/tests/test_run_store.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Create api/services/run_store.py**

```python
import asyncio
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunEntry:
    run_id: str
    status: str                          # "running"|"awaiting_approval"|"complete"|"failed"
    queue: asyncio.Queue
    graph_config: dict[str, Any]
    hitl_event: asyncio.Event
    hitl_decision: str = ""
    events: list[dict] = field(default_factory=list)
    interrupt_value: dict[str, Any] = field(default_factory=dict)
    last_drift_report: dict | None = None


_store: dict[str, RunEntry] = {}
_latest_drift_report: dict | None = None


def create_entry(run_id: str, graph_config: dict) -> RunEntry:
    entry = RunEntry(
        run_id=run_id,
        status="running",
        queue=asyncio.Queue(),
        graph_config=graph_config,
        hitl_event=asyncio.Event(),
    )
    _store[run_id] = entry
    return entry


def get_entry(run_id: str) -> RunEntry | None:
    return _store.get(run_id)


def set_latest_drift_report(report: dict) -> None:
    global _latest_drift_report
    _latest_drift_report = report


def get_latest_drift_report() -> dict | None:
    return _latest_drift_report
```

- [ ] **Step 4: Run tests — confirm pass**

```bash
uv run pytest api/tests/test_run_store.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add api/services/run_store.py api/tests/test_run_store.py
git commit -m "feat: add in-memory RunStore with RunEntry dataclass"
```

---

## Task 3: Pipeline service (mocked graph)

**Files:**
- Create: `api/services/pipeline.py`
- Create: `api/tests/conftest.py`
- Create: `api/tests/test_pipeline.py`

- [ ] **Step 1: Create api/tests/conftest.py**

```python
"""Shared fixtures for api tests."""
import asyncio
from unittest.mock import AsyncMock, MagicMock
import pandas as pd
import pytest
from httpx import AsyncClient, ASGITransport


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feature_2": [0.5, 1.5, 2.5, 3.5, 4.5],
        "target": [0, 1, 0, 1, 0],
    })


@pytest.fixture()
def mock_graph():
    """Mock LangGraph graph with astream() and aget_state()."""
    graph = MagicMock()
    # astream yields nothing by default — override per test
    graph.astream = AsyncMock(return_value=aiter([]))
    graph.aget_state = AsyncMock()
    graph.aget_state.return_value.values = {
        "dataset_path": "",
        "training_metrics": {},
        "validation_report": {},
    }
    graph.aget_state.return_value.next = ("supervisor",)
    return graph


async def aiter(items):
    for item in items:
        yield item


@pytest.fixture()
async def async_client():
    from api.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client
```

- [ ] **Step 2: Write the failing pipeline tests**

Create `api/tests/test_pipeline.py`:

```python
"""Unit tests for pipeline_task — graph is fully mocked."""
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from api.services.run_store import create_entry, get_entry
from api.services.pipeline import pipeline_task


def _routing_chunk():
    """Simulate a supervisor routing update chunk."""
    return ("ns", "updates", {"supervisor": {"next": "trainer", "reasoning": "needs training"}})


def _messages_chunk():
    """Simulate a messages chunk (agent_reasoning type)."""
    from langchain_core.messages import AIMessageChunk
    msg = AIMessageChunk(content="Analysing data...")
    meta = {"langgraph_node": "trainer"}
    return ("ns", "messages", (msg, meta))


def _interrupt_chunk():
    """Simulate an __interrupt__ update chunk."""
    return ("ns", "updates", {"__interrupt__": [{"value": {"model_uri": "runs:/abc/model"}}]})


def _run_complete_chunk():
    """Simulate a final supervisor FINISH chunk."""
    return ("ns", "updates", {"supervisor": {"next": "FINISH"}})


@pytest.mark.asyncio
async def test_pipeline_task_queues_routing_event(mock_graph, tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text("feature_1,feature_2,target\n1.0,0.5,0\n")

    async def fake_astream(*a, **kw):
        yield _routing_chunk()
        yield _run_complete_chunk()

    mock_graph.astream = fake_astream

    with patch("api.services.pipeline.graph", mock_graph), \
         patch("api.services.pipeline.run_evidently", return_value={}):
        run_id = "test-routing"
        entry = create_entry(run_id, {"configurable": {"thread_id": run_id}})
        await pipeline_task(run_id, [str(csv)])

    events = [e for e in entry.events if e["type"] == "routing"]
    assert len(events) == 1
    assert events[0]["data"]["next"] == "trainer"


@pytest.mark.asyncio
async def test_pipeline_task_queues_run_complete(mock_graph, tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text("feature_1,feature_2,target\n1.0,0.5,0\n")

    async def fake_astream(*a, **kw):
        yield _run_complete_chunk()

    mock_graph.astream = fake_astream

    with patch("api.services.pipeline.graph", mock_graph), \
         patch("api.services.pipeline.run_evidently", return_value={}):
        run_id = "test-complete"
        entry = create_entry(run_id, {"configurable": {"thread_id": run_id}})
        await pipeline_task(run_id, [str(csv)])

    assert entry.status == "complete"
    complete_events = [e for e in entry.events if e["type"] == "run_complete"]
    assert len(complete_events) == 1


@pytest.mark.asyncio
async def test_pipeline_task_sets_awaiting_on_interrupt(mock_graph, tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text("feature_1,feature_2,target\n1.0,0.5,0\n")

    async def fake_astream(*a, **kw):
        if isinstance(a[0], dict):  # first call — initial state
            yield _interrupt_chunk()
        else:                        # second call — Command(resume=...)
            yield _run_complete_chunk()

    mock_graph.astream = fake_astream

    with patch("api.services.pipeline.graph", mock_graph), \
         patch("api.services.pipeline.run_evidently", return_value={}):
        run_id = "test-hitl"
        entry = create_entry(run_id, {"configurable": {"thread_id": run_id}})

        # Simulate approval arriving after 10ms
        async def approve_later():
            await asyncio.sleep(0.01)
            entry.hitl_decision = "approve"
            entry.hitl_event.set()

        await asyncio.gather(
            pipeline_task(run_id, [str(csv)]),
            approve_later(),
        )

    assert entry.status == "complete"
    hitl_events = [e for e in entry.events if e["type"] == "hitl_request"]
    assert len(hitl_events) == 1
```

- [ ] **Step 3: Run to confirm failure**

```bash
uv run pytest api/tests/test_pipeline.py -v
```

Expected: `ImportError` — pipeline.py doesn't exist.

- [ ] **Step 4: Create api/services/pipeline.py**

```python
"""Async background task: runs the LangGraph pipeline and feeds events to RunStore."""
import time
from typing import Any

import pandas as pd
from langchain_core.messages import AIMessageChunk, ToolMessage
from langgraph.types import Command

from dashboard.pipeline_helpers import (
    build_initial_state,
    parse_stream_event,
    reset_tool_start_times,
)
from mlops_agents.graphs.mlops_graph import graph

from api.services import run_store


def run_evidently(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
    """Run Evidently DataDriftPreset and return a DriftReport-shaped dict."""
    from datetime import datetime, timezone
    from evidently import Report
    from evidently.presets import DataDriftPreset

    report = Report([DataDriftPreset()])
    result = report.run(reference_df, current_df)
    raw = result.dump_dict()

    # Flatten to DriftReport shape
    metrics = raw.get("metrics", [])
    dataset_metric = next((m for m in metrics if m.get("metric") == "DatasetDriftMetric"), {})
    column_metrics = [m for m in metrics if m.get("metric") == "ColumnDriftMetric"]

    drift_res = dataset_metric.get("result", {})
    columns = [
        {
            "column": m["result"].get("column_name", ""),
            "drift_detected": m["result"].get("drift_detected", False),
            "score": m["result"].get("drift_score", 0.0),
            "method": m["result"].get("stattest_name", ""),
        }
        for m in column_metrics
        if "result" in m
    ]

    drifted = sum(1 for c in columns if c["drift_detected"])
    drift_share = drifted / len(columns) if columns else 0.0

    return {
        "dataset_drift": drift_res.get("dataset_drift", False),
        "drift_share": drift_share,
        "columns": columns,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


async def pipeline_task(run_id: str, dataset_paths: list[str]) -> None:
    """Execute the LangGraph pipeline as an asyncio background task."""
    entry = run_store.get_entry(run_id)
    if entry is None:
        return

    reset_tool_start_times()
    initial_state = build_initial_state(dataset_paths)
    config = entry.graph_config

    async def _stream(source: Any) -> None:
        """Inner streaming loop — handles both initial run and post-HITL resume."""
        nonlocal entry
        async for chunk in graph.astream(
            source, config, stream_mode=["updates", "messages"], subgraphs=True
        ):
            namespace, mode, data = chunk

            if mode == "updates":
                if "__interrupt__" in data:
                    interrupt_list = data["__interrupt__"]
                    interrupt_val = interrupt_list[0].get("value", {}) if interrupt_list else {}
                    entry.status = "awaiting_approval"
                    entry.interrupt_value = interrupt_val
                    hitl_event: dict = {
                        "type": "hitl_request",
                        "agent": "deployer",
                        "timestamp_ms": time.time() * 1000,
                        "data": interrupt_val,
                    }
                    entry.events.append(hitl_event)
                    await entry.queue.put(hitl_event)
                    return  # exit streaming loop; wait for approval below

                if "supervisor" in data:
                    next_agent = data["supervisor"].get("next", "")
                    reasoning = data["supervisor"].get("reasoning", "")
                    if next_agent:
                        event = {
                            "type": "routing",
                            "agent": "supervisor",
                            "timestamp_ms": time.time() * 1000,
                            "data": {"next": next_agent, "reasoning": reasoning},
                        }
                        entry.events.append(event)
                        await entry.queue.put(event)

            elif mode == "messages":
                pipeline_event = parse_stream_event(data)
                if pipeline_event:
                    entry.events.append(dict(pipeline_event))
                    await entry.queue.put(dict(pipeline_event))

    try:
        await _stream(initial_state)

        # If paused for HITL, wait for approval then resume
        if entry.status == "awaiting_approval":
            await entry.hitl_event.wait()
            entry.status = "running"
            await _stream(Command(resume=entry.hitl_decision))

        # Run automatic Evidently drift detection
        final_state = (await graph.aget_state(config)).values
        dataset_path = final_state.get("dataset_path", "")
        try:
            if dataset_path:
                import pandas as pd
                current_df = pd.read_csv(dataset_path)
                reference_df = current_df  # same dataset for demo; override if ref available
                drift_report = run_evidently(reference_df, current_df)
            else:
                drift_report = {}
        except Exception:
            drift_report = {}

        entry.last_drift_report = drift_report
        run_store.set_latest_drift_report(drift_report)

        complete_event = {
            "type": "run_complete",
            "agent": "supervisor",
            "timestamp_ms": time.time() * 1000,
            "data": drift_report,
        }
        entry.events.append(complete_event)
        await entry.queue.put(complete_event)
        entry.status = "complete"

    except Exception as exc:
        error_event = {
            "type": "run_complete",
            "agent": "supervisor",
            "timestamp_ms": time.time() * 1000,
            "data": {"error": str(exc)},
        }
        entry.events.append(error_event)
        await entry.queue.put(error_event)
        entry.status = "failed"
```

- [ ] **Step 5: Run tests — confirm pass**

```bash
uv run pytest api/tests/test_pipeline.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add api/services/pipeline.py api/tests/conftest.py api/tests/test_pipeline.py
git commit -m "feat: add async pipeline_task with HITL interrupt support"
```

---

## Task 4: MLflow service

**Files:**
- Create: `api/services/mlflow_client.py`
- Create: `api/tests/test_experiments.py` (partial — service layer only)

- [ ] **Step 1: Write the failing tests**

Create `api/tests/test_experiments.py`:

```python
"""Tests for MlflowService."""
from datetime import datetime
from unittest.mock import MagicMock, patch
import pytest
from api.services.mlflow_client import MlflowService
from api.models.experiment import ExperimentOut, RunOut


def _mock_experiment(exp_id="1", name="mlops-agents"):
    exp = MagicMock()
    exp.experiment_id = exp_id
    exp.name = name
    return exp


def _mock_run(run_id="abc123", status="FINISHED", start_time_ms=1_700_000_000_000):
    run = MagicMock()
    run.info.run_id = run_id
    run.info.run_name = "run-1"
    run.info.status = status
    run.info.start_time = start_time_ms
    run.data.params = {"model_type": "RandomForest"}
    run.data.metrics = {"accuracy": 0.94, "f1": 0.91}
    return run


def _mock_metric_history(values: list[float]):
    return [MagicMock(step=i, value=v) for i, v in enumerate(values)]


def test_list_experiments():
    with patch("api.services.mlflow_client.MlflowClient") as MockClient:
        MockClient.return_value.search_experiments.return_value = [
            _mock_experiment("1", "mlops-agents"),
            _mock_experiment("2", "test-exp"),
        ]
        svc = MlflowService()
        exps = svc.list_experiments()

    assert len(exps) == 2
    assert all(isinstance(e, ExperimentOut) for e in exps)
    assert exps[0].name == "mlops-agents"


def test_get_runs_returns_run_out_list():
    with patch("api.services.mlflow_client.MlflowClient") as MockClient:
        client = MockClient.return_value
        client.search_runs.return_value = [_mock_run()]
        client.get_metric_history.side_effect = lambda run_id, metric: _mock_metric_history([0.8, 0.9])

        svc = MlflowService()
        runs = svc.get_runs("1")

    assert len(runs) == 1
    run = runs[0]
    assert isinstance(run, RunOut)
    assert run.run_id == "abc123"
    assert run.metrics["accuracy"] == 0.94
    assert len(run.metric_series) == 2  # accuracy + f1
    # line_style cycles: solid, dashed
    styles = [ms.line_style for ms in run.metric_series]
    assert styles == ["solid", "dashed"]


def test_get_runs_empty():
    with patch("api.services.mlflow_client.MlflowClient") as MockClient:
        MockClient.return_value.search_runs.return_value = []
        svc = MlflowService()
        runs = svc.get_runs("1")
    assert runs == []
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest api/tests/test_experiments.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Create api/services/mlflow_client.py**

```python
"""Typed wrapper over MLflow's MlflowClient — returns Pydantic models shaped for charts."""
from datetime import datetime, timezone
from typing import Literal

import mlflow
from mlflow.tracking import MlflowClient

from api.models.experiment import ExperimentOut, MetricSeries, RunOut
from mlops_agents.config.settings import settings

_LINE_STYLES: list[Literal["solid", "dashed", "dotted"]] = ["solid", "dashed", "dotted"]


class MlflowService:
    def __init__(self) -> None:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        self._client = MlflowClient()

    def list_experiments(self) -> list[ExperimentOut]:
        exps = self._client.search_experiments()
        return [
            ExperimentOut(experiment_id=e.experiment_id, name=e.name)
            for e in exps
        ]

    def get_runs(self, experiment_id: str, max_results: int = 50) -> list[RunOut]:
        runs = self._client.search_runs(
            experiment_ids=[experiment_id],
            order_by=["start_time DESC"],
            max_results=max_results,
        )
        return [self._to_run_out(r) for r in runs]

    def _to_run_out(self, run) -> RunOut:
        metrics: dict[str, float] = dict(run.data.metrics)

        metric_series: list[MetricSeries] = []
        for idx, metric_name in enumerate(sorted(metrics.keys())):
            history = self._client.get_metric_history(run.info.run_id, metric_name)
            if history:
                metric_series.append(MetricSeries(
                    name=metric_name,
                    steps=[m.step for m in history],
                    values=[m.value for m in history],
                    line_style=_LINE_STYLES[idx % len(_LINE_STYLES)],
                ))

        return RunOut(
            run_id=run.info.run_id,
            run_name=run.info.run_name or run.info.run_id[:8],
            status=run.info.status,
            start_time=datetime.fromtimestamp(
                run.info.start_time / 1000, tz=timezone.utc
            ),
            params=dict(run.data.params),
            metrics=metrics,
            metric_series=metric_series,
        )
```

- [ ] **Step 4: Run tests — confirm pass**

```bash
uv run pytest api/tests/test_experiments.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add api/services/mlflow_client.py api/tests/test_experiments.py
git commit -m "feat: add MlflowService wrapper with MetricSeries for line charts"
```

---

## Task 5: Runs router

**Files:**
- Create: `api/routers/runs.py`
- Create: `api/main.py` (minimal version — needed to mount router)
- Create: `api/tests/test_runs.py`

- [ ] **Step 1: Create minimal api/main.py**

```python
"""FastAPI application entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import runs, experiments, monitoring

app = FastAPI(title="MLOps Backend API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(runs.router)
app.include_router(experiments.router)
app.include_router(monitoring.router)


@app.get("/health")
async def health():
    mlflow_ok = True
    graph_ok = True
    try:
        from mlops_agents.config.settings import settings
        import mlflow
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.search_experiments()
    except Exception:
        mlflow_ok = False
    try:
        from mlops_agents.graphs.mlops_graph import graph  # noqa: F401
    except Exception:
        graph_ok = False
    return {"status": "ok", "mlflow": mlflow_ok, "graph": graph_ok}
```

- [ ] **Step 2: Create skeleton routers so app imports don't fail**

Create `api/routers/experiments.py`:
```python
from fastapi import APIRouter
router = APIRouter()
```

Create `api/routers/monitoring.py`:
```python
from fastapi import APIRouter
router = APIRouter()
```

- [ ] **Step 3: Write failing runs router tests**

Create `api/tests/test_runs.py`:

```python
"""Tests for /runs endpoints."""
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import pytest
from httpx import AsyncClient, ASGITransport
from api.main import app
import api.services.run_store as run_store_module


@pytest.fixture()
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_post_runs_returns_run_id(client):
    with patch("api.routers.runs.BackgroundTasks.add_task"):
        resp = await client.post("/runs", json={"dataset_paths": ["data/samples/iris_measurements.csv"]})
    assert resp.status_code == 200
    body = resp.json()
    assert "run_id" in body
    assert isinstance(body["run_id"], str)


@pytest.mark.asyncio
async def test_get_run_status_running(client):
    with patch("api.routers.runs.BackgroundTasks.add_task"):
        start = await client.post("/runs", json={"dataset_paths": ["data/samples/iris_measurements.csv"]})
    run_id = start.json()["run_id"]
    resp = await client.get(f"/runs/{run_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "running"


@pytest.mark.asyncio
async def test_get_run_status_unknown(client):
    resp = await client.get("/runs/does-not-exist")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_approve_run_not_awaiting_returns_400(client):
    with patch("api.routers.runs.BackgroundTasks.add_task"):
        start = await client.post("/runs", json={"dataset_paths": ["data/samples/iris_measurements.csv"]})
    run_id = start.json()["run_id"]
    resp = await client.post(f"/runs/{run_id}/approve", json={"decision": "approve"})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_approve_run_awaiting_returns_ok(client):
    with patch("api.routers.runs.BackgroundTasks.add_task"):
        start = await client.post("/runs", json={"dataset_paths": ["data/samples/iris_measurements.csv"]})
    run_id = start.json()["run_id"]
    entry = run_store_module.get_entry(run_id)
    entry.status = "awaiting_approval"
    resp = await client.post(f"/runs/{run_id}/approve", json={"decision": "approve"})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert entry.hitl_decision == "approve"
    assert entry.hitl_event.is_set()


@pytest.mark.asyncio
async def test_get_run_events_empty(client):
    with patch("api.routers.runs.BackgroundTasks.add_task"):
        start = await client.post("/runs", json={"dataset_paths": ["data/samples/iris_measurements.csv"]})
    run_id = start.json()["run_id"]
    resp = await client.get(f"/runs/{run_id}/events")
    assert resp.status_code == 200
    assert resp.json() == []
```

- [ ] **Step 4: Run to confirm failure**

```bash
uv run pytest api/tests/test_runs.py -v
```

Expected: `ImportError` — `api/routers/runs.py` doesn't exist.

- [ ] **Step 5: Create api/routers/runs.py**

```python
"""Runs router: pipeline execution, WebSocket streaming, HITL approval."""
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect

import api.services.run_store as run_store
from api.models.run import HITLDecision, RunCreate, RunStatus
from api.services.pipeline import pipeline_task

router = APIRouter()


@router.post("/runs")
async def start_run(body: RunCreate, background_tasks: BackgroundTasks):
    run_id = str(uuid4())
    config = {"configurable": {"thread_id": run_id}}
    run_store.create_entry(run_id, config)
    background_tasks.add_task(pipeline_task, run_id, body.dataset_paths)
    return {"run_id": run_id}


@router.get("/runs/{run_id}", response_model=RunStatus)
async def get_run_status(run_id: str):
    entry = run_store.get_entry(run_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return RunStatus(
        run_id=entry.run_id,
        status=entry.status,
        interrupt_value=entry.interrupt_value or None,
    )


@router.post("/runs/{run_id}/approve")
async def approve_run(run_id: str, body: HITLDecision):
    entry = run_store.get_entry(run_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if entry.status != "awaiting_approval":
        raise HTTPException(status_code=400, detail="Run is not awaiting approval")
    entry.hitl_decision = body.decision
    entry.hitl_event.set()
    entry.status = "running"
    return {"ok": True}


@router.get("/runs/{run_id}/events")
async def get_run_events(run_id: str):
    entry = run_store.get_entry(run_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return entry.events


@router.websocket("/ws/{run_id}")
async def pipeline_ws(websocket: WebSocket, run_id: str):
    entry = run_store.get_entry(run_id)
    if entry is None:
        await websocket.close(code=4004)
        return
    await websocket.accept()
    try:
        while True:
            event = await entry.queue.get()
            await websocket.send_json(event)
            if event.get("type") == "run_complete":
                break
    except WebSocketDisconnect:
        pass
```

- [ ] **Step 6: Run tests — confirm pass**

```bash
uv run pytest api/tests/test_runs.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add api/main.py api/routers/runs.py api/routers/experiments.py api/routers/monitoring.py api/tests/test_runs.py
git commit -m "feat: add runs router with POST /runs, HITL approve, WebSocket, GET /runs/{id}"
```

---

## Task 6: Experiments router

**Files:**
- Modify: `api/routers/experiments.py`
- Modify: `api/tests/test_experiments.py` (add router tests)

- [ ] **Step 1: Add router tests to test_experiments.py**

Append to `api/tests/test_experiments.py`:

```python
# ── Router-level tests ─────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_get_experiments_endpoint():
    from httpx import AsyncClient, ASGITransport
    from api.main import app
    with patch("api.routers.experiments.MlflowService") as MockSvc:
        MockSvc.return_value.list_experiments.return_value = [
            ExperimentOut(experiment_id="1", name="mlops-agents"),
        ]
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/experiments")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["name"] == "mlops-agents"


@pytest.mark.asyncio
async def test_get_experiment_runs_endpoint():
    from datetime import datetime, timezone
    from httpx import AsyncClient, ASGITransport
    from api.main import app
    with patch("api.routers.experiments.MlflowService") as MockSvc:
        MockSvc.return_value.get_runs.return_value = [
            RunOut(
                run_id="abc",
                run_name="run-1",
                status="FINISHED",
                start_time=datetime.now(timezone.utc),
                params={},
                metrics={"accuracy": 0.94},
                metric_series=[
                    MetricSeries(name="accuracy", steps=[1], values=[0.94], line_style="solid")
                ],
            )
        ]
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/experiments/1/runs")
    assert resp.status_code == 200
    runs = resp.json()
    assert runs[0]["run_id"] == "abc"
    assert runs[0]["metric_series"][0]["line_style"] == "solid"
```

- [ ] **Step 2: Run to confirm new tests fail**

```bash
uv run pytest api/tests/test_experiments.py::test_get_experiments_endpoint api/tests/test_experiments.py::test_get_experiment_runs_endpoint -v
```

Expected: FAIL — router returns 404 (endpoints don't exist yet).

- [ ] **Step 3: Implement api/routers/experiments.py**

```python
"""Experiments router: MLflow experiment and run data."""
from fastapi import APIRouter, HTTPException

from api.models.experiment import ExperimentOut, RunOut
from api.services.mlflow_client import MlflowService

router = APIRouter()


@router.get("/experiments", response_model=list[ExperimentOut])
async def list_experiments():
    try:
        svc = MlflowService()
        return svc.list_experiments()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"MLflow unavailable: {exc}")


@router.get("/experiments/{experiment_id}/runs", response_model=list[RunOut])
async def get_experiment_runs(experiment_id: str):
    try:
        svc = MlflowService()
        return svc.get_runs(experiment_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"MLflow unavailable: {exc}")
```

- [ ] **Step 4: Run all experiment tests — confirm pass**

```bash
uv run pytest api/tests/test_experiments.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add api/routers/experiments.py api/tests/test_experiments.py
git commit -m "feat: add experiments router with MLflow proxy endpoints"
```

---

## Task 7: Monitoring router

**Files:**
- Modify: `api/routers/monitoring.py`
- Create: `api/tests/test_monitoring.py`

- [ ] **Step 1: Write failing tests**

Create `api/tests/test_monitoring.py`:

```python
"""Tests for /monitoring endpoints."""
import io
from unittest.mock import patch
import pytest
from httpx import AsyncClient, ASGITransport
from api.main import app


@pytest.fixture()
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_get_monitoring_latest_no_run(client):
    import api.services.run_store as rs
    rs._latest_drift_report = None
    resp = await client.get("/monitoring/latest")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_monitoring_latest_returns_report(client):
    import api.services.run_store as rs
    from datetime import datetime, timezone
    rs._latest_drift_report = {
        "dataset_drift": False,
        "drift_share": 0.0,
        "columns": [],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    resp = await client.get("/monitoring/latest")
    assert resp.status_code == 200
    assert resp.json()["dataset_drift"] is False


@pytest.mark.asyncio
async def test_post_monitoring_drift_returns_report(client):
    csv_content = b"feature_1,feature_2,target\n1.0,0.5,0\n2.0,1.5,1\n"
    mock_report = {
        "dataset_drift": False,
        "drift_share": 0.0,
        "columns": [],
        "generated_at": "2026-04-23T00:00:00+00:00",
    }
    with patch("api.routers.monitoring.run_evidently", return_value=mock_report):
        resp = await client.post(
            "/monitoring/drift",
            data={"reference_index": "0", "current_index": "1"},
            files=[
                ("files", ("ref.csv", io.BytesIO(csv_content), "text/csv")),
                ("files", ("cur.csv", io.BytesIO(csv_content), "text/csv")),
            ],
        )
    assert resp.status_code == 200
    assert "dataset_drift" in resp.json()


@pytest.mark.asyncio
async def test_post_monitoring_drift_bad_index(client):
    csv_content = b"feature_1,target\n1.0,0\n"
    resp = await client.post(
        "/monitoring/drift",
        data={"reference_index": "0", "current_index": "5"},
        files=[("files", ("ref.csv", io.BytesIO(csv_content), "text/csv"))],
    )
    assert resp.status_code == 400
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest api/tests/test_monitoring.py -v
```

Expected: FAIL — endpoints return 404.

- [ ] **Step 3: Implement api/routers/monitoring.py**

```python
"""Monitoring router: automatic drift report + ad-hoc multi-file drift detection."""
import io
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Form, HTTPException, UploadFile

import api.services.run_store as run_store
from api.services.pipeline import run_evidently

router = APIRouter()


@router.get("/monitoring/latest")
async def get_latest_drift():
    report = run_store.get_latest_drift_report()
    if report is None:
        raise HTTPException(status_code=404, detail="No completed pipeline run yet")
    return report


@router.post("/monitoring/drift")
async def drift_compare(
    files: list[UploadFile],
    reference_index: Annotated[int, Form()],
    current_index: Annotated[int, Form()],
):
    if reference_index >= len(files) or current_index >= len(files):
        raise HTTPException(
            status_code=400,
            detail=f"Index out of range: got {len(files)} files, "
                   f"reference_index={reference_index}, current_index={current_index}",
        )
    try:
        ref_bytes = await files[reference_index].read()
        cur_bytes = await files[current_index].read()
        reference_df = pd.read_csv(io.BytesIO(ref_bytes))
        current_df = pd.read_csv(io.BytesIO(cur_bytes))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {exc}")

    try:
        report = run_evidently(reference_df, current_df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Drift detection failed: {exc}")

    return report
```

- [ ] **Step 4: Run tests — confirm pass**

```bash
uv run pytest api/tests/test_monitoring.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add api/routers/monitoring.py api/tests/test_monitoring.py
git commit -m "feat: add monitoring router with automatic drift report and multi-file upload"
```

---

## Task 8: Full test suite + smoke test

**Files:**
- No new files — run everything and fix any issues

- [ ] **Step 1: Run full unit test suite**

```bash
uv run pytest api/tests/ -m "not integration" -v
```

Expected: all tests PASS. Fix any failures before proceeding.

- [ ] **Step 2: Start the server and hit /health manually**

```bash
uv run uvicorn api.main:app --reload --port 8000
```

In a second terminal:
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "ok", "mlflow": true, "graph": true}
```

If `mlflow` is `false`, start MLflow first: `uv run mlflow ui --port 5000`.

- [ ] **Step 3: Commit final state**

```bash
git add -A
git commit -m "feat: FastAPI backend API complete — all unit tests passing"
```

---

## Task 9: Integration tests

**Files:**
- Create: `api/tests/test_integration.py`

- [ ] **Step 1: Write integration tests**

Create `api/tests/test_integration.py`:

```python
"""Integration tests — hit the real graph and MLflow. Require running services.

Run with: uv run pytest api/tests/test_integration.py -m integration -v
"""
import asyncio
import pytest
from httpx import AsyncClient, ASGITransport
from api.main import app


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_health_endpoint_with_real_services():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["graph"] is True


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_full_pipeline_run_streams_events():
    """Start a real pipeline run and consume WebSocket events until run_complete."""
    import websockets
    import json

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/runs",
            json={"dataset_paths": ["data/samples/iris_measurements.csv",
                                    "data/samples/iris_labels.csv"]},
        )
    assert resp.status_code == 200
    run_id = resp.json()["run_id"]

    # Give the background task a moment to start
    await asyncio.sleep(0.5)

    received_types: list[str] = []
    async with websockets.connect(f"ws://localhost:8000/ws/{run_id}") as ws:
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=120)
            event = json.loads(raw)
            received_types.append(event["type"])
            if event["type"] == "run_complete":
                break

    assert "routing" in received_types
    assert "run_complete" in received_types
```

- [ ] **Step 2: Run to confirm integration tests are skipped in unit mode**

```bash
uv run pytest api/tests/ -m "not integration" -v
```

Expected: integration tests skipped, all unit tests PASS.

- [ ] **Step 3: Commit**

```bash
git add api/tests/test_integration.py
git commit -m "test: add integration tests for health and full pipeline run"
```

---

## Done

The backend API is complete when:
- `uv run pytest api/tests/ -m "not integration" -v` — all PASS
- `uv run uvicorn api.main:app --port 8000` starts with no errors
- `curl http://localhost:8000/health` returns `{"status":"ok","mlflow":true,"graph":true}`
- `curl http://localhost:8000/experiments` returns a JSON array

Next sub-project: **Next.js frontend** (consumes this API via WebSocket + REST).
