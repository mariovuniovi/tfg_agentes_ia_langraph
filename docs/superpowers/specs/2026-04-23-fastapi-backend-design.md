# FastAPI Backend API — Design Spec

**Date:** 2026-04-23
**Status:** Approved
**Branch:** `feature/fastapi-backend` (forked from `claude/develop`)
**Sub-project:** 1 of 2 — Backend API (Next.js frontend is sub-project 2)

---

## Problem

The current Streamlit dashboard runs the LangGraph pipeline, MLflow queries, and Evidently drift detection entirely in-process. There is no HTTP API, making it impossible for a decoupled frontend (Next.js) to consume pipeline results, stream live agent events, or trigger HITL decisions independently.

## Goal

Expose the existing MLOps pipeline, MLflow data, and Evidently drift detection as a typed FastAPI HTTP + WebSocket API. The Streamlit dashboard remains untouched and runnable during migration. The new API lives in `api/` and is the single backend the Next.js frontend will target.

---

## Decisions

| Question | Decision |
|---|---|
| Streaming mechanism | WebSockets (bidirectional — required for HITL) |
| Pipeline execution | asyncio background task — no Celery, no Redis |
| MLflow | Proxied through FastAPI with typed Pydantic responses shaped for charts |
| Monitoring — automatic | Evidently runs after each pipeline run; stored in module-level `latest_drift_report` |
| Monitoring — ad-hoc | Multi-file upload; user picks reference + current by index; no server-side storage |
| Authentication | None (thesis demo) |
| CORS | `localhost:3000` (Next.js dev) allowed |

---

## Directory Structure

```
api/
├── main.py                  # FastAPI app, CORS, router registration, lifespan
├── routers/
│   ├── runs.py              # POST /runs, WS /ws/{run_id}, POST /runs/{id}/approve, GET /runs/{id}/events
│   ├── experiments.py       # GET /experiments, GET /experiments/{exp_id}/runs
│   └── monitoring.py        # GET /monitoring/latest, POST /monitoring/drift
├── models/
│   ├── run.py               # RunCreate, RunStatus, PipelineEvent, HITLDecision
│   ├── experiment.py        # ExperimentOut, RunOut, MetricSeries
│   └── monitoring.py        # DriftReport, ColumnDriftResult
├── services/
│   ├── run_store.py         # In-memory RunEntry store + module-level latest_drift_report
│   ├── pipeline.py          # async background task: graph.astream() → queue
│   └── mlflow_client.py     # Thin wrapper over MlflowClient, returns typed models
└── tests/
    ├── conftest.py           # shared fixtures (mock graph, mock MLflow)
    └── test_runs.py          # unit tests (mocked LLM), integration tests (@pytest.mark.integration)
```

**New dependencies** (added to `pyproject.toml`):
- `fastapi`
- `uvicorn[standard]`
- `python-multipart` (file uploads)

Everything else (`mlflow`, `evidently`, `langgraph`, `loguru`, `pydantic-settings`) is already present.

**Run command:** `uv run uvicorn api.main:app --reload --port 8000`

---

## Core State — `RunStore`

```python
# api/services/run_store.py
@dataclass
class RunEntry:
    run_id: str
    status: Literal["running", "awaiting_approval", "complete", "failed"]
    queue: asyncio.Queue           # PipelineEvent dicts → WebSocket clients
    graph_config: dict             # LangGraph thread config {configurable: {thread_id: run_id}}
    hitl_event: asyncio.Event      # set() by POST /approve; awaited by pipeline_task
    hitl_decision: str             # "approve" | "reject"
    events: list[PipelineEvent]    # full event log for GET /runs/{id}/events
    interrupt_value: dict          # payload captured from graph interrupt()
    last_drift_report: dict | None # populated at run completion

_store: dict[str, RunEntry] = {}
latest_drift_report: dict | None = None   # module-level; survives RunEntry cleanup
```

- `run_id` is `uuid4()` generated at `POST /runs`
- `hitl_event` is an `asyncio.Event` — pipeline task `await`s it at the interrupt; `POST /approve` calls `.set()`
- `events` list is append-only; returned verbatim by `GET /runs/{id}/events`
- `latest_drift_report` is overwritten on every successful run completion

---

## API Endpoints

### Runs

**`POST /runs`**
- Body: `RunCreate { dataset_paths: list[str] }` (list — supports multi-file pipelines)
- Creates `RunEntry`, spawns `pipeline_task` via `BackgroundTasks`
- Returns: `{ run_id: str }`

**`WS /ws/{run_id}`**
- Accepts WebSocket connection
- Drains `entry.queue` and forwards each `PipelineEvent` as JSON
- Closes when `{"type": "run_complete"}` is sent
- Frontend reconnects if disconnected mid-run (queue is persistent in-memory)

**`POST /runs/{run_id}/approve`**
- Body: `HITLDecision { decision: Literal["approve", "reject"], reason: str = "" }`
- Validates `entry.status == "awaiting_approval"`, sets `hitl_decision`, calls `hitl_event.set()`
- Returns: `{ ok: true }`
- Raises `400` if run is not awaiting approval

**`GET /runs/{run_id}`**
- Returns `RunStatus { run_id, status, interrupt_value? }` — used by frontend after WS reconnection to recover current run state without re-streaming all events
- Raises `404` if `run_id` unknown

**`GET /runs/{run_id}/events`**
- Returns full `list[PipelineEvent]` for a run (used by Logs page)
- Raises `404` if `run_id` unknown

**`GET /health`**
- Checks MLflow reachable and graph importable
- Returns: `{ status: "ok", mlflow: bool, graph: bool }`

### Experiments

**`GET /experiments`**
- Returns `list[{ experiment_id, name }]` — selector data only

**`GET /experiments/{exp_id}/runs`**
- Returns `list[RunOut]` sorted by `start_time DESC`, max 50 runs
- `RunOut` includes:
  - `metrics: dict[str, float]` — final scalar values (Bar + Radar charts)
  - `metric_series: list[MetricSeries]` — step arrays from `get_metric_history()` (Line chart)
- `MetricSeries.line_style` cycles `solid → dashed → dotted` per series index (a11y: not color alone)

### Monitoring

**`GET /monitoring/latest`**
- Returns `DriftReport` from the most recent completed pipeline run
- Raises `404` if no run has completed

**`POST /monitoring/drift`**
- Form data: `files: list[UploadFile]`, `reference_index: int`, `current_index: int`
- Loads only the two selected files into DataFrames; discards the rest (no server-side storage)
- Runs `DataDriftPreset` via Evidently
- Returns `DriftReport`

---

## Pydantic Models

### `PipelineEvent`
Reused from `dashboard/pipeline_helpers.py` — same TypedDict, re-exported as Pydantic model:

```python
class PipelineEvent(BaseModel):
    type: Literal["routing", "tool_call", "tool_result", "agent_reasoning",
                  "hitl_request", "run_complete"]
    agent: str
    timestamp_ms: float
    data: dict
```

### `MetricSeries`
```python
class MetricSeries(BaseModel):
    name: str
    steps: list[int]
    values: list[float]
    line_style: Literal["solid", "dashed", "dotted"]
```

### `DriftReport`
```python
class ColumnDriftResult(BaseModel):
    column: str
    drift_detected: bool
    score: float
    method: str

class DriftReport(BaseModel):
    dataset_drift: bool
    drift_share: float        # fraction of columns with drift
    columns: list[ColumnDriftResult]
    generated_at: datetime
```

---

## Pipeline Task Flow

```
POST /runs
  └─ BackgroundTasks.add_task(pipeline_task, run_id, dataset_path)
       └─ graph.astream(initial_state, config, stream_mode=["updates","messages"])
            ├─ parse_stream_event(chunk) → PipelineEvent
            ├─ entry.events.append(event)
            ├─ await entry.queue.put(event)           → WS client receives it
            │
            ├─ [interrupt detected]
            │    entry.status = "awaiting_approval"
            │    await entry.queue.put({type:"hitl_request", data:interrupt_value})
            │    await entry.hitl_event.wait()        ← blocks here
            │                                            until POST /approve calls .set()
            │    graph.astream(..., Command(resume=entry.hitl_decision))
            │
            └─ [graph complete]
                 # training_df and input_df are extracted from final graph state:
                 # graph.aget_state(config).values["dataset_preview"] → input_df
                 # graph.aget_state(config).values["training_metrics"] carries split info
                 # pipeline_task captures both DataFrames before streaming run_complete
                 run_evidently(training_df, input_df) → DriftReport
                 entry.last_drift_report = report
                 run_store.latest_drift_report = report
                 await entry.queue.put({type:"run_complete", data:report})
                 entry.status = "complete"
```

---

## HITL Flow Detail

1. `pipeline_task` detects interrupt payload via `graph.get_state(config).next == ()`
2. Sets `entry.status = "awaiting_approval"` and `entry.interrupt_value = payload`
3. Sends `{"type": "hitl_request", "data": payload}` over WebSocket
4. Calls `await entry.hitl_event.wait()` — pipeline task is suspended, event loop is free
5. Frontend shows approval panel; user clicks approve or reject
6. Frontend calls `POST /runs/{run_id}/approve` with `{"decision": "approve"}`
7. Handler sets `entry.hitl_decision = "approve"`, calls `entry.hitl_event.set()`
8. `pipeline_task` unblocks, resumes graph with `Command(resume="approve")`
9. WebSocket keeps streaming post-approval events until `run_complete`

---

## Chart Data Mapping

| Agent | Metric | Chart | Endpoint | Field |
|---|---|---|---|---|
| `trainer` | Loss / accuracy over steps | Line Chart | `GET /experiments/{id}/runs` | `metric_series` |
| `evaluator` | Precision / recall / F1 / AUC | Radar + Horiz. Bar | `GET /experiments/{id}/runs` | `metrics` |
| `deployment` | Model version comparison | Horiz. Bar (sorted desc) | `GET /experiments/{id}/runs` | `metrics` |
| `data_validator` | Drift share over runs | Area Chart | `GET /monitoring/latest` | `drift_share` |

Chart accessibility rules (from ui-ux-pro-max search):
- Multi-series line: cycle `solid/dashed/dotted` per series — not color alone
- Radar: grouped bar chart always rendered alongside as fallback (a11y grade B)
- Bar: value labels on every bar by default; CSV export button in `#D97706`

---

## Testing Strategy

- **Unit tests** (`uv run pytest -m "not integration"`): mock `graph.astream()` with `unittest.mock.AsyncMock`; mock `MlflowClient`; test queue draining, HITL event flow, Pydantic model shapes
- **Integration tests** (`@pytest.mark.integration`): real graph call on iris dataset; real MLflow; asserts endpoint response schemas
- Follows existing `tests/conftest.py` fixture patterns

---

## Out of Scope

- Persistent run storage (SQLite / DB) — in-memory only; runs are lost on server restart
- Authentication / API keys
- Multiple concurrent pipeline runs (queue per `run_id` supports it, but not tested)
- Next.js frontend (sub-project 2)
- Deployment / Docker changes (existing `docker-compose.yml` unchanged)
