"""Runs router: pipeline execution, WebSocket streaming, HITL approval."""
import asyncio
import contextlib
from uuid import uuid4

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

import api.services.run_store as run_store
from api.models.run import HITLDecision, RunCreate, RunStatus
from api.services.pipeline import pipeline_task
from mlops_agents.config.constants import GRAPH_RECURSION_LIMIT

router = APIRouter()


@router.post("/runs")
async def start_run(body: RunCreate, background_tasks: BackgroundTasks):
    run_id = str(uuid4())
    config = {"configurable": {"thread_id": run_id}, "recursion_limit": GRAPH_RECURSION_LIMIT}
    run_store.create_entry(run_id, config)
    background_tasks.add_task(pipeline_task, run_id, body.dataset_paths, body.schema_json)
    return {"run_id": run_id}


@router.get("/runs")
def list_runs(limit: int = 20):
    out = []
    for e in run_store.list_entries(limit=limit):
        out.append({
            "run_id": e.run_id,
            "status": e.status,
            "started_at_ms": getattr(e, "started_at_ms", 0),
        })
    return out


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
    entry.hitl_comment = body.comment
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
    # Deliver from the authoritative append-only log (entry.events) via a cursor,
    # using entry.queue only as a "new event arrived" doorbell. This makes reconnects
    # replay missed events and never drops an in-flight event when a transient send
    # fails on a stale socket (e.g. during the executor's long blocking training).
    # Each event carries its index as `seq` so the client can dedup replayed events.
    cursor = 0
    try:
        while True:
            while cursor < len(entry.events):
                event = entry.events[cursor]
                await websocket.send_json({**event, "seq": cursor})
                cursor += 1
                if event.get("type") == "run_complete":
                    return
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(entry.queue.get(), timeout=1.0)
    except WebSocketDisconnect:
        pass


@router.get("/runs/{run_id}/dataset-preview")
def dataset_preview(run_id: str, limit: int = 50, offset: int = 0):
    entry = run_store.get_entry(run_id)
    if entry is None:
        raise HTTPException(404, "run not found")
    path = entry.processed_dataset_path
    if not path:
        raise HTTPException(409, "no processed dataset yet")
    df = pd.read_csv(path)
    total = len(df)
    rows = df.iloc[offset : offset + limit].to_dict(orient="records")
    columns = [
        {
            "name": c,
            "dtype": str(df[c].dtype),
            "non_null_count": int(df[c].notna().sum()),
            "sample_value": (lambda v: v.item() if hasattr(v, "item") else v)(
                df[c].dropna().iloc[0]
            ) if not df[c].dropna().empty else None,
        }
        for c in df.columns
    ]
    return {"columns": columns, "rows": rows, "total_rows": total}


@router.get("/runs/{run_id}/dataset-download")
def dataset_download(run_id: str):
    entry = run_store.get_entry(run_id)
    if entry is None:
        raise HTTPException(404, "run not found")
    path = entry.processed_dataset_path
    if not path:
        raise HTTPException(409, "no processed dataset yet")

    def iter_file():
        with open(path, "rb") as f:
            yield from f

    return StreamingResponse(
        iter_file(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="run-{run_id}.csv"'},
    )
