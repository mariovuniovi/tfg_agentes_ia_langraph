"""Runs router: pipeline execution, WebSocket streaming, HITL approval."""
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect

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
    try:
        while True:
            event = await entry.queue.get()
            await websocket.send_json(event)
            if event.get("type") == "run_complete":
                break
    except WebSocketDisconnect:
        pass
