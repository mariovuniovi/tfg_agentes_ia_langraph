"""Unit tests for pipeline_task — graph is fully mocked."""
import asyncio
from unittest.mock import patch

import pytest

from api.services.pipeline import pipeline_task
from api.services.run_store import create_entry


def _routing_chunk():
    return ("ns", "updates", {"supervisor": {"next": "trainer", "reasoning": "needs training"}})


def _messages_chunk():
    from langchain_core.messages import AIMessageChunk
    msg = AIMessageChunk(content="Analysing data...")
    meta = {"langgraph_node": "trainer"}
    return ("ns", "messages", (msg, meta))


def _interrupt_chunk():
    from unittest.mock import MagicMock
    interrupt = MagicMock()
    interrupt.value = {"model_uri": "runs:/abc/model"}
    return ("ns", "updates", {"__interrupt__": [interrupt]})


def _run_complete_chunk():
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

    call_count = 0

    async def fake_astream(*a, **kw):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield _interrupt_chunk()
        else:
            yield _run_complete_chunk()

    mock_graph.astream = fake_astream

    with patch("api.services.pipeline.graph", mock_graph), \
         patch("api.services.pipeline.run_evidently", return_value={}):
        run_id = "test-hitl"
        entry = create_entry(run_id, {"configurable": {"thread_id": run_id}})

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
