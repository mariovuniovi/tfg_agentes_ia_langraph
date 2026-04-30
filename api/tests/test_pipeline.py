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


def _data_validation_interrupt_chunk():
    from unittest.mock import MagicMock
    interrupt = MagicMock()
    interrupt.value = {
        "type": "data_validation",
        "question": "Review dataset",
        "attempt": 1,
        "dataset_preview": {"shape": [10, 3], "columns": [], "sample_rows": []},
        "validation_summary": {"passed": True, "missing_values": {}, "schema_validated": True},
    }
    return ("ns", "updates", {"__interrupt__": [interrupt]})


@pytest.mark.asyncio
async def test_hitl_request_event_agent_derived_from_payload_type(mock_graph, tmp_path):
    """hitl_request event agent field should be 'data_validation', not hardcoded 'deployer'."""
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,2\n")
    call_count = 0

    async def fake_astream(*a, **kw):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield _data_validation_interrupt_chunk()
        else:
            yield _run_complete_chunk()

    mock_graph.astream = fake_astream

    with patch("api.services.pipeline.graph", mock_graph):
        run_id = "test-label"
        entry = create_entry(run_id, {"configurable": {"thread_id": run_id}})

        async def approve_later():
            await asyncio.sleep(0.01)
            entry.hitl_decision = "approve"
            entry.hitl_comment = ""
            entry.hitl_event.set()

        await asyncio.gather(pipeline_task(run_id, [str(csv)]), approve_later())

    hitl_events = [e for e in entry.events if e["type"] == "hitl_request"]
    assert hitl_events[0]["agent"] == "data_validation"


@pytest.mark.asyncio
async def test_pipeline_resumes_with_dict_containing_approved_and_comment(mock_graph, tmp_path):
    """pipeline_task must resume with {"approved": bool, "comment": str}, not a raw string."""
    from langgraph.types import Command
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,2\n")
    call_count = 0
    resume_value = {}

    async def fake_astream(source, *a, **kw):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield _interrupt_chunk()
        else:
            if isinstance(source, Command):
                resume_value["value"] = source.resume
            yield _run_complete_chunk()

    mock_graph.astream = fake_astream

    with patch("api.services.pipeline.graph", mock_graph):
        run_id = "test-resume-dict"
        entry = create_entry(run_id, {"configurable": {"thread_id": run_id}})

        async def approve_later():
            await asyncio.sleep(0.01)
            entry.hitl_decision = "approve"
            entry.hitl_comment = "looks good"
            entry.hitl_event.set()

        await asyncio.gather(pipeline_task(run_id, [str(csv)]), approve_later())

    assert resume_value["value"] == {"approved": True, "comment": "looks good"}


@pytest.mark.asyncio
async def test_pipeline_handles_two_hitl_rounds(mock_graph, tmp_path):
    """pipeline_task while loop must handle two consecutive HITL interrupts."""
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,2\n")
    call_count = 0

    async def fake_astream(*a, **kw):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield _data_validation_interrupt_chunk()
        elif call_count == 2:
            yield _interrupt_chunk()
        else:
            yield _run_complete_chunk()

    mock_graph.astream = fake_astream

    with patch("api.services.pipeline.graph", mock_graph):
        run_id = "test-two-hitl"
        entry = create_entry(run_id, {"configurable": {"thread_id": run_id}})
        approval_count = 0

        async def approve_later():
            nonlocal approval_count
            while True:
                await asyncio.sleep(0.01)
                if entry.status == "awaiting_approval":
                    entry.hitl_decision = "approve"
                    entry.hitl_comment = ""
                    entry.hitl_event.set()
                    approval_count += 1
                    if approval_count >= 2:
                        break

        await asyncio.gather(pipeline_task(run_id, [str(csv)]), approve_later())

    assert entry.status == "complete"
    assert approval_count == 2
