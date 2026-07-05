"""Tests for in-memory RunStore."""
import asyncio

from api.services.run_store import RunEntry, create_entry, get_entry


def _make_entry(run_id: str = "test-run") -> RunEntry:
    return create_entry(run_id, graph_config={"configurable": {"thread_id": run_id}})


def test_create_entry_sets_defaults():
    entry = _make_entry("r1")
    assert entry.run_id == "r1"
    assert entry.status == "running"
    assert isinstance(entry.queue, asyncio.Queue)
    assert isinstance(entry.hitl_event, asyncio.Event)
    assert entry.events == []


def test_get_entry_returns_none_for_unknown():
    assert get_entry("nonexistent") is None


def test_create_and_retrieve_entry():
    entry = _make_entry("r2")
    assert get_entry("r2") is entry


def test_run_entry_has_hitl_comment_field():
    entry = _make_entry("r-comment")
    assert entry.hitl_comment == ""
