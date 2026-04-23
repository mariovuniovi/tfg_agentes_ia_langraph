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
