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
    hitl_comment: str = ""
    events: list[dict] = field(default_factory=list)
    interrupt_value: dict[str, Any] = field(default_factory=dict)
    processed_dataset_path: str | None = None
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
