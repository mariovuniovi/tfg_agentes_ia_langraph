"""Shared evidence contract — extracted from planner.py so training.py can import it
without creating a circular import (planner.py imports CandidateSpec from training.py)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class EvidenceReference(BaseModel):
    """Reference to evidence used in a planning decision."""

    source: Literal[
        "dataset_profile",
        "task_metadata",
        "experience",
        "rule",
        "registry",
    ]
    source_id: str | None = None
    summary: str = ""
    relevance_note: str | None = None
