"""ToolTrace — records what the planner agent observed during a single run."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolTrace(BaseModel):
    called_tools: list[str] = Field(default_factory=list)
    listed_model_keys: list[str] = Field(default_factory=list)
    retrieved_experience_ids: list[str] = Field(default_factory=list)
    retrieved_rule_ids: list[str] = Field(default_factory=list)
    inspected_model_keys: list[str] = Field(default_factory=list)
    inspect_model_details_count: int = 0  # CALL count — NOT len(inspected_model_keys),
                                          # because agents can call inspect_model_details("ets") 4x
                                          # and unique-set length stays at 1 (cap would never fire).
    tool_call_count: int = 0
    raw_observations: list[dict[str, Any]] = Field(default_factory=list)
