"""Node→state update contracts.

Each contract's field names match ``AgentState`` keys exactly (the binding test
in tests/test_contracts/test_state_binding.py enforces this). A graph node builds
one contract for the state-slice it owns and writes it with ``.to_update()``:

    return Command(update=EvaluationStateUpdate(**evaluate_promotion(state)).to_update(),
                   goto="workflow_controller")

Design rules:
- ``extra="forbid"``: a stray/typo'd key (e.g. from a helper dict) fails loudly.
- every field is defaulted, so a node can build a partial/failure variant.
- ``to_update()`` uses ``by_alias=True`` so leading-underscore state keys
  (e.g. ``_planner_output_record``) are emitted via ``serialization_alias``.
- nodes that also append chat history pass ``messages=`` (merged via the reducer).
- domain modules (evaluation/, deployment/, training/) MUST NOT import this module;
  contract construction happens in the graph node layer only (Philosophy 2).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class StateUpdate(BaseModel):
    """Base for all node→state update contracts."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    def to_update(self, messages: list[Any] | None = None) -> dict[str, Any]:
        update = self.model_dump(by_alias=True)
        if messages:
            update["messages"] = messages
        return update
