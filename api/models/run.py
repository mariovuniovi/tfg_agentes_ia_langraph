from typing import Any, Literal
from pydantic import BaseModel, ConfigDict


class RunCreate(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    dataset_paths: list[str]
    schema_json: str = ""


class RunStatus(BaseModel):
    run_id: str
    status: Literal["running", "awaiting_approval", "complete", "failed"]
    interrupt_value: dict[str, Any] | None = None


class PipelineEventModel(BaseModel):
    type: Literal[
        "routing", "tool_call", "tool_result", "agent_reasoning",
        "hitl_request", "run_complete",
    ]
    agent: str
    timestamp_ms: float
    data: dict[str, Any]


class HITLDecision(BaseModel):
    decision: Literal["approve", "reject"]
    reason: str = ""
    comment: str = ""
