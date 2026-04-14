"""Shared LangGraph state definition for the MLOps pipeline."""

import operator
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """Shared state passed between all nodes in the MLOps graph.

    Fields updated by reducers (operator.add) accumulate across nodes.
    All other plain fields overwrite on update — this is intentional.
    """

    # Message history — operator.add appends instead of overwriting
    messages: Annotated[list[BaseMessage], operator.add]

    # Supervisor routing — which node to visit next
    next: str

    # Pipeline inputs
    dataset_path: str

    # Stage outputs (set by each agent node)
    validation_passed: bool
    validation_report: dict  # Evidently AI report as dict
    trained_model_path: str
    training_run_id: str      # MLflow run ID
    training_metrics: dict
    evaluation_passed: bool
    evaluation_report: dict
    best_model_uri: str

    # Deployment
    deployment_decision: str  # "approved" | "rejected" | "pending"
    deployment_status: str

    # Error tracking
    error_message: str
    retry_count: int
