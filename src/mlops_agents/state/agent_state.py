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
    dataset_paths: list[str]   # raw CSV files provided by user
    dataset_path: str          # canonical CSV written by data_validator_node

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
    agent_attempt_counts: dict[str, int]  # {"data_validator": 1, "trainer": 2, …}

    # Context isolation — built deterministically by data_validator_node
    dataset_summary: dict  # {row_count, column_names, dtypes, null_counts}

    # Task type — written once by data_validator_node before agent invocation
    problem_type: str   # "classification" | "regression" | "forecasting"

    # Task-level metadata — written once by data_validator_node
    task_metadata: dict
    schema_json: str
    # classification/regression: {"target_column": str}
    # forecasting: {
    #   "target_column": str,
    #   "datetime_column": str,
    #   "series_id_columns": list[str],
    #   "forecast_horizon": int,
    #   "frequency": str,
    # }
