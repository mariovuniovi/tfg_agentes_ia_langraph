"""Shared LangGraph state definition for the MLOps pipeline."""

import operator
from typing import Annotated

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


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
    processed_dataset_path: str   # canonical CSV written by data_validator_node

    # Stage outputs (set by each agent node)
    validation_passed: bool
    validation_report: dict  # Evidently AI report as dict
    trained_model_path: str
    training_run_id: str      # MLflow run ID
    training_metrics: dict
    # SP3 training (new)
    training_plan: dict | None              # Pydantic-dumped TrainingPlan
    train_pool_path: str | None
    test_path: str | None
    split_metadata_path: str | None
    champion_candidate: dict | None
    experience_record_path: str | None
    evaluation_passed: bool | None
    evaluation_report: dict
    best_model_uri: str

    # Deployment
    deployment_decision: str  # "approved" | "rejected" | "pending"
    deployment_status: str

    # Gate 1 — dataset approval
    dataset_approved: bool | None
    dataset_rejection_comment: str

    # Gate 2 — deployment approval
    deployment_approved: bool | None

    # Deterministic evaluation outputs
    candidate_metrics: dict
    champion_metrics: dict
    thresholds_applied: dict

    # Audit LLM output
    evaluation_report_audit: dict | None
    evaluation_report_audit_status: str   # "ok" | "retry_ok" | "stub"

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

    # Join discovery outputs — written by data_validator_node after agent run
    data_join_plan: dict | None
    data_join_base_nrows: int | None  # row count of the base dataset before any joins
    data_join_evaluations: list[dict]

    # SP5 planner outputs
    planner_analysis: str | None            # LLM-generated planning explanation artifact
    planner_evidence_used: list[dict]       # list of EvidenceReference dicts
    planner_warnings: list[str]             # list of warning strings
    planner_status: str | None              # "ok" | "retry_ok" | "failed"
    planner_retry_used: bool | None         # True if second attempt was needed
    _planner_output_record: dict | None     # private state key used by executor to get planner's output record
    planner_tool_trace: dict                # tool invocations and results during planning
    planner_validation_context: dict        # validation constraints and context for planning
