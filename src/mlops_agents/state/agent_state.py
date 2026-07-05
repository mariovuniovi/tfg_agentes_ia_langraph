"""Shared LangGraph state definition for the MLOps pipeline.

The ``AgentState`` TypedDict is the single object threaded through every node in
the graph. Fields below are grouped by the node that *produces* them, in
pipeline execution order. Each node returns a partial dict that overwrites the
keys it owns; the only accumulating field is ``messages`` (operator.add reducer).
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Shared state passed between all nodes in the MLOps graph.

    Convention: every field overwrites on update except ``messages``, which
    accumulates via the ``operator.add`` reducer. Fields are grouped by their
    producing node (see the section comments below).
    """

    # === Framework / routing ===
    messages: Annotated[list[BaseMessage], operator.add]  # chat history; operator.add appends

    # === Pipeline inputs — set once at graph entry (build_initial_state) ===
    dataset_paths: list[str]   # raw CSV file paths provided by the user
    schema_json: str           # target SchemaContract serialised as JSON

    # === data_validator node — validation, cleaning, and (multi-file) join discovery ===
    processed_dataset_path: str    # canonical CSV written by data_validator_node
    validation_passed: bool
    validation_report: dict[str, Any]        # deterministic check_data_quality output (missing values, duplicates)
    dataset_summary: dict[str, Any]          # {row_count, column_names, dtypes, null_counts} — built deterministically
    problem_type: str              # "classification" | "regression" | "forecasting"
    task_metadata: dict[str, Any]            # target_column (+ datetime/series/horizon/frequency for forecasting)
    # Join discovery — only populated when multiple raw files are provided
    data_join_plan: dict[str, Any] | None            # selected JoinPlan audit (joins, rejected candidates, warnings)
    data_join_base_nrows: int | None       # row count of the base dataset before any joins
    data_join_evaluations: list[dict[str, Any]]      # per-candidate coverage / cardinality metrics

    # === dataset_approval node — HITL gate 1 ===
    dataset_approved: bool | None
    dataset_rejection_comment: str         # operator feedback injected back to data_validator on retry

    # === planner node — ReAct agent + structured PlannerOutput ===
    training_plan: dict[str, Any] | None             # Pydantic-dumped TrainingPlan consumed by the executor
    planner_analysis: str | None           # LLM-generated planning explanation artifact
    planner_evidence_used: list[dict[str, Any]]      # EvidenceReference dicts (experiences + rules cited)
    planner_warnings: list[str]            # warning strings raised during planning
    planner_status: str | None             # "ok" | "retry_ok" | "failed"
    planner_retry_used: bool | None        # True if a second attempt was needed
    planner_tool_trace: dict[str, Any]               # tool invocations and results during planning
    planner_validation_context: dict[str, Any]       # constraints / context surfaced to the planner
    _planner_output_record: dict[str, Any] | None    # private: full planner output record read by the executor

    # === executor node — deterministic training (Optuna + MLflow) ===
    train_pool_path: str | None
    test_path: str | None
    split_metadata_path: str | None
    trained_model_path: str
    training_run_id: str                   # MLflow parent run ID
    training_metrics: dict[str, Any]                 # champion candidate metrics
    champion_candidate: dict[str, Any] | None        # winning candidate spec
    experience_record_path: str | None     # JSON experience record serialised to disk
    forecast_chart_png: str | None         # base64 PNG chart; only set for forecasting runs
    selection_score: float | None          # validation score the champion was selected on (forecasting)

    # === evaluation node — deterministic promotion decision ===
    evaluation_passed: bool | None
    evaluation_report: dict[str, Any]
    candidate_metrics: dict[str, Any]                # metrics of the new candidate
    champion_metrics: dict[str, Any]                 # metrics of the current production champion
    thresholds_applied: dict[str, Any]               # promotion thresholds used for the decision

    # === report_writer node — LLM audit report ===
    evaluation_report_audit: dict[str, Any] | None   # structured EvaluationReport (audit narrative)
    evaluation_report_audit_status: str    # "ok" | "retry_ok" | "stub"

    # === deployment_approval node — HITL gate 2 ===
    deployment_approved: bool | None

    # === deployer node — MLflow Model Registry promotion ===
    deployment_decision: str               # "approved" | "rejected" | "pending" | "deployed"
    deployment_status: str
    best_model_uri: str                    # models:/<name>/<version>

    # === Cross-cutting — written by any node ===
    error_message: str
    agent_attempt_counts: dict[str, int]   # {"data_validator": 1, "trainer": 2, …}
