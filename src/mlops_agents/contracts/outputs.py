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

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # annotation-only; avoids any contracts/ import cycle
    from mlops_agents.contracts.training import TrainingResult


class StateUpdate(BaseModel):
    """Base for all node→state update contracts."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    def to_update(self, messages: list[Any] | None = None) -> dict[str, Any]:
        update = self.model_dump(by_alias=True)
        if messages:
            update["messages"] = messages
        return update


class DataValidationStateUpdate(StateUpdate):
    """data_validator node — covers all three return paths (two early-error + main).

    All fields default to their reset value, so the early-error paths build the
    same contract with only error_message/schema_json set.
    """

    validation_passed: bool = False
    validation_report: dict = Field(default_factory=dict)
    processed_dataset_path: str = ""
    dataset_summary: dict = Field(default_factory=dict)
    problem_type: str = ""
    task_metadata: dict = Field(default_factory=dict)
    schema_json: str = ""
    data_join_plan: dict | None = None
    data_join_base_nrows: int | None = None
    data_join_evaluations: list[dict] = Field(default_factory=list)
    error_message: str = ""
    dataset_rejection_comment: str = ""


class DatasetApprovalStateUpdate(StateUpdate):
    """dataset_approval HITL gate (gate 1)."""

    dataset_approved: bool | None = None
    dataset_rejection_comment: str = ""


class PlannerStateUpdate(StateUpdate):
    """planner node — success path."""

    planner_analysis: str | None = None
    planner_evidence_used: list[dict] = Field(default_factory=list)
    planner_warnings: list[str] = Field(default_factory=list)
    planner_status: str | None = None
    planner_retry_used: bool | None = None
    training_plan: dict | None = None
    planner_tool_trace: dict = Field(default_factory=dict)
    planner_validation_context: dict = Field(default_factory=dict)
    # State key has a leading underscore — emit via serialization alias.
    planner_output_record: dict | None = Field(
        default=None, serialization_alias="_planner_output_record"
    )


class PlannerErrorStateUpdate(StateUpdate):
    """planner error wrapper — when planning fails after retry."""

    planner_status: str = "failed"
    planner_retry_used: bool = True
    error_message: str = ""


class TrainingStateUpdate(StateUpdate):
    """executor node — maps TrainingResult → state keys."""

    training_plan: dict | None = None
    train_pool_path: str | None = None
    test_path: str | None = None
    split_metadata_path: str | None = None
    trained_model_path: str = ""
    training_run_id: str = ""
    training_metrics: dict = Field(default_factory=dict)
    champion_candidate: dict | None = None
    experience_record_path: str | None = None
    forecast_chart_png: str | None = None

    @classmethod
    def from_training_result(
        cls, result: TrainingResult, *, training_plan: dict
    ) -> "TrainingStateUpdate":
        return cls(
            training_plan=training_plan,
            train_pool_path=result.train_pool_path,
            test_path=result.test_path,
            split_metadata_path=result.split_metadata_path,
            trained_model_path=result.champion_model_path,
            training_run_id=result.mlflow_parent_run_id,
            training_metrics=result.champion_metrics,
            champion_candidate=result.champion_candidate,
            experience_record_path=result.experience_record_path,
            forecast_chart_png=result.forecast_chart_png,
        )


class EvaluationStateUpdate(StateUpdate):
    """evaluation node — deterministic promotion decision."""

    evaluation_passed: bool | None = None
    candidate_metrics: dict = Field(default_factory=dict)
    champion_metrics: dict = Field(default_factory=dict)
    thresholds_applied: dict = Field(default_factory=dict)
    evaluation_report: dict = Field(default_factory=dict)


class AuditStateUpdate(StateUpdate):
    """report_writer node — LLM audit report."""

    evaluation_report_audit: dict | None = None
    evaluation_report_audit_status: str = ""


class DeploymentApprovalStateUpdate(StateUpdate):
    """deployment_approval HITL gate (gate 2)."""

    deployment_approved: bool | None = None


class DeploymentStateUpdate(StateUpdate):
    """deployer node — MLflow Model Registry promotion."""

    deployment_status: str = ""
    deployment_decision: str = ""
    best_model_uri: str = ""
