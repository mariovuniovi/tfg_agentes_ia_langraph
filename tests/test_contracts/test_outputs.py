"""Unit tests for node→state update contracts."""

import pytest
from pydantic import ValidationError

from mlops_agents.contracts.outputs import StateUpdate


class _Sample(StateUpdate):
    foo: str = "x"


def test_to_update_returns_plain_dict():
    assert _Sample(foo="hello").to_update() == {"foo": "hello"}


def test_to_update_merges_messages_when_provided():
    assert _Sample(foo="hello").to_update(messages=["m1"]) == {"foo": "hello", "messages": ["m1"]}


def test_to_update_omits_messages_key_when_none():
    assert "messages" not in _Sample(foo="hello").to_update()


def test_extra_keys_are_forbidden():
    with pytest.raises(ValidationError):
        _Sample(foo="hello", bogus=1)


from mlops_agents.contracts.outputs import (
    AuditStateUpdate,
    DataValidationStateUpdate,
    DatasetApprovalStateUpdate,
    DeploymentApprovalStateUpdate,
    DeploymentStateUpdate,
    EvaluationStateUpdate,
    PlannerErrorStateUpdate,
    PlannerStateUpdate,
    TrainingStateUpdate,
)
from mlops_agents.contracts.training import TrainingResult


def test_evaluation_contract_accepts_helper_dict_shape():
    helper_dict = {
        "evaluation_passed": True,
        "candidate_metrics": {"rmse": 1.0},
        "champion_metrics": {"rmse": 2.0},
        "thresholds_applied": {"min_delta": 0.0},
        "evaluation_report": {"candidate_metrics": {"rmse": 1.0}},
    }
    update = EvaluationStateUpdate(**helper_dict).to_update()
    assert update["evaluation_passed"] is True
    assert update["evaluation_report"]["candidate_metrics"] == {"rmse": 1.0}


def test_audit_contract_accepts_helper_dict_shape():
    helper_dict = {"evaluation_report_audit": {"x": 1}, "evaluation_report_audit_status": "ok"}
    assert AuditStateUpdate(**helper_dict).to_update() == helper_dict


def test_deployment_contract_accepts_helper_dict_shape():
    helper_dict = {
        "deployment_status": "deployed",
        "deployment_decision": "deployed",
        "best_model_uri": "models:/m/1",
    }
    assert DeploymentStateUpdate(**helper_dict).to_update() == helper_dict


def test_training_contract_maps_result_fields():
    result = TrainingResult(
        champion_candidate={"model_key": "ridge"},
        champion_model_path="/tmp/model.pkl",
        train_pool_path="/tmp/train.csv",
        test_path="/tmp/test.csv",
        split_metadata_path="/tmp/split.json",
        mlflow_parent_run_id="run123",
        experience_record_path="/tmp/exp.json",
        champion_metrics={"rmse": 1.5},
    )
    update = TrainingStateUpdate.from_training_result(
        result, training_plan={"problem_type": "regression"}
    ).to_update()
    assert update["trained_model_path"] == "/tmp/model.pkl"
    assert update["training_run_id"] == "run123"
    assert update["training_metrics"] == {"rmse": 1.5}
    assert update["training_plan"] == {"problem_type": "regression"}


def test_planner_contract_emits_underscore_alias():
    update = PlannerStateUpdate(
        planner_status="ok",
        training_plan={"problem_type": "regression"},
        planner_output_record={"k": "v"},
    ).to_update()
    assert update["_planner_output_record"] == {"k": "v"}
    assert "planner_output_record" not in update


def test_planner_error_contract_keys():
    assert PlannerErrorStateUpdate(error_message="boom").to_update() == {
        "planner_status": "failed",
        "planner_retry_used": True,
        "error_message": "boom",
    }


def test_data_validation_contract_failure_variant_uses_defaults():
    update = DataValidationStateUpdate(
        validation_passed=False, error_message="bad", schema_json=""
    ).to_update()
    assert update["validation_passed"] is False
    assert update["error_message"] == "bad"
    assert update["data_join_plan"] is None
    assert update["data_join_evaluations"] == []
    assert update["dataset_rejection_comment"] == ""


def test_approval_contracts_keys():
    assert DatasetApprovalStateUpdate(
        dataset_approved=False, dataset_rejection_comment="fix"
    ).to_update() == {"dataset_approved": False, "dataset_rejection_comment": "fix"}
    assert DeploymentApprovalStateUpdate(deployment_approved=True).to_update() == {
        "deployment_approved": True
    }
