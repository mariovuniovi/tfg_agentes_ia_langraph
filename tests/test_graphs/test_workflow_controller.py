import pytest
from langgraph.graph import END
from mlops_agents.graphs.workflow_controller import workflow_controller


def _state(**overrides):
    base = {
        "error_message": "",
        "validation_passed": False,
        "dataset_approved": None,
        "dataset_rejection_comment": "",
        "training_plan": None,
        "training_run_id": "",
        "evaluation_passed": None,
        "evaluation_report_audit": None,
        "deployment_approved": None,
        "deployment_decision": "pending",
        "agent_attempt_counts": {},
    }
    base.update(overrides)
    return base


def test_routes_to_data_validator_when_validation_not_passed():
    cmd = workflow_controller(_state())
    assert cmd.goto == "data_validator"
    assert cmd.update["agent_attempt_counts"]["data_validator"] == 1


def test_increments_data_validator_attempt_counter_on_each_route():
    cmd = workflow_controller(_state(agent_attempt_counts={"data_validator": 1}))
    assert cmd.goto == "data_validator"
    assert cmd.update["agent_attempt_counts"]["data_validator"] == 2


def test_aborts_when_error_message_set():
    cmd = workflow_controller(_state(error_message="boom"))
    assert cmd.goto == END


def test_aborts_when_data_validator_exhausted_max_attempts():
    cmd = workflow_controller(_state(agent_attempt_counts={"data_validator": 3}))
    assert cmd.goto == END
    assert cmd.update["error_message"]


def test_routes_to_dataset_approval_after_validation():
    cmd = workflow_controller(_state(validation_passed=True))
    assert cmd.goto == "dataset_approval"


def test_routes_back_to_data_validator_on_rejection_with_retry_left():
    cmd = workflow_controller(_state(
        validation_passed=True,
        dataset_approved=False,
        agent_attempt_counts={"data_validator": 1},
    ))
    assert cmd.goto == "data_validator"
    assert cmd.update["dataset_approved"] is None
    assert cmd.update["validation_passed"] is False
    assert cmd.update["agent_attempt_counts"]["data_validator"] == 2


def test_aborts_when_dataset_rejected_max_attempts_reached():
    cmd = workflow_controller(_state(
        validation_passed=True,
        dataset_approved=False,
        agent_attempt_counts={"data_validator": 3},
    ))
    assert cmd.goto == END
    assert "Dataset rejected" in cmd.update["error_message"]


def test_routes_to_planner_after_dataset_approved():
    cmd = workflow_controller(_state(
        validation_passed=True,
        dataset_approved=True,
    ))
    assert cmd.goto == "planner"


def test_routes_to_executor_after_plan_exists():
    cmd = workflow_controller(_state(
        validation_passed=True,
        dataset_approved=True,
        training_plan={"candidates": []},
    ))
    assert cmd.goto == "executor"


def test_routes_to_evaluation_after_training():
    cmd = workflow_controller(_state(
        validation_passed=True,
        dataset_approved=True,
        training_plan={"candidates": []},
        training_run_id="run-1",
    ))
    assert cmd.goto == "evaluation"


def test_routes_to_report_writer_after_evaluation():
    cmd = workflow_controller(_state(
        validation_passed=True,
        dataset_approved=True,
        training_plan={"candidates": []},
        training_run_id="run-1",
        evaluation_passed=True,
    ))
    assert cmd.goto == "report_writer"


def test_skips_gate2_when_evaluation_failed():
    cmd = workflow_controller(_state(
        validation_passed=True,
        dataset_approved=True,
        training_plan={"candidates": []},
        training_run_id="run-1",
        evaluation_passed=False,
        evaluation_report_audit={"summary": "rejected"},
    ))
    assert cmd.goto == END


def test_routes_to_deployment_approval_when_passed():
    cmd = workflow_controller(_state(
        validation_passed=True,
        dataset_approved=True,
        training_plan={"candidates": []},
        training_run_id="run-1",
        evaluation_passed=True,
        evaluation_report_audit={"summary": "ok"},
    ))
    assert cmd.goto == "deployment_approval"


def test_aborts_when_deployment_rejected():
    cmd = workflow_controller(_state(
        validation_passed=True,
        dataset_approved=True,
        training_plan={"candidates": []},
        training_run_id="run-1",
        evaluation_passed=True,
        evaluation_report_audit={"summary": "ok"},
        deployment_approved=False,
    ))
    assert cmd.goto == END


def test_routes_to_deployer_when_approved():
    cmd = workflow_controller(_state(
        validation_passed=True,
        dataset_approved=True,
        training_plan={"candidates": []},
        training_run_id="run-1",
        evaluation_passed=True,
        evaluation_report_audit={"summary": "ok"},
        deployment_approved=True,
    ))
    assert cmd.goto == "deployer"


def test_ends_after_deployment_complete():
    cmd = workflow_controller(_state(
        validation_passed=True,
        dataset_approved=True,
        training_plan={"candidates": []},
        training_run_id="run-1",
        evaluation_passed=True,
        evaluation_report_audit={"summary": "ok"},
        deployment_approved=True,
        deployment_decision="approved",
    ))
    assert cmd.goto == END
