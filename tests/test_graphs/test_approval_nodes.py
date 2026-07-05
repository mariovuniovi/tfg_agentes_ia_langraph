from unittest.mock import patch

from mlops_agents.graphs.approval_nodes import dataset_approval_node, deployment_approval_node


def test_dataset_approval_approved_clears_comment():
    with patch("mlops_agents.graphs.approval_nodes.interrupt") as mock_interrupt:
        mock_interrupt.return_value = {"approved": True, "comment": ""}
        cmd = dataset_approval_node({
            "dataset_summary": {"row_count": 100},
            "validation_report": {"passed": True},
        })

    assert cmd.goto == "workflow_controller"
    assert cmd.update["dataset_approved"] is True
    assert cmd.update["dataset_rejection_comment"] == ""


def test_dataset_approval_rejected_captures_comment():
    with patch("mlops_agents.graphs.approval_nodes.interrupt") as mock_interrupt:
        mock_interrupt.return_value = {"approved": False, "comment": "missing column X"}
        cmd = dataset_approval_node({
            "dataset_summary": {},
            "validation_report": {},
        })

    assert cmd.update["dataset_approved"] is False
    assert cmd.update["dataset_rejection_comment"] == "missing column X"


def test_dataset_approval_emits_data_validation_payload_type():
    captured = {}
    def fake_interrupt(payload):
        captured.update(payload)
        return {"approved": True}

    with patch("mlops_agents.graphs.approval_nodes.interrupt", side_effect=fake_interrupt):
        dataset_approval_node({"dataset_summary": {}, "validation_report": {}})

    assert captured["type"] == "data_validation"


def test_dataset_approval_payload_includes_attempt_count():
    captured = {}
    def fake_interrupt(payload):
        captured.update(payload)
        return {"approved": True}

    with patch("mlops_agents.graphs.approval_nodes.interrupt", side_effect=fake_interrupt):
        dataset_approval_node({
            "dataset_summary": {}, "validation_report": {},
            "agent_attempt_counts": {"data_validator": 2},
        })

    assert captured["attempt"] == 2


def test_deployment_approval_approved():
    with patch("mlops_agents.graphs.approval_nodes.interrupt") as mock_interrupt:
        mock_interrupt.return_value = {"approved": True}
        cmd = deployment_approval_node({
            "evaluation_report": {},
            "evaluation_report_audit": {"summary": "ok"},
        })

    assert cmd.goto == "workflow_controller"
    assert cmd.update["deployment_approved"] is True


def test_deployment_approval_rejected():
    with patch("mlops_agents.graphs.approval_nodes.interrupt") as mock_interrupt:
        mock_interrupt.return_value = {"approved": False, "reason": "low f1"}
        cmd = deployment_approval_node({
            "evaluation_report": {},
            "evaluation_report_audit": {},
        })

    assert cmd.update["deployment_approved"] is False
    assert "deployment_decision" not in cmd.update


def test_deployment_approval_emits_deployer_payload_type():
    captured = {}
    def fake_interrupt(payload):
        captured.update(payload)
        return {"approved": True}

    with patch("mlops_agents.graphs.approval_nodes.interrupt", side_effect=fake_interrupt):
        deployment_approval_node({"evaluation_report": {}, "evaluation_report_audit": {}})

    assert captured["type"] == "deployer"
