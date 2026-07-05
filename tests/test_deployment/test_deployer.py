import json
from unittest.mock import patch

from mlops_agents.deployment.deployer import run_deployer


def _reg_payload(version=2, model_name="mlops-agent-model"):
    return json.dumps({"version": version, "model_name": model_name})


def test_run_deployer_registers_and_assigns_champion():
    state = {"training_run_id": "run-1"}
    with patch("mlops_agents.deployment.deployer.register_model") as reg, \
         patch("mlops_agents.deployment.deployer.set_model_alias") as alias:
        reg.invoke.return_value = _reg_payload(version=3)
        result = run_deployer(state)

    reg.invoke.assert_called_once_with({"run_id": "run-1"})
    alias.invoke.assert_called_once_with(
        {"model_name": "mlops-agent-model", "alias": "champion", "version": 3}
    )
    assert result["deployment_status"] == "deployed"
    assert result["best_model_uri"] == "models:/mlops-agent-model/3"


def test_run_deployer_missing_run_id_raises():
    import pytest
    with pytest.raises(ValueError, match="training_run_id"):
        run_deployer({"training_run_id": ""})
