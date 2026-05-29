"""Deterministic deployer — registers the model and assigns the champion alias.

The HITL approval gate runs upstream in deployment_approval_node, so by the
time this function is called the human has already approved.
"""
from __future__ import annotations

import json
from typing import Any

from mlops_agents.tools.mlflow_tools import register_model, set_model_alias
from mlops_agents.utils.logging import get_logger

logger = get_logger(__name__)


def run_deployer(state: dict[str, Any]) -> dict[str, Any]:
    """Register the trained model and promote it to champion."""
    run_id = state.get("training_run_id", "")
    if not run_id:
        raise ValueError("run_deployer requires training_run_id in state")

    reg_raw = register_model.invoke({"run_id": run_id})
    reg = json.loads(reg_raw)
    version = reg["version"]
    model_name = reg["model_name"]

    set_model_alias.invoke(
        {"model_name": model_name, "alias": "champion", "version": version}
    )

    logger.info(f"[deployer] promoted {model_name} v{version} from run {run_id}")
    return {
        "deployment_status": "deployed",
        "best_model_uri": f"models:/{model_name}/{version}",
    }
