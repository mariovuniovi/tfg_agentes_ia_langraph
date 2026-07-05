"""The state-update contracts must stay in sync with AgentState.

Excluded keys are the ones no domain/gate node owns:
  messages              — the reducer channel (operator.add)
  dataset_paths         — a pipeline input, never written by a node
  agent_attempt_counts  — owned by the workflow_controller router (exempt)
"""

from mlops_agents.contracts import outputs as o
from mlops_agents.state.agent_state import AgentState

_ALL_STATE_UPDATES = [
    o.DataValidationStateUpdate,
    o.DatasetApprovalStateUpdate,
    o.PlannerStateUpdate,
    o.PlannerErrorStateUpdate,
    o.TrainingStateUpdate,
    o.EvaluationStateUpdate,
    o.AuditStateUpdate,
    o.DeploymentApprovalStateUpdate,
    o.DeploymentStateUpdate,
]

_EXCLUDED = {"messages", "dataset_paths", "agent_attempt_counts"}


def _keys_written_by_contracts() -> set[str]:
    keys: set[str] = set()
    for model in _ALL_STATE_UPDATES:
        for name, field in model.model_fields.items():
            keys.add(field.serialization_alias or name)
    return keys


def test_contracts_cover_every_writable_state_key():
    state_keys = set(AgentState.__annotations__) - _EXCLUDED
    written = _keys_written_by_contracts()
    assert written == state_keys, {
        "missing_from_contracts": sorted(state_keys - written),
        "extra_in_contracts": sorted(written - state_keys),
    }
