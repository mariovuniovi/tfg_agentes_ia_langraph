import pytest


def test_registry_rejects_old_names():
    from mlops_agents.agents.registry import get_agent
    get_agent.cache_clear()
    for old in ["supervisor", "evaluator", "trainer", "deployer"]:
        with pytest.raises(ValueError, match="Unknown agent"):
            get_agent(old)


def test_registry_rejects_unknown_name():
    from mlops_agents.agents.registry import get_agent
    get_agent.cache_clear()
    with pytest.raises(ValueError, match="data_validator, planner, report_writer"):
        get_agent("not_a_real_agent")
