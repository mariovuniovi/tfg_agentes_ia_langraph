"""Agent registry — lazy-builds agents on first access to avoid import-time LLM calls."""

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=None)
def get_agent(name: str) -> Any:
    """Return a built agent by name. Agents are cached after first build.

    Args:
        name: One of 'data_validator', 'trainer', 'evaluator', 'deployer'.

    Returns:
        A compiled agent graph.
    """
    if name == "data_validator":
        from mlops_agents.agents.data_agent import build_data_agent
        return build_data_agent()
    if name == "evaluator":
        from mlops_agents.agents.evaluation_agent import build_evaluation_agent
        return build_evaluation_agent()
    raise ValueError(f"Unknown agent: '{name}'. Valid names: data_validator, evaluator")
