"""Agent registry — lazy-builds LLM-using agents on first access.

Only LLM-backed nodes are registered. Deterministic nodes (workflow_controller,
evaluation, executor, deployer, approval gates) are imported directly by the graph.
"""

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=None)
def get_agent(name: str) -> Any:
    """Return a built agent by name. Cached after first build."""
    if name == "data_validator":
        from mlops_agents.agents.data_agent import build_data_agent
        return build_data_agent()
    if name == "planner":
        from mlops_agents.agents.planner import build_planner_agent
        return build_planner_agent()
    if name == "report_writer":
        from mlops_agents.evaluation.report_writer import build_report_writer
        return build_report_writer()
    raise ValueError(
        f"Unknown agent: '{name}'. Valid: data_validator, planner, report_writer"
    )
