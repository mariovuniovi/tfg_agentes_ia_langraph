"""Deployment Agent — registers models in MLflow Registry with HITL approval gate."""

from langchain.agents import create_agent

from mlops_agents.prompts import get_prompt
from mlops_agents.tools.mlflow_tools import register_model, set_model_alias
from mlops_agents.utils.llm import get_llm


def build_deployment_agent():
    """Build and return the deployment react agent.

    Note: The HITL interrupt() is handled at the graph node level in
    agents/supervisor.py — not inside this react agent — so the agent
    itself is unaware of the pause. This keeps the agent simple and the
    approval logic explicit in the graph topology.
    """
    return create_agent(
        model=get_llm("deployer"),
        tools=[register_model, set_model_alias],
        name="deployer",
        system_prompt=get_prompt("deployment_agent").template,
    )
