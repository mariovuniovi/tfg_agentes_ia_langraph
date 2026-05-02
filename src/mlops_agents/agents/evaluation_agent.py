"""Evaluation Agent — compares models and recommends promotion or rejection."""

from langchain.agents import create_agent

from mlops_agents.prompts import get_prompt
from mlops_agents.tools.mlflow_tools import get_best_run
from mlops_agents.utils.llm import get_llm


def build_evaluation_agent():
    """Build and return the model evaluation react agent."""
    return create_agent(
        model=get_llm("evaluator"),
        tools=[get_best_run],
        name="evaluator",
        system_prompt=get_prompt("evaluation_agent").template,
    )
