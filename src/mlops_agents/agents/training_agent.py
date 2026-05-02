"""Training Agent — tunes hyperparameters, trains models, and logs to MLflow."""

from langchain.agents import create_agent

from mlops_agents.prompts import get_prompt
from mlops_agents.tools.mlflow_tools import log_experiment
from mlops_agents.tools.training_tools import train_model, tune_hyperparameters
from mlops_agents.utils.llm import get_llm


def build_training_agent():
    """Build and return the model training react agent."""
    return create_agent(
        model=get_llm("trainer"),
        tools=[tune_hyperparameters, train_model, log_experiment],
        name="trainer",
        system_prompt=get_prompt("training_agent").template,
    )
