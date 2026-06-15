"""LLM factory — returns a ChatOpenAI instance pointed at the OpenAI API."""

from langchain_openai import ChatOpenAI
from mlops_agents.config.settings import settings
from mlops_agents.prompts.loader import get_agent_config
from mlops_agents.utils.logging import get_logger

logger = get_logger(__name__)


def get_llm(agent: str = "") -> ChatOpenAI:
    """Return the LLM for a named worker agent.

    Model and optional reasoning_effort are read from the agent's prompt YAML
    config block. Falls back to settings.openai_model if no YAML config found.
    """
    config = get_agent_config(agent) if agent else {}
    model = config.get("model", settings.openai_model)
    kwargs: dict = {"model": model, "api_key": settings.openai_api_key, "max_retries": 3}
    if reasoning_effort := config.get("reasoning_effort"):
        kwargs["use_responses_api"] = True
        kwargs["output_version"] = "responses/v1"
        kwargs["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}
    logger.info(f"LLM init — agent={agent or 'default'} model={model}")
    return ChatOpenAI(**kwargs)
