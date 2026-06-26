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
    # timeout bounds a stuck connection (else the SDK default lets a hung call
    # stall for minutes); max_retries then re-issues it. 240s sits well above
    # legitimate reasoning-call latency (~30-150s) so it never cuts a real call.
    kwargs: dict = {"model": model, "api_key": settings.openai_api_key, "max_retries": 3, "timeout": 240}
    if reasoning_effort := config.get("reasoning_effort"):
        kwargs["use_responses_api"] = True
        kwargs["output_version"] = "responses/v1"
        kwargs["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}
    logger.info(f"LLM init — agent={agent or 'default'} model={model}")
    return ChatOpenAI(**kwargs)
