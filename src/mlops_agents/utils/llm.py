"""LLM factory — returns a ChatOpenAI instance pointed at the OpenAI API."""

from langchain_openai import ChatOpenAI
from mlops_agents.config.settings import settings
from mlops_agents.utils.logging import get_logger

logger = get_logger(__name__)


def _make_llm(model: str, temperature: float, max_tokens: int) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=settings.openai_api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=3,
    )


def get_llm(agent: str = "", temperature: float = 0, max_tokens: int = 4000) -> ChatOpenAI:
    """Return the LLM for a named worker agent. Pass agent="" to get the default model."""
    model_map = {
        "data_validator": settings.openai_model_data_validator,
        "trainer":        settings.openai_model_trainer,
        "evaluator":      settings.openai_model_evaluator,
        "deployer":       settings.openai_model_deployer,
    }
    model = model_map.get(agent, settings.openai_model)
    label = agent if agent else "default"
    logger.info(f"LLM init — agent={label} model={model}")
    return _make_llm(model, temperature, max_tokens)


def get_router_llm(temperature: float = 0) -> ChatOpenAI:
    """Return the LLM for the supervisor routing node."""
    model = settings.openai_model_supervisor
    logger.info(f"LLM init — agent=supervisor model={model}")
    return _make_llm(model, temperature, max_tokens=1000)
