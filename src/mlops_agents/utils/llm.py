"""LLM factory — returns a ChatOpenAI instance pointed at GitHub Models."""

from langchain_openai import ChatOpenAI
from mlops_agents.config.settings import settings


def get_llm(temperature: float = 0, max_tokens: int = 4000) -> ChatOpenAI:
    """Return the primary LLM for worker agents (uses GITHUB_MODEL)."""
    return ChatOpenAI(
        model=settings.github_model,
        base_url=settings.github_api_base,
        api_key=settings.github_token,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=2,
    )


def get_router_llm(temperature: float = 0) -> ChatOpenAI:
    """Return the LLM for the supervisor routing node.

    Uses the same model as worker agents — max_tokens=1000 keeps it cheap.
    """
    return ChatOpenAI(
        model=settings.github_model,
        base_url=settings.github_api_base,
        api_key=settings.github_token,
        temperature=temperature,
        max_tokens=1000,
        max_retries=2,
    )
