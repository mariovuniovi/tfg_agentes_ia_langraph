"""YAML prompt loader using LangChain's load_prompt."""

from pathlib import Path
from langchain_core.prompts import load_prompt, PromptTemplate

PROMPTS_DIR = Path(__file__).parent


def get_prompt(name: str) -> PromptTemplate:
    """Load a YAML prompt template by agent name.

    Args:
        name: Agent name (e.g., 'supervisor', 'data_agent').

    Returns:
        LangChain PromptTemplate loaded from the corresponding YAML file.
    """
    path = PROMPTS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return load_prompt(str(path))
