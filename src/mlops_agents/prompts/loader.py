"""YAML prompt loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml
from langchain_core.prompts import PromptTemplate

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
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return PromptTemplate(
        template=data["template"],
        input_variables=data.get("input_variables", []),
    )


def get_agent_config(name: str) -> dict[str, Any]:
    """Return the config dict from a prompt YAML, or {} if absent."""
    path = PROMPTS_DIR / f"{name}.yaml"
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return cast("dict[str, Any]", data.get("config", {}))
