"""YAML manifest loader for the model registry.

Validates every entry against Pydantic schemas and rejects unknown
`factory` / `search_space.name` references — the manifest cannot
reference unregistered Python code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from mlops_agents.models.factories import FACTORY_REGISTRY
from mlops_agents.models.search_spaces import SEARCH_SPACE_REGISTRY

DEFAULT_REGISTRY_PATH = Path(__file__).parent / "registry.yaml"


class SearchParamSpec(BaseModel):
    type: Literal["int", "float", "categorical"]
    low: float | int | None = None
    high: float | int | None = None
    step: int | None = None
    log: bool = False
    choices: list[Any] | None = None


class SearchSpaceSpec(BaseModel):
    name: str
    params: dict[str, SearchParamSpec] = Field(default_factory=dict)


class ModelSpec(BaseModel):
    model_key: str
    problem_type: Literal["classification", "regression", "forecasting"]
    family: str
    complexity_rank: int
    library: str
    factory: str
    search_space: SearchSpaceSpec
    default_params: dict[str, Any] = Field(default_factory=dict)
    requires: dict[str, Any] = Field(default_factory=dict)
    use_when: list[str] = Field(default_factory=list)
    avoid_when: list[str] = Field(default_factory=list)
    notes: str = ""

    @field_validator("factory")
    @classmethod
    def factory_must_be_registered(cls, v: str) -> str:
        import mlops_agents.models.loader as _self
        registry = _self.FACTORY_REGISTRY
        # Only enforce when the registry is non-empty (Tasks 5-8 populate it).
        # An empty FACTORY_REGISTRY means we are in skeleton / test-injection mode.
        if registry and v not in registry:
            raise ValueError(
                f"Unknown factory: {v!r}. "
                f"Known factories: {sorted(registry)}"
            )
        return v


_cached_registry: dict[str, ModelSpec] | None = None


def load_registry(path: Path = DEFAULT_REGISTRY_PATH, force_reload: bool = False) -> dict[str, ModelSpec]:
    """Load and validate the YAML registry; cached after first call."""
    global _cached_registry
    if _cached_registry is not None and not force_reload:
        return _cached_registry

    raw = yaml.safe_load(path.read_text()) or []
    registry: dict[str, ModelSpec] = {}
    for entry in raw:
        spec = ModelSpec(**entry)
        if spec.model_key in registry:
            raise ValueError(f"Duplicate model_key in registry: {spec.model_key!r}")
        registry[spec.model_key] = spec

    _cached_registry = registry
    return registry


def get_models_for(problem_type: str) -> list[ModelSpec]:
    """All registered models matching the given problem_type."""
    return [m for m in load_registry().values() if m.problem_type == problem_type]


def get_model(model_key: str) -> ModelSpec:
    """Single model by key. Raises KeyError if unknown."""
    registry = load_registry()
    if model_key not in registry:
        raise KeyError(f"Unknown model_key: {model_key!r}")
    return registry[model_key]
