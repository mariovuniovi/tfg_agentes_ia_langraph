"""Factory functions for every registered model.

Each factory takes a hyperparameter dict (and for forecasting, a task_metadata dict)
and returns either a sklearn-compatible estimator or a forecaster object. Factories
are referenced by string name from `registry.yaml`. The string-keyed lookup is the
security boundary — the YAML cannot reference a factory that isn't here.
"""

from __future__ import annotations

from typing import Any, Callable

# Populated incrementally in Tasks 5-8. Empty at registry-skeleton task.
FACTORY_REGISTRY: dict[str, Callable[..., Any]] = {}
