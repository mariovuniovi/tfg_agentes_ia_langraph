"""Generic Optuna search-space builder driven by SearchSpaceSpec from the registry.

`build_suggest_fn(spec)` produces an Optuna-style `suggest(trial)` callable
that materializes hyperparameters from the declarative spec. No hand-written
suggest_* functions per model — the YAML is the source of truth.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import optuna

if TYPE_CHECKING:
    from mlops_agents.models.loader import SearchSpaceSpec


def build_suggest_fn(spec: SearchSpaceSpec) -> Callable[[optuna.Trial], dict[str, Any]]:
    """Return a `suggest(trial) -> dict` callable that materializes params from spec."""

    def suggest(trial: optuna.Trial) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, p in spec.params.items():
            # casts: int/float params always define low/high in the registry;
            # categorical params always define choices (SearchParamSpec convention).
            if p.type == "int":
                out[name] = trial.suggest_int(
                    name, int(cast("float", p.low)), int(cast("float", p.high)), step=p.step or 1
                )
            elif p.type == "float":
                out[name] = trial.suggest_float(
                    name, float(cast("float", p.low)), float(cast("float", p.high)), log=p.log or False
                )
            elif p.type == "categorical":
                out[name] = trial.suggest_categorical(name, cast("list[Any]", p.choices))
            else:
                raise ValueError(f"Unknown search param type {p.type!r} for {name!r}")
        return out

    return suggest

