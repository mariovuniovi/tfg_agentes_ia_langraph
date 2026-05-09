"""Generic Optuna search-space builder driven by SearchSpaceSpec from the registry.

`build_suggest_fn(spec)` produces an Optuna-style `suggest(trial)` callable
that materializes hyperparameters from the declarative spec. No hand-written
suggest_* functions per model — the YAML is the source of truth.
"""

from __future__ import annotations

from typing import Any, Callable

import optuna


def build_suggest_fn(spec) -> Callable[[optuna.Trial], dict[str, Any]]:
    """Return a `suggest(trial) -> dict` callable that materializes params from spec."""

    def suggest(trial: optuna.Trial) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, p in spec.params.items():
            if p.type == "int":
                out[name] = trial.suggest_int(name, int(p.low), int(p.high), step=p.step or 1)
            elif p.type == "float":
                out[name] = trial.suggest_float(
                    name, float(p.low), float(p.high), log=p.log or False
                )
            elif p.type == "categorical":
                out[name] = trial.suggest_categorical(name, p.choices)
            else:
                raise ValueError(f"Unknown search param type {p.type!r} for {name!r}")
        return out

    return suggest


# Populated incrementally as model entries are added: each model's search_space.name
# from registry.yaml becomes a key here pointing to a "build" function. With the generic
# builder above, every entry maps to the same factory call: `build_suggest_fn(spec)`.
# We expose the registry for symmetry with FACTORY_REGISTRY (validation path).
SEARCH_SPACE_REGISTRY: dict[str, Callable[..., Any]] = {}
