"""Unit tests for the generic Optuna search-space builder."""

import optuna

from mlops_agents.models.loader import SearchParamSpec, SearchSpaceSpec
from mlops_agents.models.search_spaces import build_suggest_fn


def test_build_suggest_fn_int():
    spec = SearchSpaceSpec(name="x", params={"n": SearchParamSpec(type="int", low=10, high=20)})
    suggest = build_suggest_fn(spec)
    study = optuna.create_study()
    study.optimize(lambda t: float(suggest(t)["n"]), n_trials=3)
    assert 10 <= study.best_params["n"] <= 20


def test_build_suggest_fn_float_log():
    spec = SearchSpaceSpec(
        name="x",
        params={"lr": SearchParamSpec(type="float", low=1e-4, high=1.0, log=True)},
    )
    suggest = build_suggest_fn(spec)
    study = optuna.create_study()
    study.optimize(lambda t: -suggest(t)["lr"], n_trials=3)
    assert 1e-4 <= study.best_params["lr"] <= 1.0


def test_build_suggest_fn_categorical():
    spec = SearchSpaceSpec(
        name="x",
        params={"k": SearchParamSpec(type="categorical", choices=["a", "b", "c"])},
    )
    suggest = build_suggest_fn(spec)
    study = optuna.create_study()
    study.optimize(lambda t: 0.0 if suggest(t)["k"] == "a" else 1.0, n_trials=5)
    assert study.best_params["k"] == "a"
