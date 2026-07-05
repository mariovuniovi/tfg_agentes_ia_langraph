"""Unit tests for the model registry loader."""


import pytest
from pydantic import ValidationError

from mlops_agents.models.loader import (
    ModelSpec,
    SearchParamSpec,
    SearchSpaceSpec,
    get_models_for,
    load_registry,
)


def test_search_param_spec_int():
    p = SearchParamSpec(type="int", low=10, high=100)
    assert p.type == "int" and p.low == 10 and p.high == 100


def test_search_param_spec_float_log():
    p = SearchParamSpec(type="float", low=0.001, high=1.0, log=True)
    assert p.log is True


def test_search_param_spec_categorical():
    p = SearchParamSpec(type="categorical", choices=["l1", "l2"])
    assert p.choices == ["l1", "l2"]


def test_model_spec_unknown_factory_rejected(monkeypatch):
    monkeypatch.setattr("mlops_agents.models.loader.FACTORY_REGISTRY", {"build_x": lambda p: None})
    monkeypatch.setattr(
        "mlops_agents.models.loader.SEARCH_SPACE_REGISTRY",
        {"x_space": lambda *_: None},
    )
    with pytest.raises(ValidationError, match="Unknown factory"):
        ModelSpec(
            model_key="x",
            problem_type="classification",
            family="test",
            complexity_rank=1,
            library="sklearn",
            factory="build_unknown",
            search_space=SearchSpaceSpec(name="x_space", params={}),
            default_params={},
        )


def test_load_registry_empty_yaml(tmp_path):
    """An empty YAML file (yaml = []) loads to an empty dict — no error."""
    empty_path = tmp_path / "registry.yaml"
    empty_path.write_text("[]\n")
    registry = load_registry(empty_path, force_reload=True)
    assert registry == {}


def test_get_models_for_filters_by_problem_type(monkeypatch):
    """get_models_for() returns only entries matching the requested problem_type."""
    # Monkeypatch registries first so ModelSpec validation accepts 'build_x' / 'x_space'.
    monkeypatch.setattr("mlops_agents.models.loader.FACTORY_REGISTRY", {"build_x": lambda p: None})
    monkeypatch.setattr(
        "mlops_agents.models.loader.SEARCH_SPACE_REGISTRY",
        {"x_space": lambda *_: None},
    )
    monkeypatch.setattr(
        "mlops_agents.models.loader._cached_registry",
        {
            "a": ModelSpec(
                model_key="a", problem_type="classification", family="test",
                complexity_rank=1, library="sklearn", factory="build_x",
                search_space=SearchSpaceSpec(name="x_space", params={}),
                default_params={},
            ),
            "b": ModelSpec(
                model_key="b", problem_type="regression", family="test",
                complexity_rank=1, library="sklearn", factory="build_x",
                search_space=SearchSpaceSpec(name="x_space", params={}),
                default_params={},
            ),
        },
    )
    cls_models = get_models_for("classification")
    assert {m.model_key for m in cls_models} == {"a"}


def test_load_actual_registry_has_20_entries():
    """Smoke test: the shipped registry.yaml loads cleanly with all expected models."""
    from mlops_agents.models.loader import get_models_for, load_registry
    registry = load_registry(force_reload=True)
    assert len(registry) == 20
    assert len(get_models_for("classification")) == 5
    assert len(get_models_for("regression")) == 5
    assert len(get_models_for("forecasting")) == 10  # 4 statistical + 6 supervised


def test_registry_complexity_ranks_set():
    from mlops_agents.models.loader import load_registry
    for model in load_registry(force_reload=True).values():
        assert model.complexity_rank >= 1
