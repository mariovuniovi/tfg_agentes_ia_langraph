from mlops_agents.models.loader import get_model, get_models_for, ModelSpec

def test_modelspec_has_supports_exogenous_field():
    spec = get_model("ets")  # statistical forecasting model
    assert spec.supports_exogenous is False

def test_modelspec_has_supports_missing_field():
    spec = get_model("ets")
    assert spec.supports_missing is False

def test_summary_dict_shape():
    spec = get_model("ets")
    summary = spec.summary_dict()
    expected_keys = {"model_key", "problem_type", "family", "complexity_rank",
                     "supports_exogenous", "supports_missing", "use_when", "avoid_when"}
    assert expected_keys == set(summary.keys())

def test_details_dict_includes_search_space_name():
    spec = get_model("ets")
    details = spec.details_dict()
    assert "search_space" in details
    assert "default_params" in details
    assert "notes" in details
    assert details["model_key"] == "ets"

def test_every_model_declares_support_flags_in_registry_yaml():
    """No silent defaults: every registered model must EXPLICITLY declare both flags
    in registry.yaml. Checking spec.supports_exogenous on a constructed instance can't
    catch missing entries because the Pydantic default is False — so we read the raw YAML."""
    import yaml
    from pathlib import Path
    raw = yaml.safe_load(Path("src/mlops_agents/models/registry.yaml").read_text())
    # Adjust top-level key if registry.yaml uses a different shape (raw["models"] vs raw).
    # registry.yaml uses a list of entries; convert to {model_key: entry} for iteration.
    if isinstance(raw, list):
        models = {entry["model_key"]: entry for entry in raw}
    else:
        models = raw.get("models", raw)
    for model_key, entry in models.items():
        assert "supports_exogenous" in entry, f"{model_key} missing supports_exogenous in YAML"
        assert "supports_missing" in entry, f"{model_key} missing supports_missing in YAML"
