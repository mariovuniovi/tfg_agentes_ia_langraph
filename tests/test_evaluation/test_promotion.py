import pytest
from mlops_agents.evaluation.promotion import _metric_for_problem_type, _thresholds_for


@pytest.mark.parametrize("problem_type,expected_metric,expected_ascending", [
    ("classification", "macro_f1", False),
    ("regression", "rmse", True),
    ("forecasting", "rmse", True),
])
def test_metric_for_problem_type(problem_type, expected_metric, expected_ascending):
    metric, ascending = _metric_for_problem_type(problem_type)
    assert metric == expected_metric
    assert ascending is expected_ascending


def test_metric_for_problem_type_unknown_raises():
    with pytest.raises(ValueError, match="Unknown problem_type"):
        _metric_for_problem_type("clustering")


def test_thresholds_for_classification():
    t = _thresholds_for("classification")
    assert t == {"accuracy_min": 0.80, "macro_f1_min": 0.75}


def test_thresholds_for_regression():
    t = _thresholds_for("regression")
    assert t == {"r2_min": 0.70}


def test_thresholds_for_forecasting():
    t = _thresholds_for("forecasting")
    assert t == {}


from mlops_agents.evaluation.promotion import _apply_thresholds


def test_apply_thresholds_classification_pass():
    candidate = {"accuracy": 0.85, "macro_f1": 0.80}
    champion = {"macro_f1": 0.78}
    assert _apply_thresholds("classification", candidate, champion) is True


def test_apply_thresholds_classification_fails_threshold():
    candidate = {"accuracy": 0.70, "macro_f1": 0.80}
    champion = {"macro_f1": 0.78}
    assert _apply_thresholds("classification", candidate, champion) is False


def test_apply_thresholds_classification_loses_to_champion():
    candidate = {"accuracy": 0.85, "macro_f1": 0.75}
    champion = {"macro_f1": 0.78}
    assert _apply_thresholds("classification", candidate, champion) is False


def test_apply_thresholds_classification_no_champion_promotes_if_passes():
    candidate = {"accuracy": 0.85, "macro_f1": 0.80}
    assert _apply_thresholds("classification", candidate, {}) is True


def test_apply_thresholds_regression_pass():
    candidate = {"r2": 0.75, "rmse": 1.2}
    champion = {"rmse": 1.5}
    assert _apply_thresholds("regression", candidate, champion) is True


def test_apply_thresholds_regression_rmse_worse_than_champion():
    candidate = {"r2": 0.75, "rmse": 2.0}
    champion = {"rmse": 1.5}
    assert _apply_thresholds("regression", candidate, champion) is False


def test_apply_thresholds_forecasting_no_champion():
    candidate = {"rmse": 1.5}
    assert _apply_thresholds("forecasting", candidate, {}) is True


def test_apply_thresholds_forecasting_with_champion_pass():
    candidate = {"rmse": 1.2}
    champion = {"rmse": 1.5}
    assert _apply_thresholds("forecasting", candidate, champion) is True


def test_apply_thresholds_forecasting_missing_rmse():
    assert _apply_thresholds("forecasting", {}, {}) is False


from unittest.mock import MagicMock, patch


def _fake_runs(metric_values: list[float]):
    runs = []
    for v in metric_values:
        r = MagicMock()
        r.info.run_id = f"run-{v}"
        r.data.metrics = {"macro_f1": v}
        runs.append(r)
    return runs


def test_fetch_current_champion_returns_top_run_metrics():
    with patch("mlops_agents.evaluation.promotion._get_client") as mock_get_client:
        client = MagicMock()
        client.get_experiment_by_name.return_value = MagicMock(experiment_id="exp-1")
        client.search_runs.return_value = _fake_runs([0.92, 0.85])
        mock_get_client.return_value = client

        from mlops_agents.evaluation.promotion import _fetch_current_champion
        result = _fetch_current_champion("macro_f1", ascending=False)
        assert result == {"macro_f1": 0.92}


def test_fetch_current_champion_no_experiment_returns_empty():
    with patch("mlops_agents.evaluation.promotion._get_client") as mock_get_client:
        client = MagicMock()
        client.get_experiment_by_name.return_value = None
        mock_get_client.return_value = client

        from mlops_agents.evaluation.promotion import _fetch_current_champion
        assert _fetch_current_champion("macro_f1", ascending=False) == {}


def test_evaluate_promotion_end_to_end_classification():
    state = {
        "problem_type": "classification",
        "training_metrics": {"accuracy": 0.85, "macro_f1": 0.80},
    }
    with patch("mlops_agents.evaluation.promotion._fetch_current_champion") as f:
        f.return_value = {"macro_f1": 0.78}
        from mlops_agents.evaluation.promotion import evaluate_promotion
        result = evaluate_promotion(state)

    assert result["evaluation_passed"] is True
    assert result["candidate_metrics"] == {"accuracy": 0.85, "macro_f1": 0.80}
    assert result["champion_metrics"] == {"macro_f1": 0.78}
    assert result["thresholds_applied"] == {"accuracy_min": 0.80, "macro_f1_min": 0.75}


def test_evaluate_promotion_writes_legacy_evaluation_report_shape():
    state = {
        "problem_type": "classification",
        "training_metrics": {"accuracy": 0.85, "macro_f1": 0.80},
        "training_run_id": "run-42",
    }
    with patch("mlops_agents.evaluation.promotion._fetch_current_champion") as f:
        f.return_value = {"macro_f1": 0.78}
        from mlops_agents.evaluation.promotion import evaluate_promotion
        result = evaluate_promotion(state)

    assert result["evaluation_report"] == {
        "candidate_metrics": {"accuracy": 0.85, "macro_f1": 0.80},
        "candidate_run_id": "run-42",
        "baseline_metrics": {"macro_f1": 0.78},
    }
