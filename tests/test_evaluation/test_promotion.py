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
