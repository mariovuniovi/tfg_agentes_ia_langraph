"""Unit tests for training contracts (exog strategies, validation settings)."""

import pytest
from pydantic import ValidationError

from mlops_agents.contracts.training import ExogStrategySettings, TrainingResult


def test_drop_strategy_is_rejected_default():
    with pytest.raises(ValidationError):
        ExogStrategySettings(default_unknown_future="drop")


def test_drop_strategy_is_rejected_per_column():
    with pytest.raises(ValidationError):
        ExogStrategySettings(per_column={"temp": "drop"})


def test_training_result_has_selection_score_default_none():
    r = TrainingResult(
        champion_candidate={"model_key": "ets"},
        champion_model_path="x.pkl",
        train_pool_path="t.csv",
        test_path="te.csv",
        split_metadata_path="s.json",
        mlflow_parent_run_id="abc",
        experience_record_path="e.json",
        champion_metrics={"rmse": 1.0},
    )
    assert r.selection_score is None
    r2 = r.model_copy(update={"selection_score": 4.74})
    assert r2.selection_score == 4.74
