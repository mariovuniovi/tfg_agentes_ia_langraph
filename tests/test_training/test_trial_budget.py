"""Tests for deterministic per-candidate trial counts."""
from mlops_agents.training.trial_budget import TRIALS_PER_ML_MODEL, deterministic_trials


def test_naive_zero_param_model_gets_one_trial():
    assert deterministic_trials("naive") == 1


def test_random_forest_forecaster_gets_ml_budget():
    assert deterministic_trials("random_forest_forecaster") == TRIALS_PER_ML_MODEL


def test_xgboost_forecaster_gets_ml_budget():
    assert deterministic_trials("xgboost_forecaster") == TRIALS_PER_ML_MODEL


def test_trials_per_ml_model_is_five():
    assert TRIALS_PER_ML_MODEL == 5
