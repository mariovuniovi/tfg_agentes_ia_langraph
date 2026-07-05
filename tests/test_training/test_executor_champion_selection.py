"""Tests for champion selection: tie tolerance + complexity_rank tie-break."""
import pytest

from mlops_agents.training.executor import _pick_champion


def test_strict_winner_no_tie():
    results = [
        {"model_key": "a", "status": "successful", "best_score": 0.95, "complexity_rank": 1},
        {"model_key": "b", "status": "successful", "best_score": 0.90, "complexity_rank": 2},
    ]
    assert _pick_champion(results, "maximize", 0.01)["model_key"] == "a"


def test_tie_within_tolerance_simpler_wins_maximize():
    """0.953 >= 0.96*(1-0.01)=0.9504 → tied → simpler wins."""
    results = [
        {"model_key": "complex", "status": "successful", "best_score": 0.96, "complexity_rank": 3},
        {"model_key": "simple",  "status": "successful", "best_score": 0.953, "complexity_rank": 1},
    ]
    assert _pick_champion(results, "maximize", 0.01)["model_key"] == "simple"


def test_tie_within_tolerance_simpler_wins_minimize():
    """RMSE: 0.0998 <= 0.099*(1+0.01)=0.09999 → tied → simpler wins."""
    results = [
        {"model_key": "complex", "status": "successful", "best_score": 0.099, "complexity_rank": 3},
        {"model_key": "simple",  "status": "successful", "best_score": 0.0998, "complexity_rank": 1},
    ]
    assert _pick_champion(results, "minimize", 0.01)["model_key"] == "simple"


def test_skips_failed_candidates():
    results = [
        {"model_key": "a", "status": "failed", "error_type": "Boom", "complexity_rank": 1},
        {"model_key": "b", "status": "successful", "best_score": 0.5, "complexity_rank": 2},
    ]
    assert _pick_champion(results, "maximize", 0.01)["model_key"] == "b"


def test_all_failed_raises():
    results = [
        {"model_key": "a", "status": "failed", "error_type": "X", "complexity_rank": 1},
        {"model_key": "b", "status": "failed", "error_type": "Y", "complexity_rank": 2},
    ]
    with pytest.raises(RuntimeError, match="All candidates failed"):
        _pick_champion(results, "maximize", 0.01)
