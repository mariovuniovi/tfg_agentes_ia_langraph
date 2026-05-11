"""Tests for iter_folds: correct count, chronological order, no future leakage."""
import pandas as pd
import pytest

from mlops_agents.contracts.training import ValidationStrategy
from mlops_agents.training.validation_folds import iter_folds


def _make_pool(rows: int) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=rows, freq="W"),
        "target": range(rows),
    })


def test_single_split_yields_one_fold():
    pool = _make_pool(100)
    strat = ValidationStrategy(type="single_split", horizon=10)
    folds = list(iter_folds(pool, strat, "date", []))
    assert len(folds) == 1
    train_idx, val_idx = folds[0]
    assert len(val_idx) == 10
    assert len(train_idx) == 90
    assert pool.loc[train_idx, "date"].max() < pool.loc[val_idx, "date"].min()


def test_expanding_window_three_folds_train_grows():
    pool = _make_pool(100)
    strat = ValidationStrategy(type="expanding_window", n_folds=3, horizon=10, step_size=10)
    folds = list(iter_folds(pool, strat, "date", []))
    assert len(folds) == 3
    train_lens = [len(t) for t, _ in folds]
    assert train_lens[0] < train_lens[1] < train_lens[2]
    for _, v in folds:
        assert len(v) == 10
    for t, v in folds:
        assert pool.loc[t, "date"].max() < pool.loc[v, "date"].min()


def test_rolling_window_three_folds_train_size_constant():
    pool = _make_pool(100)
    strat = ValidationStrategy(
        type="rolling_window", n_folds=3, horizon=10, step_size=10, window_size=50
    )
    folds = list(iter_folds(pool, strat, "date", []))
    assert len(folds) == 3
    train_lens = [len(t) for t, _ in folds]
    assert all(L == 50 for L in train_lens)


def test_iter_folds_sorts_by_date_first():
    pool = _make_pool(50).sample(frac=1, random_state=0).reset_index(drop=True)
    strat = ValidationStrategy(type="single_split", horizon=5)
    [(t, v)] = list(iter_folds(pool, strat, "date", []))
    assert pool.loc[t, "date"].max() < pool.loc[v, "date"].min()
