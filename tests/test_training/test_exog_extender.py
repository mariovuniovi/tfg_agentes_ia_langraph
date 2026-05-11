"""Tests for extend_exog: naive_carry, ets, auto_arima, failure fallback,
index alignment."""
import numpy as np
import pandas as pd

from mlops_agents.training.exog_extender import (
    extend_exog,
    _align_val_exog_index,
)


def _series(values: list, freq: str = "W") -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=len(values), freq=freq)
    return pd.Series(values, index=idx, name="oil")


def test_naive_carry_repeats_last_value():
    s = _series([10.0, 11.0, 12.0, 13.0])
    out, fail = extend_exog(s, horizon=3, strategy="naive_carry", freq="W")
    assert list(out) == [13.0, 13.0, 13.0]
    assert fail is None


def test_ets_returns_horizon_values():
    np.random.seed(0)
    history = _series(list(np.linspace(0, 1, 60) + np.random.randn(60) * 0.01))
    out, fail = extend_exog(history, horizon=5, strategy="ets", freq="W")
    assert len(out) == 5
    # fail may be set or not depending on fit; check both states are valid
    assert (fail is None) or ("strategy" in fail and fail["strategy"] == "ets")


def test_auto_arima_returns_horizon_values():
    np.random.seed(0)
    history = _series(list(np.cumsum(np.random.randn(80))))
    out, fail = extend_exog(history, horizon=5, strategy="auto_arima", freq="W")
    assert len(out) == 5


def test_ets_failure_falls_back_to_naive_carry():
    # A constant series can cause some ETS configurations to fail
    s = _series([5.0] * 10)
    out, fail = extend_exog(s, horizon=3, strategy="ets", freq="W")
    # Either ETS succeeded or it fell back to naive_carry
    assert len(out) == 3
    if fail is not None:
        # fallback was used; last value repeated
        assert list(out) == [5.0, 5.0, 5.0]
        assert fail["fallback"] == "naive_carry"


def test_align_index_matches_rangeindex_series_dict():
    val_exog = pd.DataFrame({"oil": [1.0, 2.0, 3.0]})
    series_dict = {"__single__": pd.Series([0.0] * 50, index=pd.RangeIndex(50))}
    aligned = _align_val_exog_index(
        val_exog, series_dict, train_len=50, dt_col="date", freq="W"
    )
    assert isinstance(aligned.index, pd.RangeIndex)
    assert aligned.index.start == 50
    assert aligned.index.stop == 53


def test_align_index_matches_datetimeindex_series_dict():
    val_exog = pd.DataFrame({"oil": [1.0, 2.0, 3.0]})
    train_idx = pd.date_range("2020-01-01", periods=50, freq="W")
    series_dict = {"__single__": pd.Series([0.0] * 50, index=train_idx)}
    aligned = _align_val_exog_index(
        val_exog, series_dict, train_len=50, dt_col="date", freq="W"
    )
    assert isinstance(aligned.index, pd.DatetimeIndex)
    # First future timestamp = train_idx[-1] + 1 freq step
    assert aligned.index[0] == train_idx[-1] + pd.tseries.frequencies.to_offset("W")
