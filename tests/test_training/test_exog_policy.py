import numpy as np
import pandas as pd

from mlops_agents.training.exog_policy import resolve_exog_strategies


def _df(**series):
    n = len(next(iter(series.values())))
    base = {"ds": pd.date_range("2023-01-02", periods=n, freq="W-MON")}
    base.update(series)
    return pd.DataFrame(base)


def _meta(cols):
    return {"exogenous_columns": cols, "forecast_horizon": 8}


def test_seasonal_unknown_future_uses_ets():
    t = np.arange(156)
    temp = 9 - 13 * np.cos(2 * np.pi * (t % 52) / 52)   # strong yearly seasonality
    out = resolve_exog_strategies(
        _df(temp=temp), _meta([{"name": "temp", "future_availability": "unknown_future"}]), "W"
    )
    assert out.per_column["temp"] == "ets"


def test_flat_noise_unknown_future_uses_naive_carry():
    noise = np.random.default_rng(42).normal(0, 1, 156)
    out = resolve_exog_strategies(
        _df(noise=noise), _meta([{"name": "noise", "future_availability": "unknown_future"}]), "W"
    )
    assert out.per_column["noise"] == "naive_carry"


def test_non_numeric_unknown_future_uses_naive_carry():
    cond = ["sunny", "rainy"] * 78  # 156 strings
    out = resolve_exog_strategies(
        _df(cond=cond), _meta([{"name": "cond", "future_availability": "unknown_future"}]), "W"
    )
    assert out.per_column["cond"] == "naive_carry"


def test_known_future_absent_from_per_column():
    out = resolve_exog_strategies(
        _df(holiday=np.zeros(156)), _meta([{"name": "holiday", "future_availability": "known_future"}]), "W"
    )
    assert "holiday" not in out.per_column


def test_no_exog_declared_is_empty():
    out = resolve_exog_strategies(_df(y=np.arange(156, dtype=float)), {"exogenous_columns": []}, "W")
    assert out.per_column == {}
