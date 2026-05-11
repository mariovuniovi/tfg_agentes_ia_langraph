"""Leakage firewall: extend an exogenous series into the forecast horizon.

This module only ever sees training-window history. The executor cannot
construct val_exog for unknown_future columns through any other path.
"""
from __future__ import annotations
from typing import Callable, Literal

import pandas as pd

from mlops_agents.utils.logging import get_logger

logger = get_logger(__name__)

Strategy = Literal["naive_carry", "ets", "auto_arima"]


def extend_exog(
    history: pd.Series,
    horizon: int,
    strategy: Strategy,
    freq: str | None,
) -> tuple[pd.Series, dict | None]:
    """Return (predicted_future_values, failure_info_or_None).

    naive_carry never fails. ets / auto_arima fall back to naive_carry on
    fit failure and return a failure_info dict for the experience record.
    """
    if strategy == "naive_carry":
        return _naive_carry(history, horizon), None
    if strategy == "ets":
        return _try_statistical(history, horizon, freq, _fit_ets, "ets")
    if strategy == "auto_arima":
        return _try_statistical(history, horizon, freq, _fit_auto_arima, "auto_arima")
    raise ValueError(f"Unknown exog extension strategy: {strategy!r}")


def _naive_carry(history: pd.Series, horizon: int) -> pd.Series:
    last = history.iloc[-1]
    return pd.Series([last] * horizon, name=history.name)


def _try_statistical(
    history: pd.Series,
    horizon: int,
    freq: str | None,
    fit_fn: Callable[[pd.Series, int, str | None], pd.Series | object],
    strategy_name: str,
) -> tuple[pd.Series, dict | None]:
    try:
        preds = fit_fn(history, horizon, freq)
        return pd.Series(preds, name=history.name), None
    except Exception as e:
        logger.warning(
            f"[exog_extender] {strategy_name} failed for column "
            f"{history.name!r}: {type(e).__name__}: {e}. Falling back to naive_carry."
        )
        return _naive_carry(history, horizon), {
            "strategy": strategy_name,
            "fallback": "naive_carry",
            "error_class": type(e).__name__,
            "error_msg": str(e)[:200],
        }


def _fit_ets(history: pd.Series, horizon: int, freq: str | None) -> object:
    from statsforecast.models import AutoETS

    season_length = _season_length_for_freq(freq)
    m = AutoETS(season_length=season_length)
    m.fit(history.values.astype(float))
    return m.predict(h=horizon)["mean"]


def _fit_auto_arima(history: pd.Series, horizon: int, freq: str | None) -> object:
    from statsforecast.models import AutoARIMA

    season_length = _season_length_for_freq(freq)
    m = AutoARIMA(season_length=season_length)
    m.fit(history.values.astype(float))
    return m.predict(h=horizon)["mean"]


# Duplicated from models/factories.py intentionally: exog_extender must not
# import from the models layer to avoid a circular dependency (training → models
# → training).
_FREQ_TO_SEASON: dict[str, int] = {
    "H": 24,
    "D": 7,
    "W": 52,
    "MS": 12,
    "M": 12,
    "QS": 4,
    "YS": 1,
}


def _season_length_for_freq(freq: str | None) -> int:
    if freq is None:
        return 1
    return _FREQ_TO_SEASON.get(freq, 1)


def align_val_exog_index(
    val_exog: pd.DataFrame,
    series_dict: dict[str, pd.Series],
    train_len: int,
    dt_col: str,
    freq: str | None,
) -> pd.DataFrame:
    """Match val_exog's index type to a sample series in series_dict.

    skforecast requires train_exog and val_exog to share the same index
    type as `series`. If series_dict uses RangeIndex, val_exog continues
    at `train_len`. If DatetimeIndex, val_exog continues from the last
    training timestamp at `freq` cadence.
    """
    if val_exog.empty:
        return val_exog
    sample = next(iter(series_dict.values()))
    if isinstance(sample.index, pd.RangeIndex):
        val_exog = val_exog.copy()
        val_exog.index = pd.RangeIndex(train_len, train_len + len(val_exog))
        return val_exog
    last_train_ts = sample.index[-1]
    if freq is None:
        raise ValueError(
            "align_val_exog_index requires freq when series_dict uses a "
            "DatetimeIndex (cannot infer cadence from None)"
        )
    offset = pd.tseries.frequencies.to_offset(freq)
    future_idx = pd.date_range(
        start=last_train_ts + offset,
        periods=len(val_exog),
        freq=freq,
    )
    val_exog = val_exog.copy()
    val_exog.index = future_idx
    return val_exog
