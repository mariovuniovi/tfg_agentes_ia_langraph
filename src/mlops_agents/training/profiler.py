"""Compute the bucketed dataset profile used as the retrieval join key."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlops_agents.contracts.profile import DatasetProfile

__all__ = ["build_dataset_profile", "DatasetProfile"]

# ---------------------------------------------------------------------------
# Bucketers
# ---------------------------------------------------------------------------

def _bucket_n_rows(n: int) -> str:
    if n < 500: return "very_small"
    if n < 1000: return "small"
    if n <= 50_000: return "medium"
    return "large"


def _bucket_n_features(n: int) -> str:
    if n < 10: return "small"
    if n <= 100: return "medium"
    return "large"


def _bucket_missing(rate: float) -> str:
    if rate == 0.0: return "none"
    if rate < 0.05: return "low"
    if rate <= 0.20: return "medium"
    return "high"


def _bucket_count(n: int) -> str:
    if n == 0: return "none"
    if n <= 3: return "few"
    if n <= 10: return "some"
    return "many"


def _bucket_n_classes(n: int) -> str:
    if n == 2: return "binary"
    if n <= 5: return "small_multiclass"
    return "many_classes"


def _bucket_class_balance(class_counts: pd.Series) -> str:
    if len(class_counts) == 0: return "balanced"
    ratio = class_counts.max() / max(class_counts.min(), 1)
    if ratio < 1.5: return "balanced"
    if ratio < 5: return "moderately_imbalanced"
    return "severely_imbalanced"


def _bucket_target_distribution(s: pd.Series) -> str:
    n_unique = s.nunique()
    if n_unique > 0 and n_unique < max(len(s) / 20, 5):
        return "discrete_like"
    skew = abs(s.skew())
    kurt = s.kurt()
    if kurt > 3 and skew < 1: return "heavy_tailed"
    if skew >= 1: return "skewed"
    return "near_normal"


def _bucket_n_series(n: int) -> str:
    if n == 1: return "single"
    if n <= 10: return "few"
    if n <= 100: return "moderate"
    return "many"


# Thresholds in row count, not time units. Calibrated for daily/weekly series:
# <60 ~= one quarter of weekly data; <2000 ~= ~5y of daily data.
def _bucket_history_length(n: int) -> str:
    if n < 60: return "very_short"
    if n < 200: return "short"
    if n < 2000: return "medium"
    return "long"


_HORIZON_DIFFICULTY: dict[str, list[tuple[int, str]]] = {
    "H":  [(24, "very_short"), (168, "short"), (1000, "medium")],
    "D":  [(7, "very_short"), (30, "short"), (90, "medium")],
    "W":  [(4, "very_short"), (13, "short"), (52, "medium")],
    "MS": [(3, "very_short"), (12, "short"), (24, "medium")],
    "M":  [(3, "very_short"), (12, "short"), (24, "medium")],
    "QS": [(2, "very_short"), (4, "short"), (8, "medium")],
    "YS": [(1, "very_short"), (3, "short"), (5, "medium")],
}


def _bucket_horizon_difficulty(freq: str, horizon: int) -> str:
    bands = _HORIZON_DIFFICULTY.get(freq)
    if bands is None:
        return "medium"  # safe fallback for unknown frequencies
    for max_val, label in bands:
        if horizon <= max_val:
            return label
    return "long"


# ---------------------------------------------------------------------------
# Forecasting decompositions
# ---------------------------------------------------------------------------

def _detect_per_series(series: pd.Series, freq: str) -> tuple[bool, bool, bool]:
    """Return (seasonality, trend, stationarity) for one series."""
    from statsmodels.tsa.stattools import adfuller, acf
    from scipy.stats import kendalltau

    seasonality = False
    if len(series) >= 24:
        period = {"H": 24, "D": 7, "W": 52, "MS": 12, "M": 12, "QS": 4, "YS": 1}.get(freq, 1)
        if period > 1 and len(series) > 2 * period:
            try:
                acfs = acf(series.dropna(), nlags=min(period * 2, len(series) // 2 - 1))
                seasonality = abs(acfs[period]) > 0.3 if period < len(acfs) else False
            except Exception:
                seasonality = False

    trend = False
    if len(series) >= 10:
        try:
            x = np.arange(len(series))
            tau, p = kendalltau(x, series.values)
            trend = p < 0.05 and abs(tau) > 0.1
        except Exception:
            trend = False

    stationary = False
    if len(series) >= 12:
        try:
            res = adfuller(series.dropna(), autolag="AIC")
            stationary = res[1] < 0.05
        except Exception:
            stationary = False

    return seasonality, trend, stationary


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_dataset_profile(dataset_path: Path, task_metadata: dict[str, Any]) -> DatasetProfile:
    """Compute the bucketed profile for the given canonical CSV."""
    df = pd.read_csv(dataset_path)
    problem_type = task_metadata["problem_type"]
    target = task_metadata["target_column"]

    # Universal
    n_rows = _bucket_n_rows(len(df))
    feature_df = df.drop(columns=[target], errors="ignore")
    n_features = _bucket_n_features(len(feature_df.columns))
    missing_rate = _bucket_missing(float(df.isnull().mean().mean()))
    n_cat = sum(1 for c in feature_df.columns if pd.api.types.is_object_dtype(feature_df[c]) or isinstance(feature_df[c].dtype, pd.CategoricalDtype))
    n_num = sum(1 for c in feature_df.columns if pd.api.types.is_numeric_dtype(feature_df[c]))

    profile: dict[str, Any] = {
        "schema_version": 1,
        "problem_type": problem_type,
        "n_rows": n_rows,
        "n_features": n_features,
        "missing_rate": missing_rate,
        "n_categorical_features": _bucket_count(n_cat),
        "n_numerical_features": _bucket_count(n_num),
    }

    if problem_type == "classification":
        profile["n_classes"] = _bucket_n_classes(df[target].nunique())
        profile["class_balance"] = _bucket_class_balance(df[target].value_counts())
    elif problem_type == "regression":
        profile["target_distribution"] = _bucket_target_distribution(df[target].dropna())
    elif problem_type == "forecasting":
        dt_col = task_metadata["datetime_column"]
        sid_cols = task_metadata.get("series_id_columns") or []
        freq = task_metadata["frequency"]
        horizon = int(task_metadata["forecast_horizon"])

        df[dt_col] = pd.to_datetime(df[dt_col])
        if sid_cols:
            grouped = df.groupby(sid_cols)
            n_series = grouped.ngroups
            per_series_len = int(grouped.size().min())
        else:
            n_series = 1
            per_series_len = len(df)

        # Per-series stats: take a sample if many series
        sample_groups = (
            list(df.groupby(sid_cols))[:5] if sid_cols else [(("__single__",), df)]
        )
        votes_seasonal = votes_trend = votes_stationary = 0
        for _, g in sample_groups:
            s = g.set_index(dt_col)[target].sort_index()
            seas, tren, stat = _detect_per_series(s, freq)
            votes_seasonal += int(seas)
            votes_trend += int(tren)
            votes_stationary += int(stat)
        n_voted = max(len(sample_groups), 1)

        # Exogenous = any column not target/datetime/series_id
        protected = {target, dt_col, *sid_cols}
        exogenous = any(c not in protected for c in df.columns)

        profile.update({
            "n_series": _bucket_n_series(n_series),
            "history_length": _bucket_history_length(per_series_len),
            "frequency": freq,
            "horizon_difficulty": _bucket_horizon_difficulty(freq, horizon),
            "forecast_horizon_raw": horizon,
            "exogenous_features_available": bool(exogenous),
            "seasonality_detected": (votes_seasonal / n_voted) >= 0.5,
            "trend_detected": (votes_trend / n_voted) >= 0.5,
            "stationarity": (votes_stationary / n_voted) >= 0.5,
        })

    return DatasetProfile(**profile)
