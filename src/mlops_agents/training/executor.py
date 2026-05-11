"""Deterministic multi-candidate training executor."""
from __future__ import annotations

import pickle
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold

from mlops_agents.config.settings import settings
from mlops_agents.contracts.profile import DatasetProfile
from mlops_agents.contracts.training import (
    ExogStrategySettings,
    ForecastingSettings,
    TrainingPlan,
    TrainingPlanCandidate,
    TrainingResult,
    TrialBudget,
)
from mlops_agents.models.factories import FACTORY_REGISTRY
from mlops_agents.models.loader import get_model
from mlops_agents.models.search_spaces import build_suggest_fn
from mlops_agents.training.exog_extender import align_val_exog_index, extend_exog
from mlops_agents.training.experience import build_task_id, write_experience_record
from mlops_agents.training.override_validation import narrow_search_space
from mlops_agents.training.profiler import build_dataset_profile
from mlops_agents.training.splitter import split_dataset
from mlops_agents.training.trial_budget import allocate_trials
from mlops_agents.training.validation_folds import iter_folds
from mlops_agents.training.validation_policy import (
    resolve_rolling_window_size,
    select_validation_strategy,
    validate_forecasting_plan,
)
from mlops_agents.utils.logging import get_logger

logger = get_logger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

DEFAULT_METRIC = {
    "classification": "macro_f1",
    "regression": "rmse",
    "forecasting": "rmse",
}

METRIC_DIRECTION = {
    "macro_f1": "maximize",
    "accuracy": "maximize",
    "roc_auc": "maximize",
    "r2": "maximize",
    "rmse": "minimize",
    "mae": "minimize",
    "mape": "minimize",
    "smape": "minimize",
}


def _cls_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def _reg_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _fc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    out: dict[str, float] = {"rmse": rmse, "mae": mae}
    if (y_true != 0).all():
        out["mape"] = float(np.mean(np.abs((y_true - y_pred) / y_true)))
    out["smape"] = float(
        np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-10))
    )
    return out


# ---------------------------------------------------------------------------
# Champion selection
# ---------------------------------------------------------------------------


def _pick_champion(results: list[dict], direction: str, tol: float) -> dict:
    successful = [r for r in results if r["status"] == "successful"]
    if not successful:
        raise RuntimeError(f"All candidates failed: {[r['model_key'] for r in results]}")
    if direction == "maximize":
        best_score = max(r["best_score"] for r in successful)
        threshold = best_score * (1 - tol)
        tied = [r for r in successful if r["best_score"] >= threshold]
    else:
        best_score = min(r["best_score"] for r in successful)
        threshold = best_score * (1 + tol)
        tied = [r for r in successful if r["best_score"] <= threshold]
    tied.sort(key=lambda r: r["complexity_rank"])
    return tied[0]


# ---------------------------------------------------------------------------
# Candidate runner — classification
# ---------------------------------------------------------------------------


def _run_candidate_classification(
    candidate: TrainingPlanCandidate,
    train_pool: pd.DataFrame,
    target: str,
    n_trials: int,
    metric: str,
    direction: str,
) -> dict:
    spec = get_model(candidate.model_key)
    narrowed = (
        narrow_search_space(candidate.model_key, candidate.search_space_override)
        if candidate.search_space_override
        else spec.search_space
    )
    suggest_fn = build_suggest_fn(narrowed)
    factory = FACTORY_REGISTRY[spec.factory]
    X = train_pool.drop(columns=[target])
    y = train_pool[target]
    started = time.perf_counter()

    def objective(trial: optuna.Trial) -> float:
        params = suggest_fn(trial)
        skf = StratifiedKFold(n_splits=settings.cv_folds, shuffle=True, random_state=42)
        scores = [
            _cls_metrics(
                y.iloc[vi],
                factory(params).fit(X.iloc[ti], y.iloc[ti]).predict(X.iloc[vi]),
            )[metric]
            for ti, vi in skf.split(X, y)
        ]
        return float(np.mean(scores))

    try:
        study = optuna.create_study(
            direction=direction, sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=n_trials)
        if not study.best_trial:
            raise RuntimeError("No successful trial")
        best_params = study.best_params
        best_score = study.best_value
        n_used = len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        )
    except Exception:
        try:
            skf = StratifiedKFold(n_splits=settings.cv_folds, shuffle=True, random_state=42)
            scores = [
                _cls_metrics(
                    y.iloc[vi],
                    factory(spec.default_params).fit(X.iloc[ti], y.iloc[ti]).predict(X.iloc[vi]),
                )[metric]
                for ti, vi in skf.split(X, y)
            ]
            best_params, best_score, n_used = spec.default_params, float(np.mean(scores)), 1
        except Exception as e2:
            return {
                "model_key": candidate.model_key,
                "status": "failed",
                "error_type": type(e2).__name__,
                "error_message": str(e2),
                "n_trials_used": 0,
                "duration_s": time.perf_counter() - started,
                "complexity_rank": spec.complexity_rank,
            }
    return {
        "model_key": candidate.model_key,
        "status": "successful",
        "best_params": best_params,
        "best_score": float(best_score),
        "best_score_std": 0.0,
        "n_trials_used": n_used,
        "duration_s": time.perf_counter() - started,
        "complexity_rank": spec.complexity_rank,
    }


# ---------------------------------------------------------------------------
# Candidate runner — regression
# ---------------------------------------------------------------------------


def _run_candidate_regression(
    candidate: TrainingPlanCandidate,
    train_pool: pd.DataFrame,
    target: str,
    n_trials: int,
    metric: str,
    direction: str,
) -> dict:
    spec = get_model(candidate.model_key)
    narrowed = (
        narrow_search_space(candidate.model_key, candidate.search_space_override)
        if candidate.search_space_override
        else spec.search_space
    )
    suggest_fn = build_suggest_fn(narrowed)
    factory = FACTORY_REGISTRY[spec.factory]
    X = train_pool.drop(columns=[target])
    y = train_pool[target]
    started = time.perf_counter()

    def objective(trial: optuna.Trial) -> float:
        params = suggest_fn(trial)
        kf = KFold(n_splits=settings.cv_folds, shuffle=True, random_state=42)
        scores = [
            _reg_metrics(
                y.iloc[vi],
                factory(params).fit(X.iloc[ti], y.iloc[ti]).predict(X.iloc[vi]),
            )[metric]
            for ti, vi in kf.split(X)
        ]
        return float(np.mean(scores))

    try:
        study = optuna.create_study(
            direction=direction, sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=n_trials)
        if not study.best_trial:
            raise RuntimeError("No successful trial")
        best_params = study.best_params
        best_score = study.best_value
        n_used = len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        )
    except Exception:
        try:
            kf = KFold(n_splits=settings.cv_folds, shuffle=True, random_state=42)
            scores = [
                _reg_metrics(
                    y.iloc[vi],
                    factory(spec.default_params).fit(X.iloc[ti], y.iloc[ti]).predict(X.iloc[vi]),
                )[metric]
                for ti, vi in kf.split(X)
            ]
            best_params, best_score, n_used = spec.default_params, float(np.mean(scores)), 1
        except Exception as e2:
            return {
                "model_key": candidate.model_key,
                "status": "failed",
                "error_type": type(e2).__name__,
                "error_message": str(e2),
                "n_trials_used": 0,
                "duration_s": time.perf_counter() - started,
                "complexity_rank": spec.complexity_rank,
            }
    return {
        "model_key": candidate.model_key,
        "status": "successful",
        "best_params": best_params,
        "best_score": float(best_score),
        "best_score_std": 0.0,
        "n_trials_used": n_used,
        "duration_s": time.perf_counter() - started,
        "complexity_rank": spec.complexity_rank,
    }


# ---------------------------------------------------------------------------
# Candidate runner — forecasting
# ---------------------------------------------------------------------------


def _is_statsforecast_model(model_key: str) -> bool:
    return get_model(model_key).library == "statsforecast"


def _to_sf_format(
    df: pd.DataFrame, target: str, dt_col: str, sid_cols: list[str]
) -> pd.DataFrame:
    out = df.rename(columns={target: "y", dt_col: "ds"}).copy()
    if sid_cols:
        if len(sid_cols) == 1:
            out = out.rename(columns={sid_cols[0]: "unique_id"})
        else:
            out["unique_id"] = out[sid_cols].astype(str).agg("__".join, axis=1)
    else:
        out["unique_id"] = "__single__"
    out["ds"] = pd.to_datetime(out["ds"])
    return out[["unique_id", "ds", "y"]]


def _build_exog_df(
    df: pd.DataFrame,
    dt_col: str,
    target: str,
    sid_cols: list[str],
    series_dict: dict[str, pd.Series] | None = None,
) -> pd.DataFrame | None:
    """Extract exogenous columns as a DataFrame whose index matches series_dict.

    skforecast requires exog and series to share the same index type
    (DatetimeIndex or RangeIndex). Returns None when no exogenous columns exist.
    """
    exclude = {dt_col, target} | set(sid_cols)
    exog_cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not exog_cols:
        return None
    # For panel data skip exog — multi-series exog alignment is complex
    if sid_cols:
        return None
    exog = df.sort_values(dt_col).set_index(dt_col)[exog_cols].copy()
    exog.index = pd.to_datetime(exog.index)
    # If series_dict fell back to RangeIndex we must match it
    if series_dict is not None:
        sample = next(iter(series_dict.values()))
        if isinstance(sample.index, pd.RangeIndex):
            exog = exog.reset_index(drop=True)
    return exog


def _build_series_dict(
    df: pd.DataFrame, dt_col: str, target: str, sid_cols: list[str], freq_hint: str | None = None
) -> dict[str, pd.Series]:
    """Build series_dict for skforecast with explicit freq or RangeIndex fallback."""
    def _prep(s: pd.Series) -> pd.Series:
        s = s.sort_index()
        for freq in ([freq_hint] if freq_hint else []) + ([pd.infer_freq(s.index)] if pd.infer_freq(s.index) else []):
            if not freq:
                continue
            try:
                candidate = s.asfreq(freq)
                if candidate.notna().all():
                    return candidate
            except Exception:
                pass
        return s.reset_index(drop=True)

    if sid_cols:
        return {
            sid: _prep(g.set_index(dt_col)[target])
            for sid, g in df.groupby(sid_cols[0])
        }
    return {"__single__": _prep(df.set_index(dt_col)[target])}


def _align_train_exog_index(
    exog: pd.DataFrame, series_dict: dict[str, pd.Series]
) -> pd.DataFrame:
    """Match exog's index type to a sample series in series_dict.

    skforecast requires exog and series to share the same index type
    (DatetimeIndex or RangeIndex). The training DataFrame may carry a
    RangeIndex (from CSV loading) while series_dict produces a
    DatetimeIndex when frequency can be inferred.
    """
    sample = next(iter(series_dict.values()))
    exog = exog.copy()
    if isinstance(sample.index, pd.DatetimeIndex):
        exog.index = sample.index
        return exog
    return exog.reset_index(drop=True)


def _resolve_exog_availability(df_columns: list[str], task_metadata: dict) -> dict[str, str]:
    """Return {col: 'known_future' | 'unknown_future'} for every exog column.

    If task_metadata['exogenous_columns'] is present, it is authoritative and
    unlisted non-target/non-date/non-sid columns are dropped from the exog set.
    If absent, all non-protected columns are treated as unknown_future.
    """
    target = task_metadata["target_column"]
    dt = task_metadata["datetime_column"]
    sids = set(task_metadata.get("series_id_columns") or [])
    protected = {target, dt, *sids}

    declared = task_metadata.get("exogenous_columns")
    if declared is not None:
        return {e["name"]: e["future_availability"] for e in declared}
    return {c: "unknown_future" for c in df_columns if c not in protected}


def _run_candidate_forecasting_panel(
    candidate: TrainingPlanCandidate,
    train_pool: pd.DataFrame,
    task_metadata: dict[str, Any],
    n_trials: int,
    metric: str,
    direction: str,
) -> dict:
    """Preserved original forecasting path — single-split, no leakage-safe exog.

    Used by the rewritten _run_candidate_forecasting for panel data (sid_cols
    non-empty) where the multi-target leakage-safe fold iteration is deferred
    to v2.
    """
    spec = get_model(candidate.model_key)
    target = task_metadata["target_column"]
    dt_col = task_metadata["datetime_column"]
    sid_cols = task_metadata.get("series_id_columns") or []
    horizon = int(task_metadata["forecast_horizon"])
    started = time.perf_counter()

    pool = train_pool.copy()
    pool[dt_col] = pd.to_datetime(pool[dt_col])
    if sid_cols:
        pool_sorted = pool.sort_values(sid_cols + [dt_col])
        val = pool_sorted.groupby(sid_cols).tail(horizon)
        cand_train = pool_sorted.drop(val.index)
    else:
        pool_sorted = pool.sort_values(dt_col)
        val = pool_sorted.tail(horizon)
        cand_train = pool_sorted.iloc[:-horizon]

    is_stat = _is_statsforecast_model(candidate.model_key)
    factory = FACTORY_REGISTRY[spec.factory]

    freq = task_metadata.get("frequency")

    def fit_score(params: dict) -> float:
        if is_stat:
            sf = factory({"task_metadata": task_metadata, "params": params})
            sf.fit(_to_sf_format(cand_train, target, dt_col, sid_cols))
            fcst = sf.predict(h=horizon)
            # statsforecast output: columns ['unique_id', 'ds', <ModelName>]
            model_col = [c for c in fcst.columns if c not in ("unique_id", "ds")][0]
            val_sf = _to_sf_format(val, target, dt_col, sid_cols)
            merged = val_sf.merge(fcst, on=["unique_id", "ds"])
            if merged.empty:
                # Irregular dates (e.g. yfinance weekly): align predictions by position
                val_s = val_sf.sort_values(["unique_id", "ds"]).reset_index(drop=True)
                fct_s = fcst.sort_values(["unique_id", "ds"]).reset_index(drop=True)
                n = min(len(val_s), len(fct_s))
                if n == 0:
                    raise ValueError("Statsforecast produced no predictions")
                return _fc_metrics(val_s["y"].values[:n], fct_s[model_col].values[:n])[metric]
            return _fc_metrics(merged["y"].values, merged[model_col].values)[metric]
        else:
            forecaster = factory({"task_metadata": task_metadata, "params": params})
            series_dict = _build_series_dict(cand_train, dt_col, target, sid_cols, freq)
            train_exog = _build_exog_df(cand_train, dt_col, target, sid_cols, series_dict)
            val_exog = _build_exog_df(val, dt_col, target, sid_cols, series_dict)
            # When series uses RangeIndex, val_exog must continue from train_len
            if val_exog is not None and isinstance(val_exog.index, pd.RangeIndex):
                train_len = len(next(iter(series_dict.values())))
                val_exog.index = pd.RangeIndex(train_len, train_len + len(val_exog))
            forecaster.fit(series=series_dict, exog=train_exog)
            # skforecast 0.22: predict returns DataFrame with DatetimeIndex,
            # columns ['level', 'pred']
            preds = forecaster.predict(steps=horizon, exog=val_exog)
            preds = preds.reset_index().rename(columns={"index": "ds"})
            # preds now has columns ['ds', 'level', 'pred']
            if sid_cols:
                val_long = val.rename(
                    columns={sid_cols[0]: "level", target: "y_true", dt_col: "ds"}
                )
            else:
                val_long = val.rename(columns={target: "y_true", dt_col: "ds"})
                val_long = val_long.copy()
                val_long["level"] = "__single__"
            val_long["ds"] = pd.to_datetime(val_long["ds"])
            preds["ds"] = pd.to_datetime(preds["ds"])
            joined = val_long[["level", "ds", "y_true"]].merge(
                preds[["level", "ds", "pred"]], on=["level", "ds"], how="inner"
            )
            if joined.empty:
                # fallback: align by order
                return _fc_metrics(
                    val_long["y_true"].values,
                    preds["pred"].values[: len(val_long)],
                )[metric]
            return _fc_metrics(joined["y_true"].values, joined["pred"].values)[metric]

    narrowed = (
        narrow_search_space(candidate.model_key, candidate.search_space_override)
        if candidate.search_space_override
        else spec.search_space
    )
    suggest_fn = build_suggest_fn(narrowed)

    def objective(trial: optuna.Trial) -> float:
        return fit_score(suggest_fn(trial))

    try:
        if not narrowed.params:
            best_score = fit_score(spec.default_params)
            best_params, n_used = spec.default_params, 1
        else:
            study = optuna.create_study(
                direction=direction, sampler=optuna.samplers.TPESampler(seed=42)
            )
            study.optimize(objective, n_trials=n_trials)
            if not study.best_trial:
                raise RuntimeError("No successful trial")
            best_params = study.best_params
            best_score = study.best_value
            n_used = len(
                [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            )
    except Exception:
        try:
            best_score = fit_score(spec.default_params)
            best_params, n_used = spec.default_params, 1
        except Exception as e2:
            return {
                "model_key": candidate.model_key,
                "status": "failed",
                "error_type": type(e2).__name__,
                "error_message": str(e2),
                "n_trials_used": 0,
                "duration_s": time.perf_counter() - started,
                "complexity_rank": spec.complexity_rank,
            }
    return {
        "model_key": candidate.model_key,
        "status": "successful",
        "best_params": best_params,
        "best_score": float(best_score),
        "best_score_std": 0.0,
        "n_trials_used": n_used,
        "duration_s": time.perf_counter() - started,
        "complexity_rank": spec.complexity_rank,
    }


def _run_candidate_forecasting(
    candidate: TrainingPlanCandidate,
    train_pool: pd.DataFrame,
    task_metadata: dict[str, Any],
    n_trials: int,
    metric: str,
    direction: str,
    forecasting_settings: ForecastingSettings,
    profile: DatasetProfile,
) -> dict:
    spec = get_model(candidate.model_key)
    target = task_metadata["target_column"]
    dt_col = task_metadata["datetime_column"]
    sid_cols = task_metadata.get("series_id_columns") or []
    horizon = int(task_metadata["forecast_horizon"])
    freq = task_metadata.get("frequency")
    started = time.perf_counter()

    pool = train_pool.copy()
    pool[dt_col] = pd.to_datetime(pool[dt_col])

    is_stat = _is_statsforecast_model(candidate.model_key)
    factory = FACTORY_REGISTRY[spec.factory]

    train_pool_stats = {
        "single_series": not sid_cols,
        "series_lengths": (pool.groupby(sid_cols[0]).size().to_dict() if sid_cols else None),
        "total_len": len(pool),
    }

    # Plan-level guardrail
    throwaway = TrainingPlan(
        problem_type="forecasting",
        candidates=[candidate],
        trial_budget=TrialBudget(total_trials=1, allocation_strategy="equal",
                                 min_trials_per_candidate=1, max_trials_per_candidate=1),
        forecasting_settings=forecasting_settings,
    )

    # Resolve auto window_size for rolling_window BEFORE validation
    vs = forecasting_settings.validation_strategy
    if vs.type == "rolling_window" and vs.window_size is None:
        vs = vs.model_copy(update={
            "window_size": resolve_rolling_window_size(
                len(pool), horizon, vs.n_folds, season_length=None,
            )
        })
        forecasting_settings = forecasting_settings.model_copy(update={"validation_strategy": vs})
        throwaway = throwaway.model_copy(update={"forecasting_settings": forecasting_settings})

    validate_forecasting_plan(throwaway, task_metadata, profile, train_pool_stats)

    # Multi-target panel: delegate to preserved path (no exog support in v1)
    if sid_cols:
        return _run_candidate_forecasting_panel(
            candidate, pool, task_metadata, n_trials, metric, direction,
        )

    # ─── Single-target leakage-safe path ────────────────────────────
    availability = _resolve_exog_availability(list(pool.columns), task_metadata)
    exog_columns = list(availability.keys())
    strategies = forecasting_settings.exog_strategies

    exog_cache: dict[tuple, pd.Series] = {}

    def fit_score(params: dict) -> tuple[float, list[float], list[dict]]:
        fold_scores: list[float] = []
        fold_failures: list[dict] = []

        for fold_id, (train_idx, val_idx) in enumerate(iter_folds(pool, vs, dt_col, sid_cols)):
            cand_train = pool.loc[train_idx].reset_index(drop=True)
            cand_val = pool.loc[val_idx].reset_index(drop=True)

            if is_stat:
                # statsforecast path: ignores exog (existing behavior)
                sf = factory({"task_metadata": task_metadata, "params": params})
                sf.fit(_to_sf_format(cand_train, target, dt_col, sid_cols))
                fcst = sf.predict(h=horizon)
                model_col = [c for c in fcst.columns if c not in ("unique_id", "ds")][0]
                val_sf = _to_sf_format(cand_val, target, dt_col, sid_cols)
                merged = val_sf.merge(fcst, on=["unique_id", "ds"])
                if merged.empty:
                    val_s = val_sf.sort_values(["unique_id", "ds"]).reset_index(drop=True)
                    fct_s = fcst.sort_values(["unique_id", "ds"]).reset_index(drop=True)
                    n = min(len(val_s), len(fct_s))
                    if n == 0:
                        raise ValueError("Statsforecast produced no predictions")
                    score = _fc_metrics(val_s["y"].values[:n], fct_s[model_col].values[:n])[metric]
                else:
                    score = _fc_metrics(merged["y"].values, merged[model_col].values)[metric]
                fold_scores.append(score)
                continue

            # ── Skforecast path with leakage-safe exog ────────────
            forecaster = factory({"task_metadata": task_metadata, "params": params})
            series_dict = _build_series_dict(cand_train, dt_col, target, sid_cols, freq)

            future_values: dict[str, pd.Series] = {}
            for col in exog_columns:
                avail = availability[col]
                if avail == "known_future":
                    future_values[col] = cand_val[col].reset_index(drop=True)
                    continue
                strat = strategies.per_column.get(col, strategies.default_unknown_future)
                if strat == "drop":
                    continue
                cache_key = (col, strat, fold_id, "default")
                if cache_key in exog_cache:
                    future_values[col] = exog_cache[cache_key]
                else:
                    preds, fail = extend_exog(cand_train[col], horizon, strat, freq)
                    future_values[col] = preds
                    exog_cache[cache_key] = preds
                    if fail is not None:
                        fold_failures.append(fail | {"fold_id": fold_id, "column": col})

            used_cols = list(future_values.keys())
            if used_cols:
                train_exog = _align_train_exog_index(cand_train[used_cols], series_dict)
            else:
                train_exog = None
            val_exog = None
            if used_cols:
                val_exog_raw = pd.DataFrame(future_values)
                val_exog = align_val_exog_index(
                    val_exog_raw, series_dict, train_len=len(next(iter(series_dict.values()))),
                    dt_col=dt_col, freq=freq,
                )

            forecaster.fit(series=series_dict, exog=train_exog)
            preds = forecaster.predict(steps=horizon, exog=val_exog)
            preds = preds.reset_index().rename(columns={"index": "ds"})
            val_long = cand_val.rename(columns={target: "y_true", dt_col: "ds"}).copy()
            val_long["level"] = "__single__"
            val_long["ds"] = pd.to_datetime(val_long["ds"])
            preds["ds"] = pd.to_datetime(preds["ds"])
            joined = val_long[["level", "ds", "y_true"]].merge(
                preds[["level", "ds", "pred"]], on=["level", "ds"], how="inner"
            )
            if joined.empty:
                score = _fc_metrics(
                    val_long["y_true"].values, preds["pred"].values[: len(val_long)],
                )[metric]
            else:
                score = _fc_metrics(joined["y_true"].values, joined["pred"].values)[metric]
            fold_scores.append(score)

        return float(np.mean(fold_scores)), fold_scores, fold_failures

    narrowed = (
        narrow_search_space(candidate.model_key, candidate.search_space_override)
        if candidate.search_space_override else spec.search_space
    )
    suggest_fn = build_suggest_fn(narrowed)

    last_per_fold: list[float] = []
    last_failures: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        nonlocal last_per_fold, last_failures
        params = suggest_fn(trial)
        score, per_fold, failures = fit_score(params)
        last_per_fold = per_fold
        last_failures = failures
        return score

    try:
        if not narrowed.params:
            best_score, last_per_fold, last_failures = fit_score(spec.default_params)
            best_params, n_used = spec.default_params, 1
        else:
            study = optuna.create_study(
                direction=direction, sampler=optuna.samplers.TPESampler(seed=42)
            )
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            best_params = study.best_params
            best_score = study.best_value
            n_used = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        status = "successful"
    except Exception as e:
        logger.exception(f"[{candidate.model_key}] failed: {e}")
        return {
            "model_key": candidate.model_key, "status": "failed",
            "best_params": {}, "best_score": float("inf"),
            "best_score_std": 0.0, "n_trials_used": 0,
            "duration_s": time.perf_counter() - started,
            "complexity_rank": spec.complexity_rank,
            "per_fold_scores": [], "exog_fit_failures": [],
        }

    return {
        "model_key": candidate.model_key,
        "status": status,
        "best_params": best_params,
        "best_score": float(best_score),
        "best_score_std": float(np.std(last_per_fold)) if last_per_fold else 0.0,
        "n_trials_used": n_used,
        "duration_s": time.perf_counter() - started,
        "complexity_rank": spec.complexity_rank,
        "per_fold_scores": [float(x) for x in last_per_fold],
        "exog_fit_failures": last_failures,
    }


# ---------------------------------------------------------------------------
# Champion retraining helpers
# ---------------------------------------------------------------------------


def _retrain_tabular(
    spec: Any, champion: dict, train_pool: pd.DataFrame, target: str, models_dir: Path
) -> Path:
    factory = FACTORY_REGISTRY[spec.factory]
    X = train_pool.drop(columns=[target])
    y = train_pool[target]
    model = factory(champion["best_params"])
    model.fit(X, y)
    path = models_dir / f"champion_{champion['model_key']}.pkl"
    with path.open("wb") as f:
        pickle.dump(model, f)
    return path


def _retrain_forecasting(
    spec: Any,
    champion: dict,
    train_pool: pd.DataFrame,
    task_metadata: dict[str, Any],
    models_dir: Path,
) -> Path:
    factory = FACTORY_REGISTRY[spec.factory]
    target = task_metadata["target_column"]
    dt_col = task_metadata["datetime_column"]
    sid_cols = task_metadata.get("series_id_columns") or []
    train_pool = train_pool.copy()
    train_pool[dt_col] = pd.to_datetime(train_pool[dt_col])
    path = models_dir / f"champion_{champion['model_key']}.pkl"
    if _is_statsforecast_model(champion["model_key"]):
        sf = factory({"task_metadata": task_metadata, "params": champion["best_params"]})
        sf.fit(_to_sf_format(train_pool, target, dt_col, sid_cols))
        with path.open("wb") as f:
            pickle.dump(sf, f)
        return path

    forecaster = factory({"task_metadata": task_metadata, "params": champion["best_params"]})
    freq = task_metadata.get("frequency")
    series_dict = _build_series_dict(train_pool, dt_col, target, sid_cols, freq)
    if not sid_cols:
        availability = _resolve_exog_availability(list(train_pool.columns), task_metadata)
        used_cols = [c for c in availability if c in train_pool.columns]
        if used_cols:
            train_exog = _align_train_exog_index(train_pool[used_cols], series_dict)
        else:
            train_exog = None
    else:
        train_exog = None
    forecaster.fit(series=series_dict, exog=train_exog)
    with path.open("wb") as f:
        pickle.dump(forecaster, f)
    return path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_training_plan(
    plan: TrainingPlan,
    processed_dataset_path: Path,
    target_column: str,
    task_metadata: dict[str, Any],
    output_dir: Path,
    mlflow_experiment: str,
    random_state: int = 42,
) -> TrainingResult:
    metric = plan.metric_to_optimize or DEFAULT_METRIC[plan.problem_type]
    direction = METRIC_DIRECTION[metric]

    profile = build_dataset_profile(processed_dataset_path, task_metadata)

    # Resolve forecasting_settings before any candidate runs
    fs = plan.forecasting_settings
    if fs is None and plan.problem_type == "forecasting":
        fs = ForecastingSettings(
            validation_strategy=select_validation_strategy(profile, task_metadata),
            exog_strategies=ExogStrategySettings(),
        )
        plan = plan.model_copy(update={"forecasting_settings": fs})

    train_pool_path, test_path, split_meta_path = split_dataset(
        processed_dataset_path, task_metadata, output_dir, random_state=random_state
    )
    train_pool = pd.read_csv(train_pool_path)
    allocations = allocate_trials(plan.candidates, plan.trial_budget)

    mlflow.set_experiment(mlflow_experiment)
    candidate_results: list[dict] = []
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    with mlflow.start_run(run_name=f"pipeline_{ts}") as parent:
        parent_run_id = parent.info.run_id

        for cand in sorted(plan.candidates, key=lambda c: c.priority):
            with mlflow.start_run(run_name=cand.model_key, nested=True) as child:
                if plan.problem_type == "classification":
                    res = _run_candidate_classification(
                        cand, train_pool, target_column,
                        allocations[cand.model_key], metric, direction,
                    )
                elif plan.problem_type == "regression":
                    res = _run_candidate_regression(
                        cand, train_pool, target_column,
                        allocations[cand.model_key], metric, direction,
                    )
                else:
                    res = _run_candidate_forecasting(
                        cand, train_pool, task_metadata,
                        allocations[cand.model_key], metric, direction,
                        forecasting_settings=fs,
                        profile=profile,
                    )
                res["mlflow_run_id"] = child.info.run_id
                if res["status"] == "successful":
                    mlflow.log_params(res["best_params"])
                    mlflow.log_metric(metric, res["best_score"])
                else:
                    mlflow.set_tag("status", "failed")
                candidate_results.append(res)

        champion = _pick_champion(candidate_results, direction, settings.tie_tolerance_relative)

        models_dir = output_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        spec = get_model(champion["model_key"])

        if plan.problem_type in ("classification", "regression"):
            champion_path = _retrain_tabular(spec, champion, train_pool, target_column, models_dir)
        else:
            champion_path = _retrain_forecasting(spec, champion, train_pool, task_metadata, models_dir)

        with mlflow.start_run(run_id=champion["mlflow_run_id"], nested=True):
            mlflow.set_tag("champion", "true")
            mlflow.log_artifact(str(champion_path))
        mlflow.set_tag("champion_run_id", champion["mlflow_run_id"])

        val_strategy = (
            "stratified_5_fold_cv"
            if plan.problem_type == "classification"
            else "kfold_5_shuffle"
            if plan.problem_type == "regression"
            else plan.forecasting_settings.validation_strategy.type
            if plan.forecasting_settings is not None
            else "temporal_holdout"
        )

        # Forecasting-specific MLflow params, per-fold metrics, and experience extras
        forecasting_extras: dict[str, Any] = {}
        if plan.problem_type == "forecasting" and fs is not None:
            mlflow.log_param("validation_strategy_type", fs.validation_strategy.type)
            mlflow.log_param("validation_n_folds", fs.validation_strategy.n_folds)
            mlflow.log_param("exog_default_strategy", fs.exog_strategies.default_unknown_future)
            mlflow.log_param("expected_drift", task_metadata.get("expected_drift", "low"))

            per_fold = champion.get("per_fold_scores", [])
            for i, s in enumerate(per_fold):
                mlflow.log_metric(f"fold_{i}_{metric}", s)
            if per_fold:
                mlflow.log_metric(f"fold_mean_{metric}", float(np.mean(per_fold)))
                mlflow.log_metric(f"fold_std_{metric}", float(np.std(per_fold)))

            availability = _resolve_exog_availability(list(train_pool.columns), task_metadata)
            used_strategies: dict[str, str] = {}
            for col, avail in availability.items():
                if avail == "known_future":
                    used_strategies[col] = "known_future"
                else:
                    used_strategies[col] = fs.exog_strategies.per_column.get(
                        col, fs.exog_strategies.default_unknown_future,
                    )
            forecasting_extras = {
                "validation_strategy": fs.validation_strategy.model_dump(),
                "exog_availability": availability,
                "exog_strategies": used_strategies,
                "per_fold_metrics": [
                    {"fold_id": i, metric: s}
                    for i, s in enumerate(champion.get("per_fold_scores", []))
                ],
                "exog_fit_failures": champion.get("exog_fit_failures", []),
            }

        task_id = build_task_id(processed_dataset_path.stem, plan.problem_type, run_idx=1)
        record: dict[str, Any] = {
            "task_id": task_id,
            "problem_type": plan.problem_type,
            "dataset_name": processed_dataset_path.stem,
            "dataset_profile": profile.model_dump(),
            "training_plan_input": plan.model_dump(),
            "split_artifacts": {
                "train_pool_path": str(train_pool_path),
                "test_path": str(test_path),
                "split_metadata_path": str(split_meta_path),
            },
            "mlflow": {
                "experiment_name": mlflow_experiment,
                "parent_run_id": parent_run_id,
            },
            "metric_to_optimize": metric,
            "metric_direction": direction,
            "candidate_selection_policy": {
                "primary": "best_validation_score",
                "tie_breaker": "complexity_rank",
                "tie_tolerance_relative": settings.tie_tolerance_relative,
            },
            "models_tested": [
                {k: v for k, v in r.items() if k != "traceback"}
                for r in candidate_results
            ],
            "selected_solution": {
                "model_key": champion["model_key"],
                "hyperparameters": champion["best_params"],
                "validation_strategy": val_strategy,
                "main_metric": metric,
                "validation_score": champion["best_score"],
                "validation_std": champion.get("best_score_std", 0.0),
                "complexity_rank": champion["complexity_rank"],
            },
            "experience_summary": "",
            **forecasting_extras,
        }
        record_path = write_experience_record(record, settings.experience_pool_dir)

    return TrainingResult(
        champion_candidate=champion,
        champion_model_path=str(champion_path),
        train_pool_path=str(train_pool_path),
        test_path=str(test_path),
        split_metadata_path=str(split_meta_path),
        mlflow_parent_run_id=parent_run_id,
        experience_record_path=str(record_path),
        champion_metrics={metric: champion["best_score"]},
    )
