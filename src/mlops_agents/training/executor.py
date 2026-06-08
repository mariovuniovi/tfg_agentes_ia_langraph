"""Deterministic multi-candidate training executor."""
from __future__ import annotations

import pickle
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from mlops_agents.config.settings import settings
from mlops_agents.contracts.profile import DatasetProfile
from mlops_agents.contracts.training import (
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
from mlops_agents.training.exog_policy import resolve_exog_strategies
from mlops_agents.training.experience import build_task_id, write_experience_record
from mlops_agents.training.override_validation import narrow_search_space
from mlops_agents.training.profiler import build_dataset_profile
from mlops_agents.training.splitter import split_dataset
from mlops_agents.training.trial_budget import allocate_trials
from mlops_agents.training.validation_folds import iter_folds
from mlops_agents.training.validation_policy import (
    resolve_rolling_window_size,
    resolve_validation_strategy,
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
# Tabular validation strategy selectors
# ---------------------------------------------------------------------------


def _select_cls_validation(y: pd.Series) -> tuple:
    """Return ('single_split',) or ('stratified_kfold', n_folds).

    Falls back to single_split when the dataset is too small for reliable CV:
    - fewer than settings.min_rows_for_cv total training rows, or
    - fewer than settings.min_class_count_for_cv samples in the smallest class.
    A single deterministic split is much more honest than 2-fold CV on 8 rows.
    """
    n_rows = len(y)
    min_class_count = int(y.value_counts().min())

    if min_class_count < 2:
        logger.warning(
            f"[executor] smallest class has {min_class_count} sample(s) — "
            "stratified CV impossible, using single split"
        )
        return ("single_split",)

    if n_rows < settings.min_rows_for_cv or min_class_count < settings.min_class_count_for_cv:
        logger.warning(
            f"[executor] dataset too small for reliable CV "
            f"(rows={n_rows}, min_class={min_class_count}) — using single stratified split"
        )
        return ("single_split",)

    n_folds = min(settings.cv_folds, min_class_count)
    return ("stratified_kfold", n_folds)


def _select_reg_validation(y: pd.Series) -> tuple:
    """Return ('single_split',) or ('kfold', n_folds)."""
    n_rows = len(y)

    if n_rows < settings.min_rows_for_cv:
        logger.warning(
            f"[executor] dataset too small for reliable CV "
            f"(rows={n_rows}) — using single split"
        )
        return ("single_split",)

    n_folds = min(settings.cv_folds, n_rows)
    return ("kfold", n_folds)


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
    val_strategy = _select_cls_validation(y)
    started = time.perf_counter()

    if val_strategy[0] == "single_split":
        # Fix the split once so all Optuna trials see the same validation set.
        stratify = y if int(y.value_counts().min()) >= 2 else None
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.25, stratify=stratify, random_state=42
        )

        def objective(trial: optuna.Trial) -> float:
            params = suggest_fn(trial)
            score = _cls_metrics(y_val, factory(params).fit(X_tr, y_tr).predict(X_val))[metric]
            trial.set_user_attr("fold_scores", [score])
            return float(score)

        fallback_score_fn = lambda p: _cls_metrics(  # noqa: E731
            y_val, factory(p).fit(X_tr, y_tr).predict(X_val)
        )[metric]
        val_strategy_label = "single_split"
    else:
        n_folds = val_strategy[1]
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        def objective(trial: optuna.Trial) -> float:
            params = suggest_fn(trial)
            scores = [
                _cls_metrics(y.iloc[vi], factory(params).fit(X.iloc[ti], y.iloc[ti]).predict(X.iloc[vi]))[metric]
                for ti, vi in skf.split(X, y)
            ]
            trial.set_user_attr("fold_scores", scores)
            return float(np.mean(scores))

        fallback_score_fn = lambda p: float(np.mean([  # noqa: E731
            _cls_metrics(y.iloc[vi], factory(p).fit(X.iloc[ti], y.iloc[ti]).predict(X.iloc[vi]))[metric]
            for ti, vi in skf.split(X, y)
        ]))
        val_strategy_label = f"stratified_kfold_{n_folds}"

    try:
        study = optuna.create_study(
            direction=direction, sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=n_trials)
        if not study.best_trial:
            raise RuntimeError("No successful trial")
        best_params = study.best_params
        best_score = study.best_value
        best_fold_scores = study.best_trial.user_attrs.get("fold_scores", [])
        n_used = len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        )
    except Exception:
        try:
            score = fallback_score_fn(spec.default_params)
            best_params, best_score, best_fold_scores, n_used = spec.default_params, score, [score], 1
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
        "best_score_std": float(np.std(best_fold_scores)) if best_fold_scores else 0.0,
        "validation_strategy": val_strategy_label,
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
    val_strategy = _select_reg_validation(y)
    started = time.perf_counter()

    if val_strategy[0] == "single_split":
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

        def objective(trial: optuna.Trial) -> float:
            params = suggest_fn(trial)
            score = _reg_metrics(y_val, factory(params).fit(X_tr, y_tr).predict(X_val))[metric]
            trial.set_user_attr("fold_scores", [score])
            return float(score)

        fallback_score_fn = lambda p: _reg_metrics(  # noqa: E731
            y_val, factory(p).fit(X_tr, y_tr).predict(X_val)
        )[metric]
        val_strategy_label = "single_split"
    else:
        n_folds = val_strategy[1]
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        def objective(trial: optuna.Trial) -> float:
            params = suggest_fn(trial)
            scores = [
                _reg_metrics(y.iloc[vi], factory(params).fit(X.iloc[ti], y.iloc[ti]).predict(X.iloc[vi]))[metric]
                for ti, vi in kf.split(X)
            ]
            trial.set_user_attr("fold_scores", scores)
            return float(np.mean(scores))

        fallback_score_fn = lambda p: float(np.mean([  # noqa: E731
            _reg_metrics(y.iloc[vi], factory(p).fit(X.iloc[ti], y.iloc[ti]).predict(X.iloc[vi]))[metric]
            for ti, vi in kf.split(X)
        ]))
        val_strategy_label = f"kfold_{n_folds}"

    try:
        study = optuna.create_study(
            direction=direction, sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=n_trials)
        if not study.best_trial:
            raise RuntimeError("No successful trial")
        best_params = study.best_params
        best_score = study.best_value
        best_fold_scores = study.best_trial.user_attrs.get("fold_scores", [])
        n_used = len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        )
    except Exception:
        try:
            score = fallback_score_fn(spec.default_params)
            best_params, best_score, best_fold_scores, n_used = spec.default_params, score, [score], 1
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
        "best_score_std": float(np.std(best_fold_scores)) if best_fold_scores else 0.0,
        "validation_strategy": val_strategy_label,
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
    # sid_cols is always empty in V1 (panel is out of scope)
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

    # Defensive guard — should never be reached if top-level guard in run_training_plan works
    if sid_cols:
        raise NotImplementedError(
            "Multi-target panel forecasting (series_id_columns non-empty) is out of "
            "scope for V1. V1 supports single-target forecasting with multiple "
            "exogenous predictor columns. Got series_id_columns="
            f"{sid_cols}"
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

    def objective(trial: optuna.Trial) -> float:
        params = suggest_fn(trial)
        score, per_fold, failures = fit_score(params)
        trial.set_user_attr("per_fold_scores", per_fold)
        trial.set_user_attr("exog_fit_failures", failures)
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
            best_trial = study.best_trial
            last_per_fold = best_trial.user_attrs.get("per_fold_scores", [])
            last_failures = best_trial.user_attrs.get("exog_fit_failures", [])
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
# Test-set forecast helpers
# ---------------------------------------------------------------------------


def _build_test_exog(
    train_pool: pd.DataFrame,
    test_df: pd.DataFrame,
    task_metadata: dict[str, Any],
    forecasting_settings: ForecastingSettings,
    horizon: int,
    freq: str | None,
    series_dict: dict[str, pd.Series],
) -> pd.DataFrame | None:
    """Build the test-horizon exog for a skforecast champion.

    Mirrors validation: known_future columns use the actual test values;
    unknown_future columns are extended from train history (no `drop` — all
    exog kept). No oracle peeking at unknown-future actuals.
    """
    dt_col = task_metadata["datetime_column"]
    availability = _resolve_exog_availability(list(train_pool.columns), task_metadata)
    strategies = forecasting_settings.exog_strategies
    future_values: dict[str, pd.Series] = {}
    for col, avail in availability.items():
        if col not in train_pool.columns:
            continue
        if avail == "known_future":
            future_values[col] = test_df[col].reset_index(drop=True)
        else:
            strat = strategies.per_column.get(col, strategies.default_unknown_future)
            preds_col, _ = extend_exog(train_pool[col], horizon, strat, freq)
            future_values[col] = preds_col.reset_index(drop=True)
    if not future_values:
        return None
    return align_val_exog_index(
        pd.DataFrame(future_values),
        series_dict,
        train_len=len(next(iter(series_dict.values()))),
        dt_col=dt_col,
        freq=freq,
    )


def _forecast_champion_on_test(
    champion: dict,
    champion_model_path: Path,
    train_pool: pd.DataFrame,
    test_path: Path,
    task_metadata: dict[str, Any],
    forecasting_settings: ForecastingSettings,
    metric: str,
) -> tuple[dict[str, float], list[dict]]:
    """Forecast the retrained champion across the held-out test horizon.

    Returns (test_metrics, test_preview) where test_preview is
    [{"ds": str, "y_true": float, "y_pred": float}, ...] for the chart.
    statsforecast -> predict(h); skforecast -> extend unknown-future exog from
    train history, use actual known-future values, then predict(steps, exog).
    """
    target = task_metadata["target_column"]
    dt_col = task_metadata["datetime_column"]
    sid_cols = task_metadata.get("series_id_columns") or []
    horizon = int(task_metadata["forecast_horizon"])
    freq = task_metadata.get("frequency")

    test_df = pd.read_csv(test_path)
    test_df[dt_col] = pd.to_datetime(test_df[dt_col])
    test_df = test_df.sort_values(dt_col).reset_index(drop=True)
    y_true = test_df[target].to_numpy(dtype=float)

    with champion_model_path.open("rb") as f:
        model = pickle.load(f)

    if _is_statsforecast_model(champion["model_key"]):
        fcst = model.predict(h=horizon).sort_values("ds").reset_index(drop=True)
        model_col = [c for c in fcst.columns if c not in ("unique_id", "ds")][0]
        y_pred = fcst[model_col].to_numpy(dtype=float)
        ds_vals = pd.to_datetime(fcst["ds"]).dt.strftime("%Y-%m-%d").tolist()
    else:
        pool = train_pool.copy()
        pool[dt_col] = pd.to_datetime(pool[dt_col])
        series_dict = _build_series_dict(pool, dt_col, target, sid_cols, freq)
        test_exog = _build_test_exog(
            pool, test_df, task_metadata, forecasting_settings, horizon, freq, series_dict
        )
        preds = model.predict(steps=horizon, exog=test_exog)
        preds = preds.reset_index().rename(columns={"index": "ds"})
        preds["ds"] = pd.to_datetime(preds["ds"])
        y_pred = preds["pred"].to_numpy(dtype=float)
        ds_vals = preds["ds"].dt.strftime("%Y-%m-%d").tolist()

    n = min(len(y_true), len(y_pred))
    if n == 0:
        raise ValueError("test forecast produced no overlapping points")
    y_true, y_pred, ds_vals = y_true[:n], y_pred[:n], ds_vals[:n]
    test_metrics = _fc_metrics(y_true, y_pred)
    test_preview = [
        {"ds": ds_vals[i], "y_true": float(y_true[i]), "y_pred": float(y_pred[i])}
        for i in range(n)
    ]
    return test_metrics, test_preview


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _build_forecast_chart_png(
    train_df: pd.DataFrame,
    val_preview: list[dict],
    dt_col: str,
    target_col: str,
) -> str | None:
    try:
        import base64
        import io

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        train_ds = pd.to_datetime(train_df[dt_col])
        train_y = train_df[target_col].values
        val_ds = pd.to_datetime([p["ds"] for p in val_preview])
        val_true = [p["y_true"] for p in val_preview]
        val_pred = [p["y_pred"] for p in val_preview]

        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(train_ds, train_y, color="#4f46e5", linewidth=1.5, label="Train (actual)")
        ax.plot(val_ds, val_true, color="#6b7280", linewidth=1.5, label="Test (actual)")
        ax.plot(val_ds, val_pred, color="#f97316", linewidth=1.5, linestyle="--", label="Test (predicted)")
        if len(val_ds):
            ax.axvline(val_ds[0], color="#d1d5db", linewidth=1, linestyle=":")
        ax.set_ylabel(target_col, fontsize=9)
        ax.legend(fontsize=8, framealpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception as exc:
        logger.warning(f"[executor] forecast chart generation failed: {exc}")
        return None


def run_training_plan(
    plan: TrainingPlan,
    processed_dataset_path: Path,
    target_column: str,
    task_metadata: dict[str, Any],
    output_dir: Path,
    mlflow_experiment: str,
    random_state: int = 42,
    planner_output: dict | None = None,
) -> TrainingResult:
    metric = plan.metric_to_optimize or DEFAULT_METRIC[plan.problem_type]
    direction = METRIC_DIRECTION[metric]

    if plan.problem_type == "forecasting":
        sid_cols = task_metadata.get("series_id_columns") or []
        if sid_cols:
            raise NotImplementedError(
                "Multi-target panel forecasting (series_id_columns non-empty) is out of "
                "scope for V1. V1 supports single-target forecasting with multiple "
                "exogenous predictor columns. Got series_id_columns="
                f"{sid_cols}"
            )

    profile = build_dataset_profile(processed_dataset_path, task_metadata)

    # Resolve forecasting_settings before any candidate runs (fallback for plans built
    # without the planner, e.g. the benchmark runner / direct-executor tests).
    fs = plan.forecasting_settings
    if fs is None and plan.problem_type == "forecasting":
        _full_df = pd.read_csv(processed_dataset_path)
        fs = ForecastingSettings(
            validation_strategy=resolve_validation_strategy(task_metadata, len(_full_df)),
            exog_strategies=resolve_exog_strategies(
                _full_df, task_metadata, task_metadata.get("frequency")
            ),
        )
        plan = plan.model_copy(update={"forecasting_settings": fs})

    train_pool_path, test_path, split_meta_path = split_dataset(
        processed_dataset_path, task_metadata, output_dir, random_state=random_state
    )
    train_pool = pd.read_csv(train_pool_path)

    # XGBoost and some other models require numeric class labels.
    # Encode the target in-place so all candidate runs and the champion retrain see integers.
    # Keep the encoder so we can apply the same transform to the test set later.
    label_encoder: LabelEncoder | None = None
    if plan.problem_type == "classification" and not pd.api.types.is_numeric_dtype(
        train_pool[target_column]
    ):
        label_encoder = LabelEncoder()
        train_pool[target_column] = label_encoder.fit_transform(train_pool[target_column])
        logger.info(f"[executor] label-encoded target '{target_column}': {list(label_encoder.classes_)}")

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

        selection_score: float | None = None
        if plan.problem_type in ("classification", "regression"):
            champion_path = _retrain_tabular(spec, champion, train_pool, target_column, models_dir)
            forecast_chart_png: str | None = None
        else:
            champion_path = _retrain_forecasting(spec, champion, train_pool, task_metadata, models_dir)
            forecast_chart_png = None

        with mlflow.start_run(run_id=champion["mlflow_run_id"], nested=True):
            mlflow.set_tag("champion", "true")
            mlflow.log_artifact(str(champion_path))

        # Log champion as a proper MLflow model on the parent run so that
        # register_model can find it via runs:/<parent_run_id>/model.
        with champion_path.open("rb") as _f:
            _model_obj = pickle.load(_f)
        mlflow.sklearn.log_model(_model_obj, artifact_path="model")

        mlflow.set_tag("champion_run_id", champion["mlflow_run_id"])

        # Evaluate champion on the held-out test set to get all metrics.
        # Evaluate champion on the held-out test set to get all metrics.
        # For classification/regression: predict on X_test using the pkl model.
        # For forecasting: the validation score from Optuna is already the best
        # available metric (StatsForecast models use temporal splits, not X/y pairs).
        if plan.problem_type in ("classification", "regression"):
            _test_df = pd.read_csv(test_path)
            _X_test = _test_df.drop(columns=[target_column])
            _y_test = _test_df[target_column]
            if label_encoder is not None:
                _y_test = pd.Series(label_encoder.transform(_y_test), index=_y_test.index)
            with champion_path.open("rb") as _f:
                _eval_model = pickle.load(_f)
            if plan.problem_type == "classification":
                all_champion_metrics = _cls_metrics(_y_test, _eval_model.predict(_X_test))
            else:
                all_champion_metrics = _reg_metrics(_y_test, _eval_model.predict(_X_test))
        else:
            selection_score = float(champion["best_score"])
            try:
                all_champion_metrics, _test_preview = _forecast_champion_on_test(
                    champion, champion_path, train_pool, test_path,
                    task_metadata, fs, metric,
                )
                forecast_chart_png = _build_forecast_chart_png(
                    train_pool, _test_preview, task_metadata["datetime_column"], target_column
                )
            except Exception as exc:
                logger.warning(f"[executor] test forecast failed: {exc}")
                all_champion_metrics = {}
                forecast_chart_png = None
        mlflow.log_metrics(all_champion_metrics)
        if plan.problem_type == "forecasting" and selection_score is not None:
            mlflow.log_metric(f"selection_{metric}", selection_score)
        logger.info(f"[executor] champion metrics: {all_champion_metrics}")

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
                "expected_drift": task_metadata.get("expected_drift", "low"),
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
                "tie_breaker_chain": ["complexity_rank", "priority"],
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
            "planner_output": planner_output,
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
        champion_metrics=all_champion_metrics,
        forecast_chart_png=forecast_chart_png,
        selection_score=selection_score,
    )
