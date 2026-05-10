"""Deterministic multi-candidate training executor."""
from __future__ import annotations

import json
import pickle
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold

from mlops_agents.config.settings import settings
from mlops_agents.contracts.training import TrainingPlan, TrainingPlanCandidate, TrainingResult
from mlops_agents.models.factories import FACTORY_REGISTRY
from mlops_agents.models.loader import get_model
from mlops_agents.models.search_spaces import build_suggest_fn
from mlops_agents.training.experience import build_task_id, write_experience_record
from mlops_agents.training.override_validation import narrow_search_space
from mlops_agents.training.profiler import build_dataset_profile
from mlops_agents.training.splitter import split_dataset
from mlops_agents.training.trial_budget import allocate_trials

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


def _run_candidate_forecasting(
    candidate: TrainingPlanCandidate,
    train_pool: pd.DataFrame,
    task_metadata: dict[str, Any],
    n_trials: int,
    metric: str,
    direction: str,
) -> dict:
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
            forecaster.fit(series=series_dict)
            # skforecast 0.22: predict returns DataFrame with DatetimeIndex,
            # columns ['level', 'pred']
            preds = forecaster.predict(steps=horizon)
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
    else:
        forecaster = factory({"task_metadata": task_metadata, "params": champion["best_params"]})
        freq = task_metadata.get("frequency")
        series_dict = _build_series_dict(train_pool, dt_col, target, sid_cols, freq)
        forecaster.fit(series=series_dict)
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
            else "temporal_holdout"
        )
        task_id = build_task_id(processed_dataset_path.stem, plan.problem_type, run_idx=1)
        record: dict[str, Any] = {
            "task_id": task_id,
            "problem_type": plan.problem_type,
            "dataset_name": processed_dataset_path.stem,
            "dataset_profile": profile,
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
