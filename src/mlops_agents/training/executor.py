"""Deterministic multi-candidate training executor."""
from __future__ import annotations

import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from mlops_agents.config.settings import settings
from mlops_agents.contracts.training import (
    ForecastingSettings,
    TrainingPlan,
    TrainingResult,
)
from mlops_agents.models.loader import get_model
from mlops_agents.training.exog_policy import resolve_exog_strategies
from mlops_agents.training.experience_record import build_task_id, write_experience_record
from mlops_agents.training.forecasting_runner import (
    build_forecast_chart_png,
    forecast_champion_on_test,
    resolve_exog_availability,
    retrain_forecasting,
    run_candidate_forecasting,
)
from mlops_agents.training.profiler import build_dataset_profile
from mlops_agents.training.splitter import split_dataset
from mlops_agents.training.tabular_runner import (
    cls_metrics,
    reg_metrics,
    retrain_tabular,
    run_candidate_classification,
    run_candidate_regression,
)
from mlops_agents.training.trial_budget import deterministic_trials
from mlops_agents.training.validation_policy import resolve_validation_strategy
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


# ---------------------------------------------------------------------------
# Champion selection
# ---------------------------------------------------------------------------


def _pick_champion(results: list[dict[str, Any]], direction: str, tol: float) -> dict[str, Any]:
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
    planner_output: dict[str, Any] | None = None,
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

    # Drop identifier columns (schema unique:true, excluding target/datetime/series_id)
    # from tabular features — they are row keys, not predictors. Forecasting keeps its
    # temporal index untouched.
    id_cols: list[str] = []
    if plan.problem_type in ("classification", "regression"):
        id_cols = [c for c in (task_metadata.get("id_columns") or []) if c in train_pool.columns]
        if id_cols:
            train_pool = train_pool.drop(columns=id_cols)
            logger.info(f"[executor] dropped id columns from features: {id_cols}")

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

    allocations = {c.model_key: deterministic_trials(c.model_key) for c in plan.candidates}

    mlflow.set_experiment(mlflow_experiment)
    candidate_results: list[dict[str, Any]] = []
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    dataset_name = task_metadata.get("name", "unknown")
    with mlflow.start_run(run_name=f"pipeline_{ts}") as parent:
        mlflow.set_tag("dataset_name", dataset_name)
        parent_run_id = parent.info.run_id

        for cand in sorted(plan.candidates, key=lambda c: c.priority):
            with mlflow.start_run(run_name=cand.model_key, nested=True) as child:
                if plan.problem_type == "classification":
                    res = run_candidate_classification(
                        cand, train_pool, target_column,
                        allocations[cand.model_key], metric, direction,
                    )
                elif plan.problem_type == "regression":
                    res = run_candidate_regression(
                        cand, train_pool, target_column,
                        allocations[cand.model_key], metric, direction,
                    )
                else:
                    res = run_candidate_forecasting(
                        cand, train_pool, task_metadata,
                        allocations[cand.model_key], metric, direction,
                        # cast: fs is resolved above for every forecasting plan
                        forecasting_settings=cast(ForecastingSettings, fs),
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
            champion_path = retrain_tabular(spec, champion, train_pool, target_column, models_dir)
            forecast_chart_png: str | None = None
        else:
            champion_path = retrain_forecasting(spec, champion, train_pool, task_metadata, models_dir)
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
            _X_test = _test_df.drop(columns=[target_column, *id_cols], errors="ignore")
            _y_test = _test_df[target_column]
            if label_encoder is not None:
                _y_test = pd.Series(label_encoder.transform(_y_test), index=_y_test.index)
            with champion_path.open("rb") as _f:
                _eval_model = pickle.load(_f)
            if plan.problem_type == "classification":
                all_champion_metrics = cls_metrics(_y_test, _eval_model.predict(_X_test))
            else:
                all_champion_metrics = reg_metrics(_y_test, _eval_model.predict(_X_test))
        else:
            selection_score = float(champion["best_score"])
            try:
                all_champion_metrics, _test_preview = forecast_champion_on_test(
                    champion, champion_path, train_pool, test_path,
                    # cast: fs is resolved above for every forecasting plan
                    task_metadata, cast(ForecastingSettings, fs), metric,
                )
                forecast_chart_png = build_forecast_chart_png(
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

            availability = resolve_exog_availability(list(train_pool.columns), task_metadata)
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
