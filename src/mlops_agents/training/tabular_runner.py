"""Tabular candidate runners — classification and regression.

Both modalities share the same data shape (feature matrix X + target column y),
the same size-aware validation-strategy selection (single split vs k-fold CV),
the same Optuna search scaffold with a default-params fallback, and the same
full-pool champion retrain.
"""
from __future__ import annotations

import pickle
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from mlops_agents.config.settings import settings
from mlops_agents.contracts.training import TrainingPlanCandidate
from mlops_agents.models.factories import FACTORY_REGISTRY
from mlops_agents.models.loader import get_model
from mlops_agents.models.search_spaces import build_suggest_fn
from mlops_agents.training.override_validation import narrow_search_space
from mlops_agents.training.trial_budget import make_sampler
from mlops_agents.utils.logging import get_logger

logger = get_logger(__name__)


def cls_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def reg_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _select_cls_validation(y: pd.Series) -> tuple[Any, ...]:
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


def _select_reg_validation(y: pd.Series) -> tuple[Any, ...]:
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


def run_candidate_classification(
    candidate: TrainingPlanCandidate,
    train_pool: pd.DataFrame,
    target: str,
    n_trials: int,
    metric: str,
    direction: str,
) -> dict[str, Any]:
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
            score = cls_metrics(y_val, factory(params).fit(X_tr, y_tr).predict(X_val))[metric]
            trial.set_user_attr("fold_scores", [score])
            return float(score)

        fallback_score_fn: Callable[[dict[str, Any]], float] = lambda p: cls_metrics(  # noqa: E731
            y_val, factory(p).fit(X_tr, y_tr).predict(X_val)
        )[metric]
        val_strategy_label = "single_split"
    else:
        n_folds = val_strategy[1]
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        def objective(trial: optuna.Trial) -> float:
            params = suggest_fn(trial)
            scores = [
                cls_metrics(y.iloc[vi], factory(params).fit(X.iloc[ti], y.iloc[ti]).predict(X.iloc[vi]))[metric]
                for ti, vi in skf.split(X, y)
            ]
            trial.set_user_attr("fold_scores", scores)
            return float(np.mean(scores))

        fallback_score_fn = lambda p: float(np.mean([  # noqa: E731
            cls_metrics(y.iloc[vi], factory(p).fit(X.iloc[ti], y.iloc[ti]).predict(X.iloc[vi]))[metric]
            for ti, vi in skf.split(X, y)
        ]))
        val_strategy_label = f"stratified_kfold_{n_folds}"

    try:
        sampler, eff_trials = make_sampler(narrowed, n_trials)
        study = optuna.create_study(direction=direction, sampler=sampler)
        study.optimize(objective, n_trials=eff_trials)
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


def run_candidate_regression(
    candidate: TrainingPlanCandidate,
    train_pool: pd.DataFrame,
    target: str,
    n_trials: int,
    metric: str,
    direction: str,
) -> dict[str, Any]:
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
            score = reg_metrics(y_val, factory(params).fit(X_tr, y_tr).predict(X_val))[metric]
            trial.set_user_attr("fold_scores", [score])
            return float(score)

        fallback_score_fn: Callable[[dict[str, Any]], float] = lambda p: reg_metrics(  # noqa: E731
            y_val, factory(p).fit(X_tr, y_tr).predict(X_val)
        )[metric]
        val_strategy_label = "single_split"
    else:
        n_folds = val_strategy[1]
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        def objective(trial: optuna.Trial) -> float:
            params = suggest_fn(trial)
            scores = [
                reg_metrics(y.iloc[vi], factory(params).fit(X.iloc[ti], y.iloc[ti]).predict(X.iloc[vi]))[metric]
                for ti, vi in kf.split(X)
            ]
            trial.set_user_attr("fold_scores", scores)
            return float(np.mean(scores))

        fallback_score_fn = lambda p: float(np.mean([  # noqa: E731
            reg_metrics(y.iloc[vi], factory(p).fit(X.iloc[ti], y.iloc[ti]).predict(X.iloc[vi]))[metric]
            for ti, vi in kf.split(X)
        ]))
        val_strategy_label = f"kfold_{n_folds}"

    try:
        sampler, eff_trials = make_sampler(narrowed, n_trials)
        study = optuna.create_study(direction=direction, sampler=sampler)
        study.optimize(objective, n_trials=eff_trials)
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


def retrain_tabular(
    spec: Any, champion: dict[str, Any], train_pool: pd.DataFrame, target: str, models_dir: Path
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
