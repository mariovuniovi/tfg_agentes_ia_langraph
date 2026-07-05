"""Forecasting candidate runner — everything time-series-specific.

Owns the forecasting candidate loop (Optuna search over temporal folds),
frequency-aware seasonality narrowing, leakage-safe exogenous handling
(known-future actuals vs unknown-future extension), champion retraining,
champion evaluation on the held-out test horizon, and the forecast chart PNG.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from mlops_agents.contracts.profile import DatasetProfile
from mlops_agents.contracts.training import (
    ForecastingSettings,
    TrainingPlan,
    TrainingPlanCandidate,
    ValidationStrategy,
)
from mlops_agents.forecasting.seasonality import max_season_length, season_length_grid
from mlops_agents.models.factories import FACTORY_REGISTRY
from mlops_agents.models.loader import SearchSpaceSpec, get_model
from mlops_agents.models.search_spaces import build_suggest_fn
from mlops_agents.training.exog_extender import align_val_exog_index, extend_exog
from mlops_agents.training.override_validation import narrow_search_space
from mlops_agents.training.trial_budget import make_sampler
from mlops_agents.training.validation_folds import iter_folds
from mlops_agents.training.validation_policy import (
    resolve_rolling_window_size,
    validate_forecasting_plan,
)
from mlops_agents.utils.logging import get_logger

if TYPE_CHECKING:
    from mlops_agents.training.exog_extender import Strategy

logger = get_logger(__name__)


def narrow_seasonality_to_freq(
    spec: SearchSpaceSpec, freq: str | None, model_key: str, n_obs: int
) -> SearchSpaceSpec:
    """Replace a categorical season_length grid with the model's frequency-aware,
    length-pruned candidate periods (see seasonality.season_length_grid).

    Known frequencies are authoritative even after a planner/user search-space
    override. Unknown frequencies, non-categorical spaces, and models without a
    season_length parameter keep their original search space.
    """
    param = spec.params.get("season_length")
    grid = season_length_grid(model_key, freq, n_obs)
    if grid is None or param is None or param.type != "categorical":
        return spec
    # cast: categorical params always define choices (SearchParamSpec convention)
    if list(cast("list[Any]", param.choices)) == grid:
        return spec
    new_params = dict(spec.params)
    new_params["season_length"] = param.model_copy(update={"choices": grid})
    return spec.model_copy(update={"params": new_params})


def min_fold_train_len(vs: ValidationStrategy, train_pool_len: int) -> int:
    """Smallest training set any validation fold will use.

    Season pruning must use this, not the full pool: a capped rolling window (or the
    first expanding fold) can be far shorter than the pool, so a period the pool
    could nominally estimate may be unestimable in the folds that actually run.
    """
    if vs.type == "rolling_window" and vs.window_size:
        return vs.window_size
    if vs.type == "expanding_window":
        return train_pool_len - (vs.n_folds - 1) * vs.horizon
    return train_pool_len  # single_split


def fc_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    out: dict[str, float] = {"rmse": rmse, "mae": mae}
    if (y_true != 0).all():
        out["mape"] = float(np.mean(np.abs((y_true - y_pred) / y_true)))
    out["smape"] = float(
        np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-10))
    )
    return out


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


def build_series_dict(
    df: pd.DataFrame, dt_col: str, target: str, sid_cols: list[str], freq_hint: str | None = None
) -> dict[str, pd.Series]:
    """Build series_dict for skforecast with explicit freq or RangeIndex fallback."""
    def _prep(s: pd.Series) -> pd.Series:
        s = s.sort_index()
        for freq in ([freq_hint] if freq_hint else []) + ([pd.infer_freq(s.index)] if pd.infer_freq(s.index) else []):  # type: ignore[list-item, arg-type]  # dt_col is to_datetime'd upstream so the index is datetime-like; the truthiness check filters None
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
            # cast: single-column groupby keys are the series-id values (strings)
            cast("str", sid): _prep(g.set_index(dt_col)[target])
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


def resolve_exog_availability(df_columns: list[str], task_metadata: dict[str, Any]) -> dict[str, str]:
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


def run_candidate_forecasting(
    candidate: TrainingPlanCandidate,
    train_pool: pd.DataFrame,
    task_metadata: dict[str, Any],
    n_trials: int,
    metric: str,
    direction: str,
    forecasting_settings: ForecastingSettings,
    profile: DatasetProfile,
) -> dict[str, Any]:
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
        forecasting_settings=forecasting_settings,
    )

    # Resolve auto window_size for rolling_window BEFORE validation
    vs = forecasting_settings.validation_strategy
    if vs.type == "rolling_window" and vs.window_size is None:
        vs = vs.model_copy(update={
            "window_size": resolve_rolling_window_size(
                len(pool), horizon, vs.n_folds, season_length=max_season_length(freq),
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
    availability = resolve_exog_availability(list(pool.columns), task_metadata)
    exog_columns = list(availability.keys())
    strategies = forecasting_settings.exog_strategies

    exog_cache: dict[tuple[str, str, int, str], pd.Series] = {}

    def fit_score(params: dict[str, Any]) -> tuple[float, list[float], list[dict[str, Any]]]:
        fold_scores: list[float] = []
        fold_failures: list[dict[str, Any]] = []

        for fold_id, (train_idx, val_idx) in enumerate(iter_folds(pool, vs, dt_col, sid_cols)):
            cand_train = pool.loc[train_idx].reset_index(drop=True)
            cand_val = pool.loc[val_idx].reset_index(drop=True)

            if is_stat:
                # statsforecast path: ignores exog (existing behavior). series_length is
                # the fold's actual training size, so the AutoARIMA approximation gate
                # reflects this fit (small bounded windows -> exact; long fits -> CSS).
                fit_metadata = {**task_metadata, "series_length": len(cand_train)}
                sf = factory({"task_metadata": fit_metadata, "params": params})
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
                    score = fc_metrics(val_s["y"].values[:n], fct_s[model_col].values[:n])[metric]
                else:
                    score = fc_metrics(merged["y"].values, merged[model_col].values)[metric]
                fold_scores.append(score)
                continue

            # ── Skforecast path with leakage-safe exog ────────────
            forecaster = factory({"task_metadata": task_metadata, "params": params})
            series_dict = build_series_dict(cand_train, dt_col, target, sid_cols, freq)

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
                    # cast: avail != "known_future" here, so strat is a Strategy literal
                    preds, fail = extend_exog(cand_train[col], horizon, cast("Strategy", strat), freq)
                    future_values[col] = preds
                    exog_cache[cache_key] = preds
                    if fail is not None:
                        fold_failures.append(fail | {"fold_id": fold_id, "column": col})

            used_cols = list(future_values.keys())
            train_exog = _align_train_exog_index(cand_train[used_cols], series_dict) if used_cols else None
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
                score = fc_metrics(
                    val_long["y_true"].values, preds["pred"].values[: len(val_long)],
                )[metric]
            else:
                score = fc_metrics(joined["y_true"].values, joined["pred"].values)[metric]
            fold_scores.append(score)

        return float(np.mean(fold_scores)), fold_scores, fold_failures

    narrowed = (
        narrow_search_space(candidate.model_key, candidate.search_space_override)
        if candidate.search_space_override else spec.search_space
    )
    prune_n = min_fold_train_len(vs, len(pool))
    narrowed = narrow_seasonality_to_freq(narrowed, freq, candidate.model_key, prune_n)
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
            sampler, eff_trials = make_sampler(narrowed, n_trials)
            study = optuna.create_study(direction=direction, sampler=sampler)
            study.optimize(objective, n_trials=eff_trials, show_progress_bar=False)
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


def retrain_forecasting(
    spec: Any,
    champion: dict[str, Any],
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
    fit_metadata = {**task_metadata, "series_length": len(train_pool)}
    path = models_dir / f"champion_{champion['model_key']}.pkl"
    if _is_statsforecast_model(champion["model_key"]):
        sf = factory({"task_metadata": fit_metadata, "params": champion["best_params"]})
        sf.fit(_to_sf_format(train_pool, target, dt_col, sid_cols))
        with path.open("wb") as f:
            pickle.dump(sf, f)
        return path

    forecaster = factory({"task_metadata": fit_metadata, "params": champion["best_params"]})
    freq = task_metadata.get("frequency")
    series_dict = build_series_dict(train_pool, dt_col, target, sid_cols, freq)
    if not sid_cols:
        availability = resolve_exog_availability(list(train_pool.columns), task_metadata)
        used_cols = [c for c in availability if c in train_pool.columns]
        train_exog = _align_train_exog_index(train_pool[used_cols], series_dict) if used_cols else None
    else:
        train_exog = None
    forecaster.fit(series=series_dict, exog=train_exog)
    with path.open("wb") as f:
        pickle.dump(forecaster, f)
    return path


def build_test_exog(
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
    availability = resolve_exog_availability(list(train_pool.columns), task_metadata)
    strategies = forecasting_settings.exog_strategies
    future_values: dict[str, pd.Series] = {}
    for col, avail in availability.items():
        if col not in train_pool.columns:
            continue
        if avail == "known_future":
            future_values[col] = test_df[col].reset_index(drop=True)
        else:
            strat = strategies.per_column.get(col, strategies.default_unknown_future)
            # cast: avail != "known_future" here, so strat is a Strategy literal
            preds_col, _ = extend_exog(train_pool[col], horizon, cast("Strategy", strat), freq)
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


def forecast_champion_on_test(
    champion: dict[str, Any],
    champion_model_path: Path,
    train_pool: pd.DataFrame,
    test_path: Path,
    task_metadata: dict[str, Any],
    forecasting_settings: ForecastingSettings,
    metric: str,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
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
        series_dict = build_series_dict(pool, dt_col, target, sid_cols, freq)
        test_exog = build_test_exog(
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
    test_metrics = fc_metrics(y_true, y_pred)
    test_preview = [
        {"ds": ds_vals[i], "y_true": float(y_true[i]), "y_pred": float(y_pred[i])}
        for i in range(n)
    ]
    return test_metrics, test_preview


def build_forecast_chart_png(
    train_df: pd.DataFrame,
    val_preview: list[dict[str, Any]],
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
        train_y: Any = train_df[target_col].values  # ndarray | ExtensionArray union upsets matplotlib stubs
        val_ds = pd.to_datetime([p["ds"] for p in val_preview])
        val_true = [p["y_true"] for p in val_preview]
        val_pred = [p["y_pred"] for p in val_preview]

        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(train_ds, train_y, color="#4f46e5", linewidth=1.5, label="Train (actual)")
        ax.plot(val_ds, val_true, color="#6b7280", linewidth=1.5, label="Test (actual)")
        ax.plot(val_ds, val_pred, color="#f97316", linewidth=1.5, linestyle="--", label="Test (predicted)")
        if len(val_ds):
            ax.axvline(val_ds[0], color="#d1d5db", linewidth=1, linestyle=":")  # type: ignore[arg-type]  # matplotlib accepts Timestamps on date axes; stubs only declare float
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
