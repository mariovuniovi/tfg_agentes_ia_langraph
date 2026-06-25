"""Deterministic validation-strategy policy + plan-level guard rails.

resolve_validation_strategy: picks the ValidationStrategy from capacity (history length + horizon).

validate_forecasting_plan: enforced before any modelling runs. Raises
ValueError or NotImplementedError on capacity / leakage / type violations.
"""
from __future__ import annotations

from typing import Any, Literal

from mlops_agents.contracts.profile import DatasetProfile
from mlops_agents.contracts.training import TrainingPlan, ValidationStrategy
from mlops_agents.forecasting.seasonality import max_season_length
from mlops_agents.utils.logging import get_logger

logger = get_logger(__name__)

# Forecasting capacity heuristics (rule-of-thumb thresholds)
_MIN_TRAIN_ROWS = 30          # smallest tolerable training set size
_WINDOW_SIZE_FLOOR = 50       # rolling window minimum regardless of horizon
_HORIZON_MULTIPLIER = 3       # min training rows = 3 * horizon
_DEFAULT_N_FOLDS = 3          # K=3 balances backtest stability vs. compute
_MAX_FOLDS = 5                # cap backtest folds regardless of how much history is available
_LARGE_SERIES_THRESHOLD = 2000  # train-pool rows above which we cap the window
_MIN_CYCLES_WINDOW = 2          # rolling window must hold >= 2 seasonal cycles


def resolve_validation_strategy(task_metadata: dict[str, Any], n_obs: int) -> ValidationStrategy:
    """Capacity-driven validation strategy. Fold count scales with available history,
    floors at a single split, and hard-errors when even one clean split doesn't fit.

    `n_obs` is the full series length; the held-out test split takes the last `horizon`
    rows, so the validation budget is `n_obs - horizon`.
    """
    horizon = int(task_metadata["forecast_horizon"])
    drift = task_metadata.get("expected_drift", "low")
    train_pool_len = n_obs - horizon
    min_train = max(_HORIZON_MULTIPLIER * horizon, _MIN_TRAIN_ROWS)
    k_max = (train_pool_len - min_train) // horizon
    if k_max < 1:
        raise ValueError(
            f"Not enough history for a single validation split: need >= "
            f"{min_train + 2 * horizon} observations for horizon {horizon}, have {n_obs}."
        )
    k = min(k_max, _MAX_FOLDS)
    if k == 1:
        return ValidationStrategy(type="single_split", n_folds=1, horizon=horizon)
    freq = task_metadata.get("frequency")
    if drift == "high":
        # high drift: rolling window, size resolved downstream by the executor
        return ValidationStrategy(
            type="rolling_window", n_folds=k, horizon=horizon, step_size=horizon
        )
    if train_pool_len > _LARGE_SERIES_THRESHOLD:
        # use train_pool_len (not n_obs): folds are carved from the pool AFTER the
        # final test horizon is removed, matching validate_forecasting_plan's `upper`.
        window = resolve_rolling_window_size(train_pool_len, horizon, k, max_season_length(freq))
        return ValidationStrategy(
            type="rolling_window", n_folds=k, horizon=horizon,
            step_size=horizon, window_size=window,
        )
    return ValidationStrategy(
        type="expanding_window", n_folds=k, horizon=horizon, step_size=horizon
    )


def resolve_rolling_window_size(
    total_history: int,
    horizon: int,
    n_folds: int,
    season_length: int | None,
) -> int:
    """Bounded training window for rolling-window validation.

    Sized to the larger of 3*horizon, 2 seasonal cycles, and a floor — so seasonal
    models keep enough history to estimate their period while each fit stays cheap
    on long series. Capped so the folds still fit in the available history.
    """
    base = max(_HORIZON_MULTIPLIER * horizon, _MIN_CYCLES_WINDOW * (season_length or 0), _WINDOW_SIZE_FLOOR)
    upper = total_history - n_folds * horizon
    return min(base, max(upper, horizon))


def validate_forecasting_plan(
    plan: TrainingPlan,
    task_metadata: dict[str, Any],
    profile: DatasetProfile,
    train_pool_stats: dict[str, Any],
) -> None:
    """Raise ValueError on leakage/capacity/type violations, NotImplementedError
    on panel-specific deferred behavior."""
    fs = plan.forecasting_settings
    if fs is None:
        raise ValueError("Forecasting plan missing forecasting_settings")

    horizon_meta = int(task_metadata["forecast_horizon"])
    vs = fs.validation_strategy

    # (1) horizon match
    if vs.horizon != horizon_meta:
        raise ValueError(
            f"validation_strategy.horizon={vs.horizon} != task_metadata.forecast_horizon={horizon_meta}"
        )

    # (2) n_folds invariant
    if vs.n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {vs.n_folds}")
    if (vs.type == "single_split") != (vs.n_folds == 1):
        raise ValueError(
            f"n_folds=={vs.n_folds} inconsistent with type={vs.type!r}"
        )

    # Build the availability map from task_metadata
    exog_cols_meta = task_metadata.get("exogenous_columns")
    availability: dict[str, str] = {}
    if exog_cols_meta is not None:
        for entry in exog_cols_meta:
            availability[entry["name"]] = entry["future_availability"]

    # (3) per_column keys must be valid + must reference unknown_future columns only
    # (checked before panel guardrail so single_series=True cases get precise errors)
    for col, strat in fs.exog_strategies.per_column.items():
        if exog_cols_meta is not None and col not in availability:
            raise ValueError(
                f"per_column key {col!r} is not an exogenous column declared in task_metadata"
            )
        if availability.get(col) == "known_future" and strat != "known_future":
            raise ValueError(
                f"Cannot override known_future column {col!r} with strategy {strat!r}"
            )

    # (4) panel guardrail — V1 is single-target only
    single_series = bool(train_pool_stats.get("single_series", True))
    if not single_series:
        raise NotImplementedError(
            "Multi-target panel forecasting is out of scope for V1. "
            "Use a single-target dataset with optional exogenous predictor columns."
        )

    # (5) capacity check
    if single_series:
        total_len = int(train_pool_stats["total_len"])
        min_train_len = max(_HORIZON_MULTIPLIER * horizon_meta, _MIN_TRAIN_ROWS)
        required = vs.n_folds * horizon_meta + min_train_len
        if total_len < required:
            raise ValueError(
                f"Not enough history for {vs.n_folds}-fold backtesting: "
                f"need >={required} rows, have {total_len}"
            )

    # (6) rolling_window window_size sanity
    if vs.type == "rolling_window":
        if vs.window_size is None:
            raise ValueError(
                "rolling_window plan has window_size=None; resolve via "
                "validation_policy.resolve_rolling_window_size() before validating"
            )
        upper = train_pool_stats["total_len"] - vs.n_folds * horizon_meta
        if not (horizon_meta <= vs.window_size <= upper):
            raise ValueError(
                f"rolling window_size={vs.window_size} must be in "
                f"[{horizon_meta}, {upper}]"
            )
