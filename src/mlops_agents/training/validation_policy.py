"""Deterministic validation-strategy policy + plan-level guard rails.

select_validation_strategy: picks the right ValidationStrategy from the
dataset profile and task_metadata.

validate_forecasting_plan: enforced before any modelling runs. Raises
ValueError or NotImplementedError on capacity / leakage / type violations.
"""
from __future__ import annotations

from typing import Any

from mlops_agents.contracts.profile import DatasetProfile
from mlops_agents.contracts.training import TrainingPlan, ValidationStrategy
from mlops_agents.utils.logging import get_logger

logger = get_logger(__name__)

# Forecasting capacity heuristics (rule-of-thumb thresholds)
_MIN_TRAIN_ROWS = 30          # smallest tolerable training set size
_WINDOW_SIZE_FLOOR = 50       # rolling window minimum regardless of horizon
_HORIZON_MULTIPLIER = 3       # min training rows = 3 * horizon
_DEFAULT_N_FOLDS = 3          # K=3 balances backtest stability vs. compute


def select_validation_strategy(
    profile: DatasetProfile,
    task_metadata: dict[str, Any],
) -> ValidationStrategy:
    horizon = int(task_metadata["forecast_horizon"])
    history = profile.history_length
    drift = task_metadata.get("expected_drift", "low")

    # Short history wins over drift hints: K-fold with too little data is
    # worse than a single clean split.
    if history in ("very_short", "short"):
        return ValidationStrategy(type="single_split", n_folds=1, horizon=horizon)

    if drift == "high":
        return ValidationStrategy(
            type="rolling_window",
            n_folds=_DEFAULT_N_FOLDS,
            horizon=horizon,
            step_size=horizon,
            window_size=None,
        )
    return ValidationStrategy(
        type="expanding_window",
        n_folds=_DEFAULT_N_FOLDS,
        horizon=horizon,
        step_size=horizon,
    )


def resolve_rolling_window_size(
    total_history: int,
    horizon: int,
    n_folds: int,
    season_length: int | None,
) -> int:
    # MVP: ignore season_length. TODO: max(3*horizon, 2*season_length, 50)
    base = max(_HORIZON_MULTIPLIER * horizon, _WINDOW_SIZE_FLOOR)
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

    # (4) panel guardrail
    single_series = bool(train_pool_stats.get("single_series", True))
    if not single_series:
        if fs.exog_strategies.per_column or fs.exog_strategies.default_unknown_future != "naive_carry":
            raise NotImplementedError(
                "Leakage-safe exogenous extension for multi-target panel data deferred to v2"
            )

    # (5) capacity check (single-series only here; panel handled above)
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
