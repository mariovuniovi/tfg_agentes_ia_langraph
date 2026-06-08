"""Deterministic per-column exogenous-extension policy.

Decides how each unknown-future exog column is projected forward, so the LLM
planner no longer chooses it. known_future columns use their actual future
values (handled downstream) and are omitted from the per-column map.
"""
from __future__ import annotations

import pandas as pd

from mlops_agents.contracts.training import ExogStrategySettings
from mlops_agents.training.profiler import detect_series_structure


def resolve_exog_strategies(
    df: pd.DataFrame, task_metadata: dict, freq: str | None
) -> ExogStrategySettings:
    declared = task_metadata.get("exogenous_columns") or []
    per_column: dict[str, str] = {}
    for entry in declared:
        col, avail = entry["name"], entry["future_availability"]
        if avail != "unknown_future" or col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            per_column[col] = "naive_carry"   # user-declared exog dtype is not guaranteed numeric
            continue
        seasonal, trend, _ = detect_series_structure(df[col].astype(float), freq)
        per_column[col] = "ets" if (seasonal or trend) else "naive_carry"
    return ExogStrategySettings(per_column=per_column, default_unknown_future="naive_carry")
